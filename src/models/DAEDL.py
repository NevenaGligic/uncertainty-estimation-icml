import torch
import torch.nn as nn
import os
import sys

# --- Import DAEDL Modules ---
# Point to the root of the DAEDL repo
DAEDL_ROOT = os.path.join(os.path.dirname(__file__), "..", "..", "..", "DAEDL")
sys.path.append(os.path.abspath(DAEDL_ROOT))

# Import their official functions
from utility import load_model  # <--- Using their loader
from density_estimation import fit_gda, gmm_forward

class DAEDL(nn.Module):
    """
    Wrapper for Density Aware Evidential Deep Learning.
    
    Combines the backbone (loaded via DAEDL's utility.load_model)
    with the GDA density estimator.
    """
    
    def __init__(self, 
                 num_classes=10, 
                 dataset='MNIST',
                 input_dims=(1, 28, 28),
                 dropout_rate=0.5,
                 device='cuda'):
        super().__init__()
        
        self.num_classes = num_classes
        self.device = device
        
        # --- 1. Load Backbone using Official Function ---
        # This ensures we get the exact architecture (ConvLinSeq/VGG) they used
        # We set pretrained=False because we handle loading/training in main.py
        # index=0 is a dummy value for pretrained loading (ignored here)
        print(f"DAEDL Wrapper: Loading backbone for {dataset}...")
        self.model = load_model(
            ID_dataset=dataset, 
            pretrained=False, 
            index=0, 
            dropout_rate=dropout_rate, 
            device=device
        )
        
        # Determine embedding dimension for GDA
        # We infer this by passing a dummy input, as their code doesn't explicitly expose it
        with torch.no_grad():
            self.model.eval()
            dummy_shape = (1, *input_dims)
            self.model(torch.randn(dummy_shape).to(device))
            self.embedding_dim = self.model.feature.shape[1]
            print(f"DAEDL Wrapper: Inferred embedding dim: {self.embedding_dim}")

        # Storage for the GDA (Gaussian Discriminant Analysis) estimator
        self.gda = None
        self.p_z_min = None
        self.p_z_max = None

    def fit_density(self, train_loader):
        """
        Fits the GDA on training data.
        Equivalent to 'fit_gda' call in DAEDL's main.py.
        """
        print("Fitting DAEDL Density Estimator (GDA)...")
        self.model.eval()
        
        # Their fit_gda function handles embedding extraction and GMM fitting
        # Returns: gda (model), p_z_train (log-likelihoods of training data)
        self.gda, p_z_train = fit_gda(
            self.model, 
            train_loader, 
            self.num_classes, 
            self.embedding_dim, 
            self.device
        )
        
        # Store min/max for normalization (crucial for inference)
        # See logic in conf_calibration.py
        self.p_z_min = p_z_train.min().item()
        self.p_z_max = p_z_train.max().item()
        print(f"GDA Fitted. Density Range: [{self.p_z_min:.4f}, {self.p_z_max:.4f}]")

    def forward(self, x):
        """
        Forward pass implementing the Density-Aware logic.
        """
        # 1. Get Backbone Logits
        z = self.model(x)
        
        # 2. If training or GDA not ready, return standard exponential evidence
        # This matches 'eval_daedl' in their train.py
        if self.training or self.gda is None:
            return torch.exp(z) + 1e-6
        
        # 3. Density-Aware Inference
        # Replicates logic from 'conf_calibration.py' lines 28-36
        with torch.no_grad():
            log_probs = gmm_forward(self.model, self.gda, x)
            p_z = torch.logsumexp(log_probs, dim=-1)
            
            # Normalize density
            p_z = torch.clamp(p_z, min=self.p_z_min)
            p_z_norm = (p_z - self.p_z_min) / (self.p_z_max - self.p_z_min + 1e-12)
            
        # alpha = exp(logits * density)
        alpha = torch.exp(z * p_z_norm.view(-1, 1))
        return alpha

    def load_weights(self, path):
        if os.path.exists(path):
            state = torch.load(path, map_location=self.device)
            # Handle if saved as full dict or just state dict
            if isinstance(state, dict) and 'model_state_dict' in state:
                self.model.load_state_dict(state['model_state_dict'])
            else:
                self.model.load_state_dict(state)
            print(f"Loaded DAEDL weights from {path}")
        else:
            print(f"⚠️ DAEDL weights not found at {path}")

    def __getattr__(self, name):
        try:
            return super().__getattr__(name)
        except AttributeError:
            return getattr(self.model, name)