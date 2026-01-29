import os
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from src.utils import transform_for_maf, get_weight_path
from nflows.flows.autoregressive import MaskedAutoregressiveFlow
import torchvision.models as models
import torch.nn.utils.spectral_norm as spectral_norm

# --- Import DAEDL Modules ---
# Point to the root of the DAEDL repo
DAEDL_ROOT = os.path.join(os.path.dirname(__file__), "..", "..", "..", "DAEDL")
sys.path.append(os.path.abspath(DAEDL_ROOT))

# Import their official functions
from density_estimation import fit_gda, gmm_forward

class StandardCNN(nn.Module):
    """
    Standard Classifier Backbone. 
    Uses ResNet-18 adapted for small images (CIFAR/MNIST).
    """
    def __init__(self, num_classes=10, input_dims=(1, 28, 28)):
        super().__init__()
        self.num_classes = num_classes
        self.input_dims = tuple(input_dims)
        channels = self.input_dims[0]
        
        # Load standard ResNet18
        # We perform surgery to adapt it for small inputs (28x28 or 32x32)
        self.backbone = models.resnet18(weights=None)
        
        # --- CRITICAL MODIFICATION FOR SMALL IMAGES (CIFAR/MNIST) ---
        # Standard ResNet: Conv7x7 (stride 2) -> MaxPool (stride 2). 
        # This reduces 32x32 -> 8x8 instantly, which is too small.
        # We replace it with Conv3x3 (stride 1) and NO MaxPool.
        
        self.backbone.conv1 = spectral_norm(nn.Conv2d(
            channels,             
            64, 
            kernel_size=3,        
            stride=1,             
            padding=1, 
            bias=False
        ))
        
        # Remove the first maxpool layer (pass-through)
        self.backbone.maxpool = nn.Identity()

        def apply_sn(m):
            if isinstance(m, (nn.Conv2d, nn.Linear)):
                return spectral_norm(m)
            return m
        
        self.backbone.layer1.apply(apply_sn)
        self.backbone.layer2.apply(apply_sn)
        self.backbone.layer3.apply(apply_sn)
        self.backbone.layer4.apply(apply_sn)
            
        # Modify Output Layer for our specific Num Classes
        num_ftrs = self.backbone.fc.in_features
        self.backbone.fc = spectral_norm(nn.Linear(num_ftrs, self.num_classes))

        # Attribute required by DAEDL's fit_gda
        self.feature = None

    def forward(self, x):
        """
        Modified forward pass to save features into self.feature.
        DAEDL's fit_gda function REQUIRES this side-effect.
        """
        x = self.backbone.conv1(x)
        x = self.backbone.bn1(x)
        x = self.backbone.relu(x)
        # x = self.backbone.maxpool(x) # Removed for CIFAR
        
        x = self.backbone.layer1(x)
        x = self.backbone.layer2(x)
        x = self.backbone.layer3(x)
        x = self.backbone.layer4(x)
        
        x = self.backbone.avgpool(x)
        x = torch.flatten(x, 1)
        
        # --- SAVE FEATURE FOR DAEDL GDA ---
        self.feature = x 
        # ----------------------------------
        
        x = self.backbone.fc(x)
        return x

    def get_features(self, x):
        """Used for MAF (MNIST) or manual inspection."""
        self.forward(x) # This populates self.feature
        return self.feature

class nu_EDL(nn.Module):
    """
    Posterior Network / Nu-EDL Implementation.
    Alpha = 1 + N * p(x) * P(y|x)
    """
    def __init__(self, 
                 num_classes=10, 
                 input_dims=(1, 28, 28),
                 maf_hidden_features=1024,
                 maf_num_layers=10,
                 maf_blocks=20,
                 n_train_samples=50000): 

        super().__init__()
        self.n = float(n_train_samples)
        self.input_dims = tuple(input_dims)
        self.num_classes = num_classes
        
        # 1. The Classifier (P(y|x)) - Optimized ResNet18
        self.cnn = StandardCNN(num_classes, input_dims)
        
        # 2. The Density Estimator (p(x))
        # For CIFAR-10: Use GMM (feature space)
        # For MNIST: Use MAF (pixel space)
        if input_dims == (1, 28, 28):
            self.flat_dim = int(np.prod(input_dims))
            self.dataset_mode = 'mnist'
            # MNIST: Use MAF on pixel space
            self.maf = MaskedAutoregressiveFlow(
                features=self.flat_dim, 
                hidden_features=maf_hidden_features, 
                num_layers=maf_num_layers, 
                num_blocks_per_layer=maf_blocks,
                batch_norm_within_layers=True,
                batch_norm_between_layers=True
            )
        elif input_dims == (3, 32, 32):
            self.dataset_mode = 'cifar10'
            self.embedding_dim = 512 # ResNet18 feature dim
            self.gda = None
        else:
            raise ValueError(f"Unsupported input_dims: {input_dims}")

        # NEW: Offset to prevent underflow. 
        # Buffers for Z-Score Normalization
        # We start with Identity (Offset=0, Scale=1)
        self.register_buffer('log_prob_mean', torch.tensor(0.0))
        self.register_buffer('log_prob_std', torch.tensor(1.0))

    def fit_density(self, train_loader, device):
        """
        Specific method for CIFAR-10 to fit GDA.
        """
        if self.dataset_mode != 'cifar10':
            print("fit_density called but model is not in GDA mode (MNIST?). Skipping.")
            return

        print("Fitting DAEDL Density Estimator (GDA) for nu-EDL...")
        self.cnn.eval()
        self.cnn.to(device)

        # Call DAEDL's official function
        # Note: We pass self.cnn because StandardCNN has the .feature attribute now
        self.gda, p_z_train = fit_gda(
            self.cnn, 
            train_loader, 
            self.num_classes, 
            self.embedding_dim, 
            device
        )
        
        # 2. Calculate Z-Score Stats immediately from training log-likelihoods
        # p_z_train contains the log_probs of all training samples
        mean_lp = p_z_train.mean().item()
        std_lp = p_z_train.std().item()
        
        self.log_prob_mean.fill_(mean_lp)
        self.log_prob_std.fill_(std_lp if std_lp > 1e-6 else 1.0)
        
        print(f"✅ GDA Fitted.")
        print(f"   Stats: Mean {mean_lp:.4f} | Std {std_lp:.4f}")

    def calibrate_density(self, loader, device):
        """
        MNIST only.
        Runs one pass over data to find the max log_prob.
        Shifts log_prob so that max_density is 1.0 (log_prob = 0.0).
        This prevents likelihood underflow (exp(-2000) -> 0).
        """

        if self.dataset_mode == 'cifar10':
            print("--- Calibration skipped for GDA (Handled in fit_density) ---")
            return

        print("--- Calibrating Density Offset for Nu-EDL ---")
        
        self.maf.eval()
        self.cnn.eval()
        
        # Collect stats
        log_probs = []
        with torch.no_grad():
            for i, (x, _) in enumerate(loader):
                x = x.to(device)
                maf_input = transform_for_maf(x)
                lp = self.maf.log_prob(maf_input)
                log_probs.append(lp.cpu().numpy())
                # Limit samples to avoid OOM on huge datasets
                if i > 200: break

        all_lp = np.concatenate(log_probs)
        mean_lp = np.mean(all_lp)
        std_lp = np.std(all_lp)
        
        # Update Buffers
        self.log_prob_mean.fill_(mean_lp)
        # Prevent divide by zero if std is suspiciously low
        self.log_prob_std.fill_(std_lp if std_lp > 1e-6 else 1.0)
        
        print(f"   Train Stats: Mean {mean_lp:.4f} | Std {std_lp:.4f}")
        print(f"   Normalization: (x - {mean_lp:.2f}) / {std_lp:.2f}")

    def forward(self, x):
        # 1. P(y|x): Standard Softmax from ResNet
        logits = self.cnn(x)
        probs = F.softmax(logits, dim=1)
        
        # 2. p(x): Density from MAF
        if self.dataset_mode == 'cifar10':
            # Safety check if GDA is not fitted yet
            if self.gda is None:
                return torch.exp(logits) + 1e-6
            # GDA Logic
            with torch.no_grad():
                log_probs_class = gmm_forward(self.cnn, self.gda, x)
                log_prob = torch.logsumexp(log_probs_class, dim=-1) # [B]
        else:
            # MAF Logic
            x_maf = transform_for_maf(x)
            log_prob = self.maf.log_prob(x_maf) # [B]

        # --- ROBUST SCALING (Z-Score) ---
        # This maps ID data to roughly range [-2, 2]
        # e^2 = 7.3, e^-2 = 0.13. Perfectly safe for float32.
        log_prob_scaled = (log_prob - self.log_prob_mean) / self.log_prob_std
        
        # Convert log_prob to likelihood (e^log_prob)
        likelihood = torch.exp(log_prob_scaled).unsqueeze(1) # [B, 1]
        
        # # 3. Formula: alpha = 1 + n * p(x) * P(y|x)
        evidence = self.n * likelihood * probs

        # --- Single Components ---
        # # Ablation 1a
        # evidence = torch.full_like(probs, self.n)
        # # Ablation 1b
        # evidence = likelihood.expand_as(probs)
        # # Ablation 1c
        # evidence = probs

        # --- Pair Combinations ---
        # # Ablation 2a
        # evidence = self.n * likelihood.expand_as(probs)
        # # Ablation 2b
        # evidence = self.n * probs
        # # Ablation 2c
        # evidence = likelihood * probs


        alpha = evidence + 1
        
        return alpha
        
    def calculate_uncertainties(self, x):
        with torch.no_grad():
            alpha = self.forward(x)
            S = torch.sum(alpha, dim=1, keepdim=True)
            
            prediction = alpha / S
            
            variance = alpha * (S - alpha) / (S * S * (S + 1))
            epistemic = torch.sum(variance, dim=1)
            
            aleatoric = -torch.sum(prediction * torch.log(prediction + 1e-9), dim=1)
            
            K = alpha.shape[1]
            total_u = K / S.squeeze()
            
            return prediction, epistemic, aleatoric, total_u

    def load_cnn_weights(self, cnn_filename, device='cpu'):
        """Loads weights only into the EDL sub-component."""
        
        cnn_path = get_weight_path(cnn_filename)
        self.cnn.load_state_dict(torch.load(cnn_path, map_location=device))
        self.cnn.to(device)
        self.cnn.eval()

    def load_maf_weights(self, maf_filename, device='cpu'):
        """Loads weights only into the MAF sub-component (for MNIST only)."""
        if self.dataset_mode == 'mnist':
            # Only load weights for MAF (MNIST), not for GMM (CIFAR-10)
            maf_path = get_weight_path(maf_filename)
            self.maf.load_state_dict(torch.load(maf_path, map_location=device))
            self.maf.to(device)
            self.maf.eval()
        else:
            print("Note: GMM (CIFAR-10) doesn't use saved weights - it needs to be fitted.")

    def load_pretrained_weights(self, edl_filename, maf_filename, device='cpu'):
        # This function can now call the granular loaders:
        self.load_cnn_weights(edl_filename, device)
        self.load_maf_weights(maf_filename, device)