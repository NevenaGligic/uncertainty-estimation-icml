import torch
import torch.nn as nn
import os
import sys

# --- Import PostNet Modules ---
POSTNET_ROOT = os.path.join(os.path.dirname(__file__), "..", "..", "..", "Posterior-Network")
sys.path.append(os.path.abspath(POSTNET_ROOT))
from src_postnet.posterior_networks.PosteriorNetwork import PosteriorNetwork


class PostNet(nn.Module):
    """
    Wrapper for Posterior Network (PostNet).
    Handles Double/Float precision mismatch and Forward pass signature.
    """
    def __init__(self, 
                 num_classes=10, 
                 N=None, # Class Counts (Required by PostNet)
                 input_dims=[28, 28, 1],
                 architecture='conv',
                 latent_dim=6,
                 hidden_dims=[64, 64, 64],
                 kernel_dim=5,
                 density_type='radial_flow',
                 n_density=6,
                 k_lipschitz=None,
                 budget_function='id',
                 batch_size=64,
                 lr=5e-5,
                 loss='UCE',
                 regr=1e-5,
                 seed=123):
        super().__init__()
        
        if N is None:
            raise ValueError("PostNet requires 'N' (samples per class) to be initialized.")

        # --- DIMENSION FIX ---
        # main.py passes (C, H, W) -> (3, 32, 32)
        # PostNet Notebook uses [H, W, C] -> [32, 32, 3] for initialization logic
        c, h, w = input_dims
        postnet_input_dims = [h, w, c]
        input_dims = postnet_input_dims

        # --- Initialize Original Model ---
        self.model = PosteriorNetwork(
            N=N,
            input_dims=input_dims,
            output_dim=num_classes,
            hidden_dims=hidden_dims,
            kernel_dim=kernel_dim,
            latent_dim=latent_dim,
            architecture=architecture,
            k_lipschitz=k_lipschitz,
            no_density=False, 
            density_type=density_type,
            n_density=n_density,
            budget_function=budget_function,
            batch_size=batch_size,
            lr=lr,
            loss=loss,
            regr=regr,
            seed=seed
        )
        
        # --- CRITICAL FIX: Reset Precision ---
        # PosteriorNetwork.__init__ sets torch.set_default_tensor_type(torch.DoubleTensor).
        # This is dangerous for the rest of your script. We must revert it to Float.
        # We also cast the model itself to Float32 to match your dataloaders.
        torch.set_default_tensor_type(torch.FloatTensor)
        self.model.float()
        
    def forward(self, input, soft_output=None, return_output='hard', compute_loss=True):
        """
        Forward pass handling Training vs Evaluation signatures.
        """
        if soft_output is None:
            # --- EVALUATION MODE (Benchmark) ---
            # Benchmark calls model(x). We return 'alpha' (Dirichlet params).
            # We disable loss computation to avoid crashing on missing targets.
            return self.model(input, soft_output=None, return_output='alpha', compute_loss=False)
        else:
            # --- TRAINING MODE (PostNet Train Loop) ---
            # train_postnet calls model(x, y). We pass everything through.
            return self.model(input, soft_output, return_output, compute_loss)

    def load_weights(self, path, device):
        if os.path.exists(path):
            checkpoint = torch.load(path, map_location=device)
            # Handle PostNet's saving format
            if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
                self.model.load_state_dict(checkpoint['model_state_dict'])
            else:
                self.model.load_state_dict(checkpoint)
            
            # Ensure model stays in Float32 after loading
            self.model.float()
            self.model.to(device)
            print(f"Loaded PostNet weights from {path}")
        else:
            print(f"⚠️ PostNet weights not found at {path}")

    def __getattr__(self, name):
        try:
            return super().__getattr__(name)
        except AttributeError:
            return getattr(self.model, name)


class PostNetLoaderWrapper:
    """
    Wraps a standard DataLoader to reshape targets from [Batch] to [Batch, 1].
    Required because PostNet's train.py expects 2D target tensors for scatter_.
    """
    def __init__(self, loader):
        self.loader = loader
        # Forward the 'dataset' attribute which is accessed by train.py
        self.dataset = loader.dataset 
        
    def __iter__(self):
        for x, y in self.loader:
            # Reshape y from [B] to [B, 1] if needed
            if y.dim() == 1:
                y = y.unsqueeze(1)
            yield x, y
            
    def __len__(self):
        return len(self.loader)