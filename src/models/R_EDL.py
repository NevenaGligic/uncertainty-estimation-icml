import torch
import torch.nn as nn
import os
import sys

# Import ModifiedEvidentialNet from ICLR2024-REDL
REDL_ROOT = os.path.join(os.path.dirname(__file__), "..", "..", "..", "ICLR2024-REDL", "code_classical")
sys.path.append(os.path.abspath(REDL_ROOT))
from models.ModifiedEvidentialN import ModifiedEvidentialNet  # type: ignore


class REDL(nn.Module):
    """
    Wraps ModifiedEvidentialNet to output evidence (compatible with standard EDL evaluation pipelines).
    
    The underlying ModifiedEvidentialNet outputs alpha = evidence + lamb2.
    This wrapper converts alpha to evidence by subtracting lamb2 (typically 1.0) for compatibility
    with evaluation pipelines that expect evidence and compute alpha = evidence + 1.
    """
    
    def __init__(self, 
                 num_classes=10,
                 input_dims=(1, 28, 28),
                 architecture='conv',
                 batch_size=128,
                 lr=1e-3,
                 lamb1=1.0,
                 lamb2=1.0,
                 fisher_c=1.0,
                 kernel_dim=5,
                 hidden_dims=[64, 64, 64],
                 clf_type='softplus',
                 seed=123):
        """
        Initialize R-EDL model.
        
        Args:
            num_classes: Number of output classes
            input_dims: Input dimensions tuple (e.g., (1, 28, 28) for MNIST, (3, 32, 32) for CIFAR10).
                       If None, will be inferred from sample_input or architecture.
            sample_input: Sample input tensor to infer input_dims from. Shape should be [batch, ...].
                         If provided, input_dims will be inferred from sample_input.shape[1:].
            architecture: Architecture type ('conv', 'linear', or 'vgg'). If None, will be inferred from input_dims.
            batch_size: Batch size for training
            lr: Learning rate
            lamb1: Lambda1 hyperparameter for R-EDL loss
            lamb2: Lambda2 hyperparameter for R-EDL loss (typically 1.0, but 0.1 for MNIST)
            fisher_c: Fisher information matrix constant (0.0 for MNIST, 1.0 for others)
            kernel_dim: Kernel dimension for conv architecture (5 for MNIST, None for default)
            hidden_dims: Hidden dimensions for the architecture ([64, 64, 64] by default)
            clf_type: Classifier type ('softplus')
            seed: Random seed for initialization
        """
        super().__init__()
        
        # Create the underlying ModifiedEvidentialNet
        self.model = ModifiedEvidentialNet(
            input_dims=input_dims,
            output_dim=num_classes,
            architecture=architecture,
            hidden_dims=hidden_dims,
            kernel_dim=kernel_dim,
            batch_size=batch_size,
            lr=lr,
            loss='MEDL',  # Note: config files say "MEDL" but code uses "IEDL" - they appear to be the same
            clf_type=clf_type,
            fisher_c=fisher_c,
            lamb1=lamb1,
            lamb2=lamb2,
            seed=seed
        )
        
        self.num_classes = num_classes
        self.input_dims = input_dims
    
    def forward(self, x):
        """
        Forward pass that returns alpha directly.
        
        Args:
            x: Input tensor
            
        Returns:
            alpha: Dirichlet parameters [batch_size, num_classes]
        """
        # ModifiedEvidentialNet already returns alpha = evidence + lamb2
        alpha = self.model(x, return_output='alpha', compute_loss=False)
        return alpha
    
    def load_weights(self, weight_path, device='cpu'):
        """
        Loads weights from a checkpoint file (supports both Hugging Face and local paths).
        Compatible with both the original R-EDL checkpoint format (with 'model_state_dict' key)
        and the simplified state_dict format.
        
        Args:
            weight_path: Path to the weight file (local or Hugging Face)
            device: Device to load the weights on
        """
        from ..utils import get_weight_path
        
        # Get the weight path (handles Hugging Face download if needed)
        full_path = get_weight_path(weight_path)
        
        # Load the checkpoint
        checkpoint = torch.load(full_path, map_location=device)
        
        # Handle both checkpoint formats:
        # 1. Original R-EDL format: {'model_state_dict': ..., 'epoch': ..., ...}
        # 2. Simple state_dict format: just the state_dict
        if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
            self.model.load_state_dict(checkpoint['model_state_dict'])
        else:
            # Assume it's a simple state_dict
            self.model.load_state_dict(checkpoint)
        
        self.model.to(device)
        self.model.eval()
        print(f"✅ Loaded R-EDL weights from {full_path}")
    
    def __getattr__(self, name):
        """
        Delegate attribute access to the wrapped model for training-related attributes
        (like optimizer, scheduler, step method, etc.)
        """
        try:
            return super().__getattr__(name)
        except AttributeError:
            return getattr(self.model, name)

