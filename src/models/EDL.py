import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from ..utils import get_weight_path
import torch.nn.utils.spectral_norm as spectral_norm

# ----------------------------------------
# EDL Model Definition (Prior Network Head)
# ----------------------------------------

class EDL_old(nn.Module):
    """A LeNet model with the specific layer sizes from the paper."""
    def __init__(self, num_classes=10, input_dims=(1, 28, 28)):
        super(EDL_old, self).__init__()
        self.num_classes = num_classes
        self.input_dims = tuple(input_dims)
        
        # Determine Architecture based on Input Dimensions
        if self.input_dims == (1, 28, 28): # MNIST
            filters = [20, 50]
            hidden_units = 500
            # MNIST Calculation:
            # Conv1 (Pad 2): 28 -> 28. Pool: 14.
            # Conv2 (Pad 0): 14 -> 10. Pool: 5.
            # Flatten: 50 * 5 * 5 = 1250
            flatten_dim = 50 * 5 * 5
            
        elif self.input_dims == (3, 32, 32): # CIFAR-10
            filters = [192, 192] # "192 filters at each convolutional layer"
            hidden_units = 1000  # "1000 hidden units"
            # CIFAR Calculation (Assuming same LeNet padding logic):
            # Conv1 (Pad 2): 32 -> 32. Pool: 16.
            # Conv2 (Pad 0): 16 -> 12. Pool: 6.
            # Flatten: 192 * 6 * 6 = 6912
            flatten_dim = 192 * 6 * 6
            
        else:
            raise ValueError(f"EDL: Unknown architecture for input_dims {input_dims}")
            
        # Convolutional layers
        self.conv_layers = nn.Sequential(
            # First block: Conv -> ReLU -> MaxPool
            nn.Conv2d(in_channels=input_dims[0], out_channels=filters[0], kernel_size=5, stride=1, padding=2),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            # Second block: Conv -> ReLU -> MaxPool
            nn.Conv2d(in_channels=filters[0], out_channels=filters[1], kernel_size=5, stride=1, padding=0),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        # Fully connected layers with a single 500-unit hidden layer
        self.fc_layers = nn.Sequential(
            nn.Linear(flatten_dim, hidden_units),
            nn.ReLU(),
            nn.Dropout(p=0.5),
            nn.Linear(hidden_units, self.num_classes)
        )
    
    def load_edl_weights(self, edl_filename, device='cpu'):
        """Loads weights only into the EDL sub-component."""
        
        edl_path = get_weight_path(edl_filename)
        self.load_state_dict(torch.load(edl_path, map_location=device))
        self.to(device)
        self.eval()

    def forward(self, x):
        x = self.conv_layers(x)
        x = x.view(x.size(0), -1)
        x = self.fc_layers(x)
        # Apply ReLU to get non-negative evidence, as implied by the paper
        evidence = F.relu(x)
        alpha = evidence +1
        return alpha

    def alpha_from_output(self, evidence):
        """Converts raw evidence output to Dirichlet parameters (alpha)."""
        return evidence + 1

    def calculate_uncertainties(self, x):
        """
        Predicts class probabilities and calculates standard EDL total uncertainty (K/S).
        Equivalent to your original EDL_uncertainty_fun.
        """
        # Ensure we are not tracking gradients for inference
        with torch.no_grad():
            alpha = self.forward(x)
            
            # Sum of evidence/Dirichlet strength
            S = torch.sum(alpha, dim=1, keepdim=True)
            
            # 1. Prediction (Mean of Dirichlet)
            probs = alpha / S
            
            # 2. Total Uncertainty (K / S)
            uncertainty = self.num_classes / S.squeeze() 
            
            return probs, uncertainty


class EDL_CIFAR(nn.Module):
    """
    EDL Model using a ResNet-18 backbone adapted for small images (CIFAR/MNIST).
    Replaces the older LeNet architecture for higher performance.
    """
    def __init__(self, num_classes=10, input_dims=(1, 28, 28)):
        super(EDL_CIFAR, self).__init__()
        self.num_classes = num_classes
        self.input_dims = tuple(input_dims)
        channels = self.input_dims[0]
        
        # 1. Load Standard ResNet-18
        self.backbone = models.resnet18(weights=None)
        
        # 2. Adapt for Small Images (MNIST/CIFAR)
        # Standard ResNet uses a 7x7 conv with stride 2, which destroys 
        # resolution on 32x32 images. We replace it with a 3x3 conv with stride 1.
        # --- 2. APPLY SN TO FIRST CONV ---
        self.backbone.conv1 = spectral_norm(nn.Conv2d(
            channels,             
            64, 
            kernel_size=3,        
            stride=1,             
            padding=1, 
            bias=False
        ))
        # Remove the first maxpool (also destroys resolution too early)
        self.backbone.maxpool = nn.Identity()

        def apply_sn(m):
            if isinstance(m, (nn.Conv2d, nn.Linear)):
                return spectral_norm(m)
            return m
        
        self.backbone.layer1.apply(apply_sn)
        self.backbone.layer2.apply(apply_sn)
        self.backbone.layer3.apply(apply_sn)
        self.backbone.layer4.apply(apply_sn)
            
        # 3. Modify Output Layer
        num_ftrs = self.backbone.fc.in_features
        self.backbone.fc = spectral_norm(nn.Linear(num_ftrs, self.num_classes))
    
    def load_edl_weights(self, edl_filename, device='cpu'):
        """Loads weights only into the EDL sub-component."""
        
        edl_path = get_weight_path(edl_filename)
        self.load_state_dict(torch.load(edl_path, map_location=device))
        self.to(device)
        self.eval()

    def forward(self, x):
        # 1. Get raw Logits from the ResNet backbone
        logits = self.backbone(x)
        
        # 2. EDL Activation Logic
        # The paper requires evidence >= 0. ReLU ensures this.
        evidence = F.relu(logits)
        
        # 3. Alpha construction
        alpha = evidence + 1
        return alpha

    def alpha_from_output(self, evidence):
        """Converts raw evidence output to Dirichlet parameters (alpha)."""
        return evidence + 1

    def calculate_uncertainties(self, x):
        """
        Predicts class probabilities and calculates standard EDL total uncertainty (K/S).
        """
        with torch.no_grad():
            alpha = self.forward(x)
            
            # Sum of evidence/Dirichlet strength
            S = torch.sum(alpha, dim=1, keepdim=True)
            
            # 1. Prediction (Mean of Dirichlet)
            probs = alpha / S
            
            # 2. Total Uncertainty (K / S)
            uncertainty = self.num_classes / S.squeeze() 
            
            return probs, uncertainty

    def get_features(self, x):
        """
        Extracts 512-dim semantic features.
        Used for Feature-Space Density Estimation.
        """
        # Manually pass through ResNet layers to skip the final FC layer
        x = self.backbone.conv1(x)
        x = self.backbone.bn1(x)
        x = self.backbone.relu(x)
        # x = self.backbone.maxpool(x) # (Recall: removed for CIFAR-10)

        x = self.backbone.layer1(x)
        x = self.backbone.layer2(x)
        x = self.backbone.layer3(x)
        x = self.backbone.layer4(x)

        x = self.backbone.avgpool(x)
        return torch.flatten(x, 1) # [Batch, 512]

    
class EDL(nn.Module):
    """
    Hybrid EDL Backbone:
    - MNIST: Uses the original LeNet-style architecture (Fast, Simple).
    - CIFAR-10: Uses ResNet-18 with Spectral Normalization (Robust, High Perf).
    """
    def __init__(self, num_classes=10, input_dims=(1, 28, 28)):
        super(EDL, self).__init__()
        self.num_classes = num_classes
        self.input_dims = tuple(input_dims)
        self.feature = None # Storage for GDA features (CIFAR only)
        
        # --- BRANCH 1: MNIST (Old LeNet) ---
        if self.input_dims == (1, 28, 28):
            self.mode = 'lenet'
            # Exact architecture from your EDL_old
            filters = [20, 50]
            hidden_units = 500
            flatten_dim = 50 * 5 * 5 # 1250
            
            self.conv_layers = nn.Sequential(
                nn.Conv2d(1, filters[0], kernel_size=5, stride=1, padding=2),
                nn.ReLU(),
                nn.MaxPool2d(kernel_size=2, stride=2),
                nn.Conv2d(filters[0], filters[1], kernel_size=5, stride=1, padding=0),
                nn.ReLU(),
                nn.MaxPool2d(kernel_size=2, stride=2)
            )
            self.fc_layers = nn.Sequential(
                nn.Linear(flatten_dim, hidden_units),
                nn.ReLU(),
                nn.Dropout(p=0.5),
                nn.Linear(hidden_units, self.num_classes)
            )

        # --- BRANCH 2: CIFAR-10 (ResNet-18 with SN) ---
        elif self.input_dims == (3, 32, 32):
            self.mode = 'resnet'
            self.backbone = models.resnet18(weights=None)
            
            # Conv3x3 surgery + Spectral Norm
            self.backbone.conv1 = spectral_norm(nn.Conv2d(
                3, 64, kernel_size=3, stride=1, padding=1, bias=False
            ))
            self.backbone.maxpool = nn.Identity()

            def apply_sn(m):
                if isinstance(m, (nn.Conv2d, nn.Linear)):
                    return spectral_norm(m)
                return m
            
            self.backbone.layer1.apply(apply_sn)
            self.backbone.layer2.apply(apply_sn)
            self.backbone.layer3.apply(apply_sn)
            self.backbone.layer4.apply(apply_sn)
                
            num_ftrs = self.backbone.fc.in_features
            self.backbone.fc = spectral_norm(nn.Linear(num_ftrs, self.num_classes))
            
        else:
            raise ValueError(f"EDL: Unknown architecture for input_dims {input_dims}")

    def forward(self, x):
        if self.mode == 'lenet':
            # --- MNIST Forward ---
            x = self.conv_layers(x)
            x = x.view(x.size(0), -1)
            logits = self.fc_layers(x)
            # No feature saving needed for MNIST
        else:
            # --- CIFAR-10 Forward ---
            x = self.backbone.conv1(x)
            x = self.backbone.bn1(x)
            x = self.backbone.relu(x)
            
            x = self.backbone.layer1(x)
            x = self.backbone.layer2(x)
            x = self.backbone.layer3(x)
            x = self.backbone.layer4(x)
            
            x = self.backbone.avgpool(x)
            x = torch.flatten(x, 1)
            
            self.feature = x # Save for GDA!
            
            logits = self.backbone.fc(x)

        # EDL Activation (ReLU + 1)
        evidence = F.relu(logits)
        alpha = evidence + 1
        return alpha

    def get_features(self, x):
        """Helper for CIFAR-10 GDA fitting."""
        if self.mode == 'resnet':
            self.forward(x)
            return self.feature
        else:
            # MNIST doesn't use feature-space density, but we return something to avoid crashes
            return self.forward(x)

    def load_edl_weights(self, edl_filename, device='cpu'):
        """Loads weights only into the EDL sub-component."""
        
        edl_path = get_weight_path(edl_filename)
        self.load_state_dict(torch.load(edl_path, map_location=device))
        self.to(device)
        self.eval()

    def calculate_uncertainties(self, x):
        """
        Predicts class probabilities and calculates standard EDL total uncertainty (K/S).
        """
        with torch.no_grad():
            alpha = self.forward(x)
            
            # Sum of evidence/Dirichlet strength
            S = torch.sum(alpha, dim=1, keepdim=True)
            
            # 1. Prediction (Mean of Dirichlet)
            probs = alpha / S
            
            # 2. Total Uncertainty (K / S)
            uncertainty = self.num_classes / S.squeeze() 
            
            return probs, uncertainty
    

# ----------------------------------------
# Custom EDL Loss Function (Prior Network Head)
# ----------------------------------------
def KL_divergence(alpha, num_classes, device=None):
    """
    Calculates the KL divergence between a Dirichlet distribution with parameters alpha
    and a uniform Dirichlet distribution.
    """
    beta = torch.ones([1, num_classes], dtype=torch.float32, device=device)
    S_alpha = torch.sum(alpha, dim=1, keepdim=True)
    S_beta = torch.sum(beta, dim=1, keepdim=True)

    ln_beta_alpha = torch.lgamma(S_alpha) - torch.sum(torch.lgamma(alpha), dim=1, keepdim=True)
    ln_beta_beta = torch.sum(torch.lgamma(beta), dim=1, keepdim=True) - torch.lgamma(S_beta)

    dg_S_alpha = torch.digamma(S_alpha)
    dg_alpha = torch.digamma(alpha)

    kl = torch.sum((alpha - beta) * (dg_alpha - dg_S_alpha), dim=1, keepdim=True) + ln_beta_alpha + ln_beta_beta
    return kl

def EDL_mse_loss(output, target, epoch_num, num_classes, annealing_step, device=None):
    """
    The full loss function from the original notebook.
    Combines MSE, variance, and KL divergence regularizer.
    """
    alpha = output
    evidence = alpha - 1
    y = F.one_hot(target, num_classes=num_classes)
    S = torch.sum(alpha, dim=1, keepdim=True)
    p = alpha / S

    # Term A: Mean Squared Error
    mse_loss = torch.sum((y - p)**2, dim=1)
    
    # Term B: Variance regularizer
    variance_loss = torch.sum(alpha * (S - alpha) / (S * S * (S + 1)), dim=1)

    # Annealing coefficient
    annealing_coef = torch.min(torch.tensor(1.0, dtype=torch.float32, device=device), 
                               torch.tensor(epoch_num / annealing_step, dtype=torch.float32, device=device))
    # Term C: KL Divergence regularizer
    alpha_tilde = (evidence * (1 - y)) + 1
    kl_reg = annealing_coef * KL_divergence(alpha_tilde, num_classes, device=device)

    # Combine terms
    total_loss = mse_loss + variance_loss + kl_reg.squeeze()
    return torch.mean(total_loss)
