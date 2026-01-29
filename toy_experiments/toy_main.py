import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_moons
from torch.utils.data import DataLoader, TensorDataset
import torch.nn.functional as F
from nflows.flows.base import Flow
from nflows.distributions.normal import StandardNormal
from nflows.transforms.base import CompositeTransform
from nflows.transforms.autoregressive import MaskedAffineAutoregressiveTransform
from nflows.transforms.permutations import ReversePermutation
import os
import argparse

# --- Reuse generic training logic ---
from src.train import train_EDL
from src.models.EDL import EDL_mse_loss

# ==========================================
# 1. HELPER FUNCTIONS (KL Divergence)
# ==========================================
def kl_divergence(alpha, num_classes, device=None):
    if device is None: device = alpha.device
    beta = torch.ones([1, num_classes], dtype=torch.float32, device=device)
    S_alpha = torch.sum(alpha, dim=1, keepdim=True)
    S_beta = torch.sum(beta, dim=1, keepdim=True)
    lnB = torch.lgamma(S_alpha) - torch.sum(torch.lgamma(alpha), dim=1, keepdim=True)
    lnB_uni = torch.sum(torch.lgamma(beta), dim=1, keepdim=True) - torch.lgamma(S_beta)
    dg0 = torch.digamma(S_alpha)
    dg1 = torch.digamma(alpha)
    kl = torch.sum((alpha - beta) * (dg1 - dg0), dim=1, keepdim=True) + lnB + lnB_uni
    return kl

# ==========================================
# 2. ADAPTED MODEL (Wrapper Logic)
# ==========================================

class TinyREDL(nn.Module):
    """
    Adapts the logic of the original R-EDL snippet.
    - Encapsulates optimizer.
    - Calculates loss internally during forward pass if compute_loss=True.
    """
    def __init__(self, input_dim=2, num_classes=2, hidden_dim=64, lr=1e-3):
        super().__init__()
        self.num_classes = num_classes
        
        # 1. The Neural Network Backbone
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, num_classes)
        )

        # 2. Internal Optimizer (As implied by 'model.step()')
        self.optimizer = optim.Adam(self.parameters(), lr=lr)
        self.scheduler = optim.lr_scheduler.StepLR(self.optimizer, step_size=20, gamma=0.5)
        
        # Internal state for loss tracking
        self.grad_loss = 0.0

    def forward(self, x, y=None, compute_loss=False, epoch=0, return_output='soft'):
        # A. Inference
        logits = self.net(x)
        evidence = F.relu(logits)
        alpha = evidence + 1
        
        # B. Loss Calculation (The "Robust" Digamma Loss)
        if compute_loss and y is not None:
            # 1. Digamma Loss (Maximizing Log-Likelihood)
            # L = Σ y * (digamma(S) - digamma(alpha))
            S = torch.sum(alpha, dim=1, keepdim=True)
            loss_nll = torch.sum(F.one_hot(y, self.num_classes) * (torch.digamma(S) - torch.digamma(alpha)), dim=1, keepdim=True)
            loss_nll = torch.mean(loss_nll)

            # 2. KL Regularization
            annealing_coef = min(1, epoch / 10) # 10 epoch annealing
            kl_alpha = (alpha - 1) * (1 - F.one_hot(y, self.num_classes)) + 1
            loss_kl = annealing_coef * torch.mean(kl_divergence(kl_alpha, self.num_classes))
            
            # 3. Store total loss for the step function
            self.grad_loss = loss_nll + loss_kl
        
        if return_output == 'hard':
            return torch.argmax(alpha, dim=1)
        
        return alpha

    def step(self):
        """Performs the optimization step."""
        self.optimizer.zero_grad()
        self.grad_loss.backward()
        self.optimizer.step()

# ==========================================
# 3. ADAPTED TRAINING LOOP (From your snippet)
# ==========================================

def compute_loss_accuracy_adapted(model, loader, epoch, device):
    model.eval()
    total_loss_ = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for X, Y in loader:
            X, Y = X.to(device), Y.to(device)
            
            # Call model to compute loss internally
            # Note: We don't use the return value for prediction here to match your snippet logic strictly,
            # but we need predictions for accuracy.
            probs = model(X, Y, compute_loss=True, epoch=epoch, return_output='soft')
            
            # Accumulate the loss stored in the model
            total_loss_ += model.grad_loss.item()
            
            # Calculate Accuracy
            _, predicted = torch.max(probs, 1)
            total += Y.size(0)
            correct += (predicted == Y).sum().item()

    avg_loss = total_loss_ / len(loader)
    accuracy = correct / total
    return accuracy, avg_loss

def train_redl_adapted(model, train_loader, val_loader, max_epochs=50, device=torch.device("cpu")):
    print("Starting Adapted R-EDL Training...")
    model.to(device)
    model.train()
    
    val_losses = []
    
    for epoch in range(max_epochs):
        # 1. Training Loop
        for X_train, Y_train in train_loader:
            X_train, Y_train = X_train.to(device), Y_train.to(device)
            
            model.train()
            # Forward + Compute Loss
            model(X_train, Y_train, compute_loss=True, epoch=epoch)
            # Optimize
            model.step()
        
        # 2. Scheduler Step
        model.scheduler.step()
        
        # 3. Validation / Logging (Every 10 epochs)
        if (epoch + 1) % 10 == 0:
            val_acc, val_loss = compute_loss_accuracy_adapted(model, val_loader, epoch, device)
            val_losses.append(val_loss)
            
            print(f"\033[34m Epoch {epoch+1} -> Val loss {val_loss:.4f} | Val Acc. {val_acc*100:.2f}%\033[0m")
            
            if np.isnan(val_loss):
                print("Detected NaN Loss - Stopping")
                break

    print("Training Finished.")

# ==========================================
# 1. Tiny Models for 2D Data
# ==========================================

class TinyEDL(nn.Module):
    def __init__(self, input_dim=2, num_classes=2, hidden_dim=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, num_classes)
        )
    
    def forward(self, x):
        logits = self.net(x)
        evidence = torch.relu(logits)
        alpha = evidence + 1
        return alpha

class TinyMAF(nn.Module):
    def __init__(self, input_dim=2, hidden_features=64, num_layers=5):
        super().__init__()
        base_dist = StandardNormal(shape=[input_dim])
        transforms = []
        for _ in range(num_layers):
            transforms.append(ReversePermutation(features=input_dim))
            transforms.append(MaskedAffineAutoregressiveTransform(
                features=input_dim, 
                hidden_features=hidden_features
            ))
        self.transform = CompositeTransform(transforms)
        self.flow = Flow(self.transform, base_dist)

    def log_prob(self, x):
        return self.flow.log_prob(x)


class TinyClassifier(nn.Module):
    """
    A standard MLP classifier.
    Outputs: Softmax probabilities P(y|x).
    Training: Uses Standard CrossEntropyLoss.
    """
    def __init__(self, input_dim=2, num_classes=2, hidden_dim=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, num_classes)
        )
    
    def forward(self, x):
        logits = self.net(x)
        # We return probabilities directly for the formula
        return torch.softmax(logits, dim=1)

class nu_EDL(nn.Module):
    """
    Implements the formula: Alpha = 1 + n * p(x) * P(y|x)
    """
    def __init__(self, classifier, maf, n_samples):
        super().__init__()
        self.classifier = classifier
        self.maf = maf
        self.n = float(n_samples)
    
    def forward(self, x):
        # 1. Get P(y|x) from standard classifier
        probs = self.classifier(x)
        
        # 2. Get p(x) from MAF
        # MAF gives log_prob, so we exp() it to get likelihood
        # Detach to ensure we don't backprop into MAF/Classifier during inference visualization
        log_prob = self.maf.log_prob(x).detach()
        likelihood = torch.exp(log_prob).unsqueeze(1) # Shape [Batch, 1]
        
        # 3. Apply Formula: alpha = 1 + n * p(x) * P(y|x)
        evidence = self.n * likelihood * probs
        alpha = evidence + 1
        
        return alpha


# ==========================================
# 3. Visualization
# ==========================================
def plot_uncertainty_surface(model, X, y, title, filename):
    device = next(model.parameters()).device
    
    # Create grid covering the moons + OOD space
    x_min, x_max = X[:, 0].min() - 1.5, X[:, 0].max() + 1.5
    y_min, y_max = X[:, 1].min() - 1.5, X[:, 1].max() + 1.5
    xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.05),
                         np.arange(y_min, y_max, 0.05))
    
    grid_tensor = torch.FloatTensor(np.c_[xx.ravel(), yy.ravel()]).to(device)
    
    model.eval()
    with torch.no_grad():
        # Alpha pass
        alphas = model(grid_tensor) 
        
        # Uncertainty = K / S
        S = torch.sum(alphas, dim=1)
        uncertainty = 2.0 / S 
        uncertainty = uncertainty.cpu().reshape(xx.shape).numpy()

    plt.figure(figsize=(9, 7))
    # Plot contours
    contour = plt.contourf(xx, yy, uncertainty, levels=np.linspace(0, 1.1, 20), cmap='viridis')
    cbar = plt.colorbar(contour)
    cbar.set_label('Total Uncertainty (K/S)')
    
    # Plot data points
    plt.scatter(X[y==0, 0], X[y==0, 1], c='red', s=10, alpha=0.5, label='Class 0')
    plt.scatter(X[y==1, 0], X[y==1, 1], c='blue', s=10, alpha=0.5, label='Class 1')
    
    plt.title(title)
    plt.legend(loc='upper right')
    plt.tight_layout()
    plt.savefig(filename)
    print(f"Saved plot: {filename}")
    plt.close()

# ==========================================
# 4. Main Execution
# ==========================================
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, default=11, help="Random seed")
    parser.add_argument('--load', action='store_true', help="Load trained weights instead of training")
    parser.add_argument('--model_type', type=str, default='edl', choices=['edl', 'redl', 'nu_edl'], help="Base classifier type")
    args = parser.parse_args()

    # 1. Set Seed
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Running Toy Experiment on {device} (Seed={args.seed})")

    # 2. Data
    n_samples = 1000
    X, y = make_moons(n_samples=n_samples, noise=0.1, random_state=args.seed)
    X = torch.FloatTensor(X)
    y = torch.LongTensor(y)
    
    dataset = TensorDataset(X, y)
    train_loader = DataLoader(dataset, batch_size=128, shuffle=True)
    val_loader = DataLoader(dataset, batch_size=128, shuffle=False)

    # 3. Initialize Models
    if args.model_type == 'redl':
        print("Using R-EDL backbone")
        edl_model = TinyREDL().to(device)
    else:
        print("Using Standard EDL backbone")
        edl_model = TinyEDL().to(device)
        
    maf_model = TinyMAF().to(device)
    
    weights_path_edl = f"toy_{args.model_type}_weights.pth"
    weights_path_maf = "toy_maf_weights.pth"

    # 4. Train or Load
    if args.load and os.path.exists(weights_path_edl) and os.path.exists(weights_path_maf):
        print("\n--- Loading Saved Weights ---")
        edl_model.load_state_dict(torch.load(weights_path_edl, map_location=device))
        maf_model.load_state_dict(torch.load(weights_path_maf, map_location=device))
    else:
        print(f"\n--- Training Tiny {args.model_type.upper()} ---")
        # --- BRANCHING LOGIC FOR TRAINING ---
        if args.model_type == 'redl':
            # Use the ADAPTED loop that respects internal optimizer/loss
            train_redl_adapted(
                model=edl_model,
                train_loader=train_loader,
                val_loader=val_loader,
                max_epochs=100,
                device=device
            )
            # Save manually because train_redl_adapted doesn't save automatically to specific path
            torch.save(edl_model.state_dict(), weights_path_edl)
            
        elif args.model_type == 'edl':
            # Use the STANDARD loop for standard EDL
            train_EDL(
                model_EDL=edl_model,
                train_loader=train_loader,
                device=device,
                EDL_mse_loss=EDL_mse_loss,
                num_classes=2,
                file_path=weights_path_edl,
                epochs=100,
                annealing_step=20,
                lr=1e-3
            )

        elif args.model_type == 'nu_edl':
            print("\n--- Training Standard Classifier (Cross Entropy) ---")
            classifier = TinyClassifier().to(device)
            optimizer_cls = optim.Adam(classifier.parameters(), lr=1e-3)
            criterion = nn.CrossEntropyLoss()
            
            for epoch in range(100):
                total_loss = 0
                for bx, by in train_loader:
                    bx, by = bx.to(device), by.to(device)
                    probs = classifier(bx)
                    loss = criterion(torch.log(probs + 1e-9), by) # Standard CE
                    
                    optimizer_cls.zero_grad()
                    loss.backward()
                    optimizer_cls.step()
                    total_loss += loss.item()
                if (epoch+1) % 20 == 0:
                    print(f"  Classifier Epoch {epoch+1}: Loss {total_loss/len(train_loader):.4f}")

        print("\n--- Training Tiny MAF ---")
        optimizer_maf = optim.Adam(maf_model.parameters(), lr=1e-3)
        maf_model.train()
        for epoch in range(100):
            total_loss = 0
            for batch_x, _ in train_loader:
                batch_x = batch_x.to(device)
                loss = -maf_model.log_prob(batch_x).mean()
                
                optimizer_maf.zero_grad()
                loss.backward()
                optimizer_maf.step()
                total_loss += loss.item()
            if (epoch+1) % 20 == 0:
                print(f"  MAF Epoch {epoch+1}: Loss {total_loss / len(train_loader):.4f}")
        torch.save(maf_model.state_dict(), weights_path_maf)

    # 5. 
    if args.model_type == 'nu_edl':
        nu_edl_model = nu_EDL(classifier, maf_model, n_samples=n_samples).to(device)
    
    # 7. Visualization
    print("\n--- Generating Plots ---")
    
    if args.model_type == 'nu_edl':
        # Visualize
        plot_uncertainty_surface(
            nu_edl_model, 
            X.cpu().numpy(), 
            y.cpu().numpy(), 
            title="Nu-EDL", 
            filename="toy_nu_edl.png"
        )
    else:
        # Plot 1: Baseline EDL/REDL (No Gating)
        # We pass the inner edl_model directly
        plot_uncertainty_surface(
            edl_model, 
            X.cpu().numpy(), 
            y.cpu().numpy(), 
            title=f"Baseline {args.model_type.upper()}", 
            filename=f"toy_baseline_{args.model_type}.png"
        )
    
    print("\nExperiment Complete. Check the .png files.")

if __name__ == "__main__":
    main()
