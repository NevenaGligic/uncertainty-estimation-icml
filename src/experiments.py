import torch
import numpy as np
from tqdm import tqdm
import itertools
from src.metrics import calculate_ood_auroc, get_edl_total_uncertainty, calculate_classification_accuracy, calculate_brier_score
from src.utils import stabilize_alpha
# Note: Full plotting code is verbose. We structure the logic here.

def run_benchmark(model: torch.nn.Module, test_loader_id, ood_loaders: dict, device: torch.device, num_classes: int):
    """
    Runs unified OOD evaluation for one model and calculates the final benchmark metrics.
    
    Returns: 
        id_accuracy: Accuracy on ID test set.
        id_brier: Brier Score on ID test set.
        ood_results: Dictionary of OOD results (AUROC, AUPR, OOD_Brier).
        all_id_alphas: Raw alpha outputs for the ID set (useful for downstream analysis).
    """
    model.eval()
    
    # --- 1. Get ID Scores (Alphas, Probs, Targets) ---
    all_id_alphas = []
    all_id_probs = []
    all_id_targets = []
    
    print("Collecting ID Data (Alphas, Probs, Targets)...")
    with torch.no_grad():
        for data, target in tqdm(test_loader_id, desc="ID Inference"):
            alpha = model(data.to(device)).cpu().numpy()
            
            # Convert Alpha to Probs: P = alpha / S (standard EDL mean prediction)
            S = np.sum(alpha, axis=1, keepdims=True)
            probs = alpha / S
            
            all_id_alphas.append(alpha)
            all_id_probs.append(probs)
            all_id_targets.append(target.numpy())
            
    all_id_alphas = np.concatenate(all_id_alphas, axis=0)
    all_id_probs = np.concatenate(all_id_probs, axis=0)
    all_id_targets = np.concatenate(all_id_targets, axis=0)

    # --- 2. Calculate ID Metrics ---
    id_accuracy = calculate_classification_accuracy(all_id_probs, all_id_targets)
    print(f"Accuracy ID: {id_accuracy}")
    id_brier = calculate_brier_score(all_id_probs, all_id_targets, num_classes, is_ood=False)
    print(f"Brier ID: {id_brier}")
    
    # --- 3. Calculate ID Uncertainty Score (Baseline for OOD detection) ---
    id_uncertainty_scores = get_edl_total_uncertainty(all_id_alphas)
    problematic_mask = (id_uncertainty_scores == 0) | np.isnan(id_uncertainty_scores)
    if np.any(problematic_mask):
        print(f"Problem detected — found {np.count_nonzero(problematic_mask)} zero/NaN scores in ID uncertainty.")
    
    # --- 4. Evaluate Against OOD Sets ---
    ood_results = {}
    
    for ood_name, ood_loader in ood_loaders.items():
        all_ood_alphas = []
        all_ood_probs = []
        
        print(f"Collecting OOD Data for {ood_name}...")
        with torch.no_grad():
            for data, _ in tqdm(ood_loader, desc=f"OOD Inference ({ood_name})"):
                alpha = model(data.to(device)).cpu().numpy()
                alpha = stabilize_alpha(alpha)
                S = np.sum(alpha, axis=1, keepdims=True)
                probs = alpha / S

                all_ood_alphas.append(alpha)
                all_ood_probs.append(probs)
                
        all_ood_alphas = np.concatenate(all_ood_alphas, axis=0)
        all_ood_probs = np.concatenate(all_ood_probs, axis=0)
        ood_alpha_mask = (all_ood_alphas == 0) | np.isnan(all_ood_alphas)
        ood_alpha_total = all_ood_alphas.size
        if np.any(ood_alpha_mask):
            print(f"{ood_name}: found {np.count_nonzero(ood_alpha_mask)} zero/NaN entries out of {ood_alpha_total} OOD alphas.")
        ood_prob_mask = (all_ood_probs == 0) | np.isnan(all_ood_probs)
        ood_prob_total = all_ood_probs.size
        if np.any(ood_prob_mask):
            print(f"{ood_name}: found {np.count_nonzero(ood_prob_mask)} zero/NaN entries out of {ood_prob_total} OOD probs.")

        # 4a. OOD Uncertainty Score
        ood_uncertainty_scores = get_edl_total_uncertainty(all_ood_alphas)
        ood_uncertainty_mask = (ood_uncertainty_scores == 0) | np.isnan(ood_uncertainty_scores)
        ood_uncertainty_total = ood_uncertainty_scores.size
        if np.any(ood_uncertainty_mask):
            print(f"{ood_name}: found {np.count_nonzero(ood_uncertainty_mask)} zero/NaN entries out of {ood_uncertainty_total} OOD uncertainty scores.")
        
        # 4b. AUROC/AUPR (OOD Detection)
        auroc, aupr = calculate_ood_auroc(id_uncertainty_scores, ood_uncertainty_scores)
        
        # 4c. OOD Brier Score (Proxy metric against Uniform target)
        dummy_targets = np.zeros(all_ood_probs.shape[0]) 
        ood_brier = calculate_brier_score(all_ood_probs, dummy_targets, num_classes, is_ood=True)
        
        ood_results[ood_name] = {'AUROC': auroc, 'AUPR': aupr, 'OOD_Brier': ood_brier}
        
    
    return id_accuracy, id_brier, ood_results, all_id_alphas, all_id_targets

