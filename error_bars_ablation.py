import json
import numpy as np
from collections import defaultdict

# --- CONFIGURATION ---
RESULTS_FILE = "ablation_results.jsonl"
TARGET_DATASET = "cifar10"  # Change to "cifar10" or "mnist"

# CHANGED: Moved 1a-1c to the end of the list
ROW_ORDER = [
    "2a", "2b", "2c",   # Pairs (Now first)
    "3",                # Full (Now middle)
    "1a", "1b", "1c"    # Single Components (Now last)
]

# Define which OOD datasets to look for based on the target dataset
OOD_MAP = {
    "cifar10": ("cifar100", "svhn"),
    "mnist":   ("omniglot", "kmnist")
}

def format_metric(values):
    """
    Calculates only the mean, formatted to 4 decimal places in math mode.
    Example output: $0.9955$
    """
    if not values: return "$0.0000$"
    mean = np.mean(values)
    # Returns just the mean wrapped in $...$
    return f"${mean:.4f}$"

def main():
    # 1. Setup OOD keys
    if TARGET_DATASET not in OOD_MAP:
        print(f"Error: Dataset '{TARGET_DATASET}' not defined in OOD_MAP.")
        return

    ood1, ood2 = OOD_MAP[TARGET_DATASET]
    print(f"Generating Table for: {TARGET_DATASET.upper()}")
    print(f"OOD Datasets: {ood1}, {ood2}")
    
    # 2. Load Data
    db = defaultdict(list)
    try:
        with open(RESULTS_FILE, 'r') as f:
            for line in f:
                line = line.strip()
                if not line: continue
                try:
                    entry = json.loads(line)
                except json.JSONDecodeError:
                    continue

                # Filter by dataset and model
                if entry.get('dataset') == TARGET_DATASET and entry.get('model') == 'nu_edl':
                    task = entry.get('ablation_task', 'unknown')
                    db[task].append(entry)
    except FileNotFoundError:
        print(f"File {RESULTS_FILE} not found.")
        return

    # 3. Print Header for Debugging
    print("-" * 60)
    print(f"{'Task':<5} | {'Count':<5} | {'Acc':<10} | {'OOD1 AUC':<10} | {'OOD2 AUC':<10}")
    print("-" * 60)

    # 4. Generate LaTeX Rows
    for task in ROW_ORDER:
        runs = db.get(task, [])
        
        if not runs:
            # print(f"Skipping {task} (No data found)")
            continue

        # Extract Metrics
        acc = [r.get('id_acc', 0) for r in runs]
        id_bs = [r.get('id_brier', 0) for r in runs]
        
        # Dynamically fetch OOD metrics using f-strings
        ood1_auroc = [r.get(f'{ood1}_auroc', 0) for r in runs]
        ood2_auroc = [r.get(f'{ood2}_auroc', 0) for r in runs]
        
        ood1_aupr  = [r.get(f'{ood1}_aupr', 0) for r in runs]
        ood2_aupr  = [r.get(f'{ood2}_aupr', 0) for r in runs]
        
        ood1_bs    = [r.get(f'{ood1}_brier', 0) for r in runs]
        ood2_bs    = [r.get(f'{ood2}_brier', 0) for r in runs]

        # Debug print to console
        print(f"{task:<5} | {len(runs):<5} | {np.mean(acc):.4f}     | {np.mean(ood1_auroc):.4f}     | {np.mean(ood2_auroc):.4f}")

        # Format LaTeX Row
        # Structure: Acc & ID_Brier & OOD1_AUC & OOD2_AUC & OOD1_AUPR & OOD2_AUPR & OOD1_Brier & OOD2_Brier
        row = f"% Task {task}\n"
        row += f"& {format_metric(acc)} & {format_metric(id_bs)} & "
        row += f"{format_metric(ood1_auroc)} & {format_metric(ood2_auroc)} & "
        row += f"{format_metric(ood1_aupr)} & {format_metric(ood2_aupr)} & "
        row += f"{format_metric(ood1_bs)} & {format_metric(ood2_bs)} \\\\"
        
        print("\nLATEX ROW:")
        print(row)
        print("-" * 20)

if __name__ == "__main__":
    main()