# Save this as analyze_results.py
import json
import numpy as np

DATASETS = ["cifar10", "mnist"]
MODELS = ["edl", "nu_edl", "redl", "daedl", "postnet"]
INPUT_FILE = "results_database.jsonl"

def format_cell(values):
    if not values: return "Pending..."
    # 3 seeds or 5 seeds? The code adapts automatically.
    mean = np.mean(values)
    std = np.std(values)
    return f"{mean:.4f} \\pm {std:.4f}"

# 1. Load and De-Duplicate Data
db = {} # Use a dict key to enforce uniqueness
try:
    with open(INPUT_FILE, 'r') as f:
        for line in f:
            entry = json.loads(line)
            # Create a unique key for this run
            key = f"{entry['model']}_{entry['dataset']}_{entry['seed']}"
            # This overwrites previous entries if you re-ran a seed
            db[key] = entry 
except FileNotFoundError:
    print("No results found yet!")
    exit()

unique_runs = list(db.values()) # Convert back to list
print(f"Loaded {len(unique_runs)} unique training runs.")

for dataset in DATASETS:
    print(f"\n\n=== TABLE FOR {dataset.upper()} ===")
    
    if dataset == 'cifar10':
        near, far = 'cifar100', 'svhn'
        header = "Model & Acc & BS & C100_AUROC & C100_AUPR & C100_BS & SVHN_AUROC & SVHN_AUPR & SVHN_BS"
    else:
        near, far = 'kmnist', 'omniglot'
        header = "Model & Acc & BS & KMNIST_AUROC & KMNIST_AUPR & KMNIST_BS & OMNI_AUROC & OMNI_AUPR & OMNI_BS"
    
    print(header)

    for model in MODELS:
        # Filter for this specific model/dataset
        runs = [r for r in unique_runs if r['model'] == model and r['dataset'] == dataset]
        
        # Check seed count
        if len(runs) > 0:
            print(f"% Found {len(runs)} seeds for {model}...", end="\r")

        acc = [r['id_acc'] for r in runs]
        bs = [r['id_brier'] for r in runs]
        
        near_auroc = [r.get(f'{near}_auroc', 0) for r in runs]
        near_aupr = [r.get(f'{near}_aupr', 0) for r in runs]
        near_bs = [r.get(f'{near}_brier', 0) for r in runs]
        
        far_auroc = [r.get(f'{far}_auroc', 0) for r in runs]
        far_aupr = [r.get(f'{far}_aupr', 0) for r in runs]
        far_bs = [r.get(f'{far}_brier', 0) for r in runs]

        row = f"\\textbf{{{model.upper()}}} & "
        row += f"{format_cell(acc)} & {format_cell(bs)} & "
        row += f"{format_cell(near_auroc)} & {format_cell(near_aupr)} & {format_cell(near_bs)} & "
        row += f"{format_cell(far_auroc)} & {format_cell(far_aupr)} & {format_cell(far_bs)} \\\\"
        
        print(row)