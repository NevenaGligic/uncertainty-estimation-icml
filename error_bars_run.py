import subprocess
import json
import numpy as np
import sys
import os
import time
import datetime

# --- CONFIGURATION ---
PYTHON_EXEC = sys.executable 
SCRIPT_NAME = "main.py"
SEEDS = [40] 
DATASETS = ["cifar10", "mnist"]
RESULTS_FILE = "results_database.jsonl" 

MODEL_FLAGS = {
    "edl": ["--train_edl"],
    "nu_edl": ["--train_cnn", "--train_maf"],
    "redl": ["--train_redl"],
    "daedl": ["--train_daedl"],
    "postnet": ["--train_postnet"]
}

def save_result(data):
    with open(RESULTS_FILE, "a") as f:
        f.write(json.dumps(data) + "\n")
    print(f"  [Saved result to {RESULTS_FILE}]")

def run_trial(dataset, model, flags, seed):
    
    cmd = [PYTHON_EXEC, "-u", SCRIPT_NAME, 
           "--dataset", dataset, 
           "--model", model, 
           "--seed", str(seed)] + flags
    
    start_time = time.time()
    
    # --- CHANGED: Popen allows live streaming of output ---
    process = subprocess.Popen(
        cmd, 
        stdout=subprocess.PIPE, 
        stderr=subprocess.STDOUT, # Merge stderr into stdout so we see errors too
        text=True, 
        bufsize=1 # Line buffered
    )

    full_output = []

    # Read output line by line as it is generated
    for line in process.stdout:
        print(line, end='') # Print to console immediately
        full_output.append(line)
    
    # Wait for finish
    process.wait()
    duration = (time.time() - start_time) / 60.0
    full_output_str = "".join(full_output)

    if process.returncode != 0:
        print(f"\n\nCRASH: Seed {seed} failed!")
        with open("crash_log.txt", "a") as f:
            f.write(f"[{datetime.datetime.now()}] Crash {model} {dataset} {seed}\n")
            f.write(full_output_str[-1000:] + "\n\n")
        return

    # Parse JSON output from the accumulated string
    try:
        start_tag = "__JSON_START__"
        end_tag = "__JSON_END__"
        idx_start = full_output_str.find(start_tag) + len(start_tag)
        idx_end = full_output_str.find(end_tag)
        
        if idx_start != -1 and idx_end != -1:
            raw_data = json.loads(full_output_str[idx_start:idx_end])
            
            raw_data['model'] = model
            raw_data['dataset'] = dataset
            raw_data['seed'] = seed
            raw_data['timestamp'] = str(datetime.datetime.now())
            
            save_result(raw_data)
        else:
            print("\n  Error: Could not find JSON tag in output.")
            
    except Exception as e:
        print(f"\n  Error parsing output: {e}")

# --- MAIN LOOP ---
# 1. CALCULATE TOTAL WORK
total_jobs = len(DATASETS) * len(MODEL_FLAGS) * len(SEEDS)
current_job = 0

print(f"Starting experiments. Results saving to {RESULTS_FILE}")
print(f"Total Jobs Scheduled: {total_jobs}")

for dataset in DATASETS:
    for model, flags in MODEL_FLAGS.items():
        for seed in SEEDS:
            current_job += 1
            
            # 2. UPDATE THE PRINT STATEMENT
            # Change the print inside run_trial or just print it here before calling it
            print(f"\n{'='*60}")
            print(f">> PROGRESS: [{current_job}/{total_jobs}]")
            print(f">> STARTING: {dataset.upper()} | {model.upper()} | Seed {seed}")
            print(f"{'='*60}\n")
            
            run_trial(dataset, model, flags, seed)