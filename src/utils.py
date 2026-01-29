import numpy as np
import torch
import os
from huggingface_hub import hf_hub_download, HfApi

# ----------------------------------------
# Smart Model Loading and Uploading Functions
# ----------------------------------------
# Hugging Face repository configuration
HF_REPO_ID = os.environ.get("HF_REPO_ID")

def get_weight_path(filename, repo_id=HF_REPO_ID):
    """
    Locates the weight file:
    1. Checks local disk.
    2. If missing, downloads from Hugging Face.
    Returns the absolute file path.
    """
    # 1. Check Local
    if os.path.exists(os.path.abspath(filename)):
        print(f"✅ Found local weights: {filename}")
        return filename
    
    # 2. Download from HF
    print(f"📥 {filename} not found locally. Downloading from HF ({repo_id})...")
    try:
        model_path = hf_hub_download(
            repo_id=repo_id,
            filename=filename
        )
        print(f"✅ Download complete: {model_path}")
        return model_path
    except Exception as e:
        print(f"❌ Error downloading {filename}: {e}")
        raise e

# It can now just use the helper above.
def load_model_smart(model_class, filename, device, **kwargs):
    path = get_weight_path(filename)
    model = model_class(**kwargs).to(device)
    model.load_state_dict(torch.load(path, map_location=device))
    return model

def save_and_upload_model(model, model_name, upload_to_hf=True):
    """
    Save model locally and optionally upload to Hugging Face.
    
    Args:
        model: The trained model
        model_name (str): Name of the model file
        upload_to_hf (bool): Whether to upload to Hugging Face (uses global UPLOAD_TO_HF if None)
    """
    # Create the directory if it doesn't exist
    parent_dir = os.path.dirname(model_name)
    if parent_dir and not os.path.exists(parent_dir):
        os.makedirs(parent_dir, exist_ok=True)
    # Save locally
    torch.save(model.state_dict(), model_name)
    print(f"💾 Saved {model_name} locally")
    
    # Upload to Hugging Face if requested
    if upload_to_hf:
        print(f"📤 Uploading {model_name} to Hugging Face...")
        
        try:
            api = HfApi()
            api.upload_file(
                path_or_fileobj=model_name,
                path_in_repo=model_name,
                repo_id=HF_REPO_ID,
                commit_message=f"Update {model_name}"
            )
            print(f"✅ Successfully uploaded {model_name} to Hugging Face")
            
        except Exception as e:
            print(f"❌ Failed to upload {model_name} to Hugging Face: {e}")
            print(f"💡 Make sure you have a valid Hugging Face token set in HUGGINGFACE_HUB_TOKEN")

# ----------------------------------------
# GPU Selection Functions
# ----------------------------------------
# Get the GPU with the most free memory
def get_best_device():
    """
    Selects the best available device, prioritizing CUDA, then Apple (MPS), then CPU.
    """
    # 1. Check for NVIDIA CUDA
    if torch.cuda.is_available():
        print("NVIDIA CUDA GPU found.")
        # --- OPTIMIZATION ---
        # Enable TF32 for a free speedup on Ampere+ GPUs
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

        num_gpus = torch.cuda.device_count()
        if num_gpus == 1:
            print("Using single CUDA GPU: cuda:0")
            return torch.device("cuda:0")

        # Find the GPU with the most free memory if multiple are available
        max_free_memory = 0
        best_gpu_index = 0
        for i in range(num_gpus):
            free_mem, total_mem = torch.cuda.mem_get_info(i)
            print(f"GPU {i}: {free_mem / 1e9:.2f} GB free / {total_mem / 1e9:.2f} GB total")
            if free_mem > max_free_memory:
                max_free_memory = free_mem
                best_gpu_index = i
                
        print(f"Selected GPU {best_gpu_index} with the most free memory.")
        return torch.device(f"cuda:{best_gpu_index}")

    # 2. Check for Apple M1/M2/M3 (Metal Performance Shaders)
    elif torch.backends.mps.is_available():
        print("Apple Metal (MPS) GPU found. Using mps.")
        return torch.device("mps")
    
    # 3. Fallback to CPU
    else:
        print("No GPU acceleration available (CUDA or MPS). Using CPU.")
        return torch.device("cpu")


# ----------------------------------------
# MAF helper functions
# ----------------------------------------
import torch

def transform_for_maf(x: torch.Tensor, dataset_name: str = None) -> torch.Tensor:
    """
    Applies the specific transforms MAF needs based on Papamakarios et al. (2017).
    
    Steps:
    0. (Optional) Horizontal Flip (CIFAR-10 Train only)
    1. Dequantize: (x * 255 + U[0,1]) / 256
    2. Logit Transform: logit(lambda + (1 - 2*lambda) * x)
       - lambda = 1e-6 for MNIST
       - lambda = 0.05 for CIFAR-10
    3. Flatten
    """
    # 1. Determine parameters
    if dataset_name is None:
        flat_dim = x.numel() // x.size(0) 
        if flat_dim == 784:
            lam = 1e-6
            is_cifar = False
        else:
            is_cifar = True
    else:
        is_cifar = (dataset_name.lower() == 'cifar10')
        lam = None if is_cifar else 1e-6

    if is_cifar:
        # Input is Normalized [-2, 2]. Do NOT Logit Transform.
        # Just flatten.
        return x.view(x.shape[0], -1)

    else: 
        # 2. Dequantization
        # x is [0, 1]. Map back to continuous [0, 1]
        noise = torch.rand_like(x)
        x = (x * 255.0 + noise) / 256.0
        
        # 3. Logit Transform
        x = lam + (1 - 2 * lam) * x
        x_logit = torch.log(x / (1.0 - x))
        
        # 4. Flatten
        return x_logit.view(x_logit.size(0), -1)

def stabilize_alpha(alpha: np.ndarray) -> np.ndarray:
    """
    Enforces that alpha values are finite and greater than 1 (>= 1).
    """
    # 1. Check for NaN/Inf/Negative values
    is_corrupted = ~np.isfinite(alpha) | (alpha < 1)
    
    # 2. Replace corrupted values
    if np.any(is_corrupted):
        # print(f"CRITICAL GUARD: Replacing {np.count_nonzero(is_corrupted)} corrupted alpha entries.")
        # Replace corrupted values with a 1.
        # Alpha MUST be at least 1 since these are Dirichlet distribution parameters.
        alpha[is_corrupted] = 1
    
    return alpha