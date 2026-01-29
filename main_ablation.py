import argparse
import os
import sys
import torch
import numpy as np
import json

# --- 1. Import Project Modules ---
from src.dataloaders import get_experiment_loaders
from src.train import train_EDL, train_density_estimator, train_StandardCNN
from src.models.SGA import SGA
from src.models.nu_EDL_ablation import nu_EDL
from src.models.EDL import EDL, EDL_mse_loss
from src.models.R_EDL import REDL
from src.models.DAEDL import DAEDL
from src.models.PostNet import PostNet, PostNetLoaderWrapper
from src.utils import get_best_device, save_and_upload_model 
from src.experiments import find_best_hyperparameters_shorter, run_benchmark, find_best_hyperparameters, find_best_hyperparameters_shorter


# --- 1b. Import External Modules ---
# R-EDL
REDL_ROOT = os.path.join(os.path.dirname(__file__), "..", "ICLR2024-REDL", "code_classical")
sys.path.append(os.path.abspath(REDL_ROOT))
from train import train as redl_train

# DAEDL
DAEDL_ROOT = os.path.join(os.path.dirname(__file__), "..", "DAEDL")
sys.path.append(os.path.abspath(DAEDL_ROOT))
from train_DAEDL import train_daedl

# PostNet
POSTNET_ROOT = os.path.join(os.path.dirname(__file__), "..", "Posterior-Network")
sys.path.append(os.path.abspath(POSTNET_ROOT))
from src_postnet.posterior_networks.train_postnet import train_postnet

# --- CENTRALIZED CONFIGURATION ---
# Default settings for each dataset. CLI arguments will override these.
CONFIGS = {
    'mnist': {
        'num_classes': 10,
        'input_dims': (1, 28, 28),
        
        'maf': {
            'batch_size': 128,
            'lr': 1e-4,
            'epochs': 50,
            'maf_hidden_features': 1024,
            'maf_num_layers': 10,
            'maf_blocks': 20,
            'weight_decay': 1e-5,
            'early_stopping_patience': 5,
            'scheduler_step_size': 5,
        },
        'edl': {
            'batch_size': 128,
            'lr': 1e-3,
            'weight_decay': 0.005,
            'epochs': 50,
            'annealing_step': 10, # Epochs to reach full KL divergence weight
            # Architecture params (LeNet-5 style)
            'filters': [20, 50],
            'hidden_units': 500
        },
        'cnn': {
            'batch_size': 128,
            'lr': 1e-3, # Classifier Learning Rate
            'epochs': 50,
            # It reuses the 'maf' config above for the density estimator
        },
        'redl': {
            'batch_size': 64, 
            'lr': 1e-3, 
            'epochs': 60,
            'lamb1': 1.0, 
            'lamb2': 0.1, 
            'fisher_c': 0.0,
            'hidden_dims': [64, 64, 64], 
            'kernel_dim': 5, 
            'architecture': 'conv'
        },
        'daedl': {
            'batch_size': 64, 
            'lr': 1e-3, 
            'epochs': 50,
            'reg': 5e-2, 
            'dropout_rate': 0.5
        },
        'postnet': {
            'batch_size': 64, 
            'lr': 5e-5, 
            'epochs': 50,
            'latent_dim': 6, 
            'density_type': 'radial_flow', 
            'architecture': 'conv'
        }
    },
    'cifar10': {
        'num_classes': 10,
        'input_dims': (3, 32, 32),
        'maf': {
            'batch_size': 128,
            'lr': 1e-4,
            'epochs': 50,
            'maf_hidden_features': 1024, # Increased capacity for CIFAR features
            'maf_num_layers': 10,
            'maf_blocks': 5,
            'weight_decay': 1e-6,
            'early_stopping_patience': 10,
            'scheduler_step_size': 30,
        },
        'edl': {
            'batch_size': 128,
            'lr': 1e-4,
            'weight_decay': 5e-4,
            'epochs': 100,
            'annealing_step': 10, # Slower annealing for harder dataset
            # Architecture params ("Large LeNet" from paper)
            'filters': [192, 192],
            'hidden_units': 1000
        },
        'cnn': {
            'batch_size': 128,
            'lr': 1e-3, 
            'epochs': 75,
        },
        'redl': {
            'batch_size': 64, 
            'lr': 1e-4, 
            'epochs': 200,
            'lamb1': 1.0, 
            'lamb2': 0.1, 
            'fisher_c': 0.0,
            'hidden_dims': [64, 64, 64], 
            'kernel_dim': 3, 
            'architecture': 'vgg'
            # Note: You might need to check if R-EDL supports 'vgg' or 'resnet' for CIFAR
        },
        'daedl': {
            'batch_size': 64, 
            'lr': 1e-3, 
            'epochs': 100,
            'reg': 5e-2, 
            'dropout_rate': 0.5
        },
        'postnet': {
            'batch_size': 64, 
            'lr': 5e-4, 
            'epochs': 200,
            'latent_dim': 6, 
            'density_type': 'radial_flow', 
            'architecture': 'conv'
        }
    }
}


def main():
    parser = argparse.ArgumentParser(description="Unified Benchmarking Script for Shrinkage-Gated Architecture (SGA).")
    parser.add_argument('--dataset', type=str, choices=['mnist', 'cifar10'], default='mnist', help="ID dataset for training and comparison.")
    parser.add_argument('--model', type=str, choices=['sga', 'nu_edl', 'edl', 'redl', 'daedl', 'postnet'], default='sga', help="Model to use for training and evaluation.")
    parser.add_argument('--seed', type=int, default=42, help="Random seed.")
    
    parser.add_argument('--val_split', type=float, default=0.2, help="Validation split fraction for R-EDL (also used by other models).")

    parser.add_argument('--train_edl', action='store_true', help="Train the EDL model from scratch.")
    parser.add_argument('--train_maf', action='store_true', help="Train the MAF model from scratch.")
    parser.add_argument('--train_cnn', action='store_true', help="Train the CNN model from scratch.")
    parser.add_argument('--train_redl', action='store_true', help="Train R-EDL model from scratch.")
    parser.add_argument('--train_daedl', action='store_true', help="Train DAEDL model from scratch.")
    parser.add_argument('--train_postnet', action='store_true', help="Train PostNet model from scratch.")

    # File Paths
    parser.add_argument('--edl_path', type=str, default=None, help="Path to EDL weights.")
    parser.add_argument('--maf_path', type=str, default=None, help="Path to MAF weights.")
    parser.add_argument('--cnn_path', type=str, default=None, help="Path to CNN weights.")
    parser.add_argument('--redl_path', type=str, default=None, help="Checkpoint prefix for R-EDL (\"_best\"/\"_last\" appended automatically).")
    parser.add_argument('--daedl_path', type=str, default=None, help="Path to DAEDL weights.")
    parser.add_argument('--postnet_path', type=str, default=None, help="Path to PostNet weights.")

    # SGA Hyperparameters
    parser.add_argument('--run_hp_search', action='store_true', help="Run hyperparameter search for kappa/tau.")
    parser.add_argument('--kappa', type=float, default=0.01, help="Gating slope hyperparameter.")
    parser.add_argument('--tau', type=float, default=None, help="Gating threshold hyperparameter.")

    # DAEDL Hyperparameters
    # parser.add_argument('--daedl_reg', type=float, default=5e-2)

    # PostNet Hyperparameters
    # parser.add_argument('--postnet_regr', type=float, default=1e-5)

    args = parser.parse_args()

    # --- 0. SET SEED AND LOAD CONFIGURATION ---
    def set_seed(seed):
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        np.random.seed(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    set_seed(args.seed)

    dataset_cfg = CONFIGS[args.dataset]

    # Determine Model Config
    if args.model == 'sga' or args.model == 'nu_edl':
        maf_cfg = dataset_cfg['maf']
        if args.model == 'sga':
            model_cfg = dataset_cfg['edl']
        else:
            model_cfg = dataset_cfg['cnn']
    else:
        model_cfg = dataset_cfg[args.model]
    
    # Global Constants
    NUM_CLASSES = dataset_cfg['num_classes']
    INPUT_DIMS = dataset_cfg['input_dims']
    BATCH_SIZE = model_cfg['batch_size']
    EPOCHS = model_cfg['epochs']

    # Set Paths Automatically
    save_dir = "saved_model_weights"
    os.makedirs(save_dir, exist_ok=True)
    seed_suffix = f"seed{args.seed}"
    if args.edl_path is None: args.edl_path = f"{save_dir}/{args.dataset}_SGA_EDL_model_{seed_suffix}.pth"

    if args.maf_path is None:
        if args.dataset == 'mnist':
            args.maf_path = f"{save_dir}/{args.dataset}_SGA_MAF_model_{seed_suffix}.pth"
        elif args.dataset == 'cifar10':
            if args.model == 'sga':
                args.maf_path = f"{save_dir}/{args.dataset}_SGA_MAF_model_{seed_suffix}.pth"
            elif args.model == 'nu_edl':
                args.maf_path = f"{save_dir}/{args.dataset}_nu_EDL_MAF_model_{seed_suffix}.pth"

    if args.cnn_path is None: args.cnn_path = f"{save_dir}/{args.dataset}_nu_EDL_CNN_model_{seed_suffix}.pth"
    if args.redl_path is None: args.redl_path = f"{save_dir}/{args.dataset}_R_EDL_model_{seed_suffix}"
    if args.daedl_path is None: args.daedl_path = f"{save_dir}/{args.dataset}_DAEDL_model_{seed_suffix}.pth"
    if args.postnet_path is None: args.postnet_path = f"{save_dir}/{args.dataset}_PostNet_model_{seed_suffix}.pth"

    # Default Tau for SGA
    if args.tau is None and args.dataset == 'mnist':
        args.tau = -1267
    elif args.tau is None and args.dataset == 'cifar10':
        args.tau = -796

    # --- DEVICE SETUP ---
    device = get_best_device()
    print(f"Using device: {device}")

    # --- 1. Load Data ---
    print("Loading the data.")
    train_loader, val_loader, test_loader_id, ood_loaders = get_experiment_loaders(
        args.dataset, BATCH_SIZE, val_split=args.val_split, seed=args.seed
    )
    
    # --- 2. Initialize Model ---
    # SGA wrapper is created, holding the two uninitialized models
    print("Initializing the model")

    if args.model == 'sga':
        model = SGA(
            num_classes=NUM_CLASSES, 
            input_dims=INPUT_DIMS,
            kappa=args.kappa, 
            tau=args.tau,
            maf_hidden_features=maf_cfg['maf_hidden_features'],
            maf_num_layers=maf_cfg['maf_num_layers'],
            maf_blocks=maf_cfg['maf_blocks']
            ).to(device)
    
    elif args.model == 'edl':
        model = EDL(
            num_classes=NUM_CLASSES, 
            input_dims=INPUT_DIMS
        ).to(device)

    elif args.model == 'nu_edl': # <--- NEW: Nu-EDL Initialization
        model = nu_EDL(
            num_classes=NUM_CLASSES, 
            input_dims=INPUT_DIMS,
            maf_hidden_features=maf_cfg['maf_hidden_features'],
            maf_num_layers=maf_cfg['maf_num_layers'],
            maf_blocks=maf_cfg['maf_blocks'],
            n_train_samples=len(train_loader.dataset)  # Dynamically determine N from train set size
        ).to(device)
    
    elif args.model == 'redl':
        model = REDL(
            num_classes=NUM_CLASSES,
            input_dims=INPUT_DIMS,
            architecture=model_cfg['architecture'],
            batch_size=BATCH_SIZE,
            lr=model_cfg['lr'],
            lamb1=model_cfg['lamb1'],
            lamb2=model_cfg['lamb2'],
            fisher_c=model_cfg['fisher_c'],
            hidden_dims=model_cfg['hidden_dims'],
            kernel_dim=model_cfg['kernel_dim']
        ).to(device)

    elif args.model == 'daedl':
        # Pass dataset string in format expected by load_model ("MNIST", "CIFAR-10")
        daedl_dataset_name = "MNIST" if args.dataset == 'mnist' else "CIFAR-10"
        model = DAEDL(
            num_classes=NUM_CLASSES, 
            dataset=daedl_dataset_name, 
            input_dims=INPUT_DIMS,
            dropout_rate=model_cfg['dropout_rate'],
            device=device
        ).to(device)

    elif args.model == 'postnet':
        print("Calculating class counts (N) for PostNet...")
        if hasattr(train_loader.dataset, 'targets'): 
            targets = torch.tensor(train_loader.dataset.targets)
        elif hasattr(train_loader.dataset, 'tensors'):
            targets = train_loader.dataset.tensors[1]
        else:
            targets = torch.cat([y for _, y in train_loader])
            
        unique, counts = torch.unique(targets, return_counts=True)
        N_counts = torch.zeros(NUM_CLASSES, dtype=torch.long)
        N_counts[unique.long()] = counts.long()
        print(f"Class counts: {N_counts}")
        
        model = PostNet(
            num_classes=NUM_CLASSES,
            N=N_counts, 
            input_dims=INPUT_DIMS,
            architecture=model_cfg['architecture'],
            latent_dim=model_cfg['latent_dim'],
            density_type=model_cfg['density_type'],
            lr=model_cfg['lr'],
            regr=1e-5,
            batch_size=BATCH_SIZE
        ).to(device)
    

    # --- 3. Training Phase (Independent Components) ---
    if args.model == 'sga':
        # Load EDL weights BEFORE training MAF (MAF needs EDL as feature extractor for CIFAR-10)
        if not args.train_edl and args.dataset == 'cifar10':
            print("Loading pre-trained EDL weights (required for MAF training on CIFAR-10).")
            model.load_edl_weights(args.edl_path, device)

        if args.train_edl:
            print("--- Training SGA EDL Component ---")
            train_EDL(
                model_EDL=model.edl,
                train_loader=train_loader,
                device=device,
                EDL_mse_loss=EDL_mse_loss, 
                num_classes=NUM_CLASSES,
                file_path=args.edl_path,
                epochs=EPOCHS,
                annealing_step=model_cfg['annealing_step'],
                lr=model_cfg['lr'],
                weight_decay=model_cfg['weight_decay'],
                save_and_upload_model_fn=save_and_upload_model, # From src/utils.py
                dataset_name=args.dataset
            )
        
        if args.train_maf:
            print("--- Training SGA MAF Component ---")
            train_density_estimator(
                model_wrapper=model,
                train_loader=train_loader,
                val_loader=val_loader,
                device=device,
                file_path=args.maf_path,
                learning_rate=maf_cfg['lr'],
                weight_decay=maf_cfg['weight_decay'],
                num_epochs=maf_cfg['epochs'],
                early_stopping_patience=maf_cfg['early_stopping_patience'],
                scheduler_step_size=maf_cfg['scheduler_step_size'],
                save_and_upload_model_fn=save_and_upload_model, # From src/utils.py
                dataset_name=args.dataset
            )

    elif args.model == 'edl' and args.train_edl:
        print("--- Training EDL Model (baseline) ---")
        train_EDL(
            model_EDL=model,
            train_loader=train_loader,
            device=device,
            EDL_mse_loss=EDL_mse_loss, 
            num_classes=NUM_CLASSES,
            file_path=args.edl_path,
            epochs=EPOCHS,
            annealing_step=model_cfg['annealing_step'],
            lr=model_cfg['lr'],
            weight_decay=model_cfg['weight_decay'],
            save_and_upload_model_fn=save_and_upload_model, # From src/utils.py
            dataset_name=args.dataset
        )
    
    elif args.model == 'nu_edl':
        # Load CNN weights BEFORE training MAF (MAF needs CNN as feature extractor)
        if not args.train_cnn and args.dataset == 'cifar10':
            print("Loading pre-trained CNN weights (required for MAF training).")
            model.load_cnn_weights(args.cnn_path, device)

        if args.train_cnn:
            print("--- Training CNN Component ---")
            train_StandardCNN(
            model_CNN=model.cnn,
            train_loader=train_loader,
            device=device,
            file_path=args.cnn_path,
            epochs=model_cfg['epochs'],
            lr=model_cfg['lr'],
            save_and_upload_model_fn=save_and_upload_model
        )

        if args.train_maf:
            print("--- Training nu-EDL MAF Component ---")
            train_density_estimator(
                model_wrapper=model,
                train_loader=train_loader,
                val_loader=val_loader,
                device=device,
                file_path=args.maf_path,
                learning_rate=maf_cfg['lr'],
                weight_decay=maf_cfg['weight_decay'],
                num_epochs=maf_cfg['epochs'],
                early_stopping_patience=maf_cfg['early_stopping_patience'],
                scheduler_step_size=maf_cfg['scheduler_step_size'],
                save_and_upload_model_fn=save_and_upload_model, # From src/utils.py
                dataset_name=args.dataset
            )
        
    elif args.model == 'redl' and args.train_redl:
        print("--- Training R-EDL (ModifiedEvidentialNet) ---")
        ckpt_dir = os.path.dirname(args.redl_path) or "."
        os.makedirs(ckpt_dir, exist_ok=True)

        if args.dataset == 'mnist':
            redl_config_dict = {
            'dataset_name': 'MNIST',
            'model_type': 'menet',
            'batch_size': BATCH_SIZE,
            'split': [1 - args.val_split, args.val_split],
            'loss': 'MEDL'  # Note: config says "MEDL" but code uses "IEDL" - they appear to be the same
            }
        elif args.dataset == 'cifar10':
            redl_config_dict = {
                'dataset_name': 'CIFAR10',
                'model_type': 'menet',
                'batch_size': BATCH_SIZE,
                'split': [1 - args.val_split, args.val_split],
                'loss': 'MEDL'  # Note: config says "MEDL" but code uses "IEDL" - they appear to be the same
            }
        
        # Train the underlying ModifiedEvidentialNet (not the wrapper)
        redl_train(
            model=model.model,  # Access the underlying model for training
            train_loader=train_loader,
            val_loader=val_loader,
            max_epochs=EPOCHS,
            model_path=args.redl_path,
            full_config_dict=redl_config_dict,
            use_wandb=False,
            device=device,
            is_fisher=False,
            output_dim=NUM_CLASSES
        )
        
        best_path = args.redl_path + "_best"
        if os.path.exists(best_path):
            state = torch.load(best_path, map_location=device)
            model.model.load_state_dict(state['model_state_dict'])
            print(f"Loaded best R-EDL weights from {best_path}")
            
            # Save and upload the best model to Hugging Face (similar to EDL/MAF)
            # Create a clean state_dict-only file path (remove _best/_last suffix, ensure .pth extension)
            redl_state_dict_path = args.redl_path.replace('_best', '').replace('_last', '')
            if not redl_state_dict_path.endswith('.pth'):
                redl_state_dict_path += '.pth'
            
            # Save state_dict in the same format as EDL/MAF for consistency
            save_and_upload_model(model.model, redl_state_dict_path)
        else:
            print(f"Best checkpoint not found at {best_path}; using in-memory weights.")

    elif args.model == 'daedl' and args.train_daedl:
        print("--- Training DAEDL Backbone ---")
        # DAEDL uses its own training loop which expects specific args
        train_daedl(
            model=model.model, # Pass the inner backbone
            learning_rate=model_cfg['lr'],
            reg_param=model_cfg['reg'],
            num_epochs=EPOCHS,
            trainloader=train_loader,
            validloader=val_loader,
            num_classes=NUM_CLASSES,
            device=device
        )
        save_and_upload_model(model.model, args.daedl_path)

        # CRITICAL: Fit Density (GDA) after training/loading
        model.fit_density(train_loader)

    elif args.model == 'postnet' and args.train_postnet:
        print("--- Training Posterior Network ---")
        
        # 1. Compatibility: Inject output_dim
        if not hasattr(train_loader.dataset, 'output_dim'):
            train_loader.dataset.output_dim = NUM_CLASSES
        if not hasattr(val_loader.dataset, 'output_dim'):
            val_loader.dataset.output_dim = NUM_CLASSES

        # 2. Compatibility: Wrap Loaders (Now using wrapper from PostNet.py)
        train_loader_wrapped = PostNetLoaderWrapper(train_loader)
        val_loader_wrapped = PostNetLoaderWrapper(val_loader)

        # 3. Train
        train_postnet(
            model=model.model, 
            train_loader=train_loader_wrapped,
            val_loader=val_loader_wrapped,
            max_epochs=EPOCHS,
            frequency=2,
            patience=10,
            model_path=args.postnet_path,
            full_config_dict={} 
        )

    # --- 4. Load Weights ---
    print("--- Checking and Loading Weights ---")
    
    if args.model == 'sga':
        # Note: EDL weights are loaded earlier (before MAF training) if needed for CIFAR-10
        if not args.train_edl and args.dataset == 'mnist':
            # For MNIST, EDL can be loaded later since MAF uses pixel space
            print("Loading pre-trained EDL weights.")
            model.load_edl_weights(args.edl_path, device) 

        if not args.train_maf:
            print("Loading pre-trained MAF weights.")
            model.load_maf_weights(args.maf_path, device) 
    
    elif args.model == 'edl' and not args.train_edl:
        print("Loading pre-trained EDL weights.")
        model.load_edl_weights(args.edl_path, device) 

    elif args.model == 'nu_edl':
        # Note: CNN weights are loaded earlier (before MAF training) if needed for CIFAR-10
        if not args.train_cnn and args.dataset == 'mnist':
            # For MNIST, CNN can be loaded later since MAF uses pixel space
            print("Loading pre-trained CNN weights.")
            model.load_cnn_weights(args.cnn_path, device) 

        if not args.train_maf:
            if args.dataset == 'mnist':
                print("Loading pre-trained MAF weights.")
                model.load_maf_weights(args.maf_path, device)
            elif args.dataset == 'cifar10':
                print("fitting GDA on loaded CNN weights...")
                model.fit_density(train_loader, device)
        
    elif args.model == 'redl' and not args.train_redl:
        # Try loading from the simplified state_dict format first (Hugging Face compatible)
        redl_state_dict_path = args.redl_path.replace('_best', '').replace('_last', '')
        if not redl_state_dict_path.endswith('.pth'):
            redl_state_dict_path += '.pth'
        
        # Check if simplified format exists, otherwise fall back to checkpoint format
        if os.path.exists(redl_state_dict_path):
            model.load_weights(redl_state_dict_path, device)
        else:
            # Fall back to original checkpoint format
            best_path = args.redl_path + "_best"
            last_path = args.redl_path + "_last"
            ckpt_path = best_path if os.path.exists(best_path) else last_path
            if os.path.exists(ckpt_path):
                state = torch.load(ckpt_path, map_location=device)
                model.model.load_state_dict(state['model_state_dict'])
                model.model.to(device)
                model.model.eval()
                print(f"Loaded R-EDL weights from {ckpt_path}")
            else:
                print(f"⚠️  Warning: No R-EDL weights found at {redl_state_dict_path} or {ckpt_path}")

    elif args.model == 'daedl' and not args.train_daedl:
        # 1. Load weights if we didn't just train
        model.load_weights(args.daedl_path)
            
        # 2. Fit Density Estimator (Must happen after loading/training)
        model.fit_density(train_loader)

    elif args.model == 'postnet' and not args.train_postnet:
        model.load_weights(args.postnet_path, device)

    # --- 5. Hyperparameter Optimization ---
    if args.model == 'sga' and args.run_hp_search:
        # if args.dataset == 'mnist':
        #     tau_search_values = np.arange(-2500, -800 + 10, 10).tolist()
        #     kappa_search_values = np.arange(0.01, 0.5 + 0.001, 0.001).tolist()
        # elif args.dataset == 'cifar10':
        #     tau_search_values = np.arange(-6000, -1000 + 100, 100).tolist()
        #     kappa_search_values = [0.001, 0.005, 0.01, 0.05, 0.1, 0.2, 0.5]

        # # Call the search function (which automatically sets model.kappa/tau to best values)
        # best_tau, best_kappa, _, _ = find_best_hyperparameters(
        #     model=model,
        #     test_loader_id=test_loader_id,
        #     device=device,
        #     num_classes=NUM_CLASSES,
        #     tau_values=tau_search_values,
        #     kappa_values=kappa_search_values
        # )

        # Define Kappa search space (Sharpness)
        # We check smooth slopes (0.01) to sharp cliffs (0.5)
        kappa_search_values = [0.01, 0.05, 0.1, 0.2, 0.5]
        
        # Define Tau Percentiles (The "Tightness" of the boundary)
        # 5.0 = Keep 95% of data (Standard)
        # 1.0 = Keep 99% of data (Conservative)
        tau_percentiles = [5.0, 1.0]

        best_tau, best_kappa, _, _ = find_best_hyperparameters_shorter(
            model=model,
            train_loader_id=train_loader, # <--- Pass Training Loader here
            val_loader=val_loader,    # <--- Validation/Test loader here
            device=device,
            kappa_values=kappa_search_values,
            num_classes=NUM_CLASSES,
            percentiles=tau_percentiles
        )

        # Update runtime args with the optimized values for final evaluation report
        args.kappa = best_kappa
        args.tau = best_tau

        model.kappa = best_kappa
        model.tau = best_tau

    # --- 4.5 CALIBRATION (CRITICAL FOR NU-EDL) ---
    if args.model == 'nu_edl' and args.dataset == 'mnist':
        # We need to shift densities so they don't vanish
        model.to(device)
        model.calibrate_density(train_loader, device)
    
    # --- FLOAT32 CASTING ---
    # Cast the entire model to float32 before evaluation to maximize numerical range 
    # and prevent crashes from Float16 limits.
    model.to(device)
    model.float() 
    print(f"Model cast to float32 for stable evaluation.")
    # -------------------------------

    # --- 6. Evaluation Phase ---
    print("\n--- Running Unified Benchmark ---")
    id_accuracy, id_brier, ood_results, all_id_alphas, all_id_targets = run_benchmark(
        model=model, 
        test_loader_id=test_loader_id, 
        ood_loaders=ood_loaders, 
        device=device, 
        num_classes=NUM_CLASSES
    )

    # --- 7. Reporting ---
    print("\n\n--- EVALUATION SUMMARY ---")
    print(f"ID Dataset: {args.dataset.upper()} | Model: {args.model.upper()}")
    if args.model == 'sga':
        print(f"  Hyperparams: κ={args.kappa}, τ={args.tau}")
    print(f"  Classification Accuracy (ID): {id_accuracy * 100:.2f}%")
    print(f"  Total Brier Score (ID):       {id_brier:.4f}")
    
    for ood_name, metrics in ood_results.items():
        print(f"\n  OOD vs {ood_name.upper()}:")
        print(f"    AUROC:      {metrics['AUROC']:.4f}")
        print(f"    AUPR:       {metrics['AUPR']:.4f}")
        print(f"    OOD Brier:  {metrics['OOD_Brier']:.4f} (Uniform Target Proxy)")
    print("------------------------------------------")

    def make_serializable(obj):
        if isinstance(obj, (np.float32, np.float64)): return float(obj)
        if isinstance(obj, (np.int32, np.int64)): return int(obj)
        return obj

    # Safely extract metrics (using 0.0 if a metric is missing)
    results_payload = {
        'id_acc': id_accuracy,
        'id_brier': id_brier,
        # CIFAR-10 OOD keys
        'cifar100_auroc': ood_results.get('cifar100', {}).get('AUROC', 0),
        'cifar100_aupr':  ood_results.get('cifar100', {}).get('AUPR', 0),
        'cifar100_brier': ood_results.get('cifar100', {}).get('OOD_Brier', 0),
        'svhn_auroc':     ood_results.get('svhn', {}).get('AUROC', 0),
        'svhn_aupr':      ood_results.get('svhn', {}).get('AUPR', 0),
        'svhn_brier':     ood_results.get('svhn', {}).get('OOD_Brier', 0),
        # MNIST OOD keys
        'omniglot_auroc': ood_results.get('omniglot', {}).get('AUROC', 0),
        'omniglot_aupr':  ood_results.get('omniglot', {}).get('AUPR', 0),
        'omniglot_brier': ood_results.get('omniglot', {}).get('OOD_Brier', 0),
        'kmnist_auroc':   ood_results.get('kmnist', {}).get('AUROC', 0),
        'kmnist_aupr':    ood_results.get('kmnist', {}).get('AUPR', 0),
        'kmnist_brier':   ood_results.get('kmnist', {}).get('OOD_Brier', 0),
    }

    print(f"__JSON_START__{json.dumps(results_payload, default=make_serializable)}__JSON_END__")


if __name__ == '__main__':
    main()