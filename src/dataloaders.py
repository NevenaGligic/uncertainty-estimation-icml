import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Subset, random_split
import numpy as np

# --- Configuration ---
MNIST_MEAN = (0.1307,)
MNIST_STD = (0.3081,)
CIFAR_MEAN = (0.4914, 0.4822, 0.4465)
CIFAR_STD = (0.2023, 0.1994, 0.2010)
# ---------------------

def get_base_transforms(dataset='mnist', train=False):
    """
    Returns the appropriate transforms.
    Adds augmentation (Crop/Flip) only for CIFAR-10 training.
    """
    if dataset == 'mnist':
        return transforms.Compose([transforms.ToTensor()])
        
    elif dataset == 'cifar10':
        if train:
            # Standard CIFAR-10 Augmentation
            return transforms.Compose([
                transforms.RandomCrop(32, padding=4),
                transforms.RandomHorizontalFlip(),
                transforms.RandomRotation(degrees=15),
                transforms.ToTensor(),
                transforms.Normalize(CIFAR_MEAN, CIFAR_STD)
            ])
        else:
            return transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize(CIFAR_MEAN, CIFAR_STD)
                ])
            
    return transforms.Compose([transforms.ToTensor()])

def get_experiment_loaders(id_name: str, batch_size: int, val_split: float = 0.2, seed: int = 42):
    """
    Returns the ID training/validation/testing loaders and a dictionary of OOD test loaders
    for the specified experimental track.
    """
    
    if id_name == 'mnist':
        id_transform = get_base_transforms(dataset='mnist')
        # Load ID Datasets (MNIST)
        train_set = datasets.MNIST('./data', train=True, download=True, transform=id_transform)
        test_set_id = datasets.MNIST('./data', train=False, download=True, transform=id_transform)

        # OOD Transforms
        ood_transform_omni = transforms.Compose([
            transforms.Resize((28, 28)),
            transforms.Grayscale(num_output_channels=1),
            id_transform
        ])
        
        # OOD Datasets (All 28x28, 1-channel compatible)
        ood_set_kmnist = datasets.KMNIST('./data', train=False, download=True, transform=id_transform)
        ood_set_omni = datasets.Omniglot('./data', background=False, download=True, transform=ood_transform_omni)

        ood_loaders = {
            'kmnist': DataLoader(ood_set_kmnist, batch_size=batch_size, shuffle=False),
            'omniglot': DataLoader(ood_set_omni, batch_size=batch_size, shuffle=False)
        }

    elif id_name == 'cifar10':
        # Load ID Datasets (CIFAR-10)
        train_set = datasets.CIFAR10('./data', train=True, download=True, transform=get_base_transforms(dataset='cifar10', train=True))
        val_set = datasets.CIFAR10('./data', train=True, download=True, transform=get_base_transforms(dataset='cifar10', train=False))
        test_set_id = datasets.CIFAR10('./data', train=False, download=True, transform=get_base_transforms(dataset='cifar10', train=False))

        # OOD 1: SVHN (Far OOD)
        ood_set_svhn = datasets.SVHN('./data', split='test', download=True, transform=get_base_transforms(dataset='cifar10', train=False))
        
        # OOD 2: CIFAR-100 (Near OOD)
        ood_set_cifar100 = datasets.CIFAR100('./data', train=False, download=True, transform=get_base_transforms(dataset='cifar10', train=False))

        ood_loaders = {
            'svhn': DataLoader(ood_set_svhn, batch_size=batch_size, shuffle=False),
            'cifar100': DataLoader(ood_set_cifar100, batch_size=batch_size, shuffle=False)
        }

    else:
        raise NotImplementedError(f"Dataset {id_name} not configured.")

    # Create train/val split for ID data
    # We generate indices once to ensure no overlap
    num_train_full = len(train_set)
    val_len = int(num_train_full * val_split)
    train_len = num_train_full - val_len
    
    # Generate shuffled indices
    rng = np.random.default_rng(seed)
    indices = rng.permutation(num_train_full)
    train_idx = indices[:train_len]
    val_idx = indices[train_len:]

    # Create Subsets
    if id_name == 'cifar10':
        # Map augmented dataset to train indices, clean dataset to val indices
        train_subset = Subset(train_set, train_idx)
        val_subset = Subset(val_set, val_idx)
    else:
        train_subset = Subset(train_set, train_idx)
        val_subset = Subset(train_set, val_idx)

    # Create loaders for ID data
    train_loader = DataLoader(train_subset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_subset, batch_size=batch_size, shuffle=False)
    test_loader_id = DataLoader(test_set_id, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader, test_loader_id, ood_loaders