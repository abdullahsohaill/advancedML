# data_loader.py (Updated with persistent_workers)

import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import configs

def get_cifar100_loaders():
    """
    Prepares and returns the CIFAR-100 train and test data loaders.
    """
    # Normalization stats for CIFAR-100
    cifar100_mean = (0.5071, 0.4867, 0.4408)
    cifar100_std = (0.2675, 0.2565, 0.2761)

    # Transformations for the training set: includes augmentation
    train_transform = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(cifar100_mean, cifar100_std),
    ])

    # Transformations for the test set: no augmentation
    test_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(cifar100_mean, cifar100_std),
    ])

    # Load datasets
    train_dataset = datasets.CIFAR100(
        root=configs.DATASET_PATH, train=True, download=True, transform=train_transform
    )
    test_dataset = datasets.CIFAR100(
        root=configs.DATASET_PATH, train=False, download=True, transform=test_transform
    )

    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=configs.BATCH_SIZE,
        shuffle=True,
        num_workers=configs.NUM_WORKERS,
        pin_memory=configs.PIN_MEMORY,
        persistent_workers=True  # <-- ADD THIS LINE
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=configs.BATCH_SIZE,
        shuffle=False,
        num_workers=configs.NUM_WORKERS,
        pin_memory=configs.PIN_MEMORY,
        persistent_workers=True  # <-- AND ADD THIS LINE
    )
    
    return train_loader, test_loader


def get_cifar100_color_jitter_loaders(for_training=True):
    """
    Returns CIFAR-100 loaders with specific Color Jitter settings.
    - For training the teacher: Aggressive jitter.
    - For evaluation: A consistent, milder jitter.
    """
    cifar100_mean = (0.5071, 0.4867, 0.4408)
    cifar100_std = (0.2675, 0.2565, 0.2761)

    if for_training:
        # Aggressive jitter for fine-tuning the teacher
        transform = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.2),
            transforms.ToTensor(),
            transforms.Normalize(cifar100_mean, cifar100_std),
        ])
    else:
        # Consistent, milder jitter for the final evaluation test set
        transform = transforms.Compose([
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
            transforms.ToTensor(),
            transforms.Normalize(cifar100_mean, cifar100_std),
        ])

    # For fine-tuning, we use the train set. For eval, the test set.
    dataset = datasets.CIFAR100(
        root=configs.DATASET_PATH, train=for_training, download=True, transform=transform
    )
    
    loader = DataLoader(
        dataset,
        batch_size=configs.BATCH_SIZE,
        shuffle=for_training, # Shuffle only for the training loader
        num_workers=configs.NUM_WORKERS,
        pin_memory=configs.PIN_MEMORY,
    )
    return loader