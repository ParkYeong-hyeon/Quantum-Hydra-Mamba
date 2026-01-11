"""
SEED Dataset Loader for Quantum Hydra/Mamba/Transformer Models

The SJTU Emotion EEG Dataset (SEED) from BCMI Laboratory.
- 62 channels at 200Hz
- 15 subjects × 3 sessions = 45 recordings
- 3-class emotion recognition: positive (1), negative (-1), neutral (0)
- Stimulus: 15 four-minute film clips from Chinese movies

Reference:
    Zheng W L, Lu B L. Investigating critical frequency bands and channels
    for EEG-based emotion recognition with deep neural networks[J].
    IEEE Transactions on Autonomous Mental Development, 2015, 7(3): 162-175.

Download URL: https://bcmi.sjtu.edu.cn/home/seed/index.html
Required folder: SEED/SEED_EEG/Preprocessed_EEG

Author: Junghoon Park
Date: December 2024
"""

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from typing import Tuple, Optional, Callable
import os

# TorchEEG imports
try:
    from torcheeg.datasets import SEEDDataset
    from torcheeg import transforms
    from torcheeg.model_selection import KFoldGroupbyTrial, train_test_split_groupby_trial
    TORCHEEG_AVAILABLE = True
except ImportError:
    TORCHEEG_AVAILABLE = False
    print("Warning: TorchEEG not installed. Install with: pip install torcheeg")


def load_seed_eeg(
    seed: int,
    device: torch.device,
    batch_size: int,
    chunk_size: int = 200,
    overlap: int = 0,
    num_channel: int = 62,
    root_path: str = './SEED/SEED_EEG/Preprocessed_EEG',
    io_path: str = './SEED/seed_io',
    test_size: float = 0.2,
    val_size: float = 0.1,
    num_workers: int = 0
) -> Tuple[DataLoader, DataLoader, DataLoader, Tuple[int, int, int]]:
    """
    Loads and preprocesses the SEED EEG dataset for emotion recognition.

    Compatible interface with Load_PhysioNet_EEG_NoPrompt.py for use with
    Quantum Hydra, Mamba, and Transformer models.

    Args:
        seed (int): Random seed for reproducibility.
        device (torch.device): The device to move the tensors to.
        batch_size (int): Number of samples per batch.
        chunk_size (int): Number of data points per EEG chunk. Default 200 (1 second at 200Hz).
        overlap (int): Overlapping data points between chunks. Default 0.
        num_channel (int): Number of EEG channels to use (max 62). Default 62.
        root_path (str): Path to Preprocessed_EEG folder.
        io_path (str): Path for TorchEEG's cached IO.
        test_size (float): Proportion of data for test set. Default 0.2.
        val_size (float): Proportion of training data for validation. Default 0.1.
        num_workers (int): Number of workers for data loading. Default 0.

    Returns:
        tuple: (train_loader, val_loader, test_loader, input_dim)
            - train_loader: DataLoader for training set
            - val_loader: DataLoader for validation set
            - test_loader: DataLoader for test set
            - input_dim: Shape (n_samples, n_channels, n_timesteps)
    """
    if not TORCHEEG_AVAILABLE:
        raise ImportError("TorchEEG is required. Install with: pip install torcheeg")

    print(f"Loading SEED dataset from: {root_path}")
    print(f"Chunk size: {chunk_size}, Overlap: {overlap}, Channels: {num_channel}")

    # Set random seed
    np.random.seed(seed)
    torch.manual_seed(seed)

    # Define transforms
    # offline_transform: Applied once during preprocessing (cached)
    # online_transform: Applied each time data is accessed
    offline_transform = transforms.Compose([
        transforms.BandDifferentialEntropy(),  # Common for SEED
        transforms.ToGrid(SEEDDataset.channel_location_dict())  # If using grid representation
    ])

    # For raw EEG (no DE features), use simpler transform:
    offline_transform_raw = transforms.Compose([
        transforms.BaselineRemoval(),
    ])

    # Create dataset with raw EEG for quantum models
    dataset = SEEDDataset(
        root_path=root_path,
        chunk_size=chunk_size,
        overlap=overlap,
        num_channel=num_channel,
        online_transform=transforms.ToTensor(),
        offline_transform=None,  # Keep raw EEG for quantum encoding
        label_transform=transforms.Compose([
            transforms.Select('emotion'),
            # Map -1, 0, 1 to 0, 1, 2 for classification
            transforms.Lambda(lambda x: x + 1)
        ]),
        io_path=io_path,
        io_mode='lmdb',
        num_worker=num_workers,
        verbose=True
    )

    print(f"Total samples in SEED dataset: {len(dataset)}")

    # Split by trial to avoid data leakage
    train_dataset, test_dataset = train_test_split_groupby_trial(
        dataset=dataset,
        test_size=test_size,
        shuffle=True,
        random_state=seed
    )

    # Further split training into train and validation
    train_dataset, val_dataset = train_test_split_groupby_trial(
        dataset=train_dataset,
        test_size=val_size / (1 - test_size),  # Adjust for remaining proportion
        shuffle=True,
        random_state=seed
    )

    print(f"Training samples: {len(train_dataset)}")
    print(f"Validation samples: {len(val_dataset)}")
    print(f"Test samples: {len(test_dataset)}")

    # Create DataLoaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers
    )

    # Get input dimensions
    sample_eeg, sample_label = dataset[0]
    input_dim = (len(dataset), num_channel, chunk_size)

    print(f"\nInput shape per sample: ({num_channel}, {chunk_size})")
    print(f"Number of classes: 3 (negative, neutral, positive)")

    return train_loader, val_loader, test_loader, input_dim


def load_seed_eeg_simple(
    seed: int,
    device: torch.device,
    batch_size: int,
    chunk_size: int = 200,
    root_path: str = './SEED/SEED_EEG/Preprocessed_EEG',
    io_path: str = './SEED/seed_io',
    test_size: float = 0.2,
    val_size: float = 0.1
) -> Tuple[DataLoader, DataLoader, DataLoader, Tuple[int, int, int]]:
    """
    Simplified SEED loader that returns data in format compatible with quantum models.

    Returns EEG data in shape (batch, channels, timesteps) with labels as integers.

    This version collects all data into tensors first (like PhysioNet loader),
    which is more compatible with the existing training scripts.
    """
    if not TORCHEEG_AVAILABLE:
        raise ImportError("TorchEEG is required. Install with: pip install torcheeg")

    print(f"Loading SEED dataset (simple mode) from: {root_path}")

    np.random.seed(seed)
    torch.manual_seed(seed)

    # Create dataset
    dataset = SEEDDataset(
        root_path=root_path,
        chunk_size=chunk_size,
        overlap=0,
        num_channel=62,
        online_transform=transforms.ToTensor(),
        offline_transform=None,
        label_transform=transforms.Compose([
            transforms.Select('emotion'),
            transforms.Lambda(lambda x: int(x + 1))  # Map -1,0,1 → 0,1,2
        ]),
        io_path=io_path,
        io_mode='lmdb',
        num_worker=0,
        verbose=True
    )

    print(f"Total samples: {len(dataset)}")

    # Collect all data into numpy arrays
    X_all = []
    y_all = []

    print("Loading all samples into memory...")
    for i in range(len(dataset)):
        eeg, label = dataset[i]
        X_all.append(eeg.numpy() if isinstance(eeg, torch.Tensor) else eeg)
        y_all.append(label)

        if (i + 1) % 1000 == 0:
            print(f"  Loaded {i + 1}/{len(dataset)} samples")

    X_all = np.array(X_all, dtype=np.float32)
    y_all = np.array(y_all, dtype=np.int64)

    print(f"Full dataset shape: X={X_all.shape}, y={y_all.shape}")

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X_all, y_all, test_size=test_size, random_state=seed, stratify=y_all
    )

    X_train, X_val, y_train, y_val = train_test_split(
        X_train, y_train, test_size=val_size/(1-test_size), random_state=seed, stratify=y_train
    )

    print(f"\nTraining set shape: {X_train.shape}")
    print(f"Validation set shape: {X_val.shape}")
    print(f"Test set shape: {X_test.shape}")

    # Create PyTorch datasets
    train_dataset = TensorDataset(
        torch.tensor(X_train, dtype=torch.float32).to(device),
        torch.tensor(y_train, dtype=torch.long).to(device)
    )
    val_dataset = TensorDataset(
        torch.tensor(X_val, dtype=torch.float32).to(device),
        torch.tensor(y_val, dtype=torch.long).to(device)
    )
    test_dataset = TensorDataset(
        torch.tensor(X_test, dtype=torch.float32).to(device),
        torch.tensor(y_test, dtype=torch.long).to(device)
    )

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    input_dim = X_train.shape

    return train_loader, val_loader, test_loader, input_dim


# For compatibility with existing code
def load_eeg_ts_seed(seed, device, batch_size, sampling_freq=200, sample_size=15):
    """
    Compatibility wrapper matching PhysioNet loader interface.

    Note: sampling_freq is fixed at 200Hz for SEED dataset.
    sample_size controls chunk_size (sample_size * 10 = chunk_size).
    """
    return load_seed_eeg_simple(
        seed=seed,
        device=device,
        batch_size=batch_size,
        chunk_size=sampling_freq,  # 1 second of data
        root_path='./SEED/SEED_EEG/Preprocessed_EEG',
        io_path='./SEED/seed_io'
    )


if __name__ == "__main__":
    # Test the loader
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_loader, val_loader, test_loader, input_dim = load_seed_eeg_simple(
        seed=2024,
        device=device,
        batch_size=32,
        chunk_size=200,
        root_path='./SEED/SEED_EEG/Preprocessed_EEG',
        io_path='./SEED/seed_io_test'
    )

    print(f"\nTest batch:")
    for X, y in train_loader:
        print(f"  X shape: {X.shape}")
        print(f"  y shape: {y.shape}")
        print(f"  y unique values: {torch.unique(y)}")
        break
