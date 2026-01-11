"""
FACED Dataset Loader for Quantum Hydra/Mamba/Transformer Models

The Finer-grained Affective Computing EEG Dataset (FACED) from Tsinghua.
- 30 EEG channels at 250Hz (+ 2 mastoid reference channels)
- 123 subjects
- 28 emotion-elicitation video clips
- 9-class fine-grained emotion OR 3-class valence

Emotion Categories:
    Positive: amusement (5), inspiration (6), joy (7), tenderness (8)
    Negative: anger (0), fear (2), disgust (1), sadness (3)
    Neutral: neutral (4)

Valence Labels:
    positive (1), negative (-1), neutral (0)

Reference:
    Tsinghua Laboratory of Brain and Intelligence, 2023

Download URL: https://www.synapse.org/#!Synapse:syn50614194/files/
Required folder: Processed_data/Processed_data

Author: Junghoon Park
Date: December 2024
"""

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from typing import Tuple, Optional, Literal
import os

# TorchEEG imports
try:
    from torcheeg.datasets import FACEDDataset
    from torcheeg import transforms
    from torcheeg.model_selection import train_test_split_groupby_trial
    TORCHEEG_AVAILABLE = True
except ImportError:
    TORCHEEG_AVAILABLE = False
    print("Warning: TorchEEG not installed. Install with: pip install torcheeg")


def load_faced_eeg(
    seed: int,
    device: torch.device,
    batch_size: int,
    chunk_size: int = 250,
    overlap: int = 0,
    num_channel: int = 30,
    label_type: Literal['valence', 'emotion'] = 'valence',
    root_path: str = './Processed_data/Processed_data',
    io_path: str = './Processed_data/faced_io',
    test_size: float = 0.2,
    val_size: float = 0.1,
    num_workers: int = 0
) -> Tuple[DataLoader, DataLoader, DataLoader, Tuple[int, int, int], int]:
    """
    Loads and preprocesses the FACED EEG dataset for emotion recognition.

    Compatible interface with Load_PhysioNet_EEG_NoPrompt.py for use with
    Quantum Hydra, Mamba, and Transformer models.

    Args:
        seed (int): Random seed for reproducibility.
        device (torch.device): The device to move the tensors to.
        batch_size (int): Number of samples per batch.
        chunk_size (int): Number of data points per EEG chunk. Default 250 (1 second at 250Hz).
        overlap (int): Overlapping data points between chunks. Default 0.
        num_channel (int): Number of EEG channels to use (max 30). Default 30.
        label_type (str): 'valence' for 3-class or 'emotion' for 9-class. Default 'valence'.
        root_path (str): Path to Processed_data folder.
        io_path (str): Path for TorchEEG's cached IO.
        test_size (float): Proportion of data for test set. Default 0.2.
        val_size (float): Proportion of training data for validation. Default 0.1.
        num_workers (int): Number of workers for data loading. Default 0.

    Returns:
        tuple: (train_loader, val_loader, test_loader, input_dim, num_classes)
            - train_loader: DataLoader for training set
            - val_loader: DataLoader for validation set
            - test_loader: DataLoader for test set
            - input_dim: Shape (n_samples, n_channels, n_timesteps)
            - num_classes: Number of output classes (3 for valence, 9 for emotion)
    """
    if not TORCHEEG_AVAILABLE:
        raise ImportError("TorchEEG is required. Install with: pip install torcheeg")

    print(f"Loading FACED dataset from: {root_path}")
    print(f"Label type: {label_type}")
    print(f"Chunk size: {chunk_size}, Overlap: {overlap}, Channels: {num_channel}")

    # Set random seed
    np.random.seed(seed)
    torch.manual_seed(seed)

    # Define label transformation based on label_type
    if label_type == 'valence':
        # Map valence -1, 0, 1 → 0, 1, 2
        label_transform = transforms.Compose([
            transforms.Select('valence'),
            transforms.Lambda(lambda x: int(x + 1))
        ])
        num_classes = 3
    else:
        # Discrete emotions 0-8
        label_transform = transforms.Compose([
            transforms.Select('emotion'),
            transforms.Lambda(lambda x: int(x))
        ])
        num_classes = 9

    # Create dataset
    dataset = FACEDDataset(
        root_path=root_path,
        chunk_size=chunk_size,
        overlap=overlap,
        num_channel=num_channel,
        online_transform=transforms.ToTensor(),
        offline_transform=None,  # Keep raw EEG for quantum encoding
        label_transform=label_transform,
        io_path=io_path,
        io_mode='lmdb',
        num_worker=num_workers,
        verbose=True
    )

    print(f"Total samples in FACED dataset: {len(dataset)}")
    print(f"Number of classes: {num_classes}")

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
        test_size=val_size / (1 - test_size),
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

    input_dim = (len(dataset), num_channel, chunk_size)

    print(f"\nInput shape per sample: ({num_channel}, {chunk_size})")

    return train_loader, val_loader, test_loader, input_dim, num_classes


def load_faced_eeg_simple(
    seed: int,
    device: torch.device,
    batch_size: int,
    chunk_size: int = 250,
    label_type: Literal['valence', 'emotion'] = 'valence',
    root_path: str = './Processed_data/Processed_data',
    io_path: str = './Processed_data/faced_io',
    test_size: float = 0.2,
    val_size: float = 0.1
) -> Tuple[DataLoader, DataLoader, DataLoader, Tuple[int, int, int], int]:
    """
    Simplified FACED loader that returns data in format compatible with quantum models.

    Returns EEG data in shape (batch, channels, timesteps) with labels as integers.

    This version collects all data into tensors first (like PhysioNet loader),
    which is more compatible with the existing training scripts.
    """
    if not TORCHEEG_AVAILABLE:
        raise ImportError("TorchEEG is required. Install with: pip install torcheeg")

    print(f"Loading FACED dataset (simple mode) from: {root_path}")
    print(f"Label type: {label_type}")

    np.random.seed(seed)
    torch.manual_seed(seed)

    # Number of classes based on label type
    num_classes = 3 if label_type == 'valence' else 9

    # Define label transformation
    if label_type == 'valence':
        label_transform = transforms.Compose([
            transforms.Select('valence'),
            transforms.Lambda(lambda x: int(x + 1))  # Map -1,0,1 → 0,1,2
        ])
    else:
        label_transform = transforms.Compose([
            transforms.Select('emotion'),
            transforms.Lambda(lambda x: int(x))
        ])

    # Create dataset
    dataset = FACEDDataset(
        root_path=root_path,
        chunk_size=chunk_size,
        overlap=0,
        num_channel=30,
        online_transform=transforms.ToTensor(),
        offline_transform=None,
        label_transform=label_transform,
        io_path=io_path,
        io_mode='lmdb',
        num_worker=0,
        verbose=True
    )

    print(f"Total samples: {len(dataset)}")
    print(f"Number of classes: {num_classes}")

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
    print(f"Label distribution: {np.bincount(y_all)}")

    # Split data with stratification
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

    return train_loader, val_loader, test_loader, input_dim, num_classes


def load_faced_valence(seed, device, batch_size, sampling_freq=250, sample_size=None):
    """
    Compatibility wrapper for 3-class valence classification.
    Matches PhysioNet loader interface.
    """
    train_loader, val_loader, test_loader, input_dim, num_classes = load_faced_eeg_simple(
        seed=seed,
        device=device,
        batch_size=batch_size,
        chunk_size=sampling_freq,
        label_type='valence',
        root_path='./Processed_data/Processed_data',
        io_path='./Processed_data/faced_io'
    )
    return train_loader, val_loader, test_loader, input_dim


def load_faced_emotion(seed, device, batch_size, sampling_freq=250, sample_size=None):
    """
    Compatibility wrapper for 9-class discrete emotion classification.
    Matches PhysioNet loader interface.
    """
    train_loader, val_loader, test_loader, input_dim, num_classes = load_faced_eeg_simple(
        seed=seed,
        device=device,
        batch_size=batch_size,
        chunk_size=sampling_freq,
        label_type='emotion',
        root_path='./Processed_data/Processed_data',
        io_path='./Processed_data/faced_io'
    )
    return train_loader, val_loader, test_loader, input_dim


# Emotion label mapping for reference
FACED_EMOTION_LABELS = {
    0: 'anger',
    1: 'disgust',
    2: 'fear',
    3: 'sadness',
    4: 'neutral',
    5: 'amusement',
    6: 'inspiration',
    7: 'joy',
    8: 'tenderness'
}

FACED_VALENCE_LABELS = {
    0: 'negative',   # Original -1
    1: 'neutral',    # Original 0
    2: 'positive'    # Original 1
}


if __name__ == "__main__":
    # Test the loader
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Test valence (3-class)
    print("=" * 60)
    print("Testing FACED Valence (3-class)")
    print("=" * 60)

    train_loader, val_loader, test_loader, input_dim, num_classes = load_faced_eeg_simple(
        seed=2024,
        device=device,
        batch_size=32,
        chunk_size=250,
        label_type='valence',
        root_path='./Processed_data/Processed_data',
        io_path='./Processed_data/faced_io_test'
    )

    print(f"\nTest batch (valence):")
    for X, y in train_loader:
        print(f"  X shape: {X.shape}")
        print(f"  y shape: {y.shape}")
        print(f"  y unique values: {torch.unique(y)}")
        print(f"  num_classes: {num_classes}")
        break

    # Test emotion (9-class)
    print("\n" + "=" * 60)
    print("Testing FACED Emotion (9-class)")
    print("=" * 60)

    train_loader, val_loader, test_loader, input_dim, num_classes = load_faced_eeg_simple(
        seed=2024,
        device=device,
        batch_size=32,
        chunk_size=250,
        label_type='emotion',
        root_path='./Processed_data/Processed_data',
        io_path='./Processed_data/faced_io_test_emotion'
    )

    print(f"\nTest batch (emotion):")
    for X, y in train_loader:
        print(f"  X shape: {X.shape}")
        print(f"  y shape: {y.shape}")
        print(f"  y unique values: {torch.unique(y)}")
        print(f"  num_classes: {num_classes}")
        break
