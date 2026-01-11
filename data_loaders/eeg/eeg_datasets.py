"""
EEG Datasets for Quantum Hydra/Mamba/Transformer Models

A unified module for loading all EEG datasets:
1. PhysioNet Motor Imagery - 2-class (left/right hand movement)
2. SEED Emotion - 3-class (positive/negative/neutral)
3. FACED Valence - 3-class (positive/negative/neutral)
4. FACED Emotion - 9-class (fine-grained emotions)

Usage:
    from data_loaders.eeg_datasets import load_eeg_dataset, DATASET_INFO

    # Load any EEG dataset with unified interface
    train_loader, val_loader, test_loader, input_dim, num_classes = load_eeg_dataset(
        dataset_name='seed',  # or 'physionet', 'faced_valence', 'faced_emotion'
        seed=2024,
        batch_size=32
    )

Author: Junghoon Park
Date: December 2024
"""

import torch
from torch.utils.data import DataLoader
from typing import Tuple, Literal, Optional


# ============================================================================
# Dataset Information
# ============================================================================

DATASET_INFO = {
    'physionet': {
        'name': 'PhysioNet Motor Imagery',
        'task': 'Motor Imagery Classification',
        'channels': 64,
        'sampling_rate': 160,  # Hz (can resample to 80)
        'num_classes': 2,
        'class_names': ['left_hand', 'right_hand'],
        'subjects': 109,
        'default_chunk_size': 248,  # 3.1s at 80Hz
        'reference': 'Schalk et al. (2004)',
        'download': 'Auto-downloaded via MNE'
    },
    'seed': {
        'name': 'SJTU SEED',
        'task': 'Emotion Recognition',
        'channels': 62,
        'sampling_rate': 200,  # Hz
        'num_classes': 3,
        'class_names': ['negative', 'neutral', 'positive'],
        'subjects': 15,
        'sessions': 3,  # Total 45 recordings
        'default_chunk_size': 200,  # 1 second
        'reference': 'Zheng & Lu (2015)',
        'download': 'https://bcmi.sjtu.edu.cn/home/seed/'
    },
    'faced_valence': {
        'name': 'FACED (Valence)',
        'task': 'Emotion Recognition (Valence)',
        'channels': 30,
        'sampling_rate': 250,  # Hz
        'num_classes': 3,
        'class_names': ['negative', 'neutral', 'positive'],
        'subjects': 123,
        'default_chunk_size': 250,  # 1 second
        'reference': 'Tsinghua (2023)',
        'download': 'https://www.synapse.org/#!Synapse:syn50614194'
    },
    'faced_emotion': {
        'name': 'FACED (Discrete Emotion)',
        'task': 'Fine-grained Emotion Recognition',
        'channels': 30,
        'sampling_rate': 250,  # Hz
        'num_classes': 9,
        'class_names': ['anger', 'disgust', 'fear', 'sadness', 'neutral',
                       'amusement', 'inspiration', 'joy', 'tenderness'],
        'subjects': 123,
        'default_chunk_size': 250,  # 1 second
        'reference': 'Tsinghua (2023)',
        'download': 'https://www.synapse.org/#!Synapse:syn50614194'
    }
}


# ============================================================================
# Unified EEG Loader
# ============================================================================

def load_eeg_dataset(
    dataset_name: Literal['physionet', 'seed', 'faced_valence', 'faced_emotion'],
    seed: int = 2024,
    device: Optional[torch.device] = None,
    batch_size: int = 32,
    chunk_size: Optional[int] = None,
    root_path: Optional[str] = None,
    **kwargs
) -> Tuple[DataLoader, DataLoader, DataLoader, Tuple, int]:
    """
    Unified EEG dataset loader.

    Args:
        dataset_name: One of 'physionet', 'seed', 'faced_valence', 'faced_emotion'
        seed: Random seed for reproducibility
        device: PyTorch device (auto-detect if None)
        batch_size: Batch size for DataLoaders
        chunk_size: Timesteps per sample (uses dataset default if None)
        root_path: Dataset path (uses default if None)
        **kwargs: Additional loader arguments

    Returns:
        train_loader, val_loader, test_loader, input_dim, num_classes

    Example:
        >>> loaders = load_eeg_dataset('seed', seed=2024, batch_size=32)
        >>> train_loader, val_loader, test_loader, input_dim, num_classes = loaders
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if dataset_name not in DATASET_INFO:
        raise ValueError(f"Unknown dataset: {dataset_name}. "
                        f"Available: {list(DATASET_INFO.keys())}")

    info = DATASET_INFO[dataset_name]
    chunk = chunk_size if chunk_size else info['default_chunk_size']

    print(f"\n{'='*60}")
    print(f"Loading: {info['name']}")
    print(f"Task: {info['task']}")
    print(f"Channels: {info['channels']}, Sampling Rate: {info['sampling_rate']} Hz")
    print(f"Classes: {info['num_classes']} - {info['class_names']}")
    print(f"Chunk size: {chunk}")
    print(f"{'='*60}\n")

    # Load specific dataset
    if dataset_name == 'physionet':
        from .Load_PhysioNet_EEG_NoPrompt import load_eeg_ts_revised

        sampling_freq = kwargs.get('sampling_freq', 80)
        sample_size = kwargs.get('sample_size', 50)

        train_loader, val_loader, test_loader, input_dim = load_eeg_ts_revised(
            seed=seed,
            device=device,
            batch_size=batch_size,
            sampling_freq=sampling_freq,
            sample_size=sample_size
        )
        num_classes = 2

    elif dataset_name == 'seed':
        from .Load_SEED_EEG import load_seed_eeg_simple

        path = root_path if root_path else './Preprocessed_EEG'
        io_path = kwargs.get('io_path', './seed_io')

        train_loader, val_loader, test_loader, input_dim = load_seed_eeg_simple(
            seed=seed,
            device=device,
            batch_size=batch_size,
            chunk_size=chunk,
            root_path=path,
            io_path=io_path,
            test_size=kwargs.get('test_size', 0.2),
            val_size=kwargs.get('val_size', 0.1)
        )
        num_classes = 3

    elif dataset_name == 'faced_valence':
        from .Load_FACED_EEG import load_faced_eeg_simple

        path = root_path if root_path else './Processed_data'
        io_path = kwargs.get('io_path', './faced_io')

        train_loader, val_loader, test_loader, input_dim, num_classes = load_faced_eeg_simple(
            seed=seed,
            device=device,
            batch_size=batch_size,
            chunk_size=chunk,
            label_type='valence',
            root_path=path,
            io_path=io_path,
            test_size=kwargs.get('test_size', 0.2),
            val_size=kwargs.get('val_size', 0.1)
        )

    elif dataset_name == 'faced_emotion':
        from .Load_FACED_EEG import load_faced_eeg_simple

        path = root_path if root_path else './Processed_data'
        io_path = kwargs.get('io_path', './faced_io_emotion')

        train_loader, val_loader, test_loader, input_dim, num_classes = load_faced_eeg_simple(
            seed=seed,
            device=device,
            batch_size=batch_size,
            chunk_size=chunk,
            label_type='emotion',
            root_path=path,
            io_path=io_path,
            test_size=kwargs.get('test_size', 0.2),
            val_size=kwargs.get('val_size', 0.1)
        )

    print(f"\nLoaded successfully!")
    print(f"  Input shape: {input_dim}")
    print(f"  Classes: {num_classes}")
    print(f"  Train batches: {len(train_loader)}, Val: {len(val_loader)}, Test: {len(test_loader)}")

    return train_loader, val_loader, test_loader, input_dim, num_classes


# ============================================================================
# Utility Functions
# ============================================================================

def list_datasets():
    """List all available EEG datasets."""
    return list(DATASET_INFO.keys())


def get_dataset_config(dataset_name: str) -> dict:
    """Get configuration for a specific dataset."""
    if dataset_name not in DATASET_INFO:
        raise ValueError(f"Unknown dataset: {dataset_name}")
    return DATASET_INFO[dataset_name]


def print_comparison_table():
    """Print comparison of all EEG datasets."""
    print("\n" + "=" * 85)
    print("EEG Datasets for Quantum Models")
    print("=" * 85)
    print(f"{'Dataset':<16} {'Task':<30} {'Ch':<4} {'Hz':<5} {'Classes':<8} {'Subjects':<8}")
    print("-" * 85)

    for name, info in DATASET_INFO.items():
        print(f"{name:<16} {info['task'][:29]:<30} {info['channels']:<4} "
              f"{info['sampling_rate']:<5} {info['num_classes']:<8} {info['subjects']:<8}")

    print("=" * 85)
    print("\nNotes:")
    print("  - PhysioNet: Auto-downloads via MNE library")
    print("  - SEED: Requires manual download from BCMI lab website")
    print("  - FACED: Requires manual download from Synapse (registration needed)")
    print()


# ============================================================================
# Module Test
# ============================================================================

if __name__ == "__main__":
    print_comparison_table()

    # Quick test of the interface
    print("\nTesting unified loader interface...")
    print("Available datasets:", list_datasets())

    # Show PhysioNet config
    config = get_dataset_config('physionet')
    print(f"\nPhysioNet config: {config}")
