#!/usr/bin/env python3
"""
Data Loader for Genomic Benchmarks
https://github.com/ML-Bioinfo-CEITEC/genomic_benchmarks

Supports multiple genomic sequence classification datasets with proper
train/val/test splits, stratification, and reproducible random seeds.
"""

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from pathlib import Path

# Genomic Benchmarks package
from genomic_benchmarks.loc2seq import download_dataset
from genomic_benchmarks.data_check import list_datasets, info


def one_hot_encode(sequence: str, max_len: int = None) -> np.ndarray:
    """
    One-hot encode a DNA sequence.

    Args:
        sequence: DNA sequence string (A, C, G, T, N)
        max_len: Maximum length to pad/truncate to (None = use sequence length)

    Returns:
        One-hot encoded array of shape (4, max_len or seq_len)
    """
    mapping = {'A': 0, 'C': 1, 'G': 2, 'T': 3, 'N': 4}
    seq_len = len(sequence)

    # Determine output length
    if max_len is None:
        out_len = seq_len
    else:
        out_len = max_len

    encoding = np.zeros((4, out_len), dtype=np.float32)

    # Encode up to min(seq_len, out_len)
    for i in range(min(seq_len, out_len)):
        nucleotide = sequence[i].upper()
        if nucleotide in mapping and mapping[nucleotide] < 4:
            encoding[mapping[nucleotide], i] = 1.0
        # N (unknown) is encoded as all zeros

    return encoding


def load_genomic_benchmark(
    dataset_name: str = 'human_nontata_promoters',
    seed: int = 2024,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
    batch_size: int = 32,
    device: str = 'cuda',
    max_samples: int = None,
    cache_dir: str = None
):
    """
    Load a Genomic Benchmark dataset with proper train/val/test splits.

    Args:
        dataset_name: Name of the dataset from Genomic Benchmarks
            Options: 'human_enhancers_cohn', 'human_enhancers_ensembl',
                     'human_nontata_promoters', 'human_ocr_ensembl',
                     'demo_coding_vs_intergenomic_seqs', 'demo_human_or_worm',
                     'drosophila_enhancers_stark', 'dummy_mouse_enhancers_ensembl'
        seed: Random seed for reproducibility
        val_ratio: Proportion of training data for validation
        test_ratio: Proportion of total data for testing
        batch_size: Batch size for DataLoaders
        device: Device for tensors ('cuda' or 'cpu')
        max_samples: Maximum samples to use (None for all)
        cache_dir: Directory to cache downloaded data

    Returns:
        train_loader, val_loader, test_loader, metadata
    """
    # Set random seeds
    np.random.seed(seed)
    torch.manual_seed(seed)

    # Download dataset
    if cache_dir is None:
        cache_dir = Path.home() / '.genomic_benchmarks'

    print(f"Loading dataset: {dataset_name}")

    # Check if dataset already exists to avoid race conditions in parallel jobs
    dataset_path = Path(cache_dir) / dataset_name
    if dataset_path.exists() and (dataset_path / 'train').exists():
        print(f"Using cached dataset at {dataset_path}")
    else:
        print(f"Downloading dataset to {dataset_path}")
        dataset_path = download_dataset(dataset_name, dest_path=cache_dir)

    # Load sequences and labels
    sequences = []
    labels = []

    # Dataset structure: dataset_path/train/class_0, dataset_path/train/class_1, etc.
    # Class directories can be named as integers (0, 1) or strings (negative, positive)
    class_name_to_label = {}

    for split in ['train', 'test']:
        split_path = Path(dataset_path) / split
        if not split_path.exists():
            continue

        for class_dir in sorted(split_path.iterdir()):
            if not class_dir.is_dir():
                continue

            # Handle different naming conventions
            class_name = class_dir.name
            if class_name not in class_name_to_label:
                # Try to parse as int, otherwise assign based on order
                try:
                    class_name_to_label[class_name] = int(class_name)
                except ValueError:
                    # For string names like 'negative', 'positive'
                    # Assign 0 to first class (alphabetically), 1 to second
                    class_name_to_label[class_name] = len(class_name_to_label)

            class_label = class_name_to_label[class_name]

            for seq_file in class_dir.glob('*.txt'):
                with open(seq_file, 'r') as f:
                    seq = f.read().strip()
                    sequences.append(seq)
                    labels.append(class_label)

    print(f"Class mapping: {class_name_to_label}")

    sequences = np.array(sequences)
    labels = np.array(labels)

    print(f"Total samples: {len(sequences)}")
    print(f"Sequence length: {len(sequences[0])}")
    print(f"Class distribution: {np.bincount(labels)}")

    # Subsample if max_samples specified
    if max_samples is not None and len(sequences) > max_samples:
        indices = np.random.choice(len(sequences), max_samples, replace=False)
        sequences = sequences[indices]
        labels = labels[indices]
        print(f"Subsampled to {max_samples} samples")

    # First split: separate test set
    X_trainval, X_test, y_trainval, y_test = train_test_split(
        sequences, labels,
        test_size=test_ratio,
        stratify=labels,
        random_state=seed
    )

    # Second split: separate validation set from training
    val_ratio_adjusted = val_ratio / (1 - test_ratio)
    X_train, X_val, y_train, y_val = train_test_split(
        X_trainval, y_trainval,
        test_size=val_ratio_adjusted,
        stratify=y_trainval,
        random_state=seed
    )

    print(f"\nSplit sizes:")
    print(f"  Train: {len(X_train)} (class dist: {np.bincount(y_train)})")
    print(f"  Val:   {len(X_val)} (class dist: {np.bincount(y_val)})")
    print(f"  Test:  {len(X_test)} (class dist: {np.bincount(y_test)})")

    # Find max sequence length across all splits for padding
    all_seqs = np.concatenate([X_train, X_val, X_test])
    max_seq_len = max(len(s) for s in all_seqs)
    print(f"  Max sequence length: {max_seq_len}")

    # One-hot encode sequences with padding to max length
    def encode_sequences(seqs, max_len):
        encoded = np.array([one_hot_encode(s, max_len) for s in seqs])
        return encoded  # Shape: (n_samples, 4, max_len)

    X_train_encoded = encode_sequences(X_train, max_seq_len)
    X_val_encoded = encode_sequences(X_val, max_seq_len)
    X_test_encoded = encode_sequences(X_test, max_seq_len)

    # Convert to tensors
    X_train_tensor = torch.tensor(X_train_encoded, dtype=torch.float32)
    X_val_tensor = torch.tensor(X_val_encoded, dtype=torch.float32)
    X_test_tensor = torch.tensor(X_test_encoded, dtype=torch.float32)

    y_train_tensor = torch.tensor(y_train, dtype=torch.long)
    y_val_tensor = torch.tensor(y_val, dtype=torch.long)
    y_test_tensor = torch.tensor(y_test, dtype=torch.long)

    # Create datasets
    train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
    val_dataset = TensorDataset(X_val_tensor, y_val_tensor)
    test_dataset = TensorDataset(X_test_tensor, y_test_tensor)

    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        drop_last=False
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False
    )

    # Metadata
    seq_len = max_seq_len  # Use padded length
    n_classes = len(np.unique(labels))

    metadata = {
        'dataset_name': dataset_name,
        'n_train': len(X_train),
        'n_val': len(X_val),
        'n_test': len(X_test),
        'seq_len': seq_len,
        'n_channels': 4,  # One-hot encoding dimension
        'n_classes': n_classes,
        'class_distribution': {
            'train': np.bincount(y_train).tolist(),
            'val': np.bincount(y_val).tolist(),
            'test': np.bincount(y_test).tolist()
        }
    }

    print(f"\nData shapes:")
    print(f"  X: (batch, 4, {seq_len})")
    print(f"  y: (batch,)")

    return train_loader, val_loader, test_loader, metadata


def list_available_datasets():
    """List all available datasets in Genomic Benchmarks."""
    datasets = list_datasets()
    print("Available Genomic Benchmark datasets:")
    for ds in datasets:
        print(f"  - {ds}")
    return datasets


def get_dataset_info(dataset_name: str):
    """Get information about a specific dataset."""
    return info(dataset_name)


# Convenience functions for specific datasets
def load_human_nontata_promoters(seed=2024, batch_size=32, device='cuda', max_samples=None):
    """Load Human Non-TATA Promoters dataset (31K samples, 251 bp)."""
    return load_genomic_benchmark(
        dataset_name='human_nontata_promoters',
        seed=seed,
        batch_size=batch_size,
        device=device,
        max_samples=max_samples
    )


def load_human_enhancers_cohn(seed=2024, batch_size=32, device='cuda', max_samples=None):
    """Load Human Enhancers Cohn dataset (27K samples, 500 bp)."""
    return load_genomic_benchmark(
        dataset_name='human_enhancers_cohn',
        seed=seed,
        batch_size=batch_size,
        device=device,
        max_samples=max_samples
    )


def load_demo_coding_vs_intergenomic(seed=2024, batch_size=32, device='cuda', max_samples=None):
    """Load Coding vs Intergenomic demo dataset (78K samples, 200 bp)."""
    return load_genomic_benchmark(
        dataset_name='demo_coding_vs_intergenomic_seqs',
        seed=seed,
        batch_size=batch_size,
        device=device,
        max_samples=max_samples
    )


if __name__ == '__main__':
    # Test the loader
    print("Testing Genomic Benchmarks loader...\n")

    # List available datasets
    list_available_datasets()
    print()

    # Load a dataset with a small subset for testing
    train_loader, val_loader, test_loader, metadata = load_genomic_benchmark(
        dataset_name='human_nontata_promoters',
        seed=2024,
        batch_size=32,
        max_samples=1000  # Small subset for testing
    )

    print(f"\nMetadata: {metadata}")

    # Check a batch
    for X, y in train_loader:
        print(f"\nBatch X shape: {X.shape}")
        print(f"Batch y shape: {y.shape}")
        print(f"Batch y values: {y[:10]}")
        break
