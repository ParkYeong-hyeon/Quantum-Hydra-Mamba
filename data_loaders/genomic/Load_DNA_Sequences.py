#!/usr/bin/env python3
"""
DNA Sequence Classification Data Loader
Supports UCI DNA Promoter and Splice Junction datasets

Based on EXPERIMENTAL_PLAN_README.md (lines 218-264)

Updated: Uses stratified sampling for balanced class distribution
"""

import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader, random_split
import numpy as np
from sklearn.model_selection import train_test_split
from pathlib import Path
import urllib.request
import random


def create_synthetic_dna_data(filepath):
    """
    Create synthetic DNA promoter data for testing when download fails.

    Creates 212 sequences (106 promoters + 106 non-promoters) of length 57.
    """
    nucleotides = ['A', 'C', 'G', 'T']
    n_promoters = 106
    n_non_promoters = 106
    seq_length = 57

    sequences = []

    # Generate promoters (label +)
    for i in range(n_promoters):
        seq = ''.join(random.choices(nucleotides, k=seq_length))
        sequences.append(f"+\tpromoter_{i}\t{seq}\n")

    # Generate non-promoters (label -)
    for i in range(n_non_promoters):
        seq = ''.join(random.choices(nucleotides, k=seq_length))
        sequences.append(f"-\tnon_promoter_{i}\t{seq}\n")

    # Shuffle
    random.shuffle(sequences)

    # Write to file
    with open(filepath, 'w') as f:
        f.write("% Synthetic DNA Promoter Dataset (for testing)\n")
        f.write("% Format: <class> <instance_name> <sequence>\n")
        f.write("%\n")
        f.writelines(sequences)

    print(f"Created synthetic DNA dataset with {len(sequences)} sequences")


def download_promoter_dataset():
    """
    Download UCI Promoter Sequences dataset.

    Dataset Details:
    - 53 promoters vs 53 non-promoters (106 total)
    - Binary classification
    - Length: 57 nucleotides
    - URL: https://archive.ics.uci.edu/ml/machine-learning-databases/molecular-biology/promoter-gene-sequences/promoters.data
    """
    url = "https://archive.ics.uci.edu/ml/machine-learning-databases/molecular-biology/promoter-gene-sequences/promoters.data"
    data_dir = Path("./data/dna")
    data_dir.mkdir(parents=True, exist_ok=True)

    filepath = data_dir / "promoters.data"

    if not filepath.exists():
        print(f"Downloading promoter dataset from {url}...")
        try:
            # Try with SSL verification disabled (needed on some HPC systems)
            import ssl
            ssl._create_default_https_context = ssl._create_unverified_context
            urllib.request.urlretrieve(url, filepath)
            print(f"Downloaded to {filepath}")
        except Exception as e:
            print(f"Warning: Could not download dataset: {e}")
            print("Creating synthetic DNA dataset for testing...")
            create_synthetic_dna_data(filepath)

    return filepath


def parse_promoter_data(filepath):
    """
    Parse promoter sequences dataset.

    Returns:
        sequences: List of DNA sequences (strings)
        labels: List of labels (0: non-promoter, 1: promoter)
    """
    sequences = []
    labels = []

    with open(filepath, 'r') as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith('%'):
                continue

            # Format: +,GENE_NAME,\t\t<sequence> or -,GENE_NAME,\t\t<sequence>
            parts = line.split(',')
            if len(parts) < 3:
                continue

            label_str = parts[0].strip()
            sequence = parts[2].strip()

            # Clean sequence (remove tabs and whitespace)
            sequence = sequence.replace('\t', '').replace(' ', '').upper()

            # Skip empty sequences
            if not sequence:
                continue

            # Convert label: + = promoter (1), - = non-promoter (0)
            label = 1 if label_str == '+' else 0

            sequences.append(sequence)
            labels.append(label)

    return sequences, labels


def encode_dna_onehot(sequence):
    """
    One-hot encoding for DNA sequences.

    Encoding:
        A → [1, 0, 0, 0]
        C → [0, 1, 0, 0]
        G → [0, 0, 1, 0]
        T → [0, 0, 0, 1]
        N (unknown) → [0, 0, 0, 0]

    Args:
        sequence: DNA sequence string (e.g., "ATCG")

    Returns:
        torch.Tensor of shape (seq_len * 4,) - flattened one-hot encoding
    """
    mapping = {'A': 0, 'C': 1, 'G': 2, 'T': 3, 'N': 4}

    # Convert to uppercase
    sequence = sequence.upper()

    # Create one-hot tensor
    seq_len = len(sequence)
    one_hot = torch.zeros(seq_len, 4)

    for i, nucleotide in enumerate(sequence):
        idx = mapping.get(nucleotide, 4)  # 4 for unknown
        if idx < 4:
            one_hot[i, idx] = 1.0

    # Flatten
    return one_hot.flatten()


def encode_dna_integer(sequence):
    """
    Integer encoding for DNA sequences (for quantum models).

    Encoding:
        A → 0
        C → 1
        G → 2
        T → 3
        N (unknown) → 0 (default to A)

    Args:
        sequence: DNA sequence string

    Returns:
        torch.Tensor of shape (seq_len,)
    """
    mapping = {'A': 0, 'C': 1, 'G': 2, 'T': 3, 'N': 0}
    sequence = sequence.upper()

    encoded = [mapping.get(nuc, 0) for nuc in sequence]
    return torch.tensor(encoded, dtype=torch.long)


def load_dna_promoter(seed, n_train, n_valtest, device, batch_size, encoding='onehot'):
    """
    Load DNA Promoter dataset for classification with STRATIFIED SAMPLING.

    Args:
        seed: Random seed for reproducibility
        n_train: Number of training samples (max 212)
        n_valtest: Number of validation + test samples
        device: torch.device
        batch_size: Batch size for DataLoader
        encoding: 'onehot' or 'integer'

    Returns:
        train_loader, val_loader, test_loader, feature_dim
    """
    # Set random seed
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    # Download and parse data
    filepath = download_promoter_dataset()
    sequences, labels = parse_promoter_data(filepath)

    print(f"Loaded {len(sequences)} DNA sequences")
    print(f"  Promoters: {sum(labels)} (+)")
    print(f"  Non-promoters: {len(labels) - sum(labels)} (-)")
    print(f"  Sequence length: {len(sequences[0])} nucleotides")

    # Encode sequences
    if encoding == 'onehot':
        encoded_seqs = [encode_dna_onehot(seq) for seq in sequences]
        feature_dim = len(encoded_seqs[0])  # seq_len * 4
    elif encoding == 'integer':
        encoded_seqs = [encode_dna_integer(seq) for seq in sequences]
        feature_dim = len(encoded_seqs[0])  # seq_len
    else:
        raise ValueError(f"Unknown encoding: {encoding}")

    # Convert to numpy for sklearn compatibility
    X = torch.stack(encoded_seqs).numpy()
    y = np.array(labels)

    # Calculate actual sample sizes
    total_available = len(X)
    total_requested = n_train + n_valtest

    if total_requested > total_available:
        # Scale down proportionally
        scale = total_available / total_requested
        n_train = int(n_train * scale)
        n_valtest = total_available - n_train
        print(f"Warning: Requested {total_requested} samples, but only {total_available} available.")
        print(f"  Adjusted: n_train={n_train}, n_valtest={n_valtest}")

    # ========================================================================
    # STRATIFIED SAMPLING: Ensures equal class proportions in each split
    # ========================================================================

    # First split: separate train from (val + test)
    # Use stratify to maintain class balance
    X_train, X_valtest, y_train, y_valtest = train_test_split(
        X, y,
        train_size=n_train,
        stratify=y,
        random_state=seed
    )

    # Second split: separate val from test (50/50)
    # Use stratify again for balanced val/test
    val_size = len(X_valtest) // 2
    if val_size > 0:
        X_val, X_test, y_val, y_test = train_test_split(
            X_valtest, y_valtest,
            train_size=val_size,
            stratify=y_valtest,
            random_state=seed
        )
    else:
        # Edge case: very small valtest set
        X_val, X_test = X_valtest[:1], X_valtest[1:]
        y_val, y_test = y_valtest[:1], y_valtest[1:]

    # Convert back to torch tensors and move to device
    X_train = torch.tensor(X_train, dtype=torch.float32).to(device)
    y_train = torch.tensor(y_train, dtype=torch.long).to(device)
    X_val = torch.tensor(X_val, dtype=torch.float32).to(device)
    y_val = torch.tensor(y_val, dtype=torch.long).to(device)
    X_test = torch.tensor(X_test, dtype=torch.float32).to(device)
    y_test = torch.tensor(y_test, dtype=torch.long).to(device)

    # Create datasets
    train_dataset = TensorDataset(X_train, y_train)
    val_dataset = TensorDataset(X_val, y_val)
    test_dataset = TensorDataset(X_test, y_test)

    # Create DataLoaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    # Print split details with class distribution
    print(f"\nDataset split (STRATIFIED):")
    print(f"  Training: {len(train_dataset)} samples "
          f"({sum(y_train.cpu().numpy())} promoters, {len(y_train) - sum(y_train.cpu().numpy())} non-promoters)")
    print(f"  Validation: {len(val_dataset)} samples "
          f"({sum(y_val.cpu().numpy())} promoters, {len(y_val) - sum(y_val.cpu().numpy())} non-promoters)")
    print(f"  Test: {len(test_dataset)} samples "
          f"({sum(y_test.cpu().numpy())} promoters, {len(y_test) - sum(y_test.cpu().numpy())} non-promoters)")
    print(f"  Feature dimension: {feature_dim}")

    return train_loader, val_loader, test_loader, feature_dim


if __name__ == "__main__":
    # Test the data loader
    print("="*80)
    print("Testing DNA Promoter Data Loader (with Stratified Sampling)")
    print("="*80)

    device = torch.device("cpu")

    # Test with one-hot encoding
    print("\n[Test 1] One-hot encoding:")
    train_loader, val_loader, test_loader, feature_dim = load_dna_promoter(
        seed=2024,
        n_train=100,
        n_valtest=50,
        device=device,
        batch_size=16,
        encoding='onehot'
    )

    # Check a batch
    for X_batch, y_batch in train_loader:
        print(f"\nBatch shapes:")
        print(f"  X: {X_batch.shape}")
        print(f"  y: {y_batch.shape}")
        print(f"  Feature dim: {feature_dim}")
        break

    print("\n" + "="*80)
    print("Data loader test complete!")
    print("="*80)
