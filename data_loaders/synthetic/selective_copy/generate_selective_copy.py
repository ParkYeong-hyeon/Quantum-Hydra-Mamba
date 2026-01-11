"""
Generate Selective Copy Task Dataset for Long-Range Sequence Learning

The Selective Copy Task tests a model's ability to:
1. Remember specific marked elements across a long sequence
2. Selectively filter relevant information
3. Reproduce only the marked elements in order

Task:
    Given a sequence of tokens with sparse markers, output only the marked tokens
    in their original order.

Input:
    - Token stream:  [a, b, c, d, e, f, g, h, ...]  (random tokens)
    - Marker stream: [1, 0, 0, 1, 0, 0, 1, 0, ...]  (sparse binary markers)

Output:
    - Marked tokens: [a, d, g]  (only tokens where marker=1, in order)

Why this tests long-range learning:
    - Markers are sparse (e.g., 5-10 markers in 200+ sequence)
    - Model must remember marked tokens across long distances
    - Tests both memory AND selective attention
    - Relevant for SSM's selective Delta mechanism

Evaluation:
    - For sequence output: Token-level accuracy (exact match)
    - For classification: Predict the sequence of marked tokens as a class

Implementation Notes:
    - We frame this as sequence-to-sequence with a fixed output length
    - The output length equals the number of markers (fixed per dataset)
    - Models predict the marked tokens in order

Input Shape for Models:
    - Saved: (num_samples, seq_len, num_channels)
    - Channels: [token_value, marker]
    - DataLoader transposes to (num_samples, num_channels, seq_len)

Author: Research Team
Date: January 2026
"""

import torch
import numpy as np
import argparse
from tqdm import tqdm
from pathlib import Path


def generate_selective_copy_sample(
    seq_len: int,
    num_markers: int,
    vocab_size: int,
    marker_strategy: str = "uniform"
):
    """
    Generate a single Selective Copy sample.

    Args:
        seq_len: Length of the input sequence
        num_markers: Number of markers (and output length)
        vocab_size: Size of token vocabulary (tokens are integers 0 to vocab_size-1)
        marker_strategy: How to place markers
            - "uniform": Random positions throughout
            - "spread": Evenly spread across sequence
            - "clustered": Clustered in groups

    Returns:
        tokens: (seq_len,) random token values (normalized to [0, 1])
        markers: (seq_len,) binary markers
        target_tokens: (num_markers,) the marked tokens in order
        target_indices: (num_markers,) indices of marked positions
    """
    # Generate random tokens (normalized to [0, 1] for continuous representation)
    # Using continuous values rather than discrete for easier gradient flow
    tokens = torch.rand(seq_len)

    # Initialize markers
    markers = torch.zeros(seq_len)

    # Place markers based on strategy
    if marker_strategy == "uniform":
        # Random positions
        marker_positions = torch.randperm(seq_len)[:num_markers].sort()[0]

    elif marker_strategy == "spread":
        # Evenly spread with small random jitter
        base_positions = torch.linspace(0, seq_len - 1, num_markers).long()
        jitter = torch.randint(-2, 3, (num_markers,))
        marker_positions = torch.clamp(base_positions + jitter, 0, seq_len - 1)
        marker_positions = marker_positions.unique()

        # Pad if we lost positions due to overlap
        while len(marker_positions) < num_markers:
            extra = torch.randint(0, seq_len, (1,))
            if extra not in marker_positions:
                marker_positions = torch.cat([marker_positions, extra])
        marker_positions = marker_positions[:num_markers].sort()[0]

    elif marker_strategy == "clustered":
        # Create clusters of markers
        num_clusters = max(1, num_markers // 3)
        cluster_centers = torch.randperm(seq_len - 10)[:num_clusters] + 5

        positions = []
        markers_per_cluster = num_markers // num_clusters

        for center in cluster_centers:
            cluster_positions = center + torch.randint(-3, 4, (markers_per_cluster,))
            cluster_positions = torch.clamp(cluster_positions, 0, seq_len - 1)
            positions.extend(cluster_positions.tolist())

        # Add remaining markers randomly
        while len(positions) < num_markers:
            pos = torch.randint(0, seq_len, (1,)).item()
            if pos not in positions:
                positions.append(pos)

        marker_positions = torch.tensor(sorted(positions[:num_markers]))

    else:
        raise ValueError(f"Unknown marker_strategy: {marker_strategy}")

    # Set markers
    markers[marker_positions] = 1.0

    # Extract target tokens (marked tokens in order)
    target_tokens = tokens[marker_positions]
    target_indices = marker_positions

    return tokens, markers, target_tokens, target_indices


def generate_selective_copy_dataset(
    num_samples: int,
    seq_len: int,
    num_markers: int = 8,
    vocab_size: int = 10,
    marker_strategy: str = "uniform",
    filename: str = "selective_copy_dataset.pt",
    seed: int = None
):
    """
    Generate and save the Selective Copy dataset.

    Args:
        num_samples: Number of samples to generate
        seq_len: Sequence length
        num_markers: Number of markers per sequence (also output length)
        vocab_size: Vocabulary size (unused for continuous tokens)
        marker_strategy: Marker placement strategy
        filename: Output filename
        seed: Random seed for reproducibility

    Returns:
        Dictionary containing sequences, targets, and parameters
    """
    if seed is not None:
        torch.manual_seed(seed)
        np.random.seed(seed)

    print(f"Generating Selective Copy dataset...")
    print(f"  - num_samples: {num_samples}")
    print(f"  - seq_len: {seq_len}")
    print(f"  - num_markers: {num_markers}")
    print(f"  - marker_strategy: {marker_strategy}")
    print(f"  - marker_density: {num_markers / seq_len * 100:.1f}%")

    # Allocate tensors
    # Input: (num_samples, seq_len, 2) where channels are [token, marker]
    sequences = torch.zeros(num_samples, seq_len, 2, dtype=torch.float32)

    # Target: (num_samples, num_markers) the marked tokens
    targets = torch.zeros(num_samples, num_markers, dtype=torch.float32)

    # Also store marker indices for verification
    marker_indices = torch.zeros(num_samples, num_markers, dtype=torch.long)

    # Generate samples
    for i in tqdm(range(num_samples), desc="Generating samples"):
        tokens, markers, target_tokens, target_idx = generate_selective_copy_sample(
            seq_len=seq_len,
            num_markers=num_markers,
            vocab_size=vocab_size,
            marker_strategy=marker_strategy
        )

        sequences[i, :, 0] = tokens
        sequences[i, :, 1] = markers
        targets[i] = target_tokens
        marker_indices[i] = target_idx

    # Compute statistics
    print(f"\nDataset Statistics:")
    print(f"  - Input shape: {sequences.shape}")
    print(f"  - Target shape: {targets.shape}")
    print(f"  - Target value range: [{targets.min():.4f}, {targets.max():.4f}]")

    # Baseline MSE (predicting 0.5 for all)
    baseline_mse = ((targets - 0.5) ** 2).mean().item()
    print(f"  - Baseline MSE (predict 0.5): {baseline_mse:.4f}")

    # Save dataset
    dataset = {
        'sequences': sequences,
        'targets': targets,
        'marker_indices': marker_indices,
        'params': {
            'num_samples': num_samples,
            'seq_len': seq_len,
            'num_channels': 2,
            'num_markers': num_markers,
            'output_len': num_markers,
            'vocab_size': vocab_size,
            'marker_strategy': marker_strategy,
            'marker_density': num_markers / seq_len,
            'task': 'selective_copy',
            'target_type': 'regression_sequence',
            'baseline_mse': baseline_mse,
            'seed': seed
        }
    }

    # Create directory if needed
    filepath = Path(filename)
    filepath.parent.mkdir(parents=True, exist_ok=True)

    torch.save(dataset, filename)
    print(f"\nDataset saved to {filename}")

    return dataset


def verify_dataset(filename: str):
    """Verify the generated dataset."""
    print(f"\nVerifying dataset: {filename}")

    data = torch.load(filename)
    sequences = data['sequences']
    targets = data['targets']
    marker_indices = data['marker_indices']
    params = data['params']

    print(f"  - Sequences shape: {sequences.shape}")
    print(f"  - Targets shape: {targets.shape}")
    print(f"  - Parameters: {params}")

    # Verify a few samples
    print(f"\nSample verification (checking copy correctness):")
    for i in range(min(3, len(sequences))):
        tokens = sequences[i, :, 0]
        markers = sequences[i, :, 1]
        target = targets[i]
        indices = marker_indices[i]

        # Extract marked tokens
        marked_positions = (markers == 1.0).nonzero().squeeze()
        if marked_positions.dim() == 0:
            marked_positions = marked_positions.unsqueeze(0)

        extracted_tokens = tokens[marked_positions]

        print(f"  Sample {i}:")
        print(f"    Marker positions: {marked_positions.tolist()}")
        print(f"    Stored indices: {indices.tolist()}")
        print(f"    Extracted tokens: {extracted_tokens.tolist()}")
        print(f"    Target tokens: {target.tolist()}")
        print(f"    Match: {torch.allclose(extracted_tokens, target, atol=1e-6)}")


def generate_multiple_sequence_lengths(
    base_dir: str = "./data/selective_copy",
    num_samples: int = 5000,
    seq_lengths: list = [100, 200, 500, 1000],
    num_markers: int = 8,
    seed: int = 2024
):
    """
    Generate datasets for multiple sequence lengths.

    Args:
        base_dir: Base directory for datasets
        num_samples: Samples per dataset
        seq_lengths: List of sequence lengths to generate
        num_markers: Number of markers
        seed: Base random seed
    """
    base_dir = Path(base_dir)
    base_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("Generating Selective Copy datasets for multiple lengths")
    print("=" * 60)

    for seq_len in seq_lengths:
        print(f"\n{'='*60}")
        print(f"Sequence length: {seq_len}")
        print(f"{'='*60}")

        filename = base_dir / f"selective_copy_L{seq_len}_M{num_markers}_seed{seed}.pt"

        generate_selective_copy_dataset(
            num_samples=num_samples,
            seq_len=seq_len,
            num_markers=num_markers,
            marker_strategy="uniform",
            filename=str(filename),
            seed=seed
        )

    print("\n" + "=" * 60)
    print("All datasets generated!")
    print("=" * 60)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate Selective Copy Task Dataset")
    parser.add_argument('--num_samples', type=int, default=5000,
                        help='Number of samples to generate')
    parser.add_argument('--seq_len', type=int, default=200,
                        help='Sequence length')
    parser.add_argument('--num_markers', type=int, default=8,
                        help='Number of markers per sequence')
    parser.add_argument('--marker_strategy', type=str, default='uniform',
                        choices=['uniform', 'spread', 'clustered'],
                        help='Marker placement strategy')
    parser.add_argument('--filename', type=str, default='selective_copy_dataset.pt',
                        help='Output filename')
    parser.add_argument('--seed', type=int, default=2024,
                        help='Random seed')
    parser.add_argument('--verify', action='store_true',
                        help='Verify dataset after generation')
    parser.add_argument('--multi_length', action='store_true',
                        help='Generate for multiple sequence lengths')

    args = parser.parse_args()

    if args.multi_length:
        generate_multiple_sequence_lengths(
            num_samples=args.num_samples,
            num_markers=args.num_markers,
            seed=args.seed
        )
    else:
        generate_selective_copy_dataset(
            num_samples=args.num_samples,
            seq_len=args.seq_len,
            num_markers=args.num_markers,
            marker_strategy=args.marker_strategy,
            filename=args.filename,
            seed=args.seed
        )

        if args.verify:
            verify_dataset(args.filename)
