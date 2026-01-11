"""
Generate Adding Problem Dataset for Long-Range Sequence Learning

The Adding Problem (Hochreiter & Schmidhuber, 1997) is a classic benchmark
for testing a model's ability to learn long-range dependencies.

Task:
    Given a sequence of random numbers in [0, 1] and a binary marker sequence
    with exactly two 1s, output the sum of the two marked numbers.

Input:
    - Value stream:  [0.3, 0.1, 0.7, 0.2, 0.5, ...]  (random in [0, 1])
    - Marker stream: [1,   0,   0,   0,   1,   ...]  (exactly two 1s)

Output:
    - Sum of marked values: 0.3 + 0.5 = 0.8

Why this tests long-range learning:
    - The two markers can be arbitrarily far apart
    - Model must remember the first marked value until it sees the second
    - Random baseline achieves MSE ~ 0.167 (guessing mean 0.5 + 0.5 = 1.0)
    - Perfect model achieves MSE ~ 0.0

Marker Placement Strategy:
    - First marker: randomly in first 10% of sequence
    - Second marker: randomly in last 10% of sequence
    - This maximizes the dependency distance

Input Shape for Models:
    - Saved: (num_samples, seq_len, 2) where channels are [value, marker]
    - DataLoader transposes to (num_samples, 2, seq_len) for model compatibility

Author: Research Team
Date: January 2026
Reference: Hochreiter & Schmidhuber (1997) "Long Short-Term Memory"
"""

import torch
import numpy as np
import argparse
from tqdm import tqdm
from pathlib import Path


def generate_adding_problem_sample(seq_len: int, marker_strategy: str = "extremes"):
    """
    Generate a single Adding Problem sample.

    Args:
        seq_len: Length of the sequence
        marker_strategy: How to place markers
            - "extremes": First 10% and last 10% (hardest, tests long-range)
            - "uniform": Any two positions (easier)
            - "halves": One in each half (medium)

    Returns:
        values: (seq_len,) random values in [0, 1]
        markers: (seq_len,) binary markers with exactly two 1s
        target: scalar sum of marked values
    """
    # Generate random values
    values = torch.rand(seq_len)

    # Initialize markers
    markers = torch.zeros(seq_len)

    # Place two markers based on strategy
    if marker_strategy == "extremes":
        # First marker in first 10%, second in last 10%
        first_region = max(1, seq_len // 10)
        last_region = max(1, seq_len // 10)

        pos1 = torch.randint(0, first_region, (1,)).item()
        pos2 = torch.randint(seq_len - last_region, seq_len, (1,)).item()

    elif marker_strategy == "halves":
        # One in each half
        pos1 = torch.randint(0, seq_len // 2, (1,)).item()
        pos2 = torch.randint(seq_len // 2, seq_len, (1,)).item()

    elif marker_strategy == "uniform":
        # Any two distinct positions
        positions = torch.randperm(seq_len)[:2]
        pos1, pos2 = positions[0].item(), positions[1].item()

    else:
        raise ValueError(f"Unknown marker_strategy: {marker_strategy}")

    markers[pos1] = 1.0
    markers[pos2] = 1.0

    # Target is sum of marked values
    target = values[pos1] + values[pos2]

    return values, markers, target


def generate_adding_problem_dataset(
    num_samples: int,
    seq_len: int,
    marker_strategy: str = "extremes",
    filename: str = "adding_problem_dataset.pt",
    seed: int = None
):
    """
    Generate and save the Adding Problem dataset.

    Args:
        num_samples: Number of samples to generate
        seq_len: Sequence length
        marker_strategy: Marker placement strategy
        filename: Output filename
        seed: Random seed for reproducibility

    Returns:
        Dictionary containing sequences, targets, and parameters
    """
    if seed is not None:
        torch.manual_seed(seed)
        np.random.seed(seed)

    print(f"Generating Adding Problem dataset...")
    print(f"  - num_samples: {num_samples}")
    print(f"  - seq_len: {seq_len}")
    print(f"  - marker_strategy: {marker_strategy}")

    # Allocate tensors
    # Shape: (num_samples, seq_len, 2) where dim 2 is [value, marker]
    sequences = torch.zeros(num_samples, seq_len, 2, dtype=torch.float32)
    targets = torch.zeros(num_samples, 1, dtype=torch.float32)

    # Generate samples
    for i in tqdm(range(num_samples), desc="Generating samples"):
        values, markers, target = generate_adding_problem_sample(seq_len, marker_strategy)

        sequences[i, :, 0] = values
        sequences[i, :, 1] = markers
        targets[i, 0] = target

    # Compute statistics
    target_mean = targets.mean().item()
    target_std = targets.std().item()

    # Baseline MSE (predicting mean = 1.0)
    baseline_mse = ((targets - 1.0) ** 2).mean().item()

    print(f"\nDataset Statistics:")
    print(f"  - Target mean: {target_mean:.4f} (expected ~1.0)")
    print(f"  - Target std: {target_std:.4f}")
    print(f"  - Baseline MSE (predict 1.0): {baseline_mse:.4f}")
    print(f"  - Random guess MSE: ~0.167")

    # Save dataset
    dataset = {
        'sequences': sequences,
        'targets': targets,
        'params': {
            'num_samples': num_samples,
            'seq_len': seq_len,
            'num_channels': 2,
            'marker_strategy': marker_strategy,
            'task': 'adding_problem',
            'target_type': 'regression',
            'baseline_mse': baseline_mse,
            'seed': seed
        }
    }

    # Create directory if needed
    filepath = Path(filename)
    filepath.parent.mkdir(parents=True, exist_ok=True)

    torch.save(dataset, filename)
    print(f"\nDataset saved to {filename}")
    print(f"  - Sequences shape: {sequences.shape}")
    print(f"  - Targets shape: {targets.shape}")

    return dataset


def verify_dataset(filename: str):
    """Verify the generated dataset."""
    print(f"\nVerifying dataset: {filename}")

    data = torch.load(filename)
    sequences = data['sequences']
    targets = data['targets']
    params = data['params']

    print(f"  - Sequences shape: {sequences.shape}")
    print(f"  - Targets shape: {targets.shape}")
    print(f"  - Parameters: {params}")

    # Verify a few samples
    print(f"\nSample verification (checking sum correctness):")
    for i in range(min(3, len(sequences))):
        values = sequences[i, :, 0]
        markers = sequences[i, :, 1]
        target = targets[i, 0]

        # Find marked positions
        marked_positions = (markers == 1.0).nonzero().squeeze()
        if marked_positions.dim() == 0:
            marked_positions = marked_positions.unsqueeze(0)

        computed_sum = values[marked_positions].sum()

        print(f"  Sample {i}:")
        print(f"    Marked positions: {marked_positions.tolist()}")
        print(f"    Marked values: {values[marked_positions].tolist()}")
        print(f"    Computed sum: {computed_sum:.4f}")
        print(f"    Stored target: {target:.4f}")
        print(f"    Match: {torch.isclose(computed_sum, target)}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate Adding Problem Dataset")
    parser.add_argument('--num_samples', type=int, default=5000,
                        help='Number of samples to generate')
    parser.add_argument('--seq_len', type=int, default=200,
                        help='Sequence length')
    parser.add_argument('--marker_strategy', type=str, default='extremes',
                        choices=['extremes', 'halves', 'uniform'],
                        help='Marker placement strategy')
    parser.add_argument('--filename', type=str, default='adding_problem_dataset.pt',
                        help='Output filename')
    parser.add_argument('--seed', type=int, default=2024,
                        help='Random seed')
    parser.add_argument('--verify', action='store_true',
                        help='Verify dataset after generation')

    args = parser.parse_args()

    generate_adding_problem_dataset(
        num_samples=args.num_samples,
        seq_len=args.seq_len,
        marker_strategy=args.marker_strategy,
        filename=args.filename,
        seed=args.seed
    )

    if args.verify:
        verify_dataset(args.filename)
