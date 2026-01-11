#!/usr/bin/env python3
"""
Generate all synthetic benchmark datasets.

This script generates datasets for:
  - Forrelation: seq_len = [50, 100, 200]
  - Adding Problem: seq_len = [100, 200, 500, 1000]
  - Selective Copy: seq_len = [100, 200, 500, 1000]

Usage:
    python generate_all_synthetic_datasets.py
    python generate_all_synthetic_datasets.py --data-dir ./data/synthetic_benchmarks
"""

import sys
from pathlib import Path

# Add parent directory to path (scripts/ -> Quantum-Hydra-Mamba/)
sys.path.insert(0, str(Path(__file__).parent.parent))

import argparse


def generate_forrelation_datasets(data_dir, seed=2024):
    """Generate Forrelation datasets."""
    print("\n" + "=" * 60)
    print("Generating Forrelation Datasets")
    print("=" * 60)

    from data_loaders.generate_forrelation_dataset import generate_dataset as generate_forrelation_dataset

    output_dir = data_dir / "forrelation"
    output_dir.mkdir(parents=True, exist_ok=True)

    seq_lens = [50, 100, 200]

    # Set seed for reproducibility
    import numpy as np
    import torch
    np.random.seed(seed)
    torch.manual_seed(seed)

    for seq_len in seq_lens:
        output_path = output_dir / f"forrelation_L{seq_len}_seed{seed}.pt"

        if output_path.exists():
            print(f"  Skipping {output_path} (already exists)")
            continue

        print(f"\nGenerating seq_len={seq_len}...")
        generate_forrelation_dataset(
            num_pairs=5000,
            n_bits=6,
            seq_len=seq_len,
            filename=str(output_path)
        )


def generate_adding_problem_datasets(data_dir, seed=2024):
    """Generate Adding Problem datasets."""
    print("\n" + "=" * 60)
    print("Generating Adding Problem Datasets")
    print("=" * 60)

    from data_loaders.generate_adding_problem import generate_adding_problem_dataset

    output_dir = data_dir / "adding_problem"
    output_dir.mkdir(parents=True, exist_ok=True)

    seq_lens = [100, 200, 500, 1000]

    for seq_len in seq_lens:
        output_path = output_dir / f"adding_L{seq_len}_seed{seed}.pt"

        if output_path.exists():
            print(f"  Skipping {output_path} (already exists)")
            continue

        print(f"\nGenerating seq_len={seq_len}...")
        generate_adding_problem_dataset(
            num_samples=5000,
            seq_len=seq_len,
            marker_strategy="extremes",
            filename=str(output_path),
            seed=seed
        )


def generate_selective_copy_datasets(data_dir, num_markers=8, seed=2024):
    """Generate Selective Copy datasets."""
    print("\n" + "=" * 60)
    print("Generating Selective Copy Datasets")
    print("=" * 60)

    from data_loaders.generate_selective_copy import generate_selective_copy_dataset

    output_dir = data_dir / "selective_copy"
    output_dir.mkdir(parents=True, exist_ok=True)

    seq_lens = [100, 200, 500, 1000]

    for seq_len in seq_lens:
        output_path = output_dir / f"selective_copy_L{seq_len}_M{num_markers}_seed{seed}.pt"

        if output_path.exists():
            print(f"  Skipping {output_path} (already exists)")
            continue

        print(f"\nGenerating seq_len={seq_len}...")
        generate_selective_copy_dataset(
            num_samples=5000,
            seq_len=seq_len,
            num_markers=num_markers,
            marker_strategy="uniform",
            filename=str(output_path),
            seed=seed
        )


def main():
    parser = argparse.ArgumentParser(
        description="Generate all synthetic benchmark datasets"
    )

    parser.add_argument("--data-dir", type=str, default="./data/synthetic_benchmarks",
                        help="Output directory for datasets")
    parser.add_argument("--seed", type=int, default=2024,
                        help="Random seed")
    parser.add_argument("--num-markers", type=int, default=8,
                        help="Number of markers for selective copy")
    parser.add_argument("--tasks", nargs='+',
                        default=['forrelation', 'adding_problem', 'selective_copy'],
                        help="Tasks to generate datasets for")

    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    data_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("Synthetic Benchmark Dataset Generator")
    print("=" * 60)
    print(f"Data directory: {data_dir}")
    print(f"Tasks: {args.tasks}")
    print(f"Seed: {args.seed}")

    if 'forrelation' in args.tasks:
        generate_forrelation_datasets(data_dir, args.seed)

    if 'adding_problem' in args.tasks:
        generate_adding_problem_datasets(data_dir, args.seed)

    if 'selective_copy' in args.tasks:
        generate_selective_copy_datasets(data_dir, args.num_markers, args.seed)

    print("\n" + "=" * 60)
    print("Dataset Generation Complete!")
    print("=" * 60)

    # Print summary
    print("\nGenerated datasets:")
    for task_dir in sorted(data_dir.iterdir()):
        if task_dir.is_dir():
            files = list(task_dir.glob("*.pt"))
            print(f"  {task_dir.name}: {len(files)} files")
            for f in sorted(files):
                print(f"    - {f.name}")


if __name__ == "__main__":
    main()
