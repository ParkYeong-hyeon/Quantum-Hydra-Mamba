"""
PyTorch DataLoader for the Adding Problem Dataset

This script loads the dataset created by `generate_adding_problem.py`
and prepares it for training sequence models.

The Adding Problem is a regression task where the model must output
the sum of two marked values in a sequence.

Input Shape:
    - Raw data: (batch, seq_len, 2) where channels are [value, marker]
    - For models: (batch, 2, seq_len) after transpose

Output:
    - Target: (batch, 1) sum of marked values

Evaluation:
    - Metric: Mean Squared Error (MSE)
    - Baseline (predict 1.0): MSE ~ 0.167
    - Perfect model: MSE ~ 0.0

Author: Research Team
Date: January 2026
"""

import torch
from torch.utils.data import TensorDataset, DataLoader, random_split
from pathlib import Path
import os


def get_adding_problem_dataloader(
    dataset_path: str = "adding_problem_dataset.pt",
    batch_size: int = 32,
    train_ratio: float = 0.8,
    val_ratio: float = 0.1,
    shuffle: bool = True,
    num_workers: int = 0,
    seed: int = 2024
):
    """
    Load the Adding Problem dataset and return DataLoaders.

    Args:
        dataset_path: Path to the .pt dataset file
        batch_size: Batch size for dataloaders
        train_ratio: Fraction for training set
        val_ratio: Fraction for validation set (test gets the rest)
        shuffle: Whether to shuffle training data
        num_workers: Number of worker processes
        seed: Random seed for reproducible splits

    Returns:
        train_loader: Training DataLoader
        val_loader: Validation DataLoader
        test_loader: Test DataLoader
        params: Dataset parameters dictionary
    """
    print(f"Loading Adding Problem dataset from {dataset_path}...")

    # Check if file exists
    if not os.path.exists(dataset_path):
        print(f"Error: Dataset file not found at {dataset_path}")
        print("Please run `python generate_adding_problem.py` first.")
        print("\nExample:")
        print(f"  python generate_adding_problem.py --num_samples=5000 --seq_len=200")
        return None, None, None, None

    # Load dataset
    data = torch.load(dataset_path)
    sequences = data['sequences']
    targets = data['targets']
    params = data['params']

    print("Dataset loaded successfully.")
    print(f"  - Task: {params.get('task', 'adding_problem')}")
    print(f"  - Sequence length: {params['seq_len']}")
    print(f"  - Num channels: {params['num_channels']}")
    print(f"  - Total samples: {len(sequences)}")
    print(f"  - Marker strategy: {params.get('marker_strategy', 'unknown')}")
    print(f"  - Baseline MSE: {params.get('baseline_mse', 'N/A'):.4f}")

    # Transpose to (batch, channels, seq_len) for model compatibility
    # Models expect (batch, num_channels, seq_len)
    sequences = sequences.permute(0, 2, 1)
    print(f"  - Shape for model: {sequences.shape}")

    # Create TensorDataset
    dataset = TensorDataset(sequences, targets)

    # Split dataset
    total_size = len(dataset)
    train_size = int(train_ratio * total_size)
    val_size = int(val_ratio * total_size)
    test_size = total_size - train_size - val_size

    # Set seed for reproducible splits
    generator = torch.Generator().manual_seed(seed)

    train_dataset, val_dataset, test_dataset = random_split(
        dataset,
        [train_size, val_size, test_size],
        generator=generator
    )

    print(f"  - Training set: {len(train_dataset)}")
    print(f"  - Validation set: {len(val_dataset)}")
    print(f"  - Test set: {len(test_dataset)}")

    # Create DataLoaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=True
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )

    return train_loader, val_loader, test_loader, params


def load_adding_problem_for_training(
    seq_len: int = 200,
    num_samples: int = 5000,
    batch_size: int = 32,
    marker_strategy: str = "extremes",
    data_dir: str = "./data/adding_problem",
    seed: int = 2024,
    regenerate: bool = False
):
    """
    Convenience function to load or generate Adding Problem data.

    This function will:
    1. Check if dataset exists with matching parameters
    2. Generate new dataset if needed
    3. Return ready-to-use DataLoaders

    Args:
        seq_len: Sequence length
        num_samples: Number of samples
        batch_size: Batch size
        marker_strategy: Marker placement strategy
        data_dir: Directory to store datasets
        seed: Random seed
        regenerate: Force regeneration even if file exists

    Returns:
        train_loader, val_loader, test_loader, params
    """
    from .generate_adding_problem import generate_adding_problem_dataset

    # Create data directory
    data_dir = Path(data_dir)
    data_dir.mkdir(parents=True, exist_ok=True)

    # Filename includes key parameters for caching
    filename = data_dir / f"adding_L{seq_len}_N{num_samples}_{marker_strategy}_seed{seed}.pt"

    # Generate if needed
    if regenerate or not filename.exists():
        print(f"Generating new dataset: {filename}")
        generate_adding_problem_dataset(
            num_samples=num_samples,
            seq_len=seq_len,
            marker_strategy=marker_strategy,
            filename=str(filename),
            seed=seed
        )

    # Load and return
    return get_adding_problem_dataloader(
        dataset_path=str(filename),
        batch_size=batch_size,
        seed=seed
    )


class AddingProblemDataModule:
    """
    Data module wrapper for Adding Problem dataset.

    Provides a clean interface compatible with training scripts.
    """

    def __init__(
        self,
        seq_len: int = 200,
        num_samples: int = 5000,
        batch_size: int = 32,
        marker_strategy: str = "extremes",
        data_dir: str = "./data/adding_problem",
        seed: int = 2024
    ):
        self.seq_len = seq_len
        self.num_samples = num_samples
        self.batch_size = batch_size
        self.marker_strategy = marker_strategy
        self.data_dir = Path(data_dir)
        self.seed = seed

        self.train_loader = None
        self.val_loader = None
        self.test_loader = None
        self.params = None

    def setup(self, regenerate: bool = False):
        """Setup dataloaders."""
        self.train_loader, self.val_loader, self.test_loader, self.params = \
            load_adding_problem_for_training(
                seq_len=self.seq_len,
                num_samples=self.num_samples,
                batch_size=self.batch_size,
                marker_strategy=self.marker_strategy,
                data_dir=str(self.data_dir),
                seed=self.seed,
                regenerate=regenerate
            )

    @property
    def input_dim(self):
        """Input dimension (number of channels)."""
        return 2  # [value, marker]

    @property
    def output_dim(self):
        """Output dimension."""
        return 1  # Sum (regression)

    @property
    def task_type(self):
        """Task type."""
        return "regression"


if __name__ == '__main__':
    import sys

    # Example usage
    print("=" * 60)
    print("Adding Problem DataLoader Example")
    print("=" * 60)

    # Check if dataset exists, generate if not
    default_path = "adding_problem_dataset.pt"

    if not os.path.exists(default_path):
        print(f"\nDataset not found. Generating default dataset...")
        # Import and generate
        sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
        from .generate_adding_problem import generate_adding_problem_dataset

        generate_adding_problem_dataset(
            num_samples=1000,
            seq_len=100,
            marker_strategy="extremes",
            filename=default_path,
            seed=2024
        )
        print("-" * 60)

    # Load dataset
    train_loader, val_loader, test_loader, params = get_adding_problem_dataloader(
        dataset_path=default_path,
        batch_size=4
    )

    if train_loader is not None:
        print("\n--- DataLoader Example ---")

        # Get one batch
        batch_x, batch_y = next(iter(train_loader))

        print(f"Batch input shape: {batch_x.shape}")
        print(f"  Expected: (batch_size, 2, seq_len)")
        print(f"Batch target shape: {batch_y.shape}")
        print(f"  Expected: (batch_size, 1)")

        print(f"\nSample targets (sums): {batch_y.squeeze().tolist()}")
        print(f"Target range: [{batch_y.min():.3f}, {batch_y.max():.3f}]")
        print(f"  Expected: [0.0, 2.0]")

        # Verify one sample
        print("\n--- Sample Verification ---")
        sample_x = batch_x[0]  # (2, seq_len)
        sample_y = batch_y[0]  # (1,)

        values = sample_x[0]  # Channel 0: values
        markers = sample_x[1]  # Channel 1: markers

        marked_idx = (markers == 1.0).nonzero().squeeze()
        if marked_idx.dim() == 0:
            marked_idx = marked_idx.unsqueeze(0)

        computed_sum = values[marked_idx].sum()

        print(f"Marked positions: {marked_idx.tolist()}")
        print(f"Marked values: {values[marked_idx].tolist()}")
        print(f"Computed sum: {computed_sum:.4f}")
        print(f"Target sum: {sample_y.item():.4f}")
        print(f"Match: {torch.isclose(computed_sum, sample_y.squeeze())}")

        print("\n" + "=" * 60)
        print("DataLoader test passed!")
        print("=" * 60)
