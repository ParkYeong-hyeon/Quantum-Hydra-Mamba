import torch
from torch.utils.data import TensorDataset, DataLoader

"""
Defines a standard PyTorch DataLoader for the Sequential Forrelation dataset.

This script loads the dataset file created by `generate_forrelation_dataset.py`
and prepares it for training.

It handles the necessary tensor transpositions to make the data
compatible with the models, which expect inputs of shape:
(batch_size, num_channels, seq_len)
"""

def get_forrelation_dataloader(dataset_path="forrelation_dataset.pt", batch_size=32, shuffle=True):
    """
    Loads the dataset and returns a PyTorch DataLoader.

    Args:
        dataset_path (str): Path to the .pt dataset file.
        batch_size (int): The batch size for the dataloader.
        shuffle (bool): Whether to shuffle the dataset.

    Returns:
        A tuple of (train_loader, test_loader, dataset_params)
    """
    print(f"Loading dataset from {dataset_path}...")
    
    try:
        data = torch.load(dataset_path)
        sequences = data['sequences']
        labels = data['labels']
        params = data['params']
    except FileNotFoundError:
        print(f"Error: Dataset file not found at {dataset_path}")
        print("Please run `python generate_forrelation_dataset.py` first.")
        return None, None, None

    # The models expect input of shape (batch, channels, timesteps).
    # The stored data is (batch, timesteps, channels), so we transpose it.
    # B, L, C -> B, C, L
    sequences = sequences.permute(0, 2, 1)

    print("Dataset loaded successfully.")
    print(f"  - n_bits: {params['n_bits']}")
    print(f"  - seq_len: {params['seq_len']}")
    print(f"  - num_channels: {params['num_channels']}")
    print(f"  - Total sequences: {len(sequences)}")
    print(f"  - Shape for model: {sequences.shape}")

    # Create a TensorDataset
    dataset = TensorDataset(sequences, labels)

    # Split into training and testing sets (e.g., 80/20 split)
    train_size = int(0.8 * len(dataset))
    test_size = len(dataset) - train_size
    train_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_size, test_size])

    print(f"  - Training set size: {len(train_dataset)}")
    print(f"  - Test set size: {len(test_dataset)}")

    # Create DataLoaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=shuffle)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, test_loader, params

if __name__ == '__main__':
    # This is an example of how to use the dataloader.
    
    # First, ensure the dataset exists. If not, generate it.
    import os
    if not os.path.exists("forrelation_dataset.pt"):
        print("Dataset file not found. Generating a default dataset first...")
        os.system("python generate_forrelation_dataset.py --num_pairs=1000 --n_bits=6 --seq_len=50")
        print("-" * 50)

    # Get the dataloaders
    train_loader, test_loader, params = get_forrelation_dataloader(batch_size=4)

    if train_loader:
        print("\n--- DataLoader Example ---")
        # Get one batch of data
        sample_sequences, sample_labels = next(iter(train_loader))
        
        print(f"Successfully fetched one batch of size {sample_sequences.shape[0]}.")
        print(f"Shape of the sequence batch: {sample_sequences.shape}")
        print(f"This matches the expected model input (batch, channels, timesteps).")
        print(f"Shape of the label batch: {sample_labels.shape}")
        print(f"Labels in the batch: {sample_labels.tolist()}")
        print(f"Dataset parameters: {params}")
