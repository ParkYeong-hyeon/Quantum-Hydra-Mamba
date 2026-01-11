import torch
import numpy as np
def fwht(a):
    """Recursive implementation of the Fast Walsh-Hadamard Transform."""
    h = 2
    while h <= len(a):
        for i in range(0, len(a), h):
            for j in range(i, i + h // 2):
                x = a[j]
                y = a[j + h // 2]
                a[j] = x + y
                a[j + h // 2] = x - y
        h *= 2
    return a
import argparse
from tqdm import tqdm

"""
Generates a Sequential Forrelation dataset for training sequence models.

The Forrelation problem is to decide if a Boolean function `f` is highly
correlated with the Fourier transform of another Boolean function `g`.
This script reframes this as a sequence task.

Dataset Structure:
- Each data point is a sequence of `L` samples.
- Each sample is drawn from a pair of functions (f, g).
- The model must predict if the pair has high or low forrelation based on the sequence.

Input Shape for Models:
- The generated data is saved in the shape (num_pairs, seq_len, num_channels).
- The dataloader will transpose this to (num_pairs, num_channels, seq_len) to be
  compatible with the models, which expect (batch, channels, timesteps).
"""

def fourier_transform(boolean_func_table):
    """
    Computes the Fourier transform (Walsh-Hadamard transform) of a Boolean function.
    Args:
        boolean_func_table: A numpy array of shape (N,) representing the function's truth table.
    Returns:
        A numpy array of shape (N,) representing the Fourier transform.
    """
    # The fast Walsh-Hadamard transform is the Fourier transform for Boolean functions.
    # The normalization factor is 1/sqrt(N).
    return fwht(boolean_func_table) / np.sqrt(len(boolean_func_table))

def get_forrelation(f_table, g_table):
    """
    Computes the forrelation between two Boolean functions.
    Φ(f, g) = (1/N) * Σ_x f(x) * ĝ(x)
    """
    N = len(f_table)
    g_fourier = fourier_transform(g_table)
    # Forrelation is the scaled inner product of f and the Fourier transform of g.
    return np.dot(f_table, g_fourier) / N

def generate_forrelation_pair(n_bits, high_forrelation=True):
    """
    Generates a pair of Boolean functions (f, g) with either high or low forrelation.

    Args:
        n_bits: The number of input bits for the functions (n). The domain size is N = 2^n.
        high_forrelation: If True, generate a pair with high forrelation. Otherwise, low.

    Returns:
        A tuple (f_table, g_table, forrelation_value).
    """
    N = 2**n_bits
    
    if high_forrelation:
        # To guarantee high forrelation, we construct f and g carefully.
        # Let g be a character function (a single Fourier basis function), which has a sparse Fourier transform.
        s = np.random.randint(1, N)  # Choose a non-zero frequency
        s_bits = np.array([(s >> i) & 1 for i in range(n_bits)])
        
        inputs = np.arange(N)
        input_bits = ((inputs[:, None] & (1 << np.arange(n_bits))) > 0).astype(int)
        
        # g(x) = (-1)^(s . x)
        g_table = (-1)**(input_bits @ s_bits % 2)
        
        # The Fourier transform of g will be a delta function at s.
        # To maximize forrelation, let f also be a delta function at s.
        f_table = np.zeros(N)
        f_table[s] = 1
        # Convert to {-1, 1} representation for consistency, though it doesn't affect forrelation value
        f_table = f_table * 2 - 1 
        
        forrelation = get_forrelation(f_table, g_table)
        # This construction can sometimes lead to negative forrelation, so we take abs
        if forrelation < 0:
             f_table = -f_table # Flip f to make forrelation positive
             forrelation = abs(forrelation)

    else:
        # For low forrelation, two random functions are overwhelmingly likely to have near-zero forrelation.
        while True:
            f_table = np.random.choice([-1, 1], size=N)
            g_table = np.random.choice([-1, 1], size=N)
            forrelation = get_forrelation(f_table, g_table)
            if abs(forrelation) < 0.1:  # Ensure it's low
                break
                
    return f_table, g_table, forrelation

def generate_dataset(num_pairs, n_bits, seq_len, filename="forrelation_dataset.pt"):
    """
    Generates and saves the full Sequential Forrelation dataset.
    """
    N = 2**n_bits
    num_channels = 2 * n_bits + 2  # (x_bits, f(x), y_bits, g(y))

    all_sequences = torch.zeros(num_pairs, seq_len, num_channels, dtype=torch.float32)
    all_labels = torch.zeros(num_pairs, dtype=torch.long)

    print(f"Generating dataset with {num_pairs} sequences...")
    print(f"Parameters: n_bits={n_bits} (N={N}), seq_len={seq_len}")

    for i in tqdm(range(num_pairs)):
        is_high_forrelation = (i % 2 == 0)
        label = 1 if is_high_forrelation else 0
        
        f_table, g_table, _ = generate_forrelation_pair(n_bits, high_forrelation=is_high_forrelation)

        # Generate the sequence of L samples for this (f, g) pair
        for t in range(seq_len):
            # Sample for f
            x_int = np.random.randint(0, N)
            f_x = f_table[x_int]
            x_bits = torch.tensor([(x_int >> j) & 1 for j in range(n_bits)], dtype=torch.float32)

            # Sample for g
            y_int = np.random.randint(0, N)
            g_y = g_table[y_int]
            y_bits = torch.tensor([(y_int >> j) & 1 for j in range(n_bits)], dtype=torch.float32)
            
            # Create the feature vector for this timestep
            feature_vector = torch.cat([
                x_bits,
                torch.tensor([f_x], dtype=torch.float32),
                y_bits,
                torch.tensor([g_y], dtype=torch.float32)
            ])
            all_sequences[i, t, :] = feature_vector
        
        all_labels[i] = label

    # Save the dataset to a file
    dataset = {
        'sequences': all_sequences,
        'labels': all_labels,
        'params': {
            'n_bits': n_bits,
            'seq_len': seq_len,
            'num_channels': num_channels
        }
    }
    torch.save(dataset, filename)
    print(f"\nDataset successfully saved to {filename}")
    print(f"  Sequences tensor shape: {all_sequences.shape}")
    print(f"  Labels tensor shape: {all_labels.shape}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate a Sequential Forrelation Dataset.")
    parser.add_argument('--num_pairs', type=int, default=2000, help='Total number of (f, g) pairs to generate.')
    parser.add_argument('--n_bits', type=int, default=6, help='Number of input bits for the Boolean functions (n). N=2^n.')
    parser.add_argument('--seq_len', type=int, default=50, help='Sequence length (L) of samples for each pair.')
    parser.add_argument('--filename', type=str, default='forrelation_dataset.pt', help='Output filename.')
    
    args = parser.parse_args()
    
    generate_dataset(
        num_pairs=args.num_pairs,
        n_bits=args.n_bits,
        seq_len=args.seq_len,
        filename=args.filename
    )
