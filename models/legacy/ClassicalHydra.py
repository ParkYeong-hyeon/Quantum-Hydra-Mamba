import torch
import torch.nn as nn
import torch.nn.functional as F

"""
Classical Hydra Implementation

Based on the classical Hydra state-space model (Hwang et al., 2024).
This serves as a baseline for comparison with Quantum Hydra models.

Classical Hydra equation:
    QS(X) = shift(SS(X)) + flip(shift(SS(flip(X)))) + DX
    Y = W · QS(X)

Where:
- SS: Semi-separable matrix (implemented as learned linear layers)
- shift: Circular shift operation
- flip: Sequence reversal
- D: Diagonal matrix (element-wise multiplication)
- W: Final weight matrix
"""


class SemiSeparableLayer(nn.Module):
    """
    Implements a semi-separable matrix operation.

    For simplicity, we use a linear layer to approximate the semi-separable
    structure. A true semi-separable matrix has specific structure, but
    for comparison purposes, a learnable linear transformation suffices.
    """
    def __init__(self, seq_len, feature_dim):
        super().__init__()
        self.seq_len = seq_len
        self.feature_dim = feature_dim

        # Linear transformation that acts on the sequence dimension
        self.weight = nn.Parameter(torch.randn(seq_len, seq_len) * 0.01)
        self.bias = nn.Parameter(torch.zeros(seq_len))

    def forward(self, x):
        """
        Args:
            x: (batch_size, feature_dim, seq_len)
        Returns:
            (batch_size, feature_dim, seq_len)
        """
        # Apply matrix multiplication along sequence dimension
        # x: (B, D, L) -> (B, L, D)
        x_transposed = x.transpose(1, 2)  # (B, L, D)

        # Apply semi-separable matrix: (L, L) @ (B, L, D) -> (B, L, D)
        # We apply the weight matrix to the sequence dimension
        output = torch.einsum('ij,bjd->bid', self.weight, x_transposed) + self.bias.unsqueeze(0).unsqueeze(-1)

        # Transpose back: (B, L, D) -> (B, D, L)
        return output.transpose(1, 2)


class ClassicalHydraLayer(nn.Module):
    """
    Classical Hydra layer for time-series processing.

    Implements:
        QS(X) = shift(SS(X)) + flip(shift(SS(flip(X)))) + DX

    Args:
        seq_len: Sequence length
        feature_dim: Feature dimension per timestep
        shift_amount: How many positions to shift (default: 1)
    """
    def __init__(self, seq_len, feature_dim, shift_amount=1):
        super().__init__()
        self.seq_len = seq_len
        self.feature_dim = feature_dim
        self.shift_amount = shift_amount

        # Semi-separable matrix approximation
        self.ss = SemiSeparableLayer(seq_len, feature_dim)

        # Diagonal matrix (element-wise scaling)
        self.diagonal = nn.Parameter(torch.ones(1, feature_dim, seq_len))

    def shift_sequence(self, x, amount=1):
        """
        Circular shift along the sequence dimension.

        Args:
            x: (batch_size, feature_dim, seq_len)
            amount: Number of positions to shift right
        Returns:
            Shifted tensor
        """
        return torch.roll(x, shifts=amount, dims=2)

    def flip_sequence(self, x):
        """
        Flip (reverse) the sequence dimension.

        Args:
            x: (batch_size, feature_dim, seq_len)
        Returns:
            Flipped tensor
        """
        return torch.flip(x, dims=[2])

    def forward(self, x):
        """
        Forward pass of Classical Hydra.

        Args:
            x: (batch_size, feature_dim, seq_len)
        Returns:
            (batch_size, feature_dim, seq_len)
        """
        # Branch 1: shift(SS(X))
        branch1 = self.ss(x)
        branch1 = self.shift_sequence(branch1, self.shift_amount)

        # Branch 2: flip(shift(SS(flip(X))))
        x_flipped = self.flip_sequence(x)
        branch2 = self.ss(x_flipped)
        branch2 = self.shift_sequence(branch2, self.shift_amount)
        branch2 = self.flip_sequence(branch2)

        # Branch 3: D * X (element-wise)
        branch3 = self.diagonal * x

        # Classical addition
        output = branch1 + branch2 + branch3

        return output


class ClassicalHydra(nn.Module):
    """
    Complete Classical Hydra model for EEG classification.

    Architecture:
        Input -> Feature Projection -> Hydra Layers -> Pooling -> Output

    Args:
        n_channels: Number of EEG channels (e.g., 64)
        n_timesteps: Sequence length (e.g., 160)
        hidden_dim: Hidden feature dimension
        n_hydra_layers: Number of Hydra layers to stack
        output_dim: Number of output classes (e.g., 2 for binary)
        dropout: Dropout probability
    """
    def __init__(
        self,
        n_channels,
        n_timesteps,
        hidden_dim=128,
        n_hydra_layers=2,
        output_dim=2,
        dropout=0.1
    ):
        super().__init__()

        self.n_channels = n_channels
        self.n_timesteps = n_timesteps
        self.hidden_dim = hidden_dim

        # Input projection: map channels to hidden dimension
        self.input_projection = nn.Linear(n_channels, hidden_dim)
        self.dropout = nn.Dropout(dropout)

        # Stack of Hydra layers
        self.hydra_layers = nn.ModuleList([
            ClassicalHydraLayer(
                seq_len=n_timesteps,
                feature_dim=hidden_dim,
                shift_amount=1
            ) for _ in range(n_hydra_layers)
        ])

        # Layer normalization
        self.layer_norms = nn.ModuleList([
            nn.LayerNorm([hidden_dim, n_timesteps])
            for _ in range(n_hydra_layers)
        ])

        # Temporal pooling
        self.temporal_pool = nn.AdaptiveAvgPool1d(1)

        # Output head
        self.output_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, output_dim)
        )

    def forward(self, x):
        """
        Forward pass.

        Args:
            x: (batch_size, n_channels, n_timesteps)
        Returns:
            (batch_size, output_dim)
        """
        # Transpose for projection: (B, C, T) -> (B, T, C)
        x = x.transpose(1, 2)

        # Project channels to hidden dimension: (B, T, C) -> (B, T, H)
        x = self.input_projection(self.dropout(x))

        # Transpose back: (B, T, H) -> (B, H, T)
        x = x.transpose(1, 2)

        # Apply Hydra layers with residual connections
        for hydra_layer, layer_norm in zip(self.hydra_layers, self.layer_norms):
            residual = x
            x = hydra_layer(x)
            x = layer_norm(x)
            x = x + residual  # Residual connection
            x = F.relu(x)

        # Temporal pooling: (B, H, T) -> (B, H, 1) -> (B, H)
        x = self.temporal_pool(x).squeeze(-1)

        # Output head
        output = self.output_head(x)

        return output


class ClassicalHydraSimple(nn.Module):
    """
    Simplified Classical Hydra without residual connections and layer norm.
    More directly comparable to Quantum Hydra implementations.

    Args:
        n_channels: Number of input channels
        n_timesteps: Sequence length
        hidden_dim: Hidden dimension
        output_dim: Output dimension
        dropout: Dropout probability
    """
    def __init__(
        self,
        n_channels,
        n_timesteps,
        hidden_dim=128,
        output_dim=2,
        dropout=0.1
    ):
        super().__init__()

        self.n_channels = n_channels
        self.n_timesteps = n_timesteps
        self.hidden_dim = hidden_dim

        # Input projection
        self.input_projection = nn.Linear(n_channels, hidden_dim)
        self.dropout = nn.Dropout(dropout)

        # Single Hydra layer (matching quantum versions)
        self.hydra = ClassicalHydraLayer(
            seq_len=n_timesteps,
            feature_dim=hidden_dim,
            shift_amount=1
        )

        # Output projection
        self.output_projection = nn.Linear(hidden_dim * n_timesteps, output_dim)

    def forward(self, x):
        """
        Args:
            x: (batch_size, n_channels, n_timesteps)
        Returns:
            (batch_size, output_dim)
        """
        # Transpose: (B, C, T) -> (B, T, C)
        x = x.transpose(1, 2)

        # Project: (B, T, C) -> (B, T, H)
        x = self.input_projection(self.dropout(x))

        # Transpose: (B, T, H) -> (B, H, T)
        x = x.transpose(1, 2)

        # Apply Hydra: (B, H, T) -> (B, H, T)
        x = self.hydra(x)

        # Flatten: (B, H, T) -> (B, H*T)
        x = x.reshape(x.shape[0], -1)

        # Output
        output = self.output_projection(x)

        return output


# ================================================================================
# Testing
# ================================================================================

if __name__ == "__main__":
    print("=" * 80)
    print("Classical Hydra - Testing")
    print("=" * 80)

    # Test parameters matching PhysioNet EEG
    batch_size = 8
    n_channels = 64
    n_timesteps = 160
    hidden_dim = 128
    output_dim = 2

    print("\n[1] Testing ClassicalHydraLayer...")
    layer = ClassicalHydraLayer(
        seq_len=n_timesteps,
        feature_dim=hidden_dim,
        shift_amount=1
    )

    x = torch.randn(batch_size, hidden_dim, n_timesteps)
    output = layer(x)
    print(f"  Input shape: {x.shape}")
    print(f"  Output shape: {output.shape}")
    assert output.shape == x.shape, "Output shape mismatch!"

    print("\n[2] Testing ClassicalHydra (full model)...")
    model = ClassicalHydra(
        n_channels=n_channels,
        n_timesteps=n_timesteps,
        hidden_dim=hidden_dim,
        n_hydra_layers=2,
        output_dim=output_dim,
        dropout=0.1
    )

    x = torch.randn(batch_size, n_channels, n_timesteps)
    output = model(x)
    print(f"  Input shape: {x.shape}")
    print(f"  Output shape: {output.shape}")
    print(f"  Output sample: {output[0]}")
    assert output.shape == (batch_size, output_dim), "Output shape mismatch!"

    print("\n[3] Testing ClassicalHydraSimple...")
    model_simple = ClassicalHydraSimple(
        n_channels=n_channels,
        n_timesteps=n_timesteps,
        hidden_dim=64,  # Smaller for fair comparison
        output_dim=output_dim,
        dropout=0.1
    )

    output_simple = model_simple(x)
    print(f"  Input shape: {x.shape}")
    print(f"  Output shape: {output_simple.shape}")
    print(f"  Output sample: {output_simple[0]}")

    print("\n[4] Checking trainable parameters...")
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_params_simple = sum(p.numel() for p in model_simple.parameters() if p.requires_grad)
    print(f"  ClassicalHydra parameters: {total_params:,}")
    print(f"  ClassicalHydraSimple parameters: {total_params_simple:,}")

    print("\n[5] Testing shift and flip operations...")
    test_x = torch.arange(10).float().unsqueeze(0).unsqueeze(0)  # (1, 1, 10)
    print(f"  Original: {test_x.squeeze().tolist()}")

    shifted = layer.shift_sequence(test_x, amount=2)
    print(f"  Shifted by 2: {shifted.squeeze().tolist()}")

    flipped = layer.flip_sequence(test_x)
    print(f"  Flipped: {flipped.squeeze().tolist()}")

    print("\n[6] Testing gradient flow...")
    model.train()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.CrossEntropyLoss()

    x_batch = torch.randn(4, n_channels, n_timesteps)
    y_batch = torch.randint(0, output_dim, (4,))

    optimizer.zero_grad()
    outputs = model(x_batch)
    loss = criterion(outputs, y_batch)
    loss.backward()
    optimizer.step()

    print(f"  Loss: {loss.item():.4f}")
    print(f"  Gradient norm (first layer): {model.hydra_layers[0].ss.weight.grad.norm().item():.4f}")

    print("\n" + "=" * 80)
    print("Testing complete!")
    print("=" * 80)
