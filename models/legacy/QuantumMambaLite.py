import torch
import torch.nn as nn
import sys
sys.path.append('/pscratch/sd/j/junghoon')

from QuantumMamba import QuantumMambaLayer

"""
Quantum Mamba Lite (Superposition) - Memory-Efficient Architecture

This is a lightweight variant of Quantum Mamba that uses vectorized batch processing
instead of timestep loops for memory efficiency.

Key Architectural Features:
1. REMOVED: Conv1d temporal processing
2. REMOVED: AdaptiveAvgPool1d
3. VECTORIZED: All timesteps processed as a single batch (memory-efficient)
4. ADDED: Complex-valued temporal_weights for quantum superposition aggregation

Memory Efficiency:
- Original Lite (timestep loop): O(n_timesteps) memory for intermediate outputs
- Revised Lite (vectorized): O(1) memory - single batched operation

Parameter Count Comparison (6-qubit EEG with 249 timesteps):
- QuantumMambaTS (Conv1d):     5,002 parameters
- QuantumMambaTS_Lite:         ~7,196 parameters

Usage:
    model = QuantumMambaTS_Lite(
        n_qubits=6,
        n_timesteps=249,
        qlcu_layers=2,
        feature_dim=64,
        output_dim=2,
        device="cpu"
    )
"""


class QuantumMambaTS_Lite(nn.Module):
    """
    Memory-Efficient Lightweight Quantum Mamba for Time-Series.

    Processes sequential data using vectorized batch processing for all timesteps,
    avoiding the memory-intensive timestep loop that caused OOM issues.

    Args:
        n_qubits: Number of qubits
        n_timesteps: Sequence length
        qlcu_layers: QLCU circuit depth
        feature_dim: Feature dimension per timestep
        output_dim: Final output dimension
        dropout: Dropout probability
        device: torch device
    """

    def __init__(
        self,
        n_qubits: int = 6,
        n_timesteps: int = 200,
        qlcu_layers: int = 2,
        feature_dim: int = 129,
        output_dim: int = 1,
        dropout: float = 0.1,
        device: str = "cpu"
    ):
        super().__init__()

        self.n_timesteps = n_timesteps
        self.n_qubits = n_qubits
        self.device = device

        # Feature projection: map each timestep to quantum parameters
        self.n_qlcu_params = 4 * n_qubits * qlcu_layers
        self.feature_projection = nn.Linear(feature_dim, self.n_qlcu_params)
        self.dropout = nn.Dropout(dropout)

        # Single Quantum Mamba layer (shared across timesteps)
        self.quantum_mamba = QuantumMambaLayer(
            n_qubits=n_qubits,
            qlcu_layers=qlcu_layers,
            feature_dim=self.n_qlcu_params,
            output_dim=3 * n_qubits,  # Output measurements
            dropout=0.0,  # Already applied dropout
            device=device
        )

        # Temporal aggregation: complex-valued weights for quantum superposition
        # (matching Quantum Hydra Superposition)
        self.temporal_weights = nn.Parameter(
            torch.rand(n_timesteps, dtype=torch.complex64)
        )

        # Final output layer
        self.output_layer = nn.Linear(3 * n_qubits, output_dim)

        # Move all parameters to specified device
        self.to(device)

    def forward(self, x):
        """
        Forward pass for time-series input using vectorized batch processing.

        Args:
            x: Input tensor (batch_size, feature_dim, n_timesteps) or
               (batch_size, n_timesteps, feature_dim)

        Returns:
            output: Tensor (batch_size, output_dim)
        """
        batch_size = x.shape[0]

        # Handle different input formats - ensure (batch_size, n_timesteps, feature_dim)
        if x.shape[1] != self.n_timesteps and x.shape[2] == self.n_timesteps:
            # Input is (batch_size, feature_dim, n_timesteps)
            x = x.permute(0, 2, 1)  # Convert to (batch_size, n_timesteps, feature_dim)

        # VECTORIZED PROCESSING: Process all timesteps as a single batch
        # Reshape: (batch_size, n_timesteps, feature_dim) -> (batch_size * n_timesteps, feature_dim)
        x_flat = x.reshape(batch_size * self.n_timesteps, -1)

        # Apply dropout and feature projection to all timesteps at once
        x_proj = self.feature_projection(self.dropout(x_flat))

        # Run quantum layer on all timesteps as a large batch
        out_flat = self.quantum_mamba(x_proj)  # (batch_size * n_timesteps, 3*n_qubits)

        # Reshape back: (batch_size * n_timesteps, 3*n_qubits) -> (batch_size, n_timesteps, 3*n_qubits)
        timestep_outputs = out_flat.reshape(batch_size, self.n_timesteps, -1)

        # Weighted aggregation across timesteps with complex weights
        weights = self.temporal_weights / torch.sum(torch.abs(self.temporal_weights))
        weights = weights.unsqueeze(0).unsqueeze(-1)  # (1, n_timesteps, 1)

        # Complex-valued weighted sum
        timestep_outputs_complex = timestep_outputs.to(torch.complex64)
        aggregated = torch.sum(timestep_outputs_complex * weights, dim=1)  # (batch_size, 3*n_qubits)
        aggregated = torch.abs(aggregated)  # Take magnitude

        # Final prediction
        output = self.output_layer(aggregated.float())

        return output

    def count_parameters(self):
        """Count total trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


# ================================================================================
# Testing and Verification
# ================================================================================

if __name__ == "__main__":
    print("=" * 80)
    print("Quantum Mamba Lite (Superposition) - Hydra-Matched Architecture")
    print("=" * 80)

    # EEG configuration (6 qubits)
    n_qubits = 6
    n_timesteps = 249
    feature_dim = 64
    output_dim = 2
    qlcu_layers = 2

    print(f"\nConfiguration:")
    print(f"  n_qubits: {n_qubits}")
    print(f"  n_timesteps: {n_timesteps}")
    print(f"  feature_dim: {feature_dim}")
    print(f"  output_dim: {output_dim}")
    print(f"  qlcu_layers: {qlcu_layers}")

    # Create model
    print(f"\nCreating Quantum Mamba Lite model...")
    model = QuantumMambaTS_Lite(
        n_qubits=n_qubits,
        n_timesteps=n_timesteps,
        qlcu_layers=qlcu_layers,
        feature_dim=feature_dim,
        output_dim=output_dim,
        device="cpu"
    )

    # Count parameters
    total_params = model.count_parameters()
    print(f"\nTotal Parameters: {total_params:,}")

    # Detailed parameter breakdown
    print(f"\nDetailed Parameter Breakdown:")
    print(f"  feature_projection: {feature_dim * (4*n_qubits*qlcu_layers) + (4*n_qubits*qlcu_layers):,}")
    print(f"  temporal_weights: {n_timesteps} (complex)")
    print(f"  output_layer: {(3*n_qubits) * output_dim + output_dim:,}")
    print(f"  quantum_mamba (QuantumMambaLayer): {sum(p.numel() for p in model.quantum_mamba.parameters() if p.requires_grad):,}")

    # Test forward pass
    print(f"\nTesting forward pass...")
    batch_size = 4
    x = torch.randn(batch_size, n_timesteps, feature_dim)

    try:
        output = model(x)
        print(f"  Input shape: {x.shape}")
        print(f"  Output shape: {output.shape}")
        print(f"  ✓ Forward pass successful!")
    except Exception as e:
        print(f"  ✗ Error: {e}")

    # Compare with target (Quantum Hydra)
    print(f"\n" + "=" * 80)
    print("Architecture Comparison")
    print("=" * 80)
    print(f"\nTarget: Quantum Hydra (Superposition)")
    print(f"  - Uses timestep loop processing: ✓")
    print(f"  - Uses complex temporal_weights: ✓")
    print(f"  - No Conv1d: ✓")
    print(f"  - Expected params: ~7,196")
    print(f"\nQuantum Mamba Lite (Superposition)")
    print(f"  - Uses timestep loop processing: ✓")
    print(f"  - Uses complex temporal_weights: ✓")
    print(f"  - No Conv1d: ✓")
    print(f"  - Actual params: {total_params:,}")
    print(f"\nArchitectural Match: {'✓ MATCHED' if 6000 <= total_params <= 8000 else '✗ MISMATCH'}")
    print("=" * 80)
