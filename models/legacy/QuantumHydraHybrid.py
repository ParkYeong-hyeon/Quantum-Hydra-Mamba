import torch
import torch.nn as nn
import pennylane as qml
from math import log2
from typing import Optional, List

"""
Quantum Hydra - Option B (Hybrid Classical-Quantum)

This implementation is more faithful to classical Hydra's structure.
Each quantum branch is computed and measured independently, then the
results are combined via CLASSICAL addition (not quantum superposition).

Classical Hydra equation:
    QS(X) = shift(SS(X)) + flip(shift(SS(flip(X)))) + DX
    Y = W * QS(X)

Quantum Hybrid Hydra equation:
    y₁ = Measure(Qshift(QLCU|X⟩))
    y₂ = Measure(Qflip(Qshift(QLCU(Qflip|X⟩))))
    y₃ = Measure(QD|X⟩)
    Y = W(w₁·y₁ + w₂·y₂ + w₃·y₃)  ← Classical weighted sum

Key differences from Option A (QuantumHydra.py):
- Option A: Creates quantum superposition |ψ⟩ = α|ψ₁⟩ + β|ψ₂⟩ + γ|ψ₃⟩
- Option B: Measures each branch separately, combines classically

Advantages of Option B:
- Closer to classical Hydra semantics
- Easier to interpret (no quantum interference between branches)
- Can analyze contribution of each branch independently
- More robust to decoherence (each branch measured separately)
"""

# ================================================================================
# Import Quantum Circuit Building Blocks from QuantumHydra.py
# ================================================================================
# We reuse the same quantum circuits but apply them differently

def basic_ansatz_circuit(params, wires, layers=1):
    """
    Basic parametrized quantum circuit for encoding and processing.
    """
    is_batched = params.ndim == 2
    param_idx = 0

    for _ in range(layers):
        for i in range(wires):
            angle = params[:, param_idx] if is_batched else params[param_idx]
            qml.RY(angle, wires=i)
            param_idx += 1

        for i in range(wires):
            angle = params[:, param_idx] if is_batched else params[param_idx]
            qml.CRX(angle, wires=[i, (i + 1) % wires])
            param_idx += 1

        for i in range(wires):
            angle = params[:, param_idx] if is_batched else params[param_idx]
            qml.RY(angle, wires=i)
            param_idx += 1

        for i in range(wires - 1, -1, -1):
            angle = params[:, param_idx] if is_batched else params[param_idx]
            qml.CRX(angle, wires=[i, (i - 1) % wires])
            param_idx += 1


def quantum_shift_circuit(wires, shift_amount=1):
    """Quantum cyclic shift operation."""
    for _ in range(shift_amount):
        for i in range(wires - 1):
            qml.SWAP(wires=[i, i + 1])


def quantum_flip_circuit(wires):
    """Quantum flip (reversal) operation."""
    n = wires
    for i in range(n // 2):
        qml.SWAP(wires=[i, n - 1 - i])


def qlcu_circuit(params, wires, layers=1):
    """
    Quantum Linear Combination of Unitaries circuit.
    Simulates semi-separable matrix SS(X) from classical Hydra.
    """
    is_batched = params.ndim == 2
    param_idx = 0

    for _ in range(layers):
        for i in range(wires):
            angle = params[:, param_idx] if is_batched else params[param_idx]
            qml.RY(angle, wires=i)
            param_idx += 1

        for i in range(wires):
            angle = params[:, param_idx] if is_batched else params[param_idx]
            qml.IsingXX(angle, wires=[i, (i + 1) % wires])
            param_idx += 1

        for i in range(wires):
            angle = params[:, param_idx] if is_batched else params[param_idx]
            qml.RY(angle, wires=i)
            param_idx += 1

        for i in range(wires):
            angle = params[:, param_idx] if is_batched else params[param_idx]
            qml.IsingYY(angle, wires=[i, (i + 1) % wires])
            param_idx += 1


def qd_circuit(params, wires):
    """
    Quantum diagonal matrix operation QD.
    Applies independent single-qubit rotations.
    """
    is_batched = params.ndim == 2
    param_idx = 0

    for i in range(wires):
        angle_x = params[:, param_idx] if is_batched else params[param_idx]
        qml.RX(angle_x, wires=i)
        param_idx += 1

    for i in range(wires):
        angle_y = params[:, param_idx] if is_batched else params[param_idx]
        qml.RY(angle_y, wires=i)
        param_idx += 1

    for i in range(wires):
        angle_z = params[:, param_idx] if is_batched else params[param_idx]
        qml.RZ(angle_z, wires=i)
        param_idx += 1


# ================================================================================
# Hybrid Quantum Hydra Model (Option B)
# ================================================================================

class QuantumHydraHybridLayer(nn.Module):
    """
    Quantum Hydra - Hybrid Classical-Quantum Version (Option B).

    Computes three quantum branches independently, measures each,
    then combines results via CLASSICAL weighted addition.

    Algorithm:
        1. |ψ₁⟩ = Qshift(QLCU|X⟩)         → Measure → y₁
        2. |ψ₂⟩ = Qflip(Qshift(QLCU(Qflip|X⟩))) → Measure → y₂
        3. |ψ₃⟩ = QD|X⟩                   → Measure → y₃
        4. Y = OutputLayer(w₁·y₁ + w₂·y₂ + w₃·y₃)

    Key Difference from Option A:
    - No quantum superposition between branches
    - Classical addition preserves Hydra's semantic meaning
    - Can analyze each branch's contribution independently

    Args:
        n_qubits: Number of qubits
        qlcu_layers: Circuit depth for QLCU
        shift_amount: Shift amount for Qshift
        feature_dim: Input feature dimension
        output_dim: Output dimension
        dropout: Dropout probability
        device: torch device
    """

    def __init__(
        self,
        n_qubits: int,
        qlcu_layers: int = 2,
        shift_amount: int = 1,
        feature_dim: int = 128,
        output_dim: int = 1,
        dropout: float = 0.1,
        device: str = "cpu"
    ):
        super().__init__()

        self.n_qubits = n_qubits
        self.qlcu_layers = qlcu_layers
        self.shift_amount = shift_amount
        self.device = device

        # Calculate parameter counts
        self.n_qlcu_params = 4 * n_qubits * qlcu_layers
        self.n_qd_params = 3 * n_qubits
        self.n_measurements = 3 * n_qubits  # PauliX, Y, Z on each qubit

        # Classical layers
        self.feature_projection = nn.Linear(feature_dim, self.n_qlcu_params)
        self.dropout = nn.Dropout(dropout)
        self.activation = nn.Sigmoid()  # Bound to [0, 1], then scale to [0, 2π]

        # Branch-specific output processing (optional)
        # Each branch can have its own linear transformation before combining
        self.branch1_ff = nn.Linear(self.n_measurements, self.n_measurements)
        self.branch2_ff = nn.Linear(self.n_measurements, self.n_measurements)
        self.branch3_ff = nn.Linear(self.n_measurements, self.n_measurements)

        # Trainable classical weights for combining branches
        self.w1 = nn.Parameter(torch.ones(1))
        self.w2 = nn.Parameter(torch.ones(1))
        self.w3 = nn.Parameter(torch.ones(1))

        # Final output layer
        self.output_ff = nn.Linear(self.n_measurements, output_dim)

        # Fixed quantum circuit parameters (learnable)
        self.qlcu_base_params = nn.Parameter(torch.rand(self.n_qlcu_params))
        self.qd_params = nn.Parameter(torch.rand(self.n_qd_params))

        # PennyLane quantum device - use GPU if available and requested
        if device == "cuda" and torch.cuda.is_available():
            try:
                self.dev = qml.device("lightning.gpu", wires=self.n_qubits)
            except:
                # Fallback if lightning.gpu not installed
                import warnings
                warnings.warn("lightning.gpu not available, using default.qubit (CPU)")
                self.dev = qml.device("default.qubit", wires=self.n_qubits)
        else:
            self.dev = qml.device("default.qubit", wires=self.n_qubits)

        # Setup QNodes
        self._setup_qnodes()

        # Move all parameters to specified device
        self.to(device)

    def _setup_qnodes(self):
        """Setup PennyLane QNodes for quantum operations."""

        # Branch 1: Qshift(QLCU|X⟩)
        @qml.qnode(self.dev, interface="torch", diff_method="backprop")
        def _branch1_qnode(initial_state, params):
            qml.StatePrep(initial_state, wires=range(self.n_qubits))
            qlcu_circuit(params, wires=self.n_qubits, layers=self.qlcu_layers)
            quantum_shift_circuit(wires=self.n_qubits, shift_amount=self.shift_amount)
            observables = [qml.PauliX(i) for i in range(self.n_qubits)] + \
                          [qml.PauliY(i) for i in range(self.n_qubits)] + \
                          [qml.PauliZ(i) for i in range(self.n_qubits)]
            return [qml.expval(op) for op in observables]

        # Branch 2: Qflip(Qshift(QLCU(Qflip|X⟩)))
        @qml.qnode(self.dev, interface="torch", diff_method="backprop")
        def _branch2_qnode(initial_state, params):
            qml.StatePrep(initial_state, wires=range(self.n_qubits))
            quantum_flip_circuit(wires=self.n_qubits)
            qlcu_circuit(params, wires=self.n_qubits, layers=self.qlcu_layers)
            quantum_shift_circuit(wires=self.n_qubits, shift_amount=self.shift_amount)
            quantum_flip_circuit(wires=self.n_qubits)
            observables = [qml.PauliX(i) for i in range(self.n_qubits)] + \
                          [qml.PauliY(i) for i in range(self.n_qubits)] + \
                          [qml.PauliZ(i) for i in range(self.n_qubits)]
            return [qml.expval(op) for op in observables]

        # Branch 3: QD|X⟩
        @qml.qnode(self.dev, interface="torch", diff_method="backprop")
        def _branch3_qnode(initial_state, params):
            qml.StatePrep(initial_state, wires=range(self.n_qubits))
            qd_circuit(params, wires=self.n_qubits)
            observables = [qml.PauliX(i) for i in range(self.n_qubits)] + \
                          [qml.PauliY(i) for i in range(self.n_qubits)] + \
                          [qml.PauliZ(i) for i in range(self.n_qubits)]
            return [qml.expval(op) for op in observables]

        self.branch1_qnode = _branch1_qnode
        self.branch2_qnode = _branch2_qnode
        self.branch3_qnode = _branch3_qnode

    def forward(self, x):
        """
        Forward pass of Quantum Hydra Hybrid.

        Args:
            x: Input tensor (batch_size, feature_dim)

        Returns:
            output: Tensor (batch_size, output_dim)
        """
        batch_size = x.shape[0]

        # Project input features to quantum circuit parameters
        x_projected = self.feature_projection(self.dropout(x))
        qlcu_params = self.activation(x_projected) * 2 * 3.14159  # Scale to [0, 2π]

        # Add base parameters
        qlcu_params = qlcu_params + self.qlcu_base_params.unsqueeze(0)

        # Initialize quantum state to |0...0⟩
        base_states = torch.zeros(batch_size, 2 ** self.n_qubits,
                                   dtype=torch.complex64, device=self.device)
        base_states[:, 0] = 1.0

        # QD parameters (shared across batch)
        qd_params_expanded = self.qd_params.unsqueeze(0).expand(batch_size, -1)

        # ====================================================================
        # Compute and measure each branch independently
        # ====================================================================

        # Branch 1: y₁ = Measure(Qshift(QLCU|X⟩))
        measurements1 = self.branch1_qnode(initial_state=base_states, params=qlcu_params)
        measurements1 = torch.stack(measurements1, dim=1).float()  # (batch_size, 3*n_qubits)
        y1 = self.branch1_ff(measurements1)  # Optional transformation

        # Branch 2: y₂ = Measure(Qflip(Qshift(QLCU(Qflip|X⟩))))
        measurements2 = self.branch2_qnode(initial_state=base_states, params=qlcu_params)
        measurements2 = torch.stack(measurements2, dim=1).float()
        y2 = self.branch2_ff(measurements2)

        # Branch 3: y₃ = Measure(QD|X⟩)
        measurements3 = self.branch3_qnode(initial_state=base_states, params=qd_params_expanded)
        measurements3 = torch.stack(measurements3, dim=1).float()
        y3 = self.branch3_ff(measurements3)

        # ====================================================================
        # Classical weighted combination (like classical Hydra)
        # ====================================================================

        # Normalize weights to sum to 1 (optional, can remove if not desired)
        w_sum = torch.abs(self.w1) + torch.abs(self.w2) + torch.abs(self.w3)
        w1_norm = torch.abs(self.w1) / w_sum
        w2_norm = torch.abs(self.w2) / w_sum
        w3_norm = torch.abs(self.w3) / w_sum

        # Classical addition (THIS IS KEY DIFFERENCE FROM OPTION A)
        y_combined = w1_norm * y1 + w2_norm * y2 + w3_norm * y3

        # Final output layer
        output = self.output_ff(y_combined)

        return output

    def get_branch_contributions(self, x):
        """
        Get individual branch outputs for analysis.
        Useful for understanding which branch contributes most.

        Returns:
            dict with keys: 'branch1', 'branch2', 'branch3', 'combined'
        """
        with torch.no_grad():
            batch_size = x.shape[0]

            # Project input
            x_projected = self.feature_projection(x)
            qlcu_params = self.activation(x_projected) * 2 * 3.14159
            qlcu_params = qlcu_params + self.qlcu_base_params.unsqueeze(0)

            # Initialize state
            base_states = torch.zeros(batch_size, 2 ** self.n_qubits,
                                       dtype=torch.complex64, device=self.device)
            base_states[:, 0] = 1.0

            qd_params_expanded = self.qd_params.unsqueeze(0).expand(batch_size, -1)

            # Measure each branch
            measurements1 = self.branch1_qnode(initial_state=base_states, params=qlcu_params)
            measurements1 = torch.stack(measurements1, dim=1).float()
            y1 = self.branch1_ff(measurements1)

            measurements2 = self.branch2_qnode(initial_state=base_states, params=qlcu_params)
            measurements2 = torch.stack(measurements2, dim=1).float()
            y2 = self.branch2_ff(measurements2)

            measurements3 = self.branch3_qnode(initial_state=base_states, params=qd_params_expanded)
            measurements3 = torch.stack(measurements3, dim=1).float()
            y3 = self.branch3_ff(measurements3)

            # Weighted combination
            w_sum = torch.abs(self.w1) + torch.abs(self.w2) + torch.abs(self.w3)
            w1_norm = torch.abs(self.w1) / w_sum
            w2_norm = torch.abs(self.w2) / w_sum
            w3_norm = torch.abs(self.w3) / w_sum

            y_combined = w1_norm * y1 + w2_norm * y2 + w3_norm * y3

            return {
                'branch1': y1,
                'branch2': y2,
                'branch3': y3,
                'combined': y_combined,
                'weights': {
                    'w1': w1_norm.item(),
                    'w2': w2_norm.item(),
                    'w3': w3_norm.item()
                }
            }


# ================================================================================
# Multi-Timestep Quantum Hydra Hybrid (for time-series data)
# ================================================================================

class QuantumHydraHybridTS(nn.Module):
    """
    Quantum Hydra Hybrid for Time-Series data.
    Processes sequential data using Option B (classical combination).

    Args:
        n_qubits: Number of qubits
        n_timesteps: Sequence length
        qlcu_layers: QLCU circuit depth
        shift_amount: Shift amount for Qshift
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
        shift_amount: int = 1,
        feature_dim: int = 129,
        output_dim: int = 1,
        dropout: float = 0.1,
        device: str = "cpu"
    ):
        super().__init__()

        self.n_timesteps = n_timesteps
        self.n_qubits = n_qubits
        self.device = device

        # Feature projection
        self.n_qlcu_params = 4 * n_qubits * qlcu_layers
        self.feature_projection = nn.Linear(feature_dim, self.n_qlcu_params)
        self.dropout = nn.Dropout(dropout)

        # Single Quantum Hydra Hybrid layer
        self.quantum_hydra = QuantumHydraHybridLayer(
            n_qubits=n_qubits,
            qlcu_layers=qlcu_layers,
            shift_amount=shift_amount,
            feature_dim=self.n_qlcu_params,
            output_dim=3 * n_qubits,
            dropout=0.0,
            device=device
        )

        # Temporal aggregation weights (learnable)
        self.temporal_weights = nn.Parameter(torch.ones(n_timesteps) / n_timesteps)

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

        # Handle different input formats
        if x.shape[1] != self.n_timesteps and x.shape[2] == self.n_timesteps:
            x = x.permute(0, 2, 1)

        # VECTORIZED PROCESSING: Process all timesteps as a single batch
        # Reshape: (batch_size, n_timesteps, feature_dim) -> (batch_size * n_timesteps, feature_dim)
        x_flat = x.reshape(batch_size * self.n_timesteps, -1)

        # Apply dropout and feature projection to all timesteps at once
        x_proj = self.feature_projection(self.dropout(x_flat))

        # Run quantum layer on all timesteps as a large batch
        out_flat = self.quantum_hydra(x_proj)  # (batch_size * n_timesteps, 3*n_qubits)

        # Reshape back: (batch_size * n_timesteps, 3*n_qubits) -> (batch_size, n_timesteps, 3*n_qubits)
        timestep_outputs = out_flat.reshape(batch_size, self.n_timesteps, -1)

        # Weighted temporal aggregation (classical)
        weights = torch.softmax(self.temporal_weights, dim=0)
        weights = weights.unsqueeze(0).unsqueeze(-1)  # (1, n_timesteps, 1)

        aggregated = torch.sum(timestep_outputs * weights, dim=1)  # (batch_size, 3*n_qubits)

        # Final prediction
        output = self.output_layer(aggregated)

        return output


# ================================================================================
# Testing and Comparison
# ================================================================================

if __name__ == "__main__":
    print("=" * 80)
    print("Quantum Hydra Hybrid (Option B) - Testing")
    print("=" * 80)

    device = "cpu"
    n_qubits = 4
    feature_dim = 10
    output_dim = 2
    batch_size = 8

    print("\n[1] Testing QuantumHydraHybridLayer...")
    model = QuantumHydraHybridLayer(
        n_qubits=n_qubits,
        qlcu_layers=2,
        shift_amount=1,
        feature_dim=feature_dim,
        output_dim=output_dim,
        dropout=0.1,
        device=device
    )

    x = torch.randn(batch_size, feature_dim)
    output = model(x)
    print(f"  Input shape: {x.shape}")
    print(f"  Output shape: {output.shape}")
    print(f"  Output sample: {output[0]}")

    print("\n[2] Testing branch contribution analysis...")
    contributions = model.get_branch_contributions(x[:2])  # Just 2 samples
    print(f"  Branch 1 shape: {contributions['branch1'].shape}")
    print(f"  Branch 2 shape: {contributions['branch2'].shape}")
    print(f"  Branch 3 shape: {contributions['branch3'].shape}")
    print(f"  Branch weights: w1={contributions['weights']['w1']:.3f}, "
          f"w2={contributions['weights']['w2']:.3f}, "
          f"w3={contributions['weights']['w3']:.3f}")

    print("\n[3] Testing QuantumHydraHybridTS (time-series)...")
    n_timesteps = 20
    feature_dim_ts = 64

    model_ts = QuantumHydraHybridTS(
        n_qubits=4,
        n_timesteps=n_timesteps,
        qlcu_layers=2,
        shift_amount=1,
        feature_dim=feature_dim_ts,
        output_dim=1,
        dropout=0.1,
        device=device
    )

    x_ts = torch.randn(batch_size, feature_dim_ts, n_timesteps)
    output_ts = model_ts(x_ts)
    print(f"  Input shape: {x_ts.shape}")
    print(f"  Output shape: {output_ts.shape}")
    print(f"  Output sample: {output_ts[0]}")

    print("\n[4] Checking trainable parameters...")
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Total trainable parameters: {total_params:,}")

    print("\n[5] Verifying classical weights...")
    print(f"  w1: {model.w1.data.item():.4f}")
    print(f"  w2: {model.w2.data.item():.4f}")
    print(f"  w3: {model.w3.data.item():.4f}")

    print("\n[6] Key differences from Option A:")
    print("  ✓ Option A: Quantum superposition → |ψ⟩ = α|ψ₁⟩ + β|ψ₂⟩ + γ|ψ₃⟩")
    print("  ✓ Option B: Classical combination → y = w₁·y₁ + w₂·y₂ + w₃·y₃")
    print("  ✓ Option B is more faithful to classical Hydra semantics")
    print("  ✓ Option B allows independent branch analysis")

    print("\n" + "=" * 80)
    print("Testing complete!")
    print("=" * 80)
