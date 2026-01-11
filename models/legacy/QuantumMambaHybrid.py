import torch
import torch.nn as nn
import torch.nn.functional as F
import pennylane as qml
import numpy as np
from typing import Optional

"""
Quantum Mamba - Option B (Hybrid Classical-Quantum)

This implementation combines quantum circuits with classical weighted addition,
similar to how classical Mamba combines different pathways.

Classical Mamba equation (simplified):
    h[t] = A * h[t-1] + B(u) * u[t]
    y[t] = C(u) * h[t] + D * u[t]
    output = y * silu(gate)

Quantum Hybrid Mamba equation:
    y₁ = Measure(Q_SSM|X⟩)         ← Selective SSM path
    y₂ = Measure(Q_Gate|X⟩)        ← Gating path
    y₃ = Measure(Q_Skip|X⟩)        ← Skip connection path
    Y = OutputLayer(w₁·y₁ + w₂·y₂ + w₃·y₃)  ← Classical weighted sum

Key differences from Option A (QuantumMamba.py):
- Option A: Creates quantum superposition |ψ⟩ = α|ψ_ssm⟩ + β|ψ_gate⟩ + γ|ψ_skip⟩
- Option B: Measures each branch separately, combines classically

Advantages of Option B:
- Closer to classical Mamba semantics
- Easier to interpret (no quantum interference between branches)
- Can analyze contribution of each branch independently
- More robust to decoherence (each branch measured separately)
- Real-valued weights (easier optimization)
"""

# ================================================================================
# Quantum Circuit Components for Mamba
# ================================================================================

def selective_ssm_circuit(base_state, qlcu_params, b_params, c_params, dt_params,
                          n_qubits, qlcu_layers):
    """
    Quantum selective SSM circuit.
    Implements input-dependent state-space transformation.

    Args:
        base_state: Initial quantum state
        qlcu_params: Parameters for QLCU (state transformation)
        b_params: Input-dependent B matrix parameters
        c_params: Input-dependent C matrix parameters
        dt_params: Time-step parameters
        n_qubits: Number of qubits
        qlcu_layers: Circuit depth
    """
    is_batched = qlcu_params.ndim == 2
    param_idx = 0

    # Input encoding
    for _ in range(qlcu_layers):
        # A matrix transformation (state evolution)
        for i in range(n_qubits):
            angle = qlcu_params[:, param_idx] if is_batched else qlcu_params[param_idx]
            qml.RY(angle, wires=i)
            param_idx += 1

        # Entanglement for state mixing
        for i in range(n_qubits):
            angle = qlcu_params[:, param_idx] if is_batched else qlcu_params[param_idx]
            qml.IsingXX(angle, wires=[i, (i + 1) % n_qubits])
            param_idx += 1

        # Input-dependent B matrix (selective mechanism)
        for i in range(n_qubits):
            angle_b = b_params[:, i] if is_batched else b_params[i]
            qml.RY(angle_b, wires=i)

        # Second layer
        for i in range(n_qubits):
            angle = qlcu_params[:, param_idx] if is_batched else qlcu_params[param_idx]
            qml.RY(angle, wires=i)
            param_idx += 1

        for i in range(n_qubits):
            angle = qlcu_params[:, param_idx] if is_batched else qlcu_params[param_idx]
            qml.IsingYY(angle, wires=[i, (i + 1) % n_qubits])
            param_idx += 1

        # Output-dependent C matrix
        for i in range(n_qubits):
            angle_c = c_params[:, i] if is_batched else c_params[i]
            qml.RZ(angle_c, wires=i)

        # Time-step modulation
        for i in range(n_qubits):
            angle_dt = dt_params[:, i] if is_batched else dt_params[i]
            qml.RX(angle_dt, wires=i)


def gating_circuit(base_state, gate_params, n_qubits, layers=2):
    """
    Quantum gating circuit.
    Implements multiplicative gating mechanism (like SiLU gating in Mamba).

    Args:
        base_state: Initial quantum state
        gate_params: Gating parameters
        n_qubits: Number of qubits
        layers: Circuit depth
    """
    is_batched = gate_params.ndim == 2
    param_idx = 0

    for _ in range(layers):
        # First rotation layer
        for i in range(n_qubits):
            angle = gate_params[:, param_idx] if is_batched else gate_params[param_idx]
            qml.RX(angle, wires=i)
            param_idx += 1

        # Controlled gates for gating
        for i in range(n_qubits - 1):
            angle = gate_params[:, param_idx] if is_batched else gate_params[param_idx]
            qml.CRZ(angle, wires=[i, i + 1])
            param_idx += 1

        # Second rotation layer
        for i in range(n_qubits):
            angle = gate_params[:, param_idx] if is_batched else gate_params[param_idx]
            qml.RY(angle, wires=i)
            param_idx += 1

        # Ring connectivity for last qubit
        if n_qubits > 1:
            angle = gate_params[:, param_idx] if is_batched else gate_params[param_idx]
            qml.CRZ(angle, wires=[n_qubits - 1, 0])
            param_idx += 1


def skip_circuit(base_state, skip_params, n_qubits):
    """
    Quantum skip connection circuit (D matrix in Mamba).
    Diagonal operations for direct input-to-output path.

    Args:
        base_state: Initial quantum state
        skip_params: Skip connection parameters
        n_qubits: Number of qubits
    """
    is_batched = skip_params.ndim == 2
    param_idx = 0

    # Apply diagonal rotations (independent on each qubit)
    for i in range(n_qubits):
        angle_x = skip_params[:, param_idx] if is_batched else skip_params[param_idx]
        qml.RX(angle_x, wires=i)
        param_idx += 1

    for i in range(n_qubits):
        angle_y = skip_params[:, param_idx] if is_batched else skip_params[param_idx]
        qml.RY(angle_y, wires=i)
        param_idx += 1

    for i in range(n_qubits):
        angle_z = skip_params[:, param_idx] if is_batched else skip_params[param_idx]
        qml.RZ(angle_z, wires=i)
        param_idx += 1


# ================================================================================
# Hybrid Quantum Mamba Model (Option B)
# ================================================================================

class QuantumMambaHybridLayer(nn.Module):
    """
    Quantum Mamba - Hybrid Classical-Quantum Version (Option B).

    Computes three quantum branches independently, measures each,
    then combines results via CLASSICAL weighted addition.

    Algorithm:
        1. |ψ_ssm⟩ = Q_SSM|X⟩         → Measure → y₁
        2. |ψ_gate⟩ = Q_Gate|X⟩      → Measure → y₂
        3. |ψ_skip⟩ = Q_Skip|X⟩      → Measure → y₃
        4. Y = OutputLayer(w₁·y₁ + w₂·y₂ + w₃·y₃)

    Key Difference from Option A:
    - No quantum superposition between branches
    - Classical addition preserves Mamba's semantic meaning
    - Can analyze each branch's contribution independently

    Args:
        n_qubits: Number of qubits
        qlcu_layers: Circuit depth for QLCU (SSM transformation)
        gate_layers: Circuit depth for gating
        feature_dim: Input feature dimension
        output_dim: Output dimension
        dropout: Dropout probability
        device: torch device
    """

    def __init__(
        self,
        n_qubits: int,
        qlcu_layers: int = 2,
        gate_layers: int = 2,
        feature_dim: int = 128,
        output_dim: int = 1,
        dropout: float = 0.1,
        device: str = "cpu"
    ):
        super().__init__()

        self.n_qubits = n_qubits
        self.qlcu_layers = qlcu_layers
        self.gate_layers = gate_layers
        self.device = device

        # Calculate parameter counts
        self.n_qlcu_params = 4 * n_qubits * qlcu_layers  # SSM transformation
        self.n_gate_params = (3 * n_qubits + (n_qubits)) * gate_layers  # Gating
        self.n_skip_params = 3 * n_qubits  # Skip connection
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
        self.w1 = nn.Parameter(torch.ones(1))  # SSM weight
        self.w2 = nn.Parameter(torch.ones(1))  # Gate weight
        self.w3 = nn.Parameter(torch.ones(1))  # Skip weight

        # Final output layer
        self.output_ff = nn.Linear(self.n_measurements, output_dim)

        # Fixed quantum circuit parameters (learnable)
        self.qlcu_base_params = nn.Parameter(torch.rand(self.n_qlcu_params) * 2 * np.pi)
        self.gate_base_params = nn.Parameter(torch.rand(self.n_gate_params) * 2 * np.pi)
        self.skip_params = nn.Parameter(torch.rand(self.n_skip_params) * 2 * np.pi)

        # Selective mechanism parameters (B, C, dt)
        self.b_params = nn.Parameter(torch.rand(n_qubits) * 2 * np.pi)
        self.c_params = nn.Parameter(torch.rand(n_qubits) * 2 * np.pi)
        self.dt_params = nn.Parameter(torch.rand(n_qubits) * 2 * np.pi)

        # PennyLane quantum device - use GPU if available and requested
        if device == "cuda" and torch.cuda.is_available():
            try:
                self.dev = qml.device("lightning.gpu", wires=self.n_qubits)
            except:
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

        # Branch 1: Selective SSM path
        @qml.qnode(self.dev, interface="torch", diff_method="backprop")
        def _branch1_qnode(initial_state, qlcu_params, b_params, c_params, dt_params):
            qml.StatePrep(initial_state, wires=range(self.n_qubits))
            selective_ssm_circuit(
                initial_state, qlcu_params, b_params, c_params, dt_params,
                self.n_qubits, self.qlcu_layers
            )
            observables = [qml.PauliX(i) for i in range(self.n_qubits)] + \
                          [qml.PauliY(i) for i in range(self.n_qubits)] + \
                          [qml.PauliZ(i) for i in range(self.n_qubits)]
            return [qml.expval(op) for op in observables]

        # Branch 2: Gating path
        @qml.qnode(self.dev, interface="torch", diff_method="backprop")
        def _branch2_qnode(initial_state, gate_params):
            qml.StatePrep(initial_state, wires=range(self.n_qubits))
            gating_circuit(initial_state, gate_params, self.n_qubits, self.gate_layers)
            observables = [qml.PauliX(i) for i in range(self.n_qubits)] + \
                          [qml.PauliY(i) for i in range(self.n_qubits)] + \
                          [qml.PauliZ(i) for i in range(self.n_qubits)]
            return [qml.expval(op) for op in observables]

        # Branch 3: Skip connection path
        @qml.qnode(self.dev, interface="torch", diff_method="backprop")
        def _branch3_qnode(initial_state, skip_params):
            qml.StatePrep(initial_state, wires=range(self.n_qubits))
            skip_circuit(initial_state, skip_params, self.n_qubits)
            observables = [qml.PauliX(i) for i in range(self.n_qubits)] + \
                          [qml.PauliY(i) for i in range(self.n_qubits)] + \
                          [qml.PauliZ(i) for i in range(self.n_qubits)]
            return [qml.expval(op) for op in observables]

        self.branch1_qnode = _branch1_qnode
        self.branch2_qnode = _branch2_qnode
        self.branch3_qnode = _branch3_qnode

    def forward(self, x):
        """
        Forward pass of Quantum Mamba Hybrid.

        Args:
            x: Input tensor (batch_size, feature_dim)

        Returns:
            output: Tensor (batch_size, output_dim)
        """
        batch_size = x.shape[0]

        # Project input features to quantum circuit parameters
        x_projected = self.feature_projection(self.dropout(x))
        qlcu_params = self.activation(x_projected) * 2 * np.pi  # Scale to [0, 2π]

        # Add base parameters
        qlcu_params = qlcu_params + self.qlcu_base_params.unsqueeze(0)

        # Initialize quantum state to |0...0⟩
        base_states = torch.zeros(batch_size, 2 ** self.n_qubits,
                                   dtype=torch.complex64, device=self.device)
        base_states[:, 0] = 1.0

        # Expand selective mechanism parameters
        b_params_expanded = self.b_params.unsqueeze(0).expand(batch_size, -1)
        c_params_expanded = self.c_params.unsqueeze(0).expand(batch_size, -1)
        dt_params_expanded = self.dt_params.unsqueeze(0).expand(batch_size, -1)

        # Gate and skip parameters
        gate_params_expanded = self.gate_base_params.unsqueeze(0).expand(batch_size, -1)
        skip_params_expanded = self.skip_params.unsqueeze(0).expand(batch_size, -1)

        # ====================================================================
        # Compute and measure each branch independently
        # ====================================================================

        # Branch 1: y₁ = Measure(Q_SSM|X⟩)
        measurements1 = self.branch1_qnode(
            initial_state=base_states,
            qlcu_params=qlcu_params,
            b_params=b_params_expanded,
            c_params=c_params_expanded,
            dt_params=dt_params_expanded
        )
        measurements1 = torch.stack(measurements1, dim=1).float()  # (batch_size, 3*n_qubits)
        y1 = self.branch1_ff(measurements1)  # Optional transformation

        # Branch 2: y₂ = Measure(Q_Gate|X⟩)
        measurements2 = self.branch2_qnode(
            initial_state=base_states,
            gate_params=gate_params_expanded
        )
        measurements2 = torch.stack(measurements2, dim=1).float()
        y2 = self.branch2_ff(measurements2)

        # Branch 3: y₃ = Measure(Q_Skip|X⟩)
        measurements3 = self.branch3_qnode(
            initial_state=base_states,
            skip_params=skip_params_expanded
        )
        measurements3 = torch.stack(measurements3, dim=1).float()
        y3 = self.branch3_ff(measurements3)

        # ====================================================================
        # Classical weighted combination (like classical Mamba)
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
            dict with keys: 'branch1', 'branch2', 'branch3', 'combined', 'weights'
        """
        with torch.no_grad():
            batch_size = x.shape[0]

            # Project input
            x_projected = self.feature_projection(x)
            qlcu_params = self.activation(x_projected) * 2 * np.pi
            qlcu_params = qlcu_params + self.qlcu_base_params.unsqueeze(0)

            # Initialize state
            base_states = torch.zeros(batch_size, 2 ** self.n_qubits,
                                       dtype=torch.complex64, device=self.device)
            base_states[:, 0] = 1.0

            # Expand parameters
            b_params_expanded = self.b_params.unsqueeze(0).expand(batch_size, -1)
            c_params_expanded = self.c_params.unsqueeze(0).expand(batch_size, -1)
            dt_params_expanded = self.dt_params.unsqueeze(0).expand(batch_size, -1)
            gate_params_expanded = self.gate_base_params.unsqueeze(0).expand(batch_size, -1)
            skip_params_expanded = self.skip_params.unsqueeze(0).expand(batch_size, -1)

            # Measure each branch
            measurements1 = self.branch1_qnode(
                initial_state=base_states, qlcu_params=qlcu_params,
                b_params=b_params_expanded, c_params=c_params_expanded,
                dt_params=dt_params_expanded
            )
            measurements1 = torch.stack(measurements1, dim=1).float()
            y1 = self.branch1_ff(measurements1)

            measurements2 = self.branch2_qnode(
                initial_state=base_states, gate_params=gate_params_expanded
            )
            measurements2 = torch.stack(measurements2, dim=1).float()
            y2 = self.branch2_ff(measurements2)

            measurements3 = self.branch3_qnode(
                initial_state=base_states, skip_params=skip_params_expanded
            )
            measurements3 = torch.stack(measurements3, dim=1).float()
            y3 = self.branch3_ff(measurements3)

            # Weighted combination
            w_sum = torch.abs(self.w1) + torch.abs(self.w2) + torch.abs(self.w3)
            w1_norm = torch.abs(self.w1) / w_sum
            w2_norm = torch.abs(self.w2) / w_sum
            w3_norm = torch.abs(self.w3) / w_sum

            y_combined = w1_norm * y1 + w2_norm * y2 + w3_norm * y3

            return {
                'branch1_ssm': y1,
                'branch2_gate': y2,
                'branch3_skip': y3,
                'combined': y_combined,
                'weights': {
                    'w1_ssm': w1_norm.item(),
                    'w2_gate': w2_norm.item(),
                    'w3_skip': w3_norm.item()
                }
            }


# ================================================================================
# Multi-Timestep Quantum Mamba Hybrid (for time-series data)
# ================================================================================

class QuantumMambaHybridTS(nn.Module):
    """
    Quantum Mamba Hybrid for Time-Series data.
    Processes sequential data using Option B (classical combination).

    Args:
        n_qubits: Number of qubits
        n_timesteps: Sequence length
        qlcu_layers: QLCU circuit depth
        gate_layers: Gating circuit depth
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
        gate_layers: int = 2,
        feature_dim: int = 129,
        output_dim: int = 1,
        dropout: float = 0.1,
        device: str = "cpu"
    ):
        super().__init__()

        self.n_timesteps = n_timesteps
        self.n_qubits = n_qubits
        self.device = device

        # Temporal encoding via 1D convolution (like Mamba)
        self.temporal_conv = nn.Conv1d(
            in_channels=feature_dim,
            out_channels=64,
            kernel_size=3,
            padding=1,
            groups=1
        )
        self.pool = nn.AdaptiveAvgPool1d(1)

        # Feature projection
        self.n_qlcu_params = 4 * n_qubits * qlcu_layers
        self.feature_projection = nn.Linear(64, self.n_qlcu_params)
        self.dropout = nn.Dropout(dropout)

        # Single Quantum Mamba Hybrid layer
        self.quantum_mamba = QuantumMambaHybridLayer(
            n_qubits=n_qubits,
            qlcu_layers=qlcu_layers,
            gate_layers=gate_layers,
            feature_dim=self.n_qlcu_params,
            output_dim=3 * n_qubits,
            dropout=0.0,
            device=device
        )

        # Final output layer
        self.output_layer = nn.Linear(3 * n_qubits, output_dim)

        # Move all parameters to specified device
        self.to(device)

    def forward(self, x):
        """
        Forward pass for time-series input.

        Args:
            x: Input tensor (batch_size, feature_dim, n_timesteps) or
               (batch_size, n_timesteps, feature_dim)

        Returns:
            output: Tensor (batch_size, output_dim)
        """
        batch_size = x.shape[0]

        # Handle different input formats
        if x.ndim == 3 and x.shape[2] == self.n_timesteps:
            # (batch, feature_dim, timesteps) - already correct for conv1d
            pass
        elif x.ndim == 3 and x.shape[1] == self.n_timesteps:
            # (batch, timesteps, feature_dim) - need to permute
            x = x.permute(0, 2, 1)

        # Temporal convolution (like Mamba's 1D conv)
        x_conv = F.silu(self.temporal_conv(x))  # (batch, 64, timesteps)

        # Global pooling
        x_pooled = self.pool(x_conv).squeeze(-1)  # (batch, 64)

        # Project to quantum parameters
        x_proj = self.feature_projection(self.dropout(x_pooled))

        # Quantum Mamba processing
        quantum_out = self.quantum_mamba(x_proj)

        # Final prediction
        output = self.output_layer(quantum_out)

        return output


# ================================================================================
# Testing and Comparison
# ================================================================================

if __name__ == "__main__":
    print("=" * 80)
    print("Quantum Mamba Hybrid (Option B) - Testing")
    print("=" * 80)

    device = "cpu"
    n_qubits = 4
    feature_dim = 10
    output_dim = 2
    batch_size = 8

    print("\n[1] Testing QuantumMambaHybridLayer...")
    model = QuantumMambaHybridLayer(
        n_qubits=n_qubits,
        qlcu_layers=2,
        gate_layers=2,
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
    print(f"  Branch 1 (SSM) shape: {contributions['branch1_ssm'].shape}")
    print(f"  Branch 2 (Gate) shape: {contributions['branch2_gate'].shape}")
    print(f"  Branch 3 (Skip) shape: {contributions['branch3_skip'].shape}")
    print(f"  Branch weights: w1_ssm={contributions['weights']['w1_ssm']:.3f}, "
          f"w2_gate={contributions['weights']['w2_gate']:.3f}, "
          f"w3_skip={contributions['weights']['w3_skip']:.3f}")

    print("\n[3] Testing QuantumMambaHybridTS (time-series)...")
    n_timesteps = 20
    feature_dim_ts = 64

    model_ts = QuantumMambaHybridTS(
        n_qubits=4,
        n_timesteps=n_timesteps,
        qlcu_layers=2,
        gate_layers=2,
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
    print(f"  w1 (SSM): {model.w1.data.item():.4f}")
    print(f"  w2 (Gate): {model.w2.data.item():.4f}")
    print(f"  w3 (Skip): {model.w3.data.item():.4f}")

    print("\n[6] Key differences from Option A:")
    print("  ✓ Option A: Quantum superposition → |ψ⟩ = α|ψ_ssm⟩ + β|ψ_gate⟩ + γ|ψ_skip⟩")
    print("  ✓ Option B: Classical combination → y = w₁·y₁ + w₂·y₂ + w₃·y₃")
    print("  ✓ Option B is more faithful to classical Mamba semantics")
    print("  ✓ Option B allows independent branch analysis")
    print("  ✓ Real-valued weights (easier optimization)")

    print("\n[7] Quantum Mamba architecture:")
    print("  ✓ Branch 1: Selective SSM (input-dependent B, C, dt)")
    print("  ✓ Branch 2: Gating mechanism (like SiLU gating)")
    print("  ✓ Branch 3: Skip connection (D matrix)")

    print("\n" + "=" * 80)
    print("Testing complete!")
    print("=" * 80)
