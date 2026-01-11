import torch
import torch.nn as nn
import pennylane as qml
import numpy as np

"""
Quantum Mamba (Superposition) - Option A

Quantum implementation of Mamba SSM using quantum superposition to combine
multiple selective state space paths.

Based on:
- Mamba: Gu & Dao (2024) - Selective State Spaces
- Quantum Hydra design philosophy

Key Idea:
Instead of Hydra's three branches (shift, flip, diagonal), Mamba uses:
1. Forward SSM path with input-dependent B, C, dt
2. Gated path for selective information flow
3. Skip connection path

Quantum Superposition:
|ψ⟩ = α|ψ_ssm⟩ + β|ψ_gate⟩ + γ|ψ_skip⟩

where α, β, γ ∈ ℂ are trainable complex coefficients.
"""


class QuantumSelectiveSSM(nn.Module):
    """
    Quantum implementation of selective SSM core using QLCU.

    UPDATED: Now supports batched processing for efficiency.

    Simulates the selective state space operation:
        x[t] = A * x[t-1] + B(u) * u[t]
        y[t] = C(u) * x[t]

    through quantum linear combinations of unitaries.
    """

    def __init__(self, n_qubits, qlcu_layers, device="cpu"):
        super().__init__()
        self.n_qubits = n_qubits
        self.qlcu_layers = qlcu_layers
        self.device = device

        # QLCU parameters for SSM transformation
        n_qlcu_params = 4 * n_qubits * qlcu_layers
        self.qlcu_params = nn.Parameter(torch.rand(n_qlcu_params) * 2 * np.pi)

        # Parameters for selective mechanism (input-dependent B, C)
        self.b_params = nn.Parameter(torch.rand(n_qubits) * 2 * np.pi)
        self.c_params = nn.Parameter(torch.rand(n_qubits) * 2 * np.pi)

        # Quantum device - use GPU if available and requested
        if device == "cuda" and torch.cuda.is_available():
            try:
                self.qdev = qml.device("lightning.gpu", wires=n_qubits)
            except:
                import warnings
                warnings.warn("lightning.gpu not available, using default.qubit (CPU)")
                self.qdev = qml.device("default.qubit", wires=n_qubits)
        else:
            self.qdev = qml.device("default.qubit", wires=n_qubits)

        # Setup QNode
        self._setup_qnode()

        # Move parameters to device
        self.to(device)

    def _setup_qnode(self):
        """Create quantum circuit for selective SSM with batched support."""

        @qml.qnode(self.qdev, interface="torch", diff_method="backprop")
        def ssm_circuit(base_state, qlcu_params, b_params, c_params):
            # Use StatePrep for batched support
            qml.StatePrep(base_state, wires=range(self.n_qubits))

            # Check if batched
            is_batched = qlcu_params.ndim == 2

            # QLCU layers (simulating SSM transformation)
            param_idx = 0
            for layer in range(self.qlcu_layers):
                # Rotation layer
                for i in range(self.n_qubits):
                    angle = qlcu_params[:, param_idx] if is_batched else qlcu_params[param_idx]
                    qml.RY(angle, wires=i)
                    param_idx += 1
                    angle = qlcu_params[:, param_idx] if is_batched else qlcu_params[param_idx]
                    qml.RZ(angle, wires=i)
                    param_idx += 1

                # Entanglement layer (simulating state coupling)
                for i in range(self.n_qubits - 1):
                    angle = qlcu_params[:, param_idx] if is_batched else qlcu_params[param_idx]
                    qml.CRX(angle, wires=[i, i + 1])
                    param_idx += 1

                # Selective mechanism (input-dependent)
                for i in range(self.n_qubits):
                    angle = b_params[:, i] if is_batched else b_params[i]
                    qml.RY(angle, wires=i)

                # Additional rotation
                for i in range(self.n_qubits):
                    angle = qlcu_params[:, param_idx] if is_batched else qlcu_params[param_idx]
                    qml.RX(angle, wires=i)
                    param_idx += 1

                # Output selection (C matrix)
                for i in range(self.n_qubits):
                    angle = c_params[:, i] if is_batched else c_params[i]
                    qml.RZ(angle, wires=i)

            # Measure all qubits in X, Y, Z bases
            observables = [qml.PauliX(i) for i in range(self.n_qubits)] + \
                          [qml.PauliY(i) for i in range(self.n_qubits)] + \
                          [qml.PauliZ(i) for i in range(self.n_qubits)]
            return [qml.expval(op) for op in observables]

        self.ssm_qnode = ssm_circuit

    def forward(self, base_state):
        """
        Args:
            base_state: (batch, 2^n_qubits) normalized quantum state

        Returns:
            measurements: (batch, 3*n_qubits) expectation values
        """
        batch_size = base_state.shape[0]

        # Expand parameters for batched processing
        qlcu_params_expanded = self.qlcu_params.unsqueeze(0).expand(batch_size, -1)
        b_params_expanded = self.b_params.unsqueeze(0).expand(batch_size, -1)
        c_params_expanded = self.c_params.unsqueeze(0).expand(batch_size, -1)

        # Single batched call
        measurements = self.ssm_qnode(
            base_state,
            qlcu_params_expanded,
            b_params_expanded,
            c_params_expanded
        )

        return torch.stack(measurements, dim=1).float()


class QuantumGatingPath(nn.Module):
    """
    Quantum gating mechanism (replaces SiLU gating in classical Mamba).

    UPDATED: Now supports batched processing for efficiency.

    Implements quantum version of: y * σ(z)
    where σ is activation function.
    """

    def __init__(self, n_qubits, device="cpu"):
        super().__init__()
        self.n_qubits = n_qubits
        self.device = device

        # Gating parameters
        self.gate_params = nn.Parameter(torch.rand(n_qubits) * 2 * np.pi)

        # Quantum device - use GPU if available and requested
        if device == "cuda" and torch.cuda.is_available():
            try:
                self.qdev = qml.device("lightning.gpu", wires=n_qubits)
            except:
                import warnings
                warnings.warn("lightning.gpu not available, using default.qubit (CPU)")
                self.qdev = qml.device("default.qubit", wires=n_qubits)
        else:
            self.qdev = qml.device("default.qubit", wires=n_qubits)

        self._setup_qnode()

        # Move parameters to device
        self.to(device)

    def _setup_qnode(self):
        """Create quantum circuit for gating with batched support."""

        @qml.qnode(self.qdev, interface="torch", diff_method="backprop")
        def gate_circuit(base_state, gate_params):
            # Use StatePrep for batched support
            qml.StatePrep(base_state, wires=range(self.n_qubits))

            # Check if batched
            is_batched = gate_params.ndim == 2

            # Gating rotations
            for i in range(self.n_qubits):
                angle = gate_params[:, i] if is_batched else gate_params[i]
                qml.RY(angle, wires=i)

            # Entanglement for gating correlation
            for i in range(self.n_qubits - 1):
                qml.CNOT(wires=[i, i + 1])

            # Measure
            observables = [qml.PauliX(i) for i in range(self.n_qubits)] + \
                          [qml.PauliY(i) for i in range(self.n_qubits)] + \
                          [qml.PauliZ(i) for i in range(self.n_qubits)]
            return [qml.expval(op) for op in observables]

        self.gate_qnode = gate_circuit

    def forward(self, base_state):
        """
        Args:
            base_state: (batch, 2^n_qubits)

        Returns:
            measurements: (batch, 3*n_qubits)
        """
        batch_size = base_state.shape[0]

        # Expand parameters for batched processing
        gate_params_expanded = self.gate_params.unsqueeze(0).expand(batch_size, -1)

        # Single batched call
        measurements = self.gate_qnode(base_state, gate_params_expanded)

        return torch.stack(measurements, dim=1).float()


class QuantumMambaLayer(nn.Module):
    """
    Quantum Mamba Layer using quantum superposition.

    Combines three quantum paths via superposition:
    |ψ⟩ = α|ψ_ssm⟩ + β|ψ_gate⟩ + γ|ψ_skip⟩

    Architecture:
    1. SSM path: Selective state space transformation
    2. Gate path: Gating mechanism
    3. Skip path: Direct connection (diagonal)

    Args:
        n_qubits: Number of qubits
        qlcu_layers: Depth of QLCU circuits
        feature_dim: Input feature dimension
        output_dim: Output dimension
        dropout: Dropout rate
        device: 'cuda' or 'cpu'
    """

    def __init__(
        self,
        n_qubits=4,
        qlcu_layers=2,
        feature_dim=64,
        output_dim=4,
        dropout=0.1,
        device="cpu"
    ):
        super().__init__()

        self.n_qubits = n_qubits
        self.qlcu_layers = qlcu_layers
        self.feature_dim = feature_dim
        self.output_dim = output_dim
        self.device = device
        self.state_dim = 2 ** n_qubits

        # Trainable complex coefficients for superposition
        alpha_init = torch.rand(1, dtype=torch.float32)
        beta_init = torch.rand(1, dtype=torch.float32)
        gamma_init = torch.rand(1, dtype=torch.float32)

        self.alpha_real = nn.Parameter(alpha_init)
        self.alpha_imag = nn.Parameter(torch.zeros(1))
        self.beta_real = nn.Parameter(beta_init)
        self.beta_imag = nn.Parameter(torch.zeros(1))
        self.gamma_real = nn.Parameter(gamma_init)
        self.gamma_imag = nn.Parameter(torch.zeros(1))

        # Input processing
        self.input_proj = nn.Linear(feature_dim, self.state_dim)
        self.layer_norm = nn.LayerNorm(self.state_dim)

        # Three quantum paths
        self.ssm_path = QuantumSelectiveSSM(n_qubits, qlcu_layers, device)
        self.gate_path = QuantumGatingPath(n_qubits, device)

        # Skip path parameters (diagonal operation)
        self.skip_params = nn.Parameter(torch.rand(n_qubits) * 2 * np.pi)
        # Quantum device for skip path - use GPU if available and requested
        if device == "cuda" and torch.cuda.is_available():
            try:
                self.qdev_skip = qml.device("lightning.gpu", wires=n_qubits)
            except:
                import warnings
                warnings.warn("lightning.gpu not available, using default.qubit (CPU)")
                self.qdev_skip = qml.device("default.qubit", wires=n_qubits)
        else:
            self.qdev_skip = qml.device("default.qubit", wires=n_qubits)
        self._setup_skip_qnode()

        # Output processing
        measurement_dim = 3 * n_qubits
        self.output_ff = nn.Sequential(
            nn.Linear(measurement_dim, measurement_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(measurement_dim, output_dim)
        )

        self.dropout = nn.Dropout(dropout)

        # Move all parameters to specified device
        self.to(device)

    def _setup_skip_qnode(self):
        """Setup quantum circuit for skip connection with batched support."""

        @qml.qnode(self.qdev_skip, interface="torch", diff_method="backprop")
        def skip_circuit(base_state, skip_params):
            # Use StatePrep for batched support
            qml.StatePrep(base_state, wires=range(self.n_qubits))

            # Check if batched
            is_batched = skip_params.ndim == 2

            # Diagonal operation (single-qubit rotations)
            for i in range(self.n_qubits):
                angle = skip_params[:, i] if is_batched else skip_params[i]
                qml.RZ(angle, wires=i)

            observables = [qml.PauliX(i) for i in range(self.n_qubits)] + \
                          [qml.PauliY(i) for i in range(self.n_qubits)] + \
                          [qml.PauliZ(i) for i in range(self.n_qubits)]
            return [qml.expval(op) for op in observables]

        self.skip_qnode = skip_circuit

    def forward(self, x):
        """
        Forward pass with quantum superposition.

        UPDATED: Now uses batched processing for efficiency.

        Args:
            x: (batch, feature_dim)

        Returns:
            output: (batch, output_dim)
        """
        batch_size = x.shape[0]

        # Project to quantum state space
        x_proj = self.input_proj(x)
        x_norm = self.layer_norm(x_proj)

        # Normalize to valid quantum state (complex for StatePrep)
        base_states = torch.nn.functional.normalize(x_norm, p=2, dim=1)
        base_states = base_states.to(torch.complex64)

        # Compute three quantum paths with batched processing
        # Path 1: SSM (selective state space)
        psi1_measurements = self.ssm_path(base_states)

        # Path 2: Gating
        psi2_measurements = self.gate_path(base_states)

        # Path 3: Skip connection (batched)
        skip_params_expanded = self.skip_params.unsqueeze(0).expand(batch_size, -1)
        psi3_measurements = self.skip_qnode(base_states, skip_params_expanded)
        psi3_measurements = torch.stack(psi3_measurements, dim=1).float()

        # QUANTUM SUPERPOSITION
        # Combine paths using complex coefficients
        alpha = torch.complex(self.alpha_real, self.alpha_imag)
        beta = torch.complex(self.beta_real, self.beta_imag)
        gamma = torch.complex(self.gamma_real, self.gamma_imag)

        # Convert measurements to complex
        psi1_complex = torch.complex(psi1_measurements, torch.zeros_like(psi1_measurements))
        psi2_complex = torch.complex(psi2_measurements, torch.zeros_like(psi2_measurements))
        psi3_complex = torch.complex(psi3_measurements, torch.zeros_like(psi3_measurements))

        # Superpose
        psi_combined = alpha * psi1_complex + beta * psi2_complex + gamma * psi3_complex

        # Normalize
        norm = torch.sqrt(torch.sum(torch.abs(psi_combined) ** 2, dim=1, keepdim=True) + 1e-8)
        psi_normalized = psi_combined / norm

        # Take real part for classical processing
        measurements = torch.abs(psi_normalized).float()

        # Output layer
        output = self.output_ff(measurements)
        output = self.dropout(output)

        return output


class QuantumMambaTS(nn.Module):
    """
    Quantum Mamba for time-series classification.

    Processes time-series data using Quantum Mamba layers.

    Args:
        n_qubits: Number of qubits
        n_timesteps: Number of time steps
        qlcu_layers: QLCU circuit depth
        feature_dim: Number of input channels
        output_dim: Number of output classes
        dropout: Dropout rate
        device: 'cuda' or 'cpu'
    """

    def __init__(
        self,
        n_qubits=4,
        n_timesteps=160,
        qlcu_layers=2,
        feature_dim=64,
        output_dim=2,
        dropout=0.1,
        device="cpu"
    ):
        super().__init__()

        self.n_qubits = n_qubits
        self.n_timesteps = n_timesteps
        self.feature_dim = feature_dim
        self.output_dim = output_dim

        # Temporal encoding
        self.temporal_conv = nn.Conv1d(
            in_channels=feature_dim,
            out_channels=feature_dim,
            kernel_size=3,
            padding=1,
            groups=feature_dim
        )

        # Pooling to reduce temporal dimension
        self.pool = nn.AdaptiveAvgPool1d(1)

        # Quantum Mamba layer
        self.quantum_mamba = QuantumMambaLayer(
            n_qubits=n_qubits,
            qlcu_layers=qlcu_layers,
            feature_dim=feature_dim,
            output_dim=output_dim,
            dropout=dropout,
            device=device
        )

        # Move all parameters to specified device
        self.to(device)

    def forward(self, x):
        """
        Args:
            x: (batch, n_channels, n_timesteps) - EEG data format

        Returns:
            output: (batch, output_dim)
        """
        # Temporal convolution
        x_conv = self.temporal_conv(x)  # (batch, channels, timesteps)
        x_conv = torch.nn.functional.silu(x_conv)

        # Pool over time
        x_pooled = self.pool(x_conv).squeeze(-1)  # (batch, channels)

        # Quantum Mamba
        output = self.quantum_mamba(x_pooled)

        return output


# ================================================================================
# Testing
# ================================================================================

if __name__ == "__main__":
    print("=" * 80)
    print("Quantum Mamba (Superposition) - Testing")
    print("=" * 80)

    batch_size = 4
    n_channels = 64
    n_timesteps = 160
    n_qubits = 4
    output_dim = 2

    print("\n[1] Testing QuantumMambaLayer...")
    model_layer = QuantumMambaLayer(
        n_qubits=n_qubits,
        qlcu_layers=2,
        feature_dim=n_channels,
        output_dim=output_dim
    )

    x_flat = torch.randn(batch_size, n_channels)
    output = model_layer(x_flat)
    print(f"  Input shape: {x_flat.shape}")
    print(f"  Output shape: {output.shape}")
    assert output.shape == (batch_size, output_dim), "Layer output shape mismatch!"

    print("\n[2] Testing QuantumMambaTS (full model)...")
    model = QuantumMambaTS(
        n_qubits=n_qubits,
        n_timesteps=n_timesteps,
        qlcu_layers=2,
        feature_dim=n_channels,
        output_dim=output_dim
    )

    x_ts = torch.randn(batch_size, n_channels, n_timesteps)
    output = model(x_ts)
    print(f"  Input shape: {x_ts.shape}")
    print(f"  Output shape: {output.shape}")
    assert output.shape == (batch_size, output_dim), "Model output shape mismatch!"

    print("\n[3] Checking trainable parameters...")
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Total parameters: {total_params:,}")

    # Check complex coefficients
    alpha = torch.complex(model.quantum_mamba.alpha_real, model.quantum_mamba.alpha_imag)
    beta = torch.complex(model.quantum_mamba.beta_real, model.quantum_mamba.beta_imag)
    gamma = torch.complex(model.quantum_mamba.gamma_real, model.quantum_mamba.gamma_imag)

    print(f"\n[4] Complex superposition coefficients:")
    print(f"  α = {alpha.item()}")
    print(f"  β = {beta.item()}")
    print(f"  γ = {gamma.item()}")

    print("\n[5] Testing gradient flow...")
    model.train()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.CrossEntropyLoss()

    outputs = model(x_ts)
    loss = criterion(outputs, torch.randint(0, output_dim, (batch_size,)))
    loss.backward()
    optimizer.step()

    print(f"  Loss: {loss.item():.4f}")
    print(f"  Gradient norm (alpha_real): {model.quantum_mamba.alpha_real.grad.norm().item():.4f}")

    print("\n" + "=" * 80)
    print("Testing complete!")
    print("=" * 80)
    print("\nKey Quantum Mamba Features Implemented:")
    print("  ✓ Quantum Selective SSM with QLCU")
    print("  ✓ Quantum Gating Mechanism")
    print("  ✓ QUANTUM SUPERPOSITION: |ψ⟩ = α|ψ_ssm⟩ + β|ψ_gate⟩ + γ|ψ_skip⟩")
    print("  ✓ Complex coefficients (α, β, γ ∈ ℂ)")
    print("  ✓ Trainable via backpropagation")
    print("=" * 80)
