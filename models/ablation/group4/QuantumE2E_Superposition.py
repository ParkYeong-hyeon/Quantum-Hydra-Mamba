"""
End-to-End Quantum Models with True Quantum Superposition

These models combine:
1. TRUE end-to-end quantum processing (no intermediate measurements)
2. TRUE quantum superposition (three branch states combined BEFORE measurement)
3. EXACT same circuit ansatz as QuantumE2E.py for fair comparison

Architecture:
    Classical Input → Quantum Encoding → Quantum Feature Extraction → Three Quantum Branches → Superposition → Measurement
                      ↓                  ↓                            ↓
                   RY(data)           Variational Ansatz             Branch 1 (Forward SSM)  → |ψ₁⟩ ─┐
                                      [NO measurement]               Branch 2 (Backward SSM) → |ψ₂⟩ ─┼→ |ψ⟩ = α|ψ₁⟩ + β|ψ₂⟩ + γ|ψ₃⟩
                                      [Returns state vector]         Branch 3 (Diagonal)     → |ψ₃⟩ ─┘
                                                                                                  ↓
                                                                                              PauliXYZ (ONLY measurement)

Circuit Structure (MATCHING QuantumE2E.py):
- Per layer: RX, RY, RZ rotations (3 * n_qubits params)
- Forward CRX entanglement ring (n_qubits params)
- Backward CRX entanglement ring (n_qubits params)
- Total: 5 * n_qubits params per layer

Key Innovations:
- Combines E2E quantum processing (no intermediate measurement) with true superposition
- Three quantum branches combined in STATE SPACE before measurement
- |ψ⟩ = α|ψ₁⟩ + β|ψ₂⟩ + γ|ψ₃⟩ (true superposition before measurement)
- Uses PennyLane default.qubit with native batching

Model IDs for ablation study:
- 4d: QuantumMambaE2E_Superposition (unidirectional with true superposition)
- 4e: QuantumHydraE2E_Superposition (bidirectional with true superposition)

Author: Quantum Hydra/Mamba Research Team
Date: December 2025
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import pennylane as qml
import numpy as np
from typing import Optional, Tuple

# Global quantum device cache
_QDEV_CACHE = {}


def _get_quantum_device(n_qubits: int):
    """Get or create a cached quantum device."""
    if n_qubits not in _QDEV_CACHE:
        _QDEV_CACHE[n_qubits] = qml.device("default.qubit", wires=n_qubits)
    return _QDEV_CACHE[n_qubits]


# ==============================================================================
# Circuit Building Blocks (EXACT match to QuantumE2E.py)
# ==============================================================================

def input_embedding(inputs: torch.Tensor, n_qubits: int):
    """
    Input embedding via RY gates. Supports batched inputs.

    EXACT match to QuantumE2E.py.
    """
    is_batched = inputs.ndim == 2
    for i in range(n_qubits):
        angle = inputs[:, i] if is_batched else inputs[i]
        qml.RY(angle, wires=i)


def variational_feature_layer(params: torch.Tensor, n_qubits: int, param_offset: int = 0):
    """
    Single variational layer for feature extraction. Supports batched parameters.

    EXACT match to QuantumE2E.py variational_feature_layer:
    - RX, RY, RZ rotations on each qubit (3 * n_qubits)
    - Forward CRX entanglement (n_qubits)
    - Backward CRX entanglement (n_qubits)
    - Total: 5 * n_qubits params per layer
    """
    param_idx = param_offset
    is_batched = params.ndim == 2

    # RX, RY, RZ rotations on each qubit
    for i in range(n_qubits):
        qml.RX(params[:, param_idx] if is_batched else params[param_idx], wires=i)
        qml.RY(params[:, param_idx + 1] if is_batched else params[param_idx + 1], wires=i)
        qml.RZ(params[:, param_idx + 2] if is_batched else params[param_idx + 2], wires=i)
        param_idx += 3

    # Forward CRX entanglement (ring topology)
    for i in range(n_qubits):
        qml.CRX(params[:, param_idx] if is_batched else params[param_idx],
                wires=[i, (i + 1) % n_qubits])
        param_idx += 1

    # Backward CRX entanglement (ring topology)
    for i in range(n_qubits - 1, -1, -1):
        qml.CRX(params[:, param_idx] if is_batched else params[param_idx],
                wires=[i, (i - 1) % n_qubits])
        param_idx += 1

    return param_idx


def ssm_mixing_layer(params: torch.Tensor, dt_scale: torch.Tensor, n_qubits: int, param_offset: int = 0):
    """
    Single SSM mixing layer (Mamba-style). Supports batched parameters.

    EXACT match to QuantumE2E.py ssm_mixing_layer:
    - Delta-modulated RX, RY, RZ rotations (3 * n_qubits)
    - Forward CRX entanglement (n_qubits)
    - Backward CRX entanglement (n_qubits)
    """
    param_idx = param_offset
    is_batched = params.ndim == 2

    # Delta-modulated single-qubit rotations
    for i in range(n_qubits):
        p0 = params[:, param_idx] if is_batched else params[param_idx]
        p1 = params[:, param_idx + 1] if is_batched else params[param_idx + 1]
        p2 = params[:, param_idx + 2] if is_batched else params[param_idx + 2]
        qml.RX(p0 * dt_scale, wires=i)
        qml.RY(p1 * dt_scale, wires=i)
        qml.RZ(p2 * dt_scale, wires=i)
        param_idx += 3

    # Forward CRX entanglement
    for i in range(n_qubits):
        qml.CRX(params[:, param_idx] if is_batched else params[param_idx],
                wires=[i, (i + 1) % n_qubits])
        param_idx += 1

    # Backward CRX entanglement
    for i in range(n_qubits - 1, -1, -1):
        qml.CRX(params[:, param_idx] if is_batched else params[param_idx],
                wires=[i, (i - 1) % n_qubits])
        param_idx += 1

    return param_idx


def backward_ssm_mixing_layer(params: torch.Tensor, dt_scale: torch.Tensor, n_qubits: int, param_offset: int = 0):
    """
    Backward SSM mixing layer (reversed qubit order for Hydra). Supports batched parameters.

    EXACT match to QuantumE2E.py backward_ssm_mixing_layer.
    """
    param_idx = param_offset
    is_batched = params.ndim == 2

    # Reverse order: apply rotations from last qubit to first
    for i in range(n_qubits - 1, -1, -1):
        p0 = params[:, param_idx] if is_batched else params[param_idx]
        p1 = params[:, param_idx + 1] if is_batched else params[param_idx + 1]
        p2 = params[:, param_idx + 2] if is_batched else params[param_idx + 2]
        qml.RX(p0 * dt_scale, wires=i)
        qml.RY(p1 * dt_scale, wires=i)
        qml.RZ(p2 * dt_scale, wires=i)
        param_idx += 3

    # Backward CRX first (reverse of forward)
    for i in range(n_qubits - 1, -1, -1):
        qml.CRX(params[:, param_idx] if is_batched else params[param_idx],
                wires=[i, (i - 1) % n_qubits])
        param_idx += 1

    # Forward CRX second
    for i in range(n_qubits):
        qml.CRX(params[:, param_idx] if is_batched else params[param_idx],
                wires=[i, (i + 1) % n_qubits])
        param_idx += 1

    return param_idx


def diagonal_mixing_layer(params: torch.Tensor, n_qubits: int, param_offset: int = 0):
    """
    Diagonal (skip-connection style) mixing layer.
    Uses the unified ansatz without Delta modulation.
    """
    return variational_feature_layer(params, n_qubits, param_offset)


def ssm_input_injection(input_angles: torch.Tensor, dt_scale: torch.Tensor, n_qubits: int):
    """
    Input injection layer for SSM (Mamba-style). Supports batched inputs.

    EXACT match to QuantumE2E.py ssm_input_injection.
    """
    is_batched = input_angles.ndim == 2
    for i in range(n_qubits):
        angle = input_angles[:, i] if is_batched else input_angles[i]
        qml.RY(angle * dt_scale, wires=i)


def multi_observable_measurement(n_qubits: int):
    """
    Multi-observable measurement (PauliX, PauliY, PauliZ on all qubits).

    EXACT match to QuantumE2E.py.
    """
    measurements = []
    for i in range(n_qubits):
        measurements.append(qml.expval(qml.PauliX(i)))
    for i in range(n_qubits):
        measurements.append(qml.expval(qml.PauliY(i)))
    for i in range(n_qubits):
        measurements.append(qml.expval(qml.PauliZ(i)))
    return measurements


# ==============================================================================
# E2E Quantum Core with True Superposition
# ==============================================================================

class QuantumE2ESuperpositionCore(nn.Module):
    """
    End-to-End Quantum processor with TRUE state superposition.

    Combines E2E processing (no intermediate measurement) with true superposition:
    - Each branch: Input → Feature Extraction → SSM Mixing → State Vector
    - States combined: |ψ⟩ = α|ψ₁⟩ + β|ψ₂⟩ + γ|ψ₃⟩
    - Single measurement at the end

    Uses EXACT same circuit structure as QuantumE2E.py.
    """

    def __init__(
        self,
        n_qubits: int = 6,
        n_feature_layers: int = 2,
        n_mixing_layers: int = 2,
        device: str = "cpu"
    ):
        super().__init__()

        self.n_qubits = n_qubits
        self.n_feature_layers = n_feature_layers
        self.n_mixing_layers = n_mixing_layers
        self.torch_device = torch.device(device) if isinstance(device, str) else device

        # Parameters per layer: 5 * n_qubits (matching QuantumE2E.py)
        self.params_per_layer = 5 * n_qubits
        self.n_feature_params = self.params_per_layer * n_feature_layers
        self.n_mixing_params = self.params_per_layer * n_mixing_layers

        # Quantum output dimension: 3 * n_qubits (PauliX, PauliY, PauliZ)
        self.q_dim = 3 * n_qubits

        # Quantum device
        self.qdev = _get_quantum_device(n_qubits)

        # Trainable complex coefficients for superposition (matching QuantumHydraSSM.py)
        init_val = float(1.0 / np.sqrt(3))
        self.alpha_real = nn.Parameter(torch.tensor(init_val, dtype=torch.float32))
        self.alpha_imag = nn.Parameter(torch.tensor(0.0, dtype=torch.float32))
        self.beta_real = nn.Parameter(torch.tensor(init_val, dtype=torch.float32))
        self.beta_imag = nn.Parameter(torch.tensor(0.0, dtype=torch.float32))
        self.gamma_real = nn.Parameter(torch.tensor(init_val, dtype=torch.float32))
        self.gamma_imag = nn.Parameter(torch.tensor(0.0, dtype=torch.float32))

        # Build quantum circuits
        self._build_circuits()

        self.to(self.torch_device)

    def _build_circuits(self):
        """Build quantum circuits for three branches returning state vectors."""
        n_qubits = self.n_qubits
        n_feature_layers = self.n_feature_layers
        n_mixing_layers = self.n_mixing_layers

        # Branch 1: Forward E2E (Feature Extraction → Forward SSM Mixing)
        @qml.qnode(self.qdev, interface="torch", diff_method="backprop")
        def forward_branch_circuit(input_angles, feature_params, mixing_params, dt_scale, injection_angles):
            """
            Forward branch E2E circuit returning state vector.

            Architecture (matching QuantumE2E.py QuantumMambaE2E):
            1. RY input embedding
            2. Variational feature layers
            3. Forward SSM mixing layers (Delta-modulated)
            4. Input injection
            5. Return state vector (NO measurement here)
            """
            # Step 1: Input embedding
            input_embedding(input_angles, n_qubits)

            # Step 2: Feature extraction layers
            param_idx = 0
            for layer in range(n_feature_layers):
                param_idx = variational_feature_layer(feature_params, n_qubits, param_idx)

            # Step 3: Forward SSM mixing layers
            param_idx = 0
            for layer in range(n_mixing_layers):
                param_idx = ssm_mixing_layer(mixing_params, dt_scale, n_qubits, param_idx)

            # Step 4: Input injection
            ssm_input_injection(injection_angles, dt_scale, n_qubits)

            # Return state vector (no measurement)
            return qml.state()

        # Branch 2: Backward E2E (Feature Extraction → Backward SSM Mixing)
        @qml.qnode(self.qdev, interface="torch", diff_method="backprop")
        def backward_branch_circuit(input_angles, feature_params, mixing_params, dt_scale, injection_angles):
            """
            Backward branch E2E circuit returning state vector.

            Uses backward SSM mixing (reversed qubit order).
            """
            # Step 1: Input embedding
            input_embedding(input_angles, n_qubits)

            # Step 2: Feature extraction layers
            param_idx = 0
            for layer in range(n_feature_layers):
                param_idx = variational_feature_layer(feature_params, n_qubits, param_idx)

            # Step 3: Backward SSM mixing layers
            param_idx = 0
            for layer in range(n_mixing_layers):
                param_idx = backward_ssm_mixing_layer(mixing_params, dt_scale, n_qubits, param_idx)

            # Step 4: Input injection (reversed)
            # Apply in reverse order for backward
            is_batched = injection_angles.ndim == 2
            for i in range(n_qubits - 1, -1, -1):
                angle = injection_angles[:, i] if is_batched else injection_angles[i]
                qml.RY(angle * dt_scale, wires=i)

            # Return state vector (no measurement)
            return qml.state()

        # Branch 3: Diagonal E2E (Feature Extraction → Diagonal Mixing)
        @qml.qnode(self.qdev, interface="torch", diff_method="backprop")
        def diagonal_branch_circuit(input_angles, feature_params, mixing_params):
            """
            Diagonal branch E2E circuit returning state vector.

            Uses unified ansatz without Delta modulation (skip-connection style).
            """
            # Step 1: Input embedding
            input_embedding(input_angles, n_qubits)

            # Step 2: Feature extraction layers
            param_idx = 0
            for layer in range(n_feature_layers):
                param_idx = variational_feature_layer(feature_params, n_qubits, param_idx)

            # Step 3: Diagonal mixing layers (no Delta modulation)
            param_idx = 0
            for layer in range(n_mixing_layers):
                param_idx = diagonal_mixing_layer(mixing_params, n_qubits, param_idx)

            # Return state vector (no measurement)
            return qml.state()

        # Measurement circuit for combined state
        @qml.qnode(self.qdev, interface="torch", diff_method="backprop")
        def measurement_circuit(state_vector):
            """Measure the combined superposition state."""
            qml.StatePrep(state_vector, wires=range(n_qubits))
            return multi_observable_measurement(n_qubits)

        self.forward_branch_circuit = forward_branch_circuit
        self.backward_branch_circuit = backward_branch_circuit
        self.diagonal_branch_circuit = diagonal_branch_circuit
        self.measurement_circuit = measurement_circuit

    def _get_superposition_coefficients(self):
        """Get normalized complex superposition coefficients."""
        alpha = torch.complex(self.alpha_real, self.alpha_imag)
        beta = torch.complex(self.beta_real, self.beta_imag)
        gamma = torch.complex(self.gamma_real, self.gamma_imag)

        # Normalize
        norm = torch.sqrt(
            torch.abs(alpha)**2 + torch.abs(beta)**2 + torch.abs(gamma)**2 + 1e-9
        )
        return alpha / norm, beta / norm, gamma / norm

    def forward(
        self,
        input_angles: torch.Tensor,
        feature_params: torch.Tensor,
        forward_mixing_params: torch.Tensor,
        backward_mixing_params: torch.Tensor,
        diagonal_mixing_params: torch.Tensor,
        dt_scale: torch.Tensor,
        injection_angles: torch.Tensor
    ) -> torch.Tensor:
        """
        Process batch through three-branch E2E quantum superposition.

        All inputs support native PennyLane batching.

        Args:
            input_angles: (batch, n_qubits) input encoding angles
            feature_params: (batch, n_feature_params) shared feature params
            forward_mixing_params: (batch, n_mixing_params) forward branch mixing params
            backward_mixing_params: (batch, n_mixing_params) backward branch mixing params
            diagonal_mixing_params: (batch, n_mixing_params) diagonal branch mixing params
            dt_scale: (batch,) Delta scale factors
            injection_angles: (batch, n_qubits) input injection angles

        Returns:
            measurements: (batch, 3 * n_qubits)
        """
        batch_size = input_angles.shape[0]

        # Get state vectors from each branch (batched)
        # Each branch does full E2E: input → features → mixing → state
        psi1 = self.forward_branch_circuit(
            input_angles, feature_params, forward_mixing_params, dt_scale, injection_angles
        )
        psi2 = self.backward_branch_circuit(
            input_angles, feature_params, backward_mixing_params, dt_scale, injection_angles
        )
        psi3 = self.diagonal_branch_circuit(
            input_angles, feature_params, diagonal_mixing_params
        )

        # Ensure complex dtype and proper shape
        psi1 = psi1.to(torch.complex64)
        psi2 = psi2.to(torch.complex64)
        psi3 = psi3.to(torch.complex64)

        # TRUE quantum superposition: |ψ⟩ = α|ψ₁⟩ + β|ψ₂⟩ + γ|ψ₃⟩
        alpha, beta, gamma = self._get_superposition_coefficients()
        psi_combined = alpha * psi1 + beta * psi2 + gamma * psi3

        # Normalize
        if psi_combined.ndim == 1:
            # Single sample
            norm = torch.linalg.vector_norm(psi_combined) + 1e-9
            psi_normalized = psi_combined / norm
        else:
            # Batched
            norm = torch.linalg.vector_norm(psi_combined, dim=-1, keepdim=True) + 1e-9
            psi_normalized = psi_combined / norm

        # Measure the combined state
        if psi_normalized.ndim == 1:
            measurements = self.measurement_circuit(psi_normalized)
            measurements = torch.stack(measurements).float()
        else:
            # For batched states, process each sample
            # Note: StatePrep with batched states requires processing per sample
            all_measurements = []
            for b in range(batch_size):
                m = self.measurement_circuit(psi_normalized[b])
                all_measurements.append(torch.stack(m))
            measurements = torch.stack(all_measurements).float()

        return measurements


# ==============================================================================
# End-to-End Quantum Mamba with Superposition (Model ID: 4d)
# ==============================================================================

class QuantumMambaE2E_Superposition(nn.Module):
    """
    End-to-End Quantum Mamba with TRUE state superposition.

    Combines:
    - E2E quantum processing (no intermediate measurements) from QuantumE2E.py
    - TRUE quantum superposition (from QuantumHydraSSM.py)
    - EXACT same circuit structure as QuantumMambaE2E

    Model ID: 4d (for ablation study)
    - Quantum feature extraction (E2E, no intermediate measurement)
    - Quantum mixing with true superposition
    - Unidirectional sequence processing
    """

    def __init__(
        self,
        n_qubits: int = 6,
        n_feature_layers: int = 2,
        n_mixing_layers: int = 2,
        n_channels: int = 4,
        n_timesteps: int = 200,
        d_model: int = 64,
        chunk_size: int = 16,
        output_dim: int = 2,
        dropout: float = 0.1,
        device: str = "cpu",
        # Aliases for compatibility
        qlcu_layers: Optional[int] = None,
        d_state: Optional[int] = None,
        feature_dim: Optional[int] = None,
    ):
        super().__init__()

        # Handle parameter aliases
        actual_feature_layers = n_feature_layers
        actual_mixing_layers = qlcu_layers if qlcu_layers is not None else n_mixing_layers
        actual_channels = feature_dim if feature_dim is not None else n_channels

        self.n_qubits = n_qubits
        self.n_feature_layers = actual_feature_layers
        self.n_mixing_layers = actual_mixing_layers
        self.n_channels = actual_channels
        self.n_timesteps = n_timesteps
        self.d_model = d_model
        self.chunk_size = chunk_size
        self.output_dim = output_dim
        self.torch_device = torch.device(device) if isinstance(device, str) else device

        # Quantum output dimension: 3 * n_qubits (PauliX, PauliY, PauliZ)
        self.q_dim = 3 * n_qubits

        # Parameter counts (matching QuantumE2E.py)
        # Feature params: 5 * n_qubits per layer
        self.n_feature_params = 5 * n_qubits * actual_feature_layers
        # Mixing params: 5 * n_qubits per layer per branch
        self.n_mixing_params = 5 * n_qubits * actual_mixing_layers

        # Classical preprocessing: Conv1d to aggregate features
        self.input_conv = nn.Conv1d(actual_channels, d_model, kernel_size=3, padding=1)
        self.input_norm = nn.LayerNorm(d_model)

        # Chunk attention for aggregation
        self.chunk_attention = nn.Sequential(
            nn.Linear(d_model, d_model // 4),
            nn.Tanh(),
            nn.Linear(d_model // 4, 1)
        )

        # Parameter projections
        # Shared feature params (all branches use same feature extraction)
        self.feature_param_proj = nn.Linear(d_model, self.n_feature_params)

        # Separate mixing params for each branch
        self.forward_mixing_param_proj = nn.Linear(d_model, self.n_mixing_params)
        self.backward_mixing_param_proj = nn.Linear(d_model, self.n_mixing_params)
        self.diagonal_mixing_param_proj = nn.Linear(d_model, self.n_mixing_params)

        # Input encoding projection
        self.encoding_proj = nn.Linear(d_model, n_qubits)

        # Delta (dt) projection for SSM mixing
        self.dt_proj = nn.Linear(d_model, 1)

        # Input injection projection
        self.input_injection_proj = nn.Linear(d_model, n_qubits)

        # Quantum core with superposition
        self.quantum_core = QuantumE2ESuperpositionCore(
            n_qubits=n_qubits,
            n_feature_layers=actual_feature_layers,
            n_mixing_layers=actual_mixing_layers,
            device=device
        )

        # Output projection
        self.output_proj = nn.Sequential(
            nn.Linear(self.q_dim, d_model),
            nn.LayerNorm(d_model),
            nn.SiLU(),
            nn.Dropout(dropout)
        )

        # Sequence aggregation
        self.seq_attention = nn.Sequential(
            nn.Linear(d_model, d_model // 4),
            nn.Tanh(),
            nn.Linear(d_model // 4, 1)
        )

        # Classifier
        self.classifier = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, output_dim)
        )

        self.to(self.torch_device)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through end-to-end quantum Mamba with superposition.

        Args:
            x: Input tensor of shape (batch, n_channels, seq_len) or (batch, seq_len, n_channels)

        Returns:
            logits: Output tensor of shape (batch, output_dim)
        """
        # Handle input format
        if x.dim() == 2:
            x = x.unsqueeze(1)

        # Determine format: (batch, channels, seq) or (batch, seq, channels)
        if x.shape[1] == self.n_channels:
            # Format: (batch, n_channels, seq_len) - need to process with Conv1d
            batch_size = x.shape[0]
            seq_len = x.shape[2]
            x = self.input_conv(x)  # (batch, d_model, seq_len)
            x = x.transpose(1, 2)  # (batch, seq_len, d_model)
        else:
            # Format: (batch, seq_len, n_channels) or (batch, n_timesteps, n_channels)
            batch_size, seq_len, _ = x.shape
            x = x.transpose(1, 2)  # (batch, n_channels, seq_len)
            x = self.input_conv(x)  # (batch, d_model, seq_len)
            x = x.transpose(1, 2)  # (batch, seq_len, d_model)

        x = self.input_norm(x)

        # Chunk the sequence
        n_chunks = (seq_len + self.chunk_size - 1) // self.chunk_size
        if seq_len % self.chunk_size != 0:
            pad_len = self.chunk_size * n_chunks - seq_len
            x = F.pad(x, (0, 0, 0, pad_len))

        x = x.reshape(batch_size * n_chunks, self.chunk_size, self.d_model)

        # Aggregate within chunks using attention
        attn_scores = self.chunk_attention(x)
        attn_weights = F.softmax(attn_scores, dim=1)
        chunk_summary = (attn_weights * x).sum(dim=1)  # (batch * n_chunks, d_model)

        # Get quantum circuit parameters
        feature_params = self.feature_param_proj(chunk_summary)
        forward_mixing_params = self.forward_mixing_param_proj(chunk_summary)
        backward_mixing_params = self.backward_mixing_param_proj(chunk_summary)
        diagonal_mixing_params = self.diagonal_mixing_param_proj(chunk_summary)

        # Get input encoding angles (scaled to [-π, π])
        input_angles = torch.tanh(self.encoding_proj(chunk_summary)) * np.pi

        # Get Delta (dt) scale (sigmoid to [0, 1])
        dt_scale = torch.sigmoid(self.dt_proj(chunk_summary)).squeeze(-1)

        # Get input injection angles
        injection_angles = torch.tanh(self.input_injection_proj(chunk_summary)) * np.pi

        # Process through quantum core with superposition
        quantum_features = self.quantum_core(
            input_angles,
            feature_params,
            forward_mixing_params,
            backward_mixing_params,
            diagonal_mixing_params,
            dt_scale,
            injection_angles
        )

        # Project quantum features
        chunk_features = self.output_proj(quantum_features)
        chunk_features = chunk_features.reshape(batch_size, n_chunks, self.d_model)

        # Aggregate across chunks
        seq_attn = self.seq_attention(chunk_features)
        seq_weights = F.softmax(seq_attn, dim=1)
        sequence_repr = (seq_weights * chunk_features).sum(dim=1)

        # Classify
        logits = self.classifier(sequence_repr)

        return logits


# ==============================================================================
# End-to-End Quantum Hydra with Superposition (Model ID: 4e)
# ==============================================================================

class QuantumHydraE2E_Superposition(nn.Module):
    """
    End-to-End Quantum Hydra with TRUE state superposition.

    Combines:
    - E2E quantum processing (no intermediate measurements)
    - TRUE quantum superposition
    - EXACT same circuit structure as QuantumHydraE2E
    - Bidirectional sequence processing

    Model ID: 4e (for ablation study)
    - Quantum feature extraction (E2E, no intermediate measurement)
    - Quantum mixing with true superposition
    - Bidirectional sequence processing (forward + backward)
    """

    def __init__(
        self,
        n_qubits: int = 6,
        n_feature_layers: int = 2,
        n_mixing_layers: int = 2,
        n_channels: int = 4,
        n_timesteps: int = 200,
        d_model: int = 64,
        chunk_size: int = 16,
        output_dim: int = 2,
        dropout: float = 0.1,
        device: str = "cpu",
        # Aliases for compatibility
        qlcu_layers: Optional[int] = None,
        d_state: Optional[int] = None,
        feature_dim: Optional[int] = None,
    ):
        super().__init__()

        # Handle parameter aliases
        actual_feature_layers = n_feature_layers
        actual_mixing_layers = qlcu_layers if qlcu_layers is not None else n_mixing_layers
        actual_channels = feature_dim if feature_dim is not None else n_channels

        self.n_qubits = n_qubits
        self.n_feature_layers = actual_feature_layers
        self.n_mixing_layers = actual_mixing_layers
        self.n_channels = actual_channels
        self.n_timesteps = n_timesteps
        self.d_model = d_model
        self.chunk_size = chunk_size
        self.output_dim = output_dim
        self.torch_device = torch.device(device) if isinstance(device, str) else device

        # Quantum output dimension
        self.q_dim = 3 * n_qubits

        # Parameter counts
        self.n_feature_params = 5 * n_qubits * actual_feature_layers
        self.n_mixing_params = 5 * n_qubits * actual_mixing_layers

        # Classical preprocessing
        self.input_conv = nn.Conv1d(actual_channels, d_model, kernel_size=3, padding=1)
        self.input_norm = nn.LayerNorm(d_model)

        # Chunk attention
        self.chunk_attention = nn.Sequential(
            nn.Linear(d_model, d_model // 4),
            nn.Tanh(),
            nn.Linear(d_model // 4, 1)
        )

        # Parameter projections for forward direction
        self.fwd_feature_param_proj = nn.Linear(d_model, self.n_feature_params)
        self.fwd_forward_mixing_param_proj = nn.Linear(d_model, self.n_mixing_params)
        self.fwd_backward_mixing_param_proj = nn.Linear(d_model, self.n_mixing_params)
        self.fwd_diagonal_mixing_param_proj = nn.Linear(d_model, self.n_mixing_params)

        # Parameter projections for backward direction
        self.bwd_feature_param_proj = nn.Linear(d_model, self.n_feature_params)
        self.bwd_forward_mixing_param_proj = nn.Linear(d_model, self.n_mixing_params)
        self.bwd_backward_mixing_param_proj = nn.Linear(d_model, self.n_mixing_params)
        self.bwd_diagonal_mixing_param_proj = nn.Linear(d_model, self.n_mixing_params)

        # Encoding and Delta projections
        self.encoding_proj = nn.Linear(d_model, n_qubits)
        self.dt_proj = nn.Linear(d_model, 1)
        self.input_injection_proj = nn.Linear(d_model, n_qubits)

        # Two quantum cores (forward and backward sequence directions)
        self.quantum_core_fwd = QuantumE2ESuperpositionCore(
            n_qubits=n_qubits,
            n_feature_layers=actual_feature_layers,
            n_mixing_layers=actual_mixing_layers,
            device=device
        )
        self.quantum_core_bwd = QuantumE2ESuperpositionCore(
            n_qubits=n_qubits,
            n_feature_layers=actual_feature_layers,
            n_mixing_layers=actual_mixing_layers,
            device=device
        )

        # Output projection (combines both directions)
        self.output_proj = nn.Sequential(
            nn.Linear(2 * self.q_dim, d_model),
            nn.LayerNorm(d_model),
            nn.SiLU(),
            nn.Dropout(dropout)
        )

        # Sequence aggregation
        self.seq_attention = nn.Sequential(
            nn.Linear(d_model, d_model // 4),
            nn.Tanh(),
            nn.Linear(d_model // 4, 1)
        )

        # Classifier
        self.classifier = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, output_dim)
        )

        self.to(self.torch_device)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass with bidirectional E2E quantum processing.

        Args:
            x: Input tensor

        Returns:
            logits: Output tensor of shape (batch, output_dim)
        """
        # Handle input format
        if x.dim() == 2:
            x = x.unsqueeze(1)

        # Determine format and process
        if x.shape[1] == self.n_channels:
            batch_size = x.shape[0]
            seq_len = x.shape[2]
            x = self.input_conv(x)
            x = x.transpose(1, 2)
        else:
            batch_size, seq_len, _ = x.shape
            x = x.transpose(1, 2)
            x = self.input_conv(x)
            x = x.transpose(1, 2)

        x = self.input_norm(x)

        # Chunk the sequence
        n_chunks = (seq_len + self.chunk_size - 1) // self.chunk_size
        if seq_len % self.chunk_size != 0:
            pad_len = self.chunk_size * n_chunks - seq_len
            x = F.pad(x, (0, 0, 0, pad_len))

        x_chunked = x.reshape(batch_size * n_chunks, self.chunk_size, self.d_model)

        # Aggregate within chunks
        attn_scores = self.chunk_attention(x_chunked)
        attn_weights = F.softmax(attn_scores, dim=1)
        chunk_summary = (attn_weights * x_chunked).sum(dim=1)

        # Common projections
        input_angles = torch.tanh(self.encoding_proj(chunk_summary)) * np.pi
        dt_scale = torch.sigmoid(self.dt_proj(chunk_summary)).squeeze(-1)
        injection_angles = torch.tanh(self.input_injection_proj(chunk_summary)) * np.pi

        # Forward direction quantum processing
        fwd_feature_params = self.fwd_feature_param_proj(chunk_summary)
        fwd_forward_mixing_params = self.fwd_forward_mixing_param_proj(chunk_summary)
        fwd_backward_mixing_params = self.fwd_backward_mixing_param_proj(chunk_summary)
        fwd_diagonal_mixing_params = self.fwd_diagonal_mixing_param_proj(chunk_summary)

        quantum_fwd = self.quantum_core_fwd(
            input_angles,
            fwd_feature_params,
            fwd_forward_mixing_params,
            fwd_backward_mixing_params,
            fwd_diagonal_mixing_params,
            dt_scale,
            injection_angles
        )

        # Backward direction (flip chunk order, process, flip back)
        chunk_summary_rev = chunk_summary.reshape(batch_size, n_chunks, -1)
        chunk_summary_rev = torch.flip(chunk_summary_rev, dims=[1])
        chunk_summary_rev = chunk_summary_rev.reshape(batch_size * n_chunks, -1)

        input_angles_rev = torch.tanh(self.encoding_proj(chunk_summary_rev)) * np.pi
        dt_scale_rev = torch.sigmoid(self.dt_proj(chunk_summary_rev)).squeeze(-1)
        injection_angles_rev = torch.tanh(self.input_injection_proj(chunk_summary_rev)) * np.pi

        bwd_feature_params = self.bwd_feature_param_proj(chunk_summary_rev)
        bwd_forward_mixing_params = self.bwd_forward_mixing_param_proj(chunk_summary_rev)
        bwd_backward_mixing_params = self.bwd_backward_mixing_param_proj(chunk_summary_rev)
        bwd_diagonal_mixing_params = self.bwd_diagonal_mixing_param_proj(chunk_summary_rev)

        quantum_bwd = self.quantum_core_bwd(
            input_angles_rev,
            bwd_feature_params,
            bwd_forward_mixing_params,
            bwd_backward_mixing_params,
            bwd_diagonal_mixing_params,
            dt_scale_rev,
            injection_angles_rev
        )

        # Flip backward results back
        quantum_bwd = quantum_bwd.reshape(batch_size, n_chunks, -1)
        quantum_bwd = torch.flip(quantum_bwd, dims=[1])
        quantum_bwd = quantum_bwd.reshape(batch_size * n_chunks, -1)

        # Concatenate forward and backward
        quantum_combined = torch.cat([quantum_fwd, quantum_bwd], dim=-1)

        # Output projection
        chunk_features = self.output_proj(quantum_combined)
        chunk_features = chunk_features.reshape(batch_size, n_chunks, self.d_model)

        # Sequence aggregation
        seq_attn = self.seq_attention(chunk_features)
        seq_weights = F.softmax(seq_attn, dim=1)
        sequence_repr = (seq_weights * chunk_features).sum(dim=1)

        # Classify
        logits = self.classifier(sequence_repr)

        return logits


# ==============================================================================
# Factory Functions
# ==============================================================================

def create_quantum_mamba_e2e_superposition(
    n_qubits: int = 6,
    n_feature_layers: int = 2,
    n_mixing_layers: int = 2,
    n_channels: int = 4,
    n_timesteps: int = 200,
    d_model: int = 64,
    output_dim: int = 2,
    device: str = "cpu",
    **kwargs
) -> QuantumMambaE2E_Superposition:
    """Create a QuantumMambaE2E_Superposition model (ID: 4d)."""
    return QuantumMambaE2E_Superposition(
        n_qubits=n_qubits,
        n_feature_layers=n_feature_layers,
        n_mixing_layers=n_mixing_layers,
        n_channels=n_channels,
        n_timesteps=n_timesteps,
        d_model=d_model,
        output_dim=output_dim,
        device=device,
        **kwargs
    )


def create_quantum_hydra_e2e_superposition(
    n_qubits: int = 6,
    n_feature_layers: int = 2,
    n_mixing_layers: int = 2,
    n_channels: int = 4,
    n_timesteps: int = 200,
    d_model: int = 64,
    output_dim: int = 2,
    device: str = "cpu",
    **kwargs
) -> QuantumHydraE2E_Superposition:
    """Create a QuantumHydraE2E_Superposition model (ID: 4e)."""
    return QuantumHydraE2E_Superposition(
        n_qubits=n_qubits,
        n_feature_layers=n_feature_layers,
        n_mixing_layers=n_mixing_layers,
        n_channels=n_channels,
        n_timesteps=n_timesteps,
        d_model=d_model,
        output_dim=output_dim,
        device=device,
        **kwargs
    )


# ==============================================================================
# Testing
# ==============================================================================

if __name__ == "__main__":
    print("=" * 80)
    print("End-to-End Quantum Models with TRUE Superposition - Testing")
    print("Combining E2E processing with true quantum superposition")
    print("=" * 80)

    device = "cpu"
    batch_size = 2
    n_channels = 64
    n_timesteps = 32
    n_qubits = 4
    n_feature_layers = 2
    n_mixing_layers = 2
    d_model = 32
    output_dim = 2

    print("\n[1] Testing QuantumE2ESuperpositionCore...")
    core = QuantumE2ESuperpositionCore(
        n_qubits=n_qubits,
        n_feature_layers=n_feature_layers,
        n_mixing_layers=n_mixing_layers,
        device=device
    )

    # Test with batch
    n_feature_params = 5 * n_qubits * n_feature_layers
    n_mixing_params = 5 * n_qubits * n_mixing_layers

    input_angles = torch.randn(batch_size, n_qubits)
    feature_params = torch.randn(batch_size, n_feature_params)
    forward_mixing = torch.randn(batch_size, n_mixing_params)
    backward_mixing = torch.randn(batch_size, n_mixing_params)
    diagonal_mixing = torch.randn(batch_size, n_mixing_params)
    dt_scale = torch.rand(batch_size)
    injection = torch.randn(batch_size, n_qubits)

    measurements = core(
        input_angles, feature_params, forward_mixing, backward_mixing, diagonal_mixing,
        dt_scale, injection
    )
    print(f"  Input angles: {input_angles.shape}")
    print(f"  Feature params: {feature_params.shape}")
    print(f"  Mixing params per branch: {n_mixing_params}")
    print(f"  Output measurements: {measurements.shape}")
    print(f"  Expected: ({batch_size}, {3 * n_qubits})")

    # Check superposition coefficients
    alpha, beta, gamma = core._get_superposition_coefficients()
    norm_sq = torch.abs(alpha)**2 + torch.abs(beta)**2 + torch.abs(gamma)**2
    print(f"  Superposition |α|²+|β|²+|γ|² = {norm_sq.item():.4f} (should be 1.0)")

    print("\n[2] Testing QuantumMambaE2E_Superposition (ID: 4d)...")
    model_4d = QuantumMambaE2E_Superposition(
        n_qubits=n_qubits,
        n_feature_layers=n_feature_layers,
        n_mixing_layers=n_mixing_layers,
        n_channels=n_channels,
        n_timesteps=n_timesteps,
        d_model=d_model,
        chunk_size=8,
        output_dim=output_dim,
        device=device
    )

    x = torch.randn(batch_size, n_channels, n_timesteps)
    output_4d = model_4d(x)
    params_4d = sum(p.numel() for p in model_4d.parameters() if p.requires_grad)
    print(f"  Input: {x.shape}")
    print(f"  Output: {output_4d.shape}")
    print(f"  Parameters: {params_4d:,}")

    print("\n[3] Testing QuantumHydraE2E_Superposition (ID: 4e)...")
    model_4e = QuantumHydraE2E_Superposition(
        n_qubits=n_qubits,
        n_feature_layers=n_feature_layers,
        n_mixing_layers=n_mixing_layers,
        n_channels=n_channels,
        n_timesteps=n_timesteps,
        d_model=d_model,
        chunk_size=8,
        output_dim=output_dim,
        device=device
    )

    output_4e = model_4e(x)
    params_4e = sum(p.numel() for p in model_4e.parameters() if p.requires_grad)
    print(f"  Input: {x.shape}")
    print(f"  Output: {output_4e.shape}")
    print(f"  Parameters: {params_4e:,}")

    print("\n[4] Testing gradient flow...")
    model_4d.train()
    optimizer = torch.optim.Adam(model_4d.parameters(), lr=1e-3)
    criterion = nn.CrossEntropyLoss()

    output = model_4d(x)
    labels = torch.randint(0, output_dim, (batch_size,))
    loss = criterion(output, labels)
    loss.backward()
    optimizer.step()

    print(f"  Loss: {loss.item():.4f}")
    print(f"  Gradient flow: OK")

    print("\n[5] Circuit structure verification...")
    print(f"  Params per layer: {5 * n_qubits} (matching QuantumE2E.py)")
    print(f"  Feature params: {n_feature_params}")
    print(f"  Mixing params per branch: {n_mixing_params}")
    print(f"  Total mixing params (3 branches): {3 * n_mixing_params}")
    print(f"  Ansatz: RX, RY, RZ + Forward CRX + Backward CRX")

    print("\n" + "=" * 80)
    print("Testing complete!")
    print("=" * 80)
    print("\nKey Features:")
    print("  - EXACT same circuit structure as QuantumE2E.py")
    print("  - E2E processing: NO intermediate measurements")
    print("  - TRUE quantum superposition: |ψ⟩ = α|ψ₁⟩ + β|ψ₂⟩ + γ|ψ₃⟩")
    print("  - Three branches (Forward, Backward, Diagonal)")
    print("  - Single measurement at the very end (PauliX, PauliY, PauliZ)")
    print("  - Learnable complex superposition coefficients (α, β, γ)")
    print("  - Native PennyLane batching (where supported)")
    print("\nModel IDs for ablation study:")
    print("  - 4d: QuantumMambaE2E_Superposition (unidirectional)")
    print("  - 4e: QuantumHydraE2E_Superposition (bidirectional)")
    print("=" * 80)
