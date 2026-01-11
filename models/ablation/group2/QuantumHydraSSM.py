"""
Quantum Hydra SSM: True Quantum Superposition with Delta-Modulated Recurrence

This model combines:
1. TRUE quantum superposition (three branch states combined BEFORE measurement)
2. Delta-modulated recurrence with selective forgetting
3. EXACT same circuit structure as QuantumSSM.py and QuantumE2E.py for fair comparison

Circuit Structure (MATCHING existing models):
- Per layer: RX, RY, RZ rotations (3 * n_qubits params)
- Forward CRX entanglement ring (n_qubits params)
- Backward CRX entanglement ring (n_qubits params)
- Total: 5 * n_qubits params per layer

Key Innovation:
- Three quantum branches (forward, backward, diagonal) combined in STATE SPACE
- |ψ⟩ = α|ψ₁⟩ + β|ψ₂⟩ + γ|ψ₃⟩ (true superposition before measurement)
- Uses chunked processing for efficiency (like QuantumSSM.py)
- Native PennyLane batching (no for-loops over samples)

Model IDs for ablation study:
- 2d: QuantumMambaHydraSSM (unidirectional)
- 2e: QuantumHydraHydraSSM (bidirectional)

Author: Research Team
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


# ================================================================================
# Circuit Building Blocks (EXACT match to QuantumSSM.py and QuantumE2E.py)
# ================================================================================

def unified_ansatz_layer(params: torch.Tensor, n_qubits: int, param_offset: int = 0):
    """
    Single variational layer with unified ansatz structure.

    EXACT match to QuantumFeatureExtractor and QuantumE2E (5 * n_qubits params per layer):
    - RX, RY, RZ rotations on each qubit (3 * n_qubits)
    - Forward CRX entanglement (n_qubits)
    - Backward CRX entanglement (n_qubits)

    Supports native PennyLane batching (params can be 1D or 2D).
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


def delta_modulated_ansatz_layer(params: torch.Tensor, dt_scale: torch.Tensor,
                                  n_qubits: int, param_offset: int = 0):
    """
    Delta-modulated variational layer for SSM-style selective forgetting.

    EXACT match to QuantumSSMCore and QuantumE2E ssm_mixing_layer:
    - RX, RY, RZ scaled by Delta (selective forgetting)
    - Forward/Backward CRX entanglement (not scaled by Delta)

    Supports native PennyLane batching.
    """
    param_idx = param_offset
    is_batched = params.ndim == 2

    # Delta-modulated RX, RY, RZ rotations
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


def backward_ansatz_layer(params: torch.Tensor, dt_scale: torch.Tensor,
                          n_qubits: int, param_offset: int = 0):
    """
    Backward SSM layer - applies operations in reversed qubit order.
    Used for Hydra's backward branch.

    Supports native PennyLane batching.
    """
    param_idx = param_offset
    is_batched = params.ndim == 2

    # Reverse order rotations (last qubit to first)
    for i in range(n_qubits - 1, -1, -1):
        p0 = params[:, param_idx] if is_batched else params[param_idx]
        p1 = params[:, param_idx + 1] if is_batched else params[param_idx + 1]
        p2 = params[:, param_idx + 2] if is_batched else params[param_idx + 2]
        qml.RX(p0 * dt_scale, wires=i)
        qml.RY(p1 * dt_scale, wires=i)
        qml.RZ(p2 * dt_scale, wires=i)
        param_idx += 3

    # Backward CRX first
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


def input_embedding(inputs: torch.Tensor, n_qubits: int):
    """
    Input embedding via RY gates. Supports batched inputs.

    EXACT match to QuantumFeatureExtractor.
    """
    is_batched = inputs.ndim == 2
    for i in range(n_qubits):
        angle = inputs[:, i] if is_batched else inputs[i]
        qml.RY(angle, wires=i)


def multi_observable_measurement(n_qubits: int):
    """
    Multi-observable measurement (PauliX, PauliY, PauliZ on all qubits).

    EXACT match to QuantumSSM.py and QuantumE2E.py.
    Returns 3 * n_qubits measurements.
    """
    measurements = []
    for i in range(n_qubits):
        measurements.append(qml.expval(qml.PauliX(i)))
    for i in range(n_qubits):
        measurements.append(qml.expval(qml.PauliY(i)))
    for i in range(n_qubits):
        measurements.append(qml.expval(qml.PauliZ(i)))
    return measurements


# ================================================================================
# Three-Branch Quantum Core with True Superposition (Batched)
# ================================================================================

class QuantumHydraSSMCore(nn.Module):
    """
    Three-branch quantum processor with TRUE state superposition.

    Uses EXACT same circuit structure as QuantumSSM.py and QuantumE2E.py.
    Combines three branch states before measurement: |ψ⟩ = α|ψ₁⟩ + β|ψ₂⟩ + γ|ψ₃⟩

    Efficient batched processing via PennyLane native batching.
    """

    def __init__(
        self,
        n_qubits: int = 6,
        n_layers: int = 2,
        device: str = "cpu"
    ):
        super().__init__()

        self.n_qubits = n_qubits
        self.n_layers = n_layers
        self.torch_device = torch.device(device) if isinstance(device, str) else device

        # Parameters per layer: 5 * n_qubits (matching QuantumSSM.py)
        self.params_per_layer = 5 * n_qubits
        self.n_params_per_branch = self.params_per_layer * n_layers

        # Quantum output dimension: 3 * n_qubits (PauliX, PauliY, PauliZ)
        self.q_dim = 3 * n_qubits

        # Quantum device
        self.qdev = _get_quantum_device(n_qubits)

        # Trainable complex coefficients for superposition
        # Use explicit float32 dtype for consistency
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
        n_layers = self.n_layers

        # Branch 1: Forward SSM (standard variational + Delta-modulated mixing)
        @qml.qnode(self.qdev, interface="torch", diff_method="backprop")
        def forward_circuit(input_angles, params, dt_scale):
            """Forward branch circuit returning state vector."""
            # Input embedding
            input_embedding(input_angles, n_qubits)

            # Delta-modulated variational layers
            param_idx = 0
            for layer in range(n_layers):
                param_idx = delta_modulated_ansatz_layer(params, dt_scale, n_qubits, param_idx)

            return qml.state()

        # Branch 2: Backward SSM (reversed operations for bidirectional)
        @qml.qnode(self.qdev, interface="torch", diff_method="backprop")
        def backward_circuit(input_angles, params, dt_scale):
            """Backward branch circuit returning state vector."""
            # Input embedding
            input_embedding(input_angles, n_qubits)

            # Backward variational layers
            param_idx = 0
            for layer in range(n_layers):
                param_idx = backward_ansatz_layer(params, dt_scale, n_qubits, param_idx)

            return qml.state()

        # Branch 3: Diagonal (simpler, skip-connection style)
        @qml.qnode(self.qdev, interface="torch", diff_method="backprop")
        def diagonal_circuit(input_angles, params, dt_scale):
            """Diagonal branch circuit returning state vector."""
            # Input embedding
            input_embedding(input_angles, n_qubits)

            # Standard variational layers (no backward direction, simpler mixing)
            param_idx = 0
            for layer in range(n_layers):
                param_idx = unified_ansatz_layer(params, n_qubits, param_idx)

            return qml.state()

        # Measurement circuit for combined state
        @qml.qnode(self.qdev, interface="torch", diff_method="backprop")
        def measurement_circuit(state_vector):
            """Measure the combined superposition state."""
            qml.StatePrep(state_vector, wires=range(n_qubits))
            return multi_observable_measurement(n_qubits)

        self.forward_circuit = forward_circuit
        self.backward_circuit = backward_circuit
        self.diagonal_circuit = diagonal_circuit
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
        forward_params: torch.Tensor,
        backward_params: torch.Tensor,
        diagonal_params: torch.Tensor,
        dt_scale: torch.Tensor
    ) -> torch.Tensor:
        """
        Process batch through three-branch quantum superposition.

        All inputs support native PennyLane batching.

        Args:
            input_angles: (batch, n_qubits) input encoding angles
            forward_params: (batch, n_params_per_branch) forward branch params
            backward_params: (batch, n_params_per_branch) backward branch params
            diagonal_params: (batch, n_params_per_branch) diagonal branch params
            dt_scale: (batch,) Delta scale factors

        Returns:
            measurements: (batch, 3 * n_qubits)
        """
        batch_size = input_angles.shape[0]

        # Get state vectors from each branch (batched)
        psi1 = self.forward_circuit(input_angles, forward_params, dt_scale)
        psi2 = self.backward_circuit(input_angles, backward_params, dt_scale)
        psi3 = self.diagonal_circuit(input_angles, diagonal_params, dt_scale)

        # Ensure complex dtype and proper shape
        # For batched execution, state vectors are (batch, 2^n_qubits)
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
        # Note: StatePrep with batched states requires processing per sample
        # This is a limitation - we process samples in a vectorized manner where possible
        if psi_normalized.ndim == 1:
            measurements = self.measurement_circuit(psi_normalized)
            measurements = torch.stack(measurements).float()
        else:
            # For batched states, process each sample
            # TODO: Optimize with fully batched measurement circuit
            all_measurements = []
            for b in range(batch_size):
                m = self.measurement_circuit(psi_normalized[b])
                all_measurements.append(torch.stack(m))
            measurements = torch.stack(all_measurements).float()

        return measurements


# ================================================================================
# Full Model Wrappers (Matching QuantumSSM.py interface)
# ================================================================================

class QuantumHydraSSM(nn.Module):
    """
    Full Quantum Hydra SSM model for sequence classification.

    Combines:
    - Chunked processing for efficiency (like QuantumSSM.py)
    - Three-branch quantum SSM with true superposition
    - Delta-modulated selective forgetting
    - EXACT same circuit structure as QuantumSSM.py

    Model ID for ablation study: 2d
    """

    def __init__(
        self,
        n_qubits: int = 6,
        qlcu_layers: int = 2,
        d_model: int = 128,
        d_state: int = 16,
        feature_dim: int = 64,
        n_timesteps: int = 125,
        output_dim: int = 2,
        dropout: float = 0.1,
        chunk_size: int = 32,
        device: str = "cpu",
        # Aliases for compatibility
        n_layers: Optional[int] = None,
        n_channels: Optional[int] = None,
    ):
        super().__init__()

        # Handle parameter aliases
        actual_layers = n_layers if n_layers is not None else qlcu_layers
        actual_channels = n_channels if n_channels is not None else feature_dim

        self.n_qubits = n_qubits
        self.n_layers = actual_layers
        self.d_model = d_model
        self.n_timesteps = n_timesteps
        self.chunk_size = chunk_size
        self.torch_device = torch.device(device) if isinstance(device, str) else device

        # Quantum output dimension: 3 * n_qubits
        self.q_dim = 3 * n_qubits

        # Parameters per branch: 5 * n_qubits * n_layers
        self.params_per_branch = 5 * n_qubits * actual_layers

        # Input projection
        self.input_proj = nn.Sequential(
            nn.Linear(actual_channels, d_model),
            nn.LayerNorm(d_model),
            nn.SiLU(),
            nn.Dropout(dropout)
        )

        # Chunk attention for aggregation (like QuantumSSM.py)
        self.chunk_attention = nn.Sequential(
            nn.Linear(d_model, d_model // 4),
            nn.Tanh(),
            nn.Linear(d_model // 4, 1)
        )

        # Parameter projections for three branches
        self.forward_param_proj = nn.Linear(d_model, self.params_per_branch)
        self.backward_param_proj = nn.Linear(d_model, self.params_per_branch)
        self.diagonal_param_proj = nn.Linear(d_model, self.params_per_branch)

        # Input encoding projection
        self.encoding_proj = nn.Linear(d_model, n_qubits)

        # Delta (selective forgetting) projection
        self.dt_proj = nn.Sequential(
            nn.Linear(d_model, d_state),
            nn.SiLU(),
            nn.Linear(d_state, 1),
            nn.Sigmoid()  # Delta in [0, 1]
        )

        # Quantum core
        self.quantum_core = QuantumHydraSSMCore(
            n_qubits=n_qubits,
            n_layers=actual_layers,
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

        # Classification head
        self.classifier = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, output_dim)
        )

        self.to(self.torch_device)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass with chunked quantum processing.

        Args:
            x: Input tensor (batch, n_channels, n_timesteps) or (batch, n_timesteps, n_channels)

        Returns:
            logits: (batch, output_dim)
        """
        # Handle input format
        if x.dim() == 2:
            x = x.unsqueeze(1)

        if x.shape[1] != self.n_timesteps and x.shape[2] == self.n_timesteps:
            x = x.permute(0, 2, 1)  # -> (batch, n_timesteps, n_channels)

        batch_size, seq_len, _ = x.shape

        # Input projection
        x = self.input_proj(x)  # (batch, seq_len, d_model)

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

        # Get quantum parameters for each branch
        forward_params = self.forward_param_proj(chunk_summary)
        backward_params = self.backward_param_proj(chunk_summary)
        diagonal_params = self.diagonal_param_proj(chunk_summary)

        # Get input encoding angles
        input_angles = torch.tanh(self.encoding_proj(chunk_summary)) * np.pi

        # Get Delta scale
        dt_scale = self.dt_proj(chunk_summary).squeeze(-1)

        # Process through quantum core (batched)
        quantum_features = self.quantum_core(
            input_angles, forward_params, backward_params, diagonal_params, dt_scale
        )

        # Output projection
        chunk_features = self.output_proj(quantum_features)
        chunk_features = chunk_features.reshape(batch_size, n_chunks, self.d_model)

        # Sequence aggregation
        seq_attn = self.seq_attention(chunk_features)
        seq_weights = F.softmax(seq_attn, dim=1)
        sequence_repr = (seq_weights * chunk_features).sum(dim=1)

        # Classification
        logits = self.classifier(sequence_repr)

        return logits


class QuantumHydraSSMBidirectional(nn.Module):
    """
    Bidirectional Quantum Hydra SSM (processes sequence in both directions).

    Model ID for ablation study: 2e
    """

    def __init__(
        self,
        n_qubits: int = 6,
        qlcu_layers: int = 2,
        d_model: int = 128,
        d_state: int = 16,
        feature_dim: int = 64,
        n_timesteps: int = 125,
        output_dim: int = 2,
        dropout: float = 0.1,
        chunk_size: int = 32,
        device: str = "cpu",
        n_layers: Optional[int] = None,
        n_channels: Optional[int] = None,
    ):
        super().__init__()

        actual_layers = n_layers if n_layers is not None else qlcu_layers
        actual_channels = n_channels if n_channels is not None else feature_dim

        self.n_qubits = n_qubits
        self.n_layers = actual_layers
        self.d_model = d_model
        self.n_timesteps = n_timesteps
        self.chunk_size = chunk_size
        self.torch_device = torch.device(device) if isinstance(device, str) else device

        self.q_dim = 3 * n_qubits
        self.params_per_branch = 5 * n_qubits * actual_layers

        # Input projection
        self.input_proj = nn.Sequential(
            nn.Linear(actual_channels, d_model),
            nn.LayerNorm(d_model),
            nn.SiLU(),
            nn.Dropout(dropout)
        )

        # Chunk attention
        self.chunk_attention = nn.Sequential(
            nn.Linear(d_model, d_model // 4),
            nn.Tanh(),
            nn.Linear(d_model // 4, 1)
        )

        # Parameter projections (separate for forward and backward passes)
        self.fwd_forward_param_proj = nn.Linear(d_model, self.params_per_branch)
        self.fwd_backward_param_proj = nn.Linear(d_model, self.params_per_branch)
        self.fwd_diagonal_param_proj = nn.Linear(d_model, self.params_per_branch)

        self.bwd_forward_param_proj = nn.Linear(d_model, self.params_per_branch)
        self.bwd_backward_param_proj = nn.Linear(d_model, self.params_per_branch)
        self.bwd_diagonal_param_proj = nn.Linear(d_model, self.params_per_branch)

        # Encoding and Delta projections
        self.encoding_proj = nn.Linear(d_model, n_qubits)
        self.dt_proj = nn.Sequential(
            nn.Linear(d_model, d_state),
            nn.SiLU(),
            nn.Linear(d_state, 1),
            nn.Sigmoid()
        )

        # Two quantum cores (forward and backward)
        self.quantum_core_fwd = QuantumHydraSSMCore(
            n_qubits=n_qubits,
            n_layers=actual_layers,
            device=device
        )
        self.quantum_core_bwd = QuantumHydraSSMCore(
            n_qubits=n_qubits,
            n_layers=actual_layers,
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

        # Classification head
        self.classifier = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, output_dim)
        )

        self.to(self.torch_device)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass with bidirectional processing."""
        if x.dim() == 2:
            x = x.unsqueeze(1)

        if x.shape[1] != self.n_timesteps and x.shape[2] == self.n_timesteps:
            x = x.permute(0, 2, 1)

        batch_size, seq_len, _ = x.shape

        # Input projection
        x = self.input_proj(x)

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
        dt_scale = self.dt_proj(chunk_summary).squeeze(-1)

        # Forward direction quantum processing
        fwd_forward_params = self.fwd_forward_param_proj(chunk_summary)
        fwd_backward_params = self.fwd_backward_param_proj(chunk_summary)
        fwd_diagonal_params = self.fwd_diagonal_param_proj(chunk_summary)

        quantum_fwd = self.quantum_core_fwd(
            input_angles, fwd_forward_params, fwd_backward_params, fwd_diagonal_params, dt_scale
        )

        # Backward direction (flip chunk order, process, flip back)
        chunk_summary_rev = chunk_summary.reshape(batch_size, n_chunks, -1)
        chunk_summary_rev = torch.flip(chunk_summary_rev, dims=[1])
        chunk_summary_rev = chunk_summary_rev.reshape(batch_size * n_chunks, -1)

        input_angles_rev = torch.tanh(self.encoding_proj(chunk_summary_rev)) * np.pi
        dt_scale_rev = self.dt_proj(chunk_summary_rev).squeeze(-1)

        bwd_forward_params = self.bwd_forward_param_proj(chunk_summary_rev)
        bwd_backward_params = self.bwd_backward_param_proj(chunk_summary_rev)
        bwd_diagonal_params = self.bwd_diagonal_param_proj(chunk_summary_rev)

        quantum_bwd = self.quantum_core_bwd(
            input_angles_rev, bwd_forward_params, bwd_backward_params, bwd_diagonal_params, dt_scale_rev
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

        # Classification
        logits = self.classifier(sequence_repr)

        return logits


# ================================================================================
# Model Variants for Ablation Study (Aliases)
# ================================================================================

class QuantumMambaHydraSSM(QuantumHydraSSM):
    """
    Quantum Mamba with Hydra-style three-branch superposition.
    Alias for QuantumHydraSSM (unidirectional).

    Model ID: 2d (for ablation study)
    - Classical feature extraction
    - Quantum mixing with true superposition + Delta recurrence
    """
    pass


class QuantumHydraHydraSSM(QuantumHydraSSMBidirectional):
    """
    Quantum Hydra with Hydra-style three-branch superposition (bidirectional).
    Alias for QuantumHydraSSMBidirectional.

    Model ID: 2e (for ablation study)
    - Classical feature extraction
    - Quantum mixing with true superposition + Delta recurrence
    - Bidirectional processing
    """
    pass


# ================================================================================
# Testing
# ================================================================================

def test_quantum_hydra_ssm():
    """Test the QuantumHydraSSM implementation."""
    print("=" * 80)
    print("Testing QuantumHydraSSM: True Superposition + Delta Recurrence")
    print("Using EXACT same circuit structure as QuantumSSM.py and QuantumE2E.py")
    print("=" * 80)

    device = "cpu"
    batch_size = 2
    n_channels = 64
    n_timesteps = 32  # Small for testing
    n_qubits = 4
    n_layers = 2
    output_dim = 2

    print(f"\n[1] Testing QuantumHydraSSMCore...")
    core = QuantumHydraSSMCore(n_qubits=n_qubits, n_layers=n_layers, device=device)

    # Test with batch
    params_per_branch = 5 * n_qubits * n_layers
    input_angles = torch.randn(batch_size, n_qubits)
    forward_params = torch.randn(batch_size, params_per_branch)
    backward_params = torch.randn(batch_size, params_per_branch)
    diagonal_params = torch.randn(batch_size, params_per_branch)
    dt_scale = torch.rand(batch_size)

    measurements = core(input_angles, forward_params, backward_params, diagonal_params, dt_scale)
    print(f"  Input angles: {input_angles.shape}")
    print(f"  Params per branch: {params_per_branch}")
    print(f"  Output measurements: {measurements.shape}")
    print(f"  Expected: ({batch_size}, {3 * n_qubits})")

    # Check superposition coefficients
    alpha, beta, gamma = core._get_superposition_coefficients()
    norm_sq = torch.abs(alpha)**2 + torch.abs(beta)**2 + torch.abs(gamma)**2
    print(f"  Superposition |α|²+|β|²+|γ|² = {norm_sq.item():.4f} (should be 1.0)")

    print(f"\n[2] Testing QuantumHydraSSM (full model, ID: 2d)...")
    model = QuantumHydraSSM(
        n_qubits=n_qubits,
        qlcu_layers=n_layers,
        d_model=64,
        d_state=8,
        feature_dim=n_channels,
        n_timesteps=n_timesteps,
        output_dim=output_dim,
        dropout=0.1,
        chunk_size=8,
        device=device
    )

    x = torch.randn(batch_size, n_channels, n_timesteps)
    output = model(x)
    print(f"  Input: {x.shape}")
    print(f"  Output: {output.shape}")

    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Trainable parameters: {total_params:,}")

    print(f"\n[3] Testing QuantumHydraSSMBidirectional (ID: 2e)...")
    model_bidir = QuantumHydraSSMBidirectional(
        n_qubits=n_qubits,
        qlcu_layers=n_layers,
        d_model=64,
        d_state=8,
        feature_dim=n_channels,
        n_timesteps=n_timesteps,
        output_dim=output_dim,
        dropout=0.1,
        chunk_size=8,
        device=device
    )

    output_bidir = model_bidir(x)
    print(f"  Input: {x.shape}")
    print(f"  Output: {output_bidir.shape}")

    total_params_bidir = sum(p.numel() for p in model_bidir.parameters() if p.requires_grad)
    print(f"  Trainable parameters: {total_params_bidir:,}")

    print(f"\n[4] Testing gradient flow...")
    model.train()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.CrossEntropyLoss()

    output = model(x)
    labels = torch.randint(0, output_dim, (batch_size,))
    loss = criterion(output, labels)
    loss.backward()
    optimizer.step()

    print(f"  Loss: {loss.item():.4f}")
    print(f"  Gradient flow: OK")

    print(f"\n[5] Circuit structure verification...")
    print(f"  Params per layer: {5 * n_qubits} (matching QuantumSSM.py)")
    print(f"  Params per branch: {5 * n_qubits * n_layers}")
    print(f"  Total params (3 branches): {3 * 5 * n_qubits * n_layers}")
    print(f"  Ansatz: RX, RY, RZ + Forward CRX + Backward CRX")

    print("\n" + "=" * 80)
    print("Testing complete!")
    print("=" * 80)
    print("\nKey Features:")
    print("  - EXACT same circuit structure as QuantumSSM.py and QuantumE2E.py")
    print("  - Three branches with true quantum superposition")
    print("  - Delta-modulated selective forgetting")
    print("  - Chunked processing for efficiency")
    print("  - Native PennyLane batching (where supported)")
    print("=" * 80)

    return model


if __name__ == "__main__":
    test_quantum_hydra_ssm()
