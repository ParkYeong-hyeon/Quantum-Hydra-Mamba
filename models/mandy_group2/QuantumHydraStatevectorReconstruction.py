"""
QuantumHydra Statevector Reconstruction SSM (Mandy Group 2)

This variant mirrors the quantum superposition logic by:
1) extracting statevectors from each branch,
2) combining them with complex coefficients,
3) measuring the combined state.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import pennylane as qml
import numpy as np
from typing import Optional

# Global quantum device cache
_QDEV_CACHE = {}


def _get_quantum_device(n_qubits: int):
    """Get or create a cached quantum device."""
    if n_qubits not in _QDEV_CACHE:
        _QDEV_CACHE[n_qubits] = qml.device("default.qubit", wires=n_qubits)
    return _QDEV_CACHE[n_qubits]


# ================================================================================
# Circuit Building Blocks (matching QuantumHydraSSM.py)
# ================================================================================

def unified_ansatz_layer(params: torch.Tensor, n_qubits: int, param_offset: int = 0):
    """Unified ansatz layer (5 * n_qubits params per layer)."""
    param_idx = param_offset
    is_batched = params.ndim == 2

    for i in range(n_qubits):
        qml.RX(params[:, param_idx] if is_batched else params[param_idx], wires=i)
        qml.RY(params[:, param_idx + 1] if is_batched else params[param_idx + 1], wires=i)
        qml.RZ(params[:, param_idx + 2] if is_batched else params[param_idx + 2], wires=i)
        param_idx += 3

    for i in range(n_qubits):
        qml.CRX(params[:, param_idx] if is_batched else params[param_idx],
                wires=[i, (i + 1) % n_qubits])
        param_idx += 1

    for i in range(n_qubits - 1, -1, -1):
        qml.CRX(params[:, param_idx] if is_batched else params[param_idx],
                wires=[i, (i - 1) % n_qubits])
        param_idx += 1

    return param_idx


def delta_modulated_ansatz_layer(params: torch.Tensor, dt_scale: torch.Tensor,
                                 n_qubits: int, param_offset: int = 0):
    """Delta-modulated variational layer for SSM-style selective forgetting."""
    param_idx = param_offset
    is_batched = params.ndim == 2

    for i in range(n_qubits):
        p0 = params[:, param_idx] if is_batched else params[param_idx]
        p1 = params[:, param_idx + 1] if is_batched else params[param_idx + 1]
        p2 = params[:, param_idx + 2] if is_batched else params[param_idx + 2]
        qml.RX(p0 * dt_scale, wires=i)
        qml.RY(p1 * dt_scale, wires=i)
        qml.RZ(p2 * dt_scale, wires=i)
        param_idx += 3

    for i in range(n_qubits):
        qml.CRX(params[:, param_idx] if is_batched else params[param_idx],
                wires=[i, (i + 1) % n_qubits])
        param_idx += 1

    for i in range(n_qubits - 1, -1, -1):
        qml.CRX(params[:, param_idx] if is_batched else params[param_idx],
                wires=[i, (i - 1) % n_qubits])
        param_idx += 1

    return param_idx


def backward_ansatz_layer(params: torch.Tensor, dt_scale: torch.Tensor,
                          n_qubits: int, param_offset: int = 0):
    """Backward SSM layer (reversed qubit order)."""
    param_idx = param_offset
    is_batched = params.ndim == 2

    for i in range(n_qubits - 1, -1, -1):
        p0 = params[:, param_idx] if is_batched else params[param_idx]
        p1 = params[:, param_idx + 1] if is_batched else params[param_idx + 1]
        p2 = params[:, param_idx + 2] if is_batched else params[param_idx + 2]
        qml.RX(p0 * dt_scale, wires=i)
        qml.RY(p1 * dt_scale, wires=i)
        qml.RZ(p2 * dt_scale, wires=i)
        param_idx += 3

    for i in range(n_qubits - 1, -1, -1):
        qml.CRX(params[:, param_idx] if is_batched else params[param_idx],
                wires=[i, (i - 1) % n_qubits])
        param_idx += 1

    for i in range(n_qubits):
        qml.CRX(params[:, param_idx] if is_batched else params[param_idx],
                wires=[i, (i + 1) % n_qubits])
        param_idx += 1

    return param_idx


def input_embedding(inputs: torch.Tensor, n_qubits: int):
    """Input embedding via RY gates."""
    is_batched = inputs.ndim == 2
    for i in range(n_qubits):
        angle = inputs[:, i] if is_batched else inputs[i]
        qml.RY(angle, wires=i)


def multi_observable_measurement(n_qubits: int):
    """Measure PauliX, PauliY, PauliZ on all qubits."""
    measurements = []
    for i in range(n_qubits):
        measurements.append(qml.expval(qml.PauliX(i)))
    for i in range(n_qubits):
        measurements.append(qml.expval(qml.PauliY(i)))
    for i in range(n_qubits):
        measurements.append(qml.expval(qml.PauliZ(i)))
    return measurements


# ================================================================================
# Classical Reconstruction Core (statevector mixing + measurement)
# ================================================================================

class ClassicalReconstructionHydraSSMCore(nn.Module):
    """
    Three-branch processor with full statevector reconstruction.

    This mirrors the quantum superposition logic by:
    1) extracting statevectors from each branch,
    2) combining them with complex coefficients,
    3) measuring the combined state.
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

        self.params_per_layer = 5 * n_qubits
        self.n_params_per_branch = self.params_per_layer * n_layers
        self.q_dim = 3 * n_qubits

        self.qdev = _get_quantum_device(n_qubits)

        # Trainable complex coefficients for superposition
        init_val = float(1.0 / np.sqrt(3))
        self.alpha_real = nn.Parameter(torch.tensor(init_val, dtype=torch.float32))
        self.alpha_imag = nn.Parameter(torch.tensor(0.0, dtype=torch.float32))
        self.beta_real = nn.Parameter(torch.tensor(init_val, dtype=torch.float32))
        self.beta_imag = nn.Parameter(torch.tensor(0.0, dtype=torch.float32))
        self.gamma_real = nn.Parameter(torch.tensor(init_val, dtype=torch.float32))
        self.gamma_imag = nn.Parameter(torch.tensor(0.0, dtype=torch.float32))

        self._build_circuits()
        self._build_observables()
        self.to(self.torch_device)

    def _build_circuits(self):
        """Build quantum circuits for three branches returning state vectors."""
        n_qubits = self.n_qubits
        n_layers = self.n_layers

        @qml.qnode(self.qdev, interface="torch", diff_method="backprop")
        def forward_circuit(input_angles, params, dt_scale):
            input_embedding(input_angles, n_qubits)
            param_idx = 0
            for _ in range(n_layers):
                param_idx = delta_modulated_ansatz_layer(params, dt_scale, n_qubits, param_idx)
            return qml.state()

        @qml.qnode(self.qdev, interface="torch", diff_method="backprop")
        def backward_circuit(input_angles, params, dt_scale):
            input_embedding(input_angles, n_qubits)
            param_idx = 0
            for _ in range(n_layers):
                param_idx = backward_ansatz_layer(params, dt_scale, n_qubits, param_idx)
            return qml.state()

        @qml.qnode(self.qdev, interface="torch", diff_method="backprop")
        def diagonal_circuit(input_angles, params, _dt_scale):
            input_embedding(input_angles, n_qubits)
            param_idx = 0
            for _ in range(n_layers):
                param_idx = unified_ansatz_layer(params, n_qubits, param_idx)
            return qml.state()

        self.forward_circuit = forward_circuit
        self.backward_circuit = backward_circuit
        self.diagonal_circuit = diagonal_circuit

    def _build_observables(self):
        """Precompute Pauli X/Y/Z matrices for all qubits in PennyLane wire order."""
        wire_order = list(range(self.n_qubits))
        mats = []
        for i in range(self.n_qubits):
            mats.append(torch.tensor(qml.matrix(qml.PauliX(i), wire_order=wire_order),
                                     dtype=torch.complex64))
        for i in range(self.n_qubits):
            mats.append(torch.tensor(qml.matrix(qml.PauliY(i), wire_order=wire_order),
                                     dtype=torch.complex64))
        for i in range(self.n_qubits):
            mats.append(torch.tensor(qml.matrix(qml.PauliZ(i), wire_order=wire_order),
                                     dtype=torch.complex64))
        self._obs_mats = mats

    def _get_superposition_coefficients(self):
        alpha = torch.complex(self.alpha_real, self.alpha_imag)
        beta = torch.complex(self.beta_real, self.beta_imag)
        gamma = torch.complex(self.gamma_real, self.gamma_imag)

        norm = torch.sqrt(
            torch.abs(alpha) ** 2 + torch.abs(beta) ** 2 + torch.abs(gamma) ** 2 + 1e-9
        )
        return alpha / norm, beta / norm, gamma / norm

    def _measure_statevector(self, psi: torch.Tensor) -> torch.Tensor:
        """
        Compute Pauli X/Y/Z expectations from a statevector without a circuit.

        Supports single state (2**n_qubits,) or batched (batch, 2**n_qubits).
        Returns (3 * n_qubits,) or (batch, 3 * n_qubits).
        """
        squeeze = False
        if psi.ndim == 1:
            psi = psi.unsqueeze(0)
            squeeze = True

        psi_conj = psi.conj()
        expvals = []

        for mat in self._obs_mats:
            # call matrix function from PennyLane
            mat_device = mat.to(device=psi.device, dtype=psi.dtype)
            psi_mat = psi @ mat_device.T
            expvals.append((psi_conj * psi_mat).sum(dim=1).real)

        measurements = torch.stack(expvals, dim=1)
        return measurements.squeeze(0) if squeeze else measurements

    def forward(
        self,
        input_angles: torch.Tensor,
        forward_params: torch.Tensor,
        backward_params: torch.Tensor,
        diagonal_params: torch.Tensor,
        dt_scale: torch.Tensor
    ) -> torch.Tensor:
        psi1 = self.forward_circuit(input_angles, forward_params, dt_scale)
        psi2 = self.backward_circuit(input_angles, backward_params, dt_scale)
        psi3 = self.diagonal_circuit(input_angles, diagonal_params, dt_scale)

        psi1 = psi1.to(torch.complex64)
        psi2 = psi2.to(torch.complex64)
        psi3 = psi3.to(torch.complex64)

        alpha, beta, gamma = self._get_superposition_coefficients()
        psi_combined = alpha * psi1 + beta * psi2 + gamma * psi3

        if psi_combined.ndim == 1:
            norm = torch.linalg.vector_norm(psi_combined) + 1e-9
            psi_normalized = psi_combined / norm
        else:
            norm = torch.linalg.vector_norm(psi_combined, dim=-1, keepdim=True) + 1e-9
            psi_normalized = psi_combined / norm

        measurements = self._measure_statevector(psi_normalized).float()

        return measurements


# ================================================================================
# Full Model Wrappers (matching QuantumHydraSSM.py interface)
# ================================================================================

class ClassicalReconstructionHydraSSM(nn.Module):
    """
    Unidirectional model using full statevector reconstruction.
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

        self.input_proj = nn.Sequential(
            nn.Linear(actual_channels, d_model),
            nn.LayerNorm(d_model),
            nn.SiLU(),
            nn.Dropout(dropout)
        )

        self.chunk_attention = nn.Sequential(
            nn.Linear(d_model, d_model // 4),
            nn.Tanh(),
            nn.Linear(d_model // 4, 1)
        )

        self.forward_param_proj = nn.Linear(d_model, self.params_per_branch)
        self.backward_param_proj = nn.Linear(d_model, self.params_per_branch)
        self.diagonal_param_proj = nn.Linear(d_model, self.params_per_branch)

        self.encoding_proj = nn.Linear(d_model, n_qubits)

        self.dt_proj = nn.Sequential(
            nn.Linear(d_model, d_state),
            nn.SiLU(),
            nn.Linear(d_state, 1),
            nn.Sigmoid()
        )

        self.mixing_core = ClassicalReconstructionHydraSSMCore(
            n_qubits=n_qubits,
            n_layers=actual_layers,
            device=device
        )

        self.output_proj = nn.Sequential(
            nn.Linear(self.q_dim, d_model),
            nn.LayerNorm(d_model),
            nn.SiLU(),
            nn.Dropout(dropout)
        )

        self.seq_attention = nn.Sequential(
            nn.Linear(d_model, d_model // 4),
            nn.Tanh(),
            nn.Linear(d_model // 4, 1)
        )

        self.classifier = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, output_dim)
        )

        self.to(self.torch_device)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() == 2:
            x = x.unsqueeze(1)

        if x.shape[1] != self.n_timesteps and x.shape[2] == self.n_timesteps:
            x = x.permute(0, 2, 1)

        batch_size, seq_len, _ = x.shape

        x = self.input_proj(x)

        n_chunks = (seq_len + self.chunk_size - 1) // self.chunk_size
        if seq_len % self.chunk_size != 0:
            pad_len = self.chunk_size * n_chunks - seq_len
            x = F.pad(x, (0, 0, 0, pad_len))

        x = x.reshape(batch_size * n_chunks, self.chunk_size, self.d_model)

        attn_scores = self.chunk_attention(x)
        attn_weights = F.softmax(attn_scores, dim=1)
        chunk_summary = (attn_weights * x).sum(dim=1)

        forward_params = self.forward_param_proj(chunk_summary)
        backward_params = self.backward_param_proj(chunk_summary)
        diagonal_params = self.diagonal_param_proj(chunk_summary)

        input_angles = torch.tanh(self.encoding_proj(chunk_summary)) * np.pi
        dt_scale = self.dt_proj(chunk_summary).squeeze(-1)

        mixed_measurements = self.mixing_core(
            input_angles, forward_params, backward_params, diagonal_params, dt_scale
        )

        chunk_features = self.output_proj(mixed_measurements)
        chunk_features = chunk_features.reshape(batch_size, n_chunks, self.d_model)

        seq_attn = self.seq_attention(chunk_features)
        seq_weights = F.softmax(seq_attn, dim=1)
        sequence_repr = (seq_weights * chunk_features).sum(dim=1)

        logits = self.classifier(sequence_repr)
        return logits


class ClassicalReconstructionHydraSSMBidirectional(nn.Module):
    """
    Bidirectional model using full statevector reconstruction.
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

        self.input_proj = nn.Sequential(
            nn.Linear(actual_channels, d_model),
            nn.LayerNorm(d_model),
            nn.SiLU(),
            nn.Dropout(dropout)
        )

        self.chunk_attention = nn.Sequential(
            nn.Linear(d_model, d_model // 4),
            nn.Tanh(),
            nn.Linear(d_model // 4, 1)
        )

        self.fwd_forward_param_proj = nn.Linear(d_model, self.params_per_branch)
        self.fwd_backward_param_proj = nn.Linear(d_model, self.params_per_branch)
        self.fwd_diagonal_param_proj = nn.Linear(d_model, self.params_per_branch)

        self.bwd_forward_param_proj = nn.Linear(d_model, self.params_per_branch)
        self.bwd_backward_param_proj = nn.Linear(d_model, self.params_per_branch)
        self.bwd_diagonal_param_proj = nn.Linear(d_model, self.params_per_branch)

        self.encoding_proj = nn.Linear(d_model, n_qubits)
        self.dt_proj = nn.Sequential(
            nn.Linear(d_model, d_state),
            nn.SiLU(),
            nn.Linear(d_state, 1),
            nn.Sigmoid()
        )

        self.mixing_core_fwd = ClassicalReconstructionHydraSSMCore(
            n_qubits=n_qubits,
            n_layers=actual_layers,
            device=device
        )
        self.mixing_core_bwd = ClassicalReconstructionHydraSSMCore(
            n_qubits=n_qubits,
            n_layers=actual_layers,
            device=device
        )

        self.output_proj = nn.Sequential(
            nn.Linear(2 * self.q_dim, d_model),
            nn.LayerNorm(d_model),
            nn.SiLU(),
            nn.Dropout(dropout)
        )

        self.seq_attention = nn.Sequential(
            nn.Linear(d_model, d_model // 4),
            nn.Tanh(),
            nn.Linear(d_model // 4, 1)
        )

        self.classifier = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, output_dim)
        )

        self.to(self.torch_device)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() == 2:
            x = x.unsqueeze(1)

        if x.shape[1] != self.n_timesteps and x.shape[2] == self.n_timesteps:
            x = x.permute(0, 2, 1)

        batch_size, seq_len, _ = x.shape

        x = self.input_proj(x)

        n_chunks = (seq_len + self.chunk_size - 1) // self.chunk_size
        if seq_len % self.chunk_size != 0:
            pad_len = self.chunk_size * n_chunks - seq_len
            x = F.pad(x, (0, 0, 0, pad_len))

        x_chunked = x.reshape(batch_size * n_chunks, self.chunk_size, self.d_model)

        attn_scores = self.chunk_attention(x_chunked)
        attn_weights = F.softmax(attn_scores, dim=1)
        chunk_summary = (attn_weights * x_chunked).sum(dim=1)

        input_angles = torch.tanh(self.encoding_proj(chunk_summary)) * np.pi
        dt_scale = self.dt_proj(chunk_summary).squeeze(-1)

        fwd_forward_params = self.fwd_forward_param_proj(chunk_summary)
        fwd_backward_params = self.fwd_backward_param_proj(chunk_summary)
        fwd_diagonal_params = self.fwd_diagonal_param_proj(chunk_summary)

        mixed_fwd = self.mixing_core_fwd(
            input_angles, fwd_forward_params, fwd_backward_params, fwd_diagonal_params, dt_scale
        )

        chunk_summary_rev = chunk_summary.reshape(batch_size, n_chunks, -1)
        chunk_summary_rev = torch.flip(chunk_summary_rev, dims=[1])
        chunk_summary_rev = chunk_summary_rev.reshape(batch_size * n_chunks, -1)

        input_angles_rev = torch.tanh(self.encoding_proj(chunk_summary_rev)) * np.pi
        dt_scale_rev = self.dt_proj(chunk_summary_rev).squeeze(-1)

        bwd_forward_params = self.bwd_forward_param_proj(chunk_summary_rev)
        bwd_backward_params = self.bwd_backward_param_proj(chunk_summary_rev)
        bwd_diagonal_params = self.bwd_diagonal_param_proj(chunk_summary_rev)

        mixed_bwd = self.mixing_core_bwd(
            input_angles_rev, bwd_forward_params, bwd_backward_params, bwd_diagonal_params, dt_scale_rev
        )

        mixed_bwd = mixed_bwd.reshape(batch_size, n_chunks, -1)
        mixed_bwd = torch.flip(mixed_bwd, dims=[1])
        mixed_bwd = mixed_bwd.reshape(batch_size * n_chunks, -1)

        mixed_combined = torch.cat([mixed_fwd, mixed_bwd], dim=-1)

        chunk_features = self.output_proj(mixed_combined)
        chunk_features = chunk_features.reshape(batch_size, n_chunks, self.d_model)

        seq_attn = self.seq_attention(chunk_features)
        seq_weights = F.softmax(seq_attn, dim=1)
        sequence_repr = (seq_weights * chunk_features).sum(dim=1)

        logits = self.classifier(sequence_repr)
        return logits


class ClassicalReconstructionMambaHydraSSM(ClassicalReconstructionHydraSSM):
    """Alias for unidirectional reconstruction model."""
    pass


class ClassicalReconstructionHydraHydraSSM(ClassicalReconstructionHydraSSMBidirectional):
    """Alias for bidirectional reconstruction model."""
    pass
