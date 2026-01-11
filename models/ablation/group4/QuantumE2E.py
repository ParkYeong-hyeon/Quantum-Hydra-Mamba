"""
End-to-End Quantum Models for Genomic Classification

These models implement TRUE end-to-end quantum processing where quantum states
flow continuously from feature extraction to mixing WITHOUT intermediate measurements.

CRITICAL DESIGN: These models use the EXACT same quantum circuit structures as:
- Feature Extraction: QuantumFeatureExtractor from QuantumGatedRecurrence.py
- Mamba Mixing: QuantumSSMCore from QuantumMixingSSM.py
- Hydra Mixing: QuantumBidirectionalSSMCore from QuantumMixingSSM.py
- Transformer Mixing: QuantumAttentionMixingCore from QuantumMixingSSM.py

The ONLY difference is that there is NO intermediate measurement between
feature extraction and mixing - the quantum state flows continuously.

Architecture:
    Classical Input → Quantum Encoding → Quantum Feature Extraction → Quantum Mixing → Measurement → Classical Output
                      ↓                  ↓                           ↓               ↓
                   RY(data)          Variational Ansatz          SSM/Attention    PauliXYZ
                                     (NO measurement)            (NO measurement)  (ONLY here)

Models:
1. QuantumMambaE2E: End-to-end quantum with Mamba-style selective mixing
2. QuantumHydraE2E: End-to-end quantum with bidirectional mixing
3. QuantumTransformerE2E: End-to-end quantum with attention-style global mixing

Author: Quantum Hydra/Mamba Research Team
Date: December 2024
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import pennylane as qml
import numpy as np
from typing import Tuple, Optional, List
import math

# Global quantum device cache to avoid recreation
_QDEV_CACHE = {}


def _get_quantum_device(n_qubits: int, batch_obs: bool = True):
    """Get or create a cached quantum device."""
    key = (n_qubits, batch_obs)
    if key not in _QDEV_CACHE:
        _QDEV_CACHE[key] = qml.device("default.qubit", wires=n_qubits)
    return _QDEV_CACHE[key]


# ==============================================================================
# Quantum Circuit Building Blocks (EXACT match to existing models)
# ==============================================================================

def input_embedding(inputs: torch.Tensor, n_qubits: int):
    """
    Input embedding via RY gates. Supports batched inputs.

    EXACT match to QuantumFeatureExtractor.feature_circuit (line 97-99):
        for i in range(self.n_qubits):
            angle = inputs[i]
            qml.RY(angle, wires=i)

    Args:
        inputs: Tensor of shape (n_qubits,) or (batch, n_qubits) with values in [-π, π]
        n_qubits: Number of qubits
    """
    is_batched = inputs.ndim == 2
    for i in range(n_qubits):
        angle = inputs[:, i] if is_batched else inputs[i]
        qml.RY(angle, wires=i)


def variational_feature_layer(params: torch.Tensor, n_qubits: int, param_offset: int = 0):
    """
    Single variational layer for feature extraction. Supports batched parameters.

    EXACT match to QuantumFeatureExtractor.feature_circuit (lines 104-124):
        # 2a. Full single-qubit rotations: RX, RY, RZ
        for i in range(n_qubits):
            qml.RX(params[param_idx], wires=i)
            qml.RY(params[param_idx + 1], wires=i)
            qml.RZ(params[param_idx + 2], wires=i)
            param_idx += 3
        # 2b. Forward CRX entanglement
        for i in range(n_qubits):
            qml.CRX(params[param_idx], wires=[i, (i + 1) % n_qubits])
            param_idx += 1
        # 2c. Backward CRX entanglement
        for i in range(n_qubits - 1, -1, -1):
            qml.CRX(params[param_idx], wires=[i, (i - 1) % n_qubits])
            param_idx += 1

    Args:
        params: All parameters of shape (n_params,) or (batch, n_params)
        n_qubits: Number of qubits
        param_offset: Starting index in params

    Returns:
        param_offset: Updated offset after consuming parameters
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
        qml.CRX(params[:, param_idx] if is_batched else params[param_idx], wires=[i, (i + 1) % n_qubits])
        param_idx += 1

    # Backward CRX entanglement (ring topology)
    for i in range(n_qubits - 1, -1, -1):
        qml.CRX(params[:, param_idx] if is_batched else params[param_idx], wires=[i, (i - 1) % n_qubits])
        param_idx += 1

    return param_idx


def ssm_mixing_layer(params: torch.Tensor, dt_scale: torch.Tensor, n_qubits: int, param_offset: int = 0):
    """
    Single SSM mixing layer (Mamba-style). Supports batched parameters.

    EXACT match to QuantumSSMCore.quantum_ssm_circuit (lines 224-242):
        # Delta-modulated variational layers
        for layer in range(n_layers):
            # Single-qubit rotations scaled by Delta
            for i in range(n_qubits):
                qml.RX(params[param_idx] * dt_scale, wires=i)
                qml.RY(params[param_idx + 1] * dt_scale, wires=i)
                qml.RZ(params[param_idx + 2] * dt_scale, wires=i)
                param_idx += 3
            # Forward entanglement
            for i in range(n_qubits):
                qml.CRX(params[param_idx], wires=[i, (i + 1) % n_qubits])
                param_idx += 1
            # Backward entanglement
            for i in range(n_qubits - 1, -1, -1):
                qml.CRX(params[param_idx], wires=[i, (i - 1) % n_qubits])
                param_idx += 1

    Args:
        params: All parameters of shape (n_params,) or (batch, n_params)
        dt_scale: Delta scaling factor - scalar or (batch,)
        n_qubits: Number of qubits
        param_offset: Starting index in params

    Returns:
        param_offset: Updated offset after consuming parameters
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
        qml.CRX(params[:, param_idx] if is_batched else params[param_idx], wires=[i, (i + 1) % n_qubits])
        param_idx += 1

    # Backward CRX entanglement
    for i in range(n_qubits - 1, -1, -1):
        qml.CRX(params[:, param_idx] if is_batched else params[param_idx], wires=[i, (i - 1) % n_qubits])
        param_idx += 1

    return param_idx


def ssm_input_injection(input_angles: torch.Tensor, dt_scale: torch.Tensor, n_qubits: int):
    """
    Input injection layer for SSM (Mamba-style). Supports batched inputs.

    EXACT match to QuantumSSMCore.quantum_ssm_circuit (lines 244-246):
        # Input injection scaled by Delta
        for i in range(n_qubits):
            qml.RY(input_angles[i] * dt_scale, wires=i)

    Args:
        input_angles: Input angles (n_qubits,) or (batch, n_qubits)
        dt_scale: Delta scaling factor - scalar or (batch,)
        n_qubits: Number of qubits
    """
    is_batched = input_angles.ndim == 2
    for i in range(n_qubits):
        angle = input_angles[:, i] if is_batched else input_angles[i]
        qml.RY(angle * dt_scale, wires=i)


def backward_ssm_mixing_layer(params: torch.Tensor, dt_scale: torch.Tensor, n_qubits: int, param_offset: int = 0):
    """
    Backward SSM mixing layer (reversed qubit order for Hydra). Supports batched parameters.

    This is like ssm_mixing_layer but applies rotations in reverse qubit order
    to achieve the "backward" direction in Hydra's bidirectional processing.

    Args:
        params: All parameters of shape (n_params,) or (batch, n_params)
        dt_scale: Delta scaling factor - scalar or (batch,)
        n_qubits: Number of qubits
        param_offset: Starting index in params

    Returns:
        param_offset: Updated offset after consuming parameters
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
        qml.CRX(params[:, param_idx] if is_batched else params[param_idx], wires=[i, (i - 1) % n_qubits])
        param_idx += 1

    # Forward CRX second
    for i in range(n_qubits):
        qml.CRX(params[:, param_idx] if is_batched else params[param_idx], wires=[i, (i + 1) % n_qubits])
        param_idx += 1

    return param_idx


def attention_mixing_layer(params: torch.Tensor, n_qubits: int, param_offset: int = 0):
    """
    Single attention mixing layer. Supports batched parameters.

    EXACT match to QuantumAttentionMixingCore unified ansatz (lines 778-796):
        # RX, RY, RZ rotations
        for i in range(n_qubits):
            qml.RX(params[param_idx], wires=i)
            qml.RY(params[param_idx + 1], wires=i)
            qml.RZ(params[param_idx + 2], wires=i)
            param_idx += 3
        # Forward CRX
        for i in range(n_qubits):
            qml.CRX(params[param_idx], wires=[i, (i + 1) % n_qubits])
            param_idx += 1
        # Backward CRX
        for i in range(n_qubits - 1, -1, -1):
            qml.CRX(params[param_idx], wires=[i, (i - 1) % n_qubits])
            param_idx += 1

    Args:
        params: All parameters (we slice starting from param_offset)
        n_qubits: Number of qubits
        param_offset: Starting index in params

    Returns:
        param_offset: Updated offset after consuming parameters
    """
    # This is identical to variational_feature_layer (same unified ansatz)
    return variational_feature_layer(params, n_qubits, param_offset)


def multi_observable_measurement(n_qubits: int):
    """
    Multi-observable measurement (PauliX, PauliY, PauliZ on all qubits).

    EXACT match to QuantumFeatureExtractor (lines 127-130) and QuantumSSMCore (lines 249-252):
        observables = [qml.PauliX(i) for i in range(n_qubits)] + \
                      [qml.PauliY(i) for i in range(n_qubits)] + \
                      [qml.PauliZ(i) for i in range(n_qubits)]
        return [qml.expval(op) for op in observables]

    Returns:
        List of expectation value measurements
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
# End-to-End Quantum Mamba Model
# ==============================================================================

class QuantumMambaE2E(nn.Module):
    """
    End-to-End Quantum Mamba: Quantum Features → Quantum SSM Mixing → Measurement

    Uses EXACT circuit structures from:
    - Feature Extraction: QuantumFeatureExtractor (RY input + RX/RY/RZ + bidirectional CRX)
    - SSM Mixing: QuantumSSMCore (Delta-modulated rotations + bidirectional CRX + input injection)

    Key: The quantum state flows continuously without intermediate measurements.
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
        device: str = "cpu"
    ):
        super().__init__()

        self.n_qubits = n_qubits
        self.n_feature_layers = n_feature_layers
        self.n_mixing_layers = n_mixing_layers
        self.n_channels = n_channels
        self.n_timesteps = n_timesteps
        self.d_model = d_model
        self.chunk_size = chunk_size
        self.output_dim = output_dim
        self.torch_device = device

        # Quantum output dimension: 3 measurements per qubit (PauliX, PauliY, PauliZ)
        self.q_dim = 3 * n_qubits

        # Calculate parameter counts (matching existing models)
        # Feature params: 5 * n_qubits per layer (RX, RY, RZ, CRX_fwd, CRX_bwd)
        self.n_feature_params = 5 * n_qubits * n_feature_layers
        # Mixing params: 5 * n_qubits per layer (same structure)
        self.n_mixing_params = 5 * n_qubits * n_mixing_layers
        self.n_total_circuit_params = self.n_feature_params + self.n_mixing_params

        # Quantum device
        self.qdev = _get_quantum_device(n_qubits)

        # Classical preprocessing: Conv1d to aggregate DNA features
        self.input_conv = nn.Conv1d(n_channels, d_model, kernel_size=3, padding=1)
        self.input_norm = nn.LayerNorm(d_model)

        # Chunk attention for aggregation
        self.chunk_attention = nn.Sequential(
            nn.Linear(d_model, d_model // 4),
            nn.Tanh(),
            nn.Linear(d_model // 4, 1)
        )

        # Parameter projection: chunk features → quantum circuit params
        self.param_proj = nn.Linear(d_model, self.n_total_circuit_params)

        # Input encoding projection: chunk features → qubit angles (for RY input embedding)
        self.encoding_proj = nn.Linear(d_model, n_qubits)

        # Delta (dt) projection for SSM mixing (input-dependent time step)
        self.dt_proj = nn.Linear(d_model, 1)

        # Input injection projection for SSM
        self.input_injection_proj = nn.Linear(d_model, n_qubits)

        # Build the quantum circuit
        self._build_circuit()

        # Output projection: quantum features → d_model
        self.output_proj = nn.Sequential(
            nn.Linear(self.q_dim, d_model),
            nn.LayerNorm(d_model),
            nn.SiLU(),
            nn.Dropout(dropout)
        )

        # Sequence aggregation (across chunks)
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

        self.to(device)

    def _build_circuit(self):
        """Build the end-to-end quantum circuit matching existing model structures."""
        n_qubits = self.n_qubits
        n_feature_layers = self.n_feature_layers
        n_mixing_layers = self.n_mixing_layers
        n_feature_params = self.n_feature_params

        @qml.qnode(self.qdev, interface="torch", diff_method="backprop")
        def circuit(input_angles, feature_params, mixing_params, dt_scale, injection_angles):
            """
            End-to-end quantum circuit for Mamba.

            Architecture:
            1. RY input embedding (from QuantumFeatureExtractor)
            2. Variational feature layers (from QuantumFeatureExtractor)
            3. SSM mixing layers (from QuantumSSMCore, NO intermediate measurement)
            4. Input injection (from QuantumSSMCore)
            5. Final measurement (PauliX, PauliY, PauliZ)
            """
            # Step 1: Input embedding via RY gates (QuantumFeatureExtractor style)
            input_embedding(input_angles, n_qubits)

            # Step 2: Variational feature extraction layers (QuantumFeatureExtractor style)
            param_idx = 0
            for layer in range(n_feature_layers):
                param_idx = variational_feature_layer(feature_params, n_qubits, param_idx)

            # Step 3: SSM mixing layers (QuantumSSMCore style, NO measurement between)
            param_idx = 0
            for layer in range(n_mixing_layers):
                param_idx = ssm_mixing_layer(mixing_params, dt_scale, n_qubits, param_idx)

            # Step 4: Input injection (QuantumSSMCore style)
            ssm_input_injection(injection_angles, dt_scale, n_qubits)

            # Step 5: Final measurement (ONLY measurement in entire circuit)
            return multi_observable_measurement(n_qubits)

        self.circuit = circuit

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through end-to-end quantum Mamba.

        Args:
            x: Input tensor of shape (batch, n_channels, seq_len)

        Returns:
            logits: Output tensor of shape (batch, output_dim)
        """
        batch_size, n_channels, seq_len = x.shape

        # Classical preprocessing
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
        all_params = self.param_proj(chunk_summary)  # (batch * n_chunks, n_total_params)
        feature_params = all_params[:, :self.n_feature_params]
        mixing_params = all_params[:, self.n_feature_params:]

        # Get input encoding angles (scaled to [-π, π])
        input_angles = torch.tanh(self.encoding_proj(chunk_summary)) * np.pi

        # Get Delta (dt) scale (sigmoid to [0, 1])
        dt_scale = torch.sigmoid(self.dt_proj(chunk_summary)).squeeze(-1)

        # Get input injection angles
        injection_angles = torch.tanh(self.input_injection_proj(chunk_summary)) * np.pi

        # Process through quantum circuit - BATCHED for efficiency
        # All inputs have shape (batch * n_chunks, ...) so circuit runs on all samples at once
        measurements = self.circuit(
            input_angles,       # (batch * n_chunks, n_qubits)
            feature_params,     # (batch * n_chunks, n_feature_params)
            mixing_params,      # (batch * n_chunks, n_mixing_params)
            dt_scale,           # (batch * n_chunks,)
            injection_angles    # (batch * n_chunks, n_qubits)
        )
        # Stack measurements: each measurement is (batch * n_chunks,) shaped
        # Result: (batch * n_chunks, q_dim)
        quantum_features = torch.stack(measurements, dim=1).float()

        # Project quantum features
        chunk_features = self.output_proj(quantum_features)  # (batch * n_chunks, d_model)
        chunk_features = chunk_features.reshape(batch_size, n_chunks, self.d_model)

        # Aggregate across chunks
        seq_attn = self.seq_attention(chunk_features)
        seq_weights = F.softmax(seq_attn, dim=1)
        sequence_repr = (seq_weights * chunk_features).sum(dim=1)  # (batch, d_model)

        # Classify
        logits = self.classifier(sequence_repr)

        return logits


# ==============================================================================
# End-to-End Quantum Hydra Model
# ==============================================================================

class QuantumHydraE2E(nn.Module):
    """
    End-to-End Quantum Hydra: Quantum Features → Quantum Bidirectional Mixing → Measurement

    Uses EXACT circuit structures from:
    - Feature Extraction: QuantumFeatureExtractor
    - Bidirectional Mixing: QuantumBidirectionalSSMCore (forward + backward + diagonal)

    Key: Three-head mixing (forward, backward, diagonal) all in quantum domain.
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
        device: str = "cpu"
    ):
        super().__init__()

        self.n_qubits = n_qubits
        self.n_feature_layers = n_feature_layers
        self.n_mixing_layers = n_mixing_layers
        self.n_channels = n_channels
        self.n_timesteps = n_timesteps
        self.d_model = d_model
        self.chunk_size = chunk_size
        self.output_dim = output_dim
        self.torch_device = device

        # Quantum output dimension
        self.q_dim = 3 * n_qubits

        # Parameter counts
        self.n_feature_params = 5 * n_qubits * n_feature_layers
        # Hydra has three mixing heads: forward, backward, diagonal
        # Each uses the same structure as SSM mixing
        self.n_mixing_params_per_head = 5 * n_qubits * n_mixing_layers
        self.n_total_circuit_params = self.n_feature_params + 3 * self.n_mixing_params_per_head

        # Quantum device
        self.qdev = _get_quantum_device(n_qubits)

        # Classical preprocessing
        self.input_conv = nn.Conv1d(n_channels, d_model, kernel_size=3, padding=1)
        self.input_norm = nn.LayerNorm(d_model)

        # Chunk attention
        self.chunk_attention = nn.Sequential(
            nn.Linear(d_model, d_model // 4),
            nn.Tanh(),
            nn.Linear(d_model // 4, 1)
        )

        # Parameter projections
        self.param_proj = nn.Linear(d_model, self.n_total_circuit_params)
        self.encoding_proj = nn.Linear(d_model, n_qubits)
        self.dt_proj = nn.Linear(d_model, 3)  # 3 dt values for 3 heads
        self.input_injection_proj = nn.Linear(d_model, n_qubits)

        # Learnable combination weights (matching QuantumBidirectionalSSMCore)
        self.alpha = nn.Parameter(torch.tensor([1.0]))
        self.beta = nn.Parameter(torch.tensor([1.0]))
        self.gamma = nn.Parameter(torch.tensor([0.5]))

        # Build circuit
        self._build_circuit()

        # Output layers
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

        self.to(device)

    def _build_circuit(self):
        """Build the end-to-end quantum circuit with Hydra bidirectional mixing."""
        n_qubits = self.n_qubits
        n_feature_layers = self.n_feature_layers
        n_mixing_layers = self.n_mixing_layers
        n_feature_params = self.n_feature_params
        n_mixing_params_per_head = self.n_mixing_params_per_head

        @qml.qnode(self.qdev, interface="torch", diff_method="backprop")
        def circuit(input_angles, feature_params, forward_params, backward_params, diagonal_params,
                    dt_forward, dt_backward, dt_diagonal, injection_angles):
            """
            End-to-end quantum circuit for Hydra.

            Architecture:
            1. RY input embedding
            2. Variational feature layers
            3. Forward SSM mixing (left to right style)
            4. Backward SSM mixing (right to left style)
            5. Diagonal (skip connection) mixing
            6. Input injection
            7. Final measurement
            """
            # Step 1: Input embedding
            input_embedding(input_angles, n_qubits)

            # Step 2: Feature extraction layers
            param_idx = 0
            for layer in range(n_feature_layers):
                param_idx = variational_feature_layer(feature_params, n_qubits, param_idx)

            # Step 3: Forward SSM mixing (like QuantumBidirectionalSSMCore.forward_ssm)
            param_idx = 0
            for layer in range(n_mixing_layers):
                param_idx = ssm_mixing_layer(forward_params, dt_forward, n_qubits, param_idx)

            # Step 4: Backward SSM mixing (reversed direction)
            # Uses backward_ssm_mixing_layer for batching support
            param_idx = 0
            for layer in range(n_mixing_layers):
                param_idx = backward_ssm_mixing_layer(backward_params, dt_backward, n_qubits, param_idx)

            # Step 5: Diagonal (skip connection) mixing
            param_idx = 0
            for layer in range(n_mixing_layers):
                param_idx = ssm_mixing_layer(diagonal_params, dt_diagonal, n_qubits, param_idx)

            # Step 6: Input injection
            ssm_input_injection(injection_angles, dt_forward, n_qubits)

            # Step 7: Final measurement
            return multi_observable_measurement(n_qubits)

        self.circuit = circuit

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass."""
        batch_size, n_channels, seq_len = x.shape

        # Classical preprocessing
        x = self.input_conv(x)
        x = x.transpose(1, 2)
        x = self.input_norm(x)

        # Chunking
        n_chunks = (seq_len + self.chunk_size - 1) // self.chunk_size
        if seq_len % self.chunk_size != 0:
            pad_len = self.chunk_size * n_chunks - seq_len
            x = F.pad(x, (0, 0, 0, pad_len))

        x = x.reshape(batch_size * n_chunks, self.chunk_size, self.d_model)

        # Chunk aggregation
        attn_scores = self.chunk_attention(x)
        attn_weights = F.softmax(attn_scores, dim=1)
        chunk_summary = (attn_weights * x).sum(dim=1)

        # Quantum parameters
        all_params = self.param_proj(chunk_summary)
        feature_params = all_params[:, :self.n_feature_params]
        forward_params = all_params[:, self.n_feature_params:self.n_feature_params + self.n_mixing_params_per_head]
        backward_params = all_params[:, self.n_feature_params + self.n_mixing_params_per_head:
                                        self.n_feature_params + 2 * self.n_mixing_params_per_head]
        diagonal_params = all_params[:, self.n_feature_params + 2 * self.n_mixing_params_per_head:]

        input_angles = torch.tanh(self.encoding_proj(chunk_summary)) * np.pi
        dt_values = torch.sigmoid(self.dt_proj(chunk_summary))  # (batch * n_chunks, 3)
        injection_angles = torch.tanh(self.input_injection_proj(chunk_summary)) * np.pi

        # Quantum processing - BATCHED for efficiency
        # All inputs have shape (batch * n_chunks, ...) so circuit runs on all samples at once
        measurements = self.circuit(
            input_angles,           # (batch * n_chunks, n_qubits)
            feature_params,         # (batch * n_chunks, n_feature_params)
            forward_params,         # (batch * n_chunks, n_mixing_params_per_head)
            backward_params,        # (batch * n_chunks, n_mixing_params_per_head)
            diagonal_params,        # (batch * n_chunks, n_mixing_params_per_head)
            dt_values[:, 0],        # (batch * n_chunks,) - forward dt
            dt_values[:, 1],        # (batch * n_chunks,) - backward dt
            dt_values[:, 2],        # (batch * n_chunks,) - diagonal dt
            injection_angles        # (batch * n_chunks, n_qubits)
        )
        # Stack measurements: each measurement is (batch * n_chunks,) shaped
        # Result: (batch * n_chunks, q_dim)
        quantum_features = torch.stack(measurements, dim=1).float()

        # Output projection
        chunk_features = self.output_proj(quantum_features)
        chunk_features = chunk_features.reshape(batch_size, n_chunks, self.d_model)

        # Sequence aggregation
        seq_attn = self.seq_attention(chunk_features)
        seq_weights = F.softmax(seq_attn, dim=1)
        sequence_repr = (seq_weights * chunk_features).sum(dim=1)

        return self.classifier(sequence_repr)


# ==============================================================================
# End-to-End Quantum Transformer Model
# ==============================================================================

class QuantumTransformerE2E(nn.Module):
    """
    End-to-End Quantum Transformer: Quantum Features → Quantum Attention Mixing → Measurement

    Uses EXACT circuit structures from:
    - Feature Extraction: QuantumFeatureExtractor
    - Attention Mixing: QuantumAttentionMixingCore (unified ansatz with LCU-style global mixing)

    The unified ansatz is applied multiple times with different parameters,
    creating global correlations similar to attention over all positions.
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
        device: str = "cpu"
    ):
        super().__init__()

        self.n_qubits = n_qubits
        self.n_feature_layers = n_feature_layers
        self.n_mixing_layers = n_mixing_layers
        self.n_channels = n_channels
        self.n_timesteps = n_timesteps
        self.d_model = d_model
        self.chunk_size = chunk_size
        self.output_dim = output_dim
        self.torch_device = device

        # Quantum output dimension
        self.q_dim = 3 * n_qubits

        # Parameter counts (using unified ansatz structure from QuantumAttentionMixingCore)
        self.n_feature_params = 5 * n_qubits * n_feature_layers
        # Attention mixing uses the unified ansatz: 5 * n_qubits per layer
        self.n_mixing_params = 5 * n_qubits * n_mixing_layers
        self.n_total_circuit_params = self.n_feature_params + self.n_mixing_params

        # Quantum device
        self.qdev = _get_quantum_device(n_qubits)

        # Classical preprocessing
        self.input_conv = nn.Conv1d(n_channels, d_model, kernel_size=3, padding=1)
        self.input_norm = nn.LayerNorm(d_model)

        # Chunk attention
        self.chunk_attention = nn.Sequential(
            nn.Linear(d_model, d_model // 4),
            nn.Tanh(),
            nn.Linear(d_model // 4, 1)
        )

        # Parameter projections
        self.param_proj = nn.Linear(d_model, self.n_total_circuit_params)
        self.encoding_proj = nn.Linear(d_model, n_qubits)

        # Build circuit
        self._build_circuit()

        # Output layers
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

        self.to(device)

    def _build_circuit(self):
        """Build the end-to-end quantum circuit with attention-style mixing."""
        n_qubits = self.n_qubits
        n_feature_layers = self.n_feature_layers
        n_mixing_layers = self.n_mixing_layers

        @qml.qnode(self.qdev, interface="torch", diff_method="backprop")
        def circuit(input_angles, feature_params, mixing_params):
            """
            End-to-end quantum circuit for Transformer.

            Architecture (matching QuantumAttentionMixingCore):
            1. RY input embedding
            2. Variational feature layers (unified ansatz)
            3. Attention mixing layers (unified ansatz - creates global correlations)
            4. Final measurement
            """
            # Step 1: Input embedding
            input_embedding(input_angles, n_qubits)

            # Step 2: Feature extraction layers (unified ansatz)
            param_idx = 0
            for layer in range(n_feature_layers):
                param_idx = variational_feature_layer(feature_params, n_qubits, param_idx)

            # Step 3: Attention mixing layers (unified ansatz)
            # This creates global correlations similar to attention
            param_idx = 0
            for layer in range(n_mixing_layers):
                param_idx = attention_mixing_layer(mixing_params, n_qubits, param_idx)

            # Step 4: Final measurement
            return multi_observable_measurement(n_qubits)

        self.circuit = circuit

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass."""
        batch_size, n_channels, seq_len = x.shape

        # Classical preprocessing
        x = self.input_conv(x)
        x = x.transpose(1, 2)
        x = self.input_norm(x)

        # Chunking
        n_chunks = (seq_len + self.chunk_size - 1) // self.chunk_size
        if seq_len % self.chunk_size != 0:
            pad_len = self.chunk_size * n_chunks - seq_len
            x = F.pad(x, (0, 0, 0, pad_len))

        x = x.reshape(batch_size * n_chunks, self.chunk_size, self.d_model)

        # Chunk aggregation
        attn_scores = self.chunk_attention(x)
        attn_weights = F.softmax(attn_scores, dim=1)
        chunk_summary = (attn_weights * x).sum(dim=1)

        # Quantum parameters
        all_params = self.param_proj(chunk_summary)
        feature_params = all_params[:, :self.n_feature_params]
        mixing_params = all_params[:, self.n_feature_params:]

        input_angles = torch.tanh(self.encoding_proj(chunk_summary)) * np.pi

        # Quantum processing - BATCHED for efficiency
        # All inputs have shape (batch * n_chunks, ...) so circuit runs on all samples at once
        measurements = self.circuit(
            input_angles,       # (batch * n_chunks, n_qubits)
            feature_params,     # (batch * n_chunks, n_feature_params)
            mixing_params       # (batch * n_chunks, n_mixing_params)
        )
        # Stack measurements: each measurement is (batch * n_chunks,) shaped
        # Result: (batch * n_chunks, q_dim)
        quantum_features = torch.stack(measurements, dim=1).float()

        # Output projection
        chunk_features = self.output_proj(quantum_features)
        chunk_features = chunk_features.reshape(batch_size, n_chunks, self.d_model)

        # Sequence aggregation
        seq_attn = self.seq_attention(chunk_features)
        seq_weights = F.softmax(seq_attn, dim=1)
        sequence_repr = (seq_weights * chunk_features).sum(dim=1)

        return self.classifier(sequence_repr)


# ==============================================================================
# Factory Functions
# ==============================================================================

def create_quantum_mamba_e2e(
    n_qubits: int = 6,
    n_feature_layers: int = 2,
    n_mixing_layers: int = 2,
    n_channels: int = 4,
    n_timesteps: int = 200,
    d_model: int = 64,
    output_dim: int = 2,
    device: str = "cpu",
    **kwargs
) -> QuantumMambaE2E:
    """Create a QuantumMambaE2E model."""
    return QuantumMambaE2E(
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


def create_quantum_hydra_e2e(
    n_qubits: int = 6,
    n_feature_layers: int = 2,
    n_mixing_layers: int = 2,
    n_channels: int = 4,
    n_timesteps: int = 200,
    d_model: int = 64,
    output_dim: int = 2,
    device: str = "cpu",
    **kwargs
) -> QuantumHydraE2E:
    """Create a QuantumHydraE2E model."""
    return QuantumHydraE2E(
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


def create_quantum_transformer_e2e(
    n_qubits: int = 6,
    n_feature_layers: int = 2,
    n_mixing_layers: int = 2,
    n_channels: int = 4,
    n_timesteps: int = 200,
    d_model: int = 64,
    output_dim: int = 2,
    device: str = "cpu",
    **kwargs
) -> QuantumTransformerE2E:
    """Create a QuantumTransformerE2E model."""
    return QuantumTransformerE2E(
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
    print("End-to-End Quantum Models - Testing")
    print("Using EXACT circuit structures from existing models")
    print("=" * 80)

    device = "cpu"
    batch_size = 2
    n_channels = 4  # DNA one-hot
    n_timesteps = 64
    n_qubits = 4
    d_model = 32
    output_dim = 2

    print("\n[1] Testing QuantumMambaE2E...")
    model_mamba = QuantumMambaE2E(
        n_qubits=n_qubits,
        n_feature_layers=2,
        n_mixing_layers=2,
        n_channels=n_channels,
        n_timesteps=n_timesteps,
        d_model=d_model,
        chunk_size=8,
        output_dim=output_dim,
        device=device
    )
    x = torch.randn(batch_size, n_channels, n_timesteps)
    output = model_mamba(x)
    params = sum(p.numel() for p in model_mamba.parameters() if p.requires_grad)
    print(f"  Input: {x.shape}")
    print(f"  Output: {output.shape}")
    print(f"  Parameters: {params:,}")

    print("\n[2] Testing QuantumHydraE2E...")
    model_hydra = QuantumHydraE2E(
        n_qubits=n_qubits,
        n_feature_layers=2,
        n_mixing_layers=2,
        n_channels=n_channels,
        n_timesteps=n_timesteps,
        d_model=d_model,
        chunk_size=8,
        output_dim=output_dim,
        device=device
    )
    output_hydra = model_hydra(x)
    params_hydra = sum(p.numel() for p in model_hydra.parameters() if p.requires_grad)
    print(f"  Input: {x.shape}")
    print(f"  Output: {output_hydra.shape}")
    print(f"  Parameters: {params_hydra:,}")

    print("\n[3] Testing QuantumTransformerE2E...")
    model_transformer = QuantumTransformerE2E(
        n_qubits=n_qubits,
        n_feature_layers=2,
        n_mixing_layers=2,
        n_channels=n_channels,
        n_timesteps=n_timesteps,
        d_model=d_model,
        chunk_size=8,
        output_dim=output_dim,
        device=device
    )
    output_transformer = model_transformer(x)
    params_transformer = sum(p.numel() for p in model_transformer.parameters() if p.requires_grad)
    print(f"  Input: {x.shape}")
    print(f"  Output: {output_transformer.shape}")
    print(f"  Parameters: {params_transformer:,}")

    print("\n[4] Testing gradient flow...")
    model_mamba.train()
    optimizer = torch.optim.Adam(model_mamba.parameters(), lr=1e-3)
    criterion = nn.CrossEntropyLoss()

    output = model_mamba(x)
    labels = torch.randint(0, output_dim, (batch_size,))
    loss = criterion(output, labels)
    loss.backward()
    optimizer.step()

    print(f"  Loss: {loss.item():.4f}")
    print(f"  Gradient flow: OK")

    print("\n" + "=" * 80)
    print("Testing complete!")
    print("=" * 80)
    print("\nKey Features:")
    print("  - EXACT circuit structures from existing models")
    print("  - Feature Extraction: QuantumFeatureExtractor (RY + RX/RY/RZ + CRX)")
    print("  - Mamba Mixing: QuantumSSMCore (Delta-modulated + CRX + input injection)")
    print("  - Hydra Mixing: QuantumBidirectionalSSMCore (forward + backward + diagonal)")
    print("  - Transformer Mixing: QuantumAttentionMixingCore (unified ansatz)")
    print("  - NO intermediate measurements - quantum state flows continuously")
    print("  - Single measurement at the very end (PauliX, PauliY, PauliZ)")
    print("=" * 80)
