"""
Quantum Mixing SSM Models - Classical Features -> Quantum Mixing

This module implements the inverse architecture of QuantumSSM.py:
- Current QuantumSSM: Quantum Feature Extraction -> Classical SSM Mixing
- This module: Classical Feature Extraction -> Quantum SSM Mixing

Key Design:
- ClassicalMambaQuantumSSM: Classical embedding + Quantum Mamba-style SSM
- ClassicalHydraQuantumSSM: Classical embedding + Quantum Hydra-style bidirectional SSM

The quantum mixing uses parameterized quantum circuits for the SSM operations:
- Input-dependent parameters (Delta, B, C) computed classically
- State transitions performed via quantum circuits
- Measurements provide SSM output

This creates a controlled comparison:
- QuantumMambaSSM/HydraSSM: Quantum at INPUT stage, Classical at MIXING stage
- ClassicalMambaQuantumSSM/HydraQuantumSSM: Classical at INPUT stage, Quantum at MIXING stage

Author: Junghoon Park
Date: December 2024
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import pennylane as qml
import numpy as np
from typing import Optional, Tuple, List


# ================================================================================
# Quantum Device Setup
# ================================================================================

def _get_quantum_device(n_qubits, batch_obs=True):
    """Get optimized quantum device."""
    return qml.device("default.qubit", wires=n_qubits)


# ================================================================================
# Classical Feature Extractor (IDENTICAL to TrueClassicalMamba's embedding)
# ================================================================================

class ClassicalFeatureExtractor(nn.Module):
    """
    Classical feature extraction module.

    IDENTICAL to TrueClassicalMamba/TrueClassicalHydra/ClassicalTransformer:
    - Simple linear embedding from input channels to d_model
    - Dropout for regularization

    This ensures fair comparison between:
    - Classical Features -> Classical Mixing (ClassicalMamba, ClassicalHydra, ClassicalTransformer)
    - Classical Features -> Quantum Mixing (ClassicalMambaQuantumSSM, ClassicalHydraQuantumSSM, ClassicalQuantumAttention)

    Note: Conv1d and LayerNorm removed for consistency with other classical models.
    The mixing mechanism (SSM/Attention) should handle sequence context, not feature extraction.
    """

    def __init__(
        self,
        n_channels: int,
        d_model: int,
        d_conv: int = 4,  # Kept for interface compatibility, but not used
        dropout: float = 0.1
    ):
        super().__init__()

        self.n_channels = n_channels
        self.d_model = d_model

        # Simple linear embedding (IDENTICAL to TrueClassicalMamba)
        self.embedding = nn.Linear(n_channels, d_model)

        # Dropout for regularization
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Extract classical features from input.

        Args:
            x: (batch, n_channels, n_timesteps) or (batch, n_timesteps, n_channels)

        Returns:
            features: (batch, n_timesteps, d_model)
        """
        # Handle input format
        if x.dim() == 2:
            x = x.unsqueeze(1)  # (batch, 1, features)

        # Ensure (batch, n_timesteps, n_channels)
        if x.shape[1] == self.n_channels:
            x = x.transpose(1, 2)

        # Simple linear embedding (identical to TrueClassicalMamba)
        x = self.embedding(x)  # (batch, seq_len, d_model)
        x = self.dropout(x)

        return x


# ================================================================================
# Quantum SSM Core - Quantum Mixing Module
# ================================================================================

class QuantumSSMCore(nn.Module):
    """
    Quantum State Space Model Core for mixing.

    Implements SSM operations using quantum circuits:
    - Input-dependent Delta, B, C computed classically
    - State transitions performed via parameterized quantum circuits
    - Measurements provide SSM output

    This is the QUANTUM version of MambaSelectiveSSM from QuantumSSM.py.

    SSM equations:
        h[t] = A_bar * h[t-1] + B_bar * u[t]  (quantum state transition)
        y[t] = C[t] * h[t] + D * u[t]

    The quantum circuit encodes the state transition matrix exp(Delta * A)
    using variational gates, enabling quantum superposition effects.
    """

    def __init__(
        self,
        d_model: int,
        d_state: int = 16,
        n_qubits: int = 6,
        n_layers: int = 2,
        dt_rank: str = "auto",
        dt_min: float = 0.001,
        dt_max: float = 0.1,
        device: str = "cpu"
    ):
        super().__init__()

        self.d_model = d_model
        self.d_state = d_state
        self.n_qubits = n_qubits
        self.n_layers = n_layers
        self.dt_rank = d_model // 16 if dt_rank == "auto" else dt_rank
        self.torch_device = device

        # ============================================
        # Classical projections for selective parameters
        # ============================================
        # Input -> (Delta_raw, B, C)
        self.x_proj = nn.Linear(d_model, self.dt_rank + d_state * 2, bias=False)

        # Delta projection
        self.dt_proj = nn.Linear(self.dt_rank, d_model, bias=True)

        # Initialize Delta bias for desired time step range
        dt = torch.exp(
            torch.rand(d_model) * (np.log(dt_max) - np.log(dt_min)) + np.log(dt_min)
        )
        inv_dt = dt + torch.log(-torch.expm1(-dt))
        with torch.no_grad():
            self.dt_proj.bias.copy_(inv_dt)

        # ============================================
        # Quantum circuit for state transition
        # ============================================
        self.qdev = _get_quantum_device(n_qubits)

        # Parameters: 5 per qubit per layer (RX, RY, RZ, CRX_fwd, CRX_bwd)
        self.n_circuit_params = 5 * n_qubits * n_layers
        self.circuit_params = nn.Parameter(
            torch.randn(self.n_circuit_params) * 0.1
        )

        # Projection from d_model to n_qubits for quantum input
        self.quantum_input_proj = nn.Linear(d_model, n_qubits)

        # Projection from quantum output (3 * n_qubits) to d_model
        self.quantum_output_proj = nn.Linear(3 * n_qubits, d_model)

        # D parameter (skip connection)
        self.D = nn.Parameter(torch.ones(d_model))

        # Output projection
        self.out_proj = nn.Linear(d_model, d_model)

        self._build_qnode()

    def _build_qnode(self):
        """Build quantum circuit for state transition with BATCHED support."""

        n_qubits = self.n_qubits
        n_layers = self.n_layers

        @qml.qnode(self.qdev, interface="torch", diff_method="best")
        def quantum_ssm_circuit(
            state_angles: torch.Tensor,
            input_angles: torch.Tensor,
            dt_scale: torch.Tensor,
            circuit_params: torch.Tensor
        ) -> List[torch.Tensor]:
            """
            Quantum SSM state transition circuit with BATCHED parameter support.

            Implements h[t] = f_quantum(h[t-1], u[t], Delta[t])

            The circuit:
            1. Encodes previous state via RY gates
            2. Applies Delta-modulated variational layers
            3. Injects input scaled by Delta
            4. Returns measurements for next state

            Args:
                state_angles: Previous state (n_qubits,) or (batch, n_qubits)
                input_angles: Current input (n_qubits,) or (batch, n_qubits)
                dt_scale: Delta scaling factor (1,) or (batch,)
                circuit_params: Variational parameters (n_params,)
            """
            # Check if inputs are batched
            is_batched = state_angles.ndim == 2

            # 1. Encode previous state
            for i in range(n_qubits):
                angle = state_angles[:, i] if is_batched else state_angles[i]
                qml.RY(angle, wires=i)

            # 2. Delta-modulated variational layers
            param_idx = 0
            for layer in range(n_layers):
                # Single-qubit rotations scaled by Delta
                for i in range(n_qubits):
                    dt = dt_scale if not is_batched else dt_scale
                    qml.RX(circuit_params[param_idx] * dt, wires=i)
                    qml.RY(circuit_params[param_idx + 1] * dt, wires=i)
                    qml.RZ(circuit_params[param_idx + 2] * dt, wires=i)
                    param_idx += 3

                # Forward entanglement
                for i in range(n_qubits):
                    qml.CRX(circuit_params[param_idx], wires=[i, (i + 1) % n_qubits])
                    param_idx += 1

                # Backward entanglement
                for i in range(n_qubits - 1, -1, -1):
                    qml.CRX(circuit_params[param_idx], wires=[i, (i - 1) % n_qubits])
                    param_idx += 1

            # 3. Input injection scaled by Delta
            for i in range(n_qubits):
                if is_batched:
                    angle = input_angles[:, i] * dt_scale
                else:
                    angle = input_angles[i] * dt_scale
                qml.RY(angle, wires=i)

            # 4. Measurements
            observables = [qml.PauliX(i) for i in range(n_qubits)] + \
                          [qml.PauliY(i) for i in range(n_qubits)] + \
                          [qml.PauliZ(i) for i in range(n_qubits)]
            return [qml.expval(op) for op in observables]

        self.qnode = quantum_ssm_circuit

    def forward(
        self,
        x: torch.Tensor,
        h: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Process sequence through quantum SSM.

        Args:
            x: (batch, seq_len, d_model) input sequence
            h: Optional initial hidden state

        Returns:
            y: (batch, seq_len, d_model) output sequence
            h_final: Final hidden state for continuation
        """
        batch_size, seq_len, _ = x.shape
        device = x.device

        # Compute selective parameters
        x_proj = self.x_proj(x)  # (batch, seq_len, dt_rank + 2*d_state)

        delta_raw = x_proj[..., :self.dt_rank]
        B = x_proj[..., self.dt_rank:self.dt_rank + self.d_state]
        C = x_proj[..., self.dt_rank + self.d_state:]

        # Compute Delta
        delta = F.softplus(self.dt_proj(delta_raw))  # (batch, seq_len, d_model)

        # Scale Delta to [0, 1] for quantum modulation
        delta_scale = torch.sigmoid(delta.mean(dim=-1, keepdim=True))  # (batch, seq_len, 1)

        # Project to quantum dimension
        x_quantum = torch.tanh(self.quantum_input_proj(x)) * np.pi  # (batch, seq_len, n_qubits)

        # Initialize hidden state
        if h is None:
            h = torch.zeros(batch_size, self.n_qubits, device=device)

        # Sequential quantum SSM processing with BATCHED circuit calls
        # Note: We still loop over timesteps (SSM is sequential), but process
        # entire batch in ONE circuit call per timestep
        outputs = []
        for t in range(seq_len):
            x_t = x_quantum[:, t, :]  # (batch, n_qubits)
            dt_t = delta_scale[:, t, 0]  # (batch,)

            # Process ENTIRE BATCH through quantum circuit in ONE call
            measurements = self.qnode(
                h,           # (batch, n_qubits)
                x_t,         # (batch, n_qubits)
                dt_t,        # (batch,)
                self.circuit_params
            )
            # measurements is a list of (batch,) tensors
            q_features = torch.stack(measurements, dim=1).float()  # (batch, 3*n_qubits)

            # Update state from Z measurements (last n_qubits measurements)
            h = torch.tanh(q_features[:, -self.n_qubits:]) * np.pi  # (batch, n_qubits)

            # Project quantum output back to d_model
            y_t = self.quantum_output_proj(q_features)  # (batch, d_model)

            # Add skip connection
            y_t = y_t + self.D * x[:, t, :]

            outputs.append(y_t)

        y = torch.stack(outputs, dim=1)  # (batch, seq_len, d_model)
        y = self.out_proj(y)

        return y, h


# ================================================================================
# Quantum Bidirectional SSM Core (Hydra-style)
# ================================================================================

class QuantumBidirectionalSSMCore(nn.Module):
    """
    Quantum Bidirectional SSM for Hydra-style mixing.

    Combines:
    - Forward quantum SSM pass
    - Backward quantum SSM pass
    - Diagonal (skip) connection

    Output: y = alpha * forward + beta * backward + gamma * diagonal
    """

    def __init__(
        self,
        d_model: int,
        d_state: int = 16,
        n_qubits: int = 6,
        n_layers: int = 2,
        dt_rank: str = "auto",
        device: str = "cpu"
    ):
        super().__init__()

        self.d_model = d_model

        # Forward and backward quantum SSM
        self.forward_ssm = QuantumSSMCore(
            d_model=d_model,
            d_state=d_state,
            n_qubits=n_qubits,
            n_layers=n_layers,
            dt_rank=dt_rank,
            device=device
        )

        self.backward_ssm = QuantumSSMCore(
            d_model=d_model,
            d_state=d_state,
            n_qubits=n_qubits,
            n_layers=n_layers,
            dt_rank=dt_rank,
            device=device
        )

        # Diagonal connection
        self.diagonal = nn.Linear(d_model, d_model)

        # Learnable combination weights
        self.alpha = nn.Parameter(torch.tensor([1.0]))
        self.beta = nn.Parameter(torch.tensor([1.0]))
        self.gamma = nn.Parameter(torch.tensor([0.5]))

        # Output projection
        self.out_proj = nn.Linear(d_model, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Bidirectional quantum SSM processing.

        Args:
            x: (batch, seq_len, d_model)

        Returns:
            y: (batch, seq_len, d_model)
        """
        # Forward pass
        forward_out, _ = self.forward_ssm(x)

        # Backward pass (reverse, process, reverse back)
        x_reversed = torch.flip(x, dims=[1])
        backward_out, _ = self.backward_ssm(x_reversed)
        backward_out = torch.flip(backward_out, dims=[1])

        # Diagonal
        diagonal_out = self.diagonal(x)

        # Normalize weights
        total = torch.abs(self.alpha) + torch.abs(self.beta) + torch.abs(self.gamma) + 1e-8
        alpha_norm = self.alpha / total
        beta_norm = self.beta / total
        gamma_norm = self.gamma / total

        # Combine
        y = alpha_norm * forward_out + beta_norm * backward_out + gamma_norm * diagonal_out
        y = self.out_proj(y)

        return y


# ================================================================================
# Full Models: ClassicalMambaQuantumSSM and ClassicalHydraQuantumSSM
# ================================================================================

class ClassicalMambaQuantumSSM(nn.Module):
    """
    Classical Feature Extraction + Quantum Mamba SSM Mixing.

    Architecture:
    1. Classical embedding (Linear + LayerNorm + Conv1D)
    2. Quantum SSM for mixing (parameterized quantum circuits)
    3. Classification head

    This is the inverse of QuantumMambaSSM which uses:
    1. Quantum feature extraction (QuantumSuperpositionBranches)
    2. Classical SSM for mixing (MambaBlock)

    The comparison between these models reveals whether quantum
    advantage comes from feature extraction or mixing.
    """

    def __init__(
        self,
        n_qubits: int = 6,
        n_timesteps: int = 200,
        qlcu_layers: int = 2,
        feature_dim: int = 4,
        d_model: int = 64,
        d_state: int = 16,
        n_layers: int = 1,
        output_dim: int = 2,
        dropout: float = 0.1,
        chunk_size: int = 16,
        device: str = "cpu"
    ):
        super().__init__()

        self.n_qubits = n_qubits
        self.n_timesteps = n_timesteps
        self.feature_dim = feature_dim
        self.d_model = d_model
        self.chunk_size = chunk_size
        self.device = torch.device(device) if isinstance(device, str) else device

        # 1. CLASSICAL Feature Extraction
        self.feature_extractor = ClassicalFeatureExtractor(
            n_channels=feature_dim,
            d_model=d_model,
            d_conv=4,
            dropout=dropout
        )

        # 2. Chunked processing to reduce quantum calls
        # Aggregate timesteps within chunks before quantum processing
        self.chunk_attention = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.Tanh(),
            nn.Linear(d_model // 2, 1)
        )

        # 3. QUANTUM SSM Mixing layers
        self.quantum_ssm_layers = nn.ModuleList([
            QuantumSSMCore(
                d_model=d_model,
                d_state=d_state,
                n_qubits=n_qubits,
                n_layers=qlcu_layers,
                device=device
            )
            for _ in range(n_layers)
        ])

        # 4. Output layers
        self.output_norm = nn.LayerNorm(d_model)
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.classifier = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, output_dim)
        )

        self.dropout = nn.Dropout(dropout)
        self.to(self.device)

    def _chunk_aggregate(self, x: torch.Tensor) -> torch.Tensor:
        """
        Aggregate timesteps within chunks using attention.

        Args:
            x: (batch, seq_len, d_model)

        Returns:
            chunks: (batch, n_chunks, d_model)
        """
        batch_size, seq_len, d_model = x.shape

        # Calculate number of chunks
        n_chunks = (seq_len + self.chunk_size - 1) // self.chunk_size

        # Pad if needed
        if seq_len % self.chunk_size != 0:
            pad_len = self.chunk_size * n_chunks - seq_len
            x = F.pad(x, (0, 0, 0, pad_len))

        # Reshape to chunks
        x_chunked = x.reshape(batch_size * n_chunks, self.chunk_size, d_model)

        # Attention aggregation
        attn_scores = self.chunk_attention(x_chunked)  # (B*n_chunks, chunk_size, 1)
        attn_weights = F.softmax(attn_scores, dim=1)
        chunk_summary = (attn_weights * x_chunked).sum(dim=1)  # (B*n_chunks, d_model)

        # Reshape back
        chunks = chunk_summary.reshape(batch_size, n_chunks, d_model)

        return chunks

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Args:
            x: (batch, feature_dim, n_timesteps) or (batch, n_timesteps, feature_dim)

        Returns:
            output: (batch, output_dim)
        """
        # 1. Classical feature extraction
        x = self.feature_extractor(x)  # (batch, seq_len, d_model)

        # 2. Chunk aggregation (reduces quantum calls)
        x = self._chunk_aggregate(x)  # (batch, n_chunks, d_model)
        x = self.dropout(x)

        # 3. Quantum SSM mixing
        h = None
        for quantum_layer in self.quantum_ssm_layers:
            x_new, h = quantum_layer(x, h)
            x = x + x_new  # Residual connection

        # 4. Output
        x = self.output_norm(x)
        x = x.transpose(1, 2)  # (batch, d_model, n_chunks)
        x = self.pool(x).squeeze(-1)  # (batch, d_model)
        output = self.classifier(x)

        return output


class ClassicalHydraQuantumSSM(nn.Module):
    """
    Classical Feature Extraction + Quantum Hydra (Bidirectional) SSM Mixing.

    Architecture:
    1. Classical embedding (Linear + LayerNorm + Conv1D)
    2. Quantum Bidirectional SSM for mixing (forward + backward + diagonal)
    3. Classification head

    This is the inverse of QuantumHydraSSM which uses:
    1. Quantum feature extraction (3x QuantumSuperpositionBranches)
    2. Classical bidirectional SSM for mixing (HydraBlock)
    """

    def __init__(
        self,
        n_qubits: int = 6,
        n_timesteps: int = 200,
        qlcu_layers: int = 2,
        feature_dim: int = 4,
        d_model: int = 64,
        d_state: int = 16,
        n_layers: int = 1,
        output_dim: int = 2,
        dropout: float = 0.1,
        chunk_size: int = 16,
        device: str = "cpu"
    ):
        super().__init__()

        self.n_qubits = n_qubits
        self.n_timesteps = n_timesteps
        self.feature_dim = feature_dim
        self.d_model = d_model
        self.chunk_size = chunk_size
        self.device = torch.device(device) if isinstance(device, str) else device

        # 1. CLASSICAL Feature Extraction
        self.feature_extractor = ClassicalFeatureExtractor(
            n_channels=feature_dim,
            d_model=d_model,
            d_conv=4,
            dropout=dropout
        )

        # 2. Chunk aggregation
        self.chunk_attention = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.Tanh(),
            nn.Linear(d_model // 2, 1)
        )

        # 3. QUANTUM Bidirectional SSM Mixing layers
        self.quantum_hydra_layers = nn.ModuleList([
            QuantumBidirectionalSSMCore(
                d_model=d_model,
                d_state=d_state,
                n_qubits=n_qubits,
                n_layers=qlcu_layers,
                device=device
            )
            for _ in range(n_layers)
        ])

        # 4. Output layers
        self.output_norm = nn.LayerNorm(d_model)
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.classifier = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, output_dim)
        )

        self.dropout = nn.Dropout(dropout)
        self.to(self.device)

    def _chunk_aggregate(self, x: torch.Tensor) -> torch.Tensor:
        """Aggregate timesteps within chunks using attention."""
        batch_size, seq_len, d_model = x.shape

        n_chunks = (seq_len + self.chunk_size - 1) // self.chunk_size

        if seq_len % self.chunk_size != 0:
            pad_len = self.chunk_size * n_chunks - seq_len
            x = F.pad(x, (0, 0, 0, pad_len))

        x_chunked = x.reshape(batch_size * n_chunks, self.chunk_size, d_model)

        attn_scores = self.chunk_attention(x_chunked)
        attn_weights = F.softmax(attn_scores, dim=1)
        chunk_summary = (attn_weights * x_chunked).sum(dim=1)

        chunks = chunk_summary.reshape(batch_size, n_chunks, d_model)

        return chunks

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Args:
            x: (batch, feature_dim, n_timesteps) or (batch, n_timesteps, feature_dim)

        Returns:
            output: (batch, output_dim)
        """
        # 1. Classical feature extraction
        x = self.feature_extractor(x)  # (batch, seq_len, d_model)

        # 2. Chunk aggregation
        x = self._chunk_aggregate(x)  # (batch, n_chunks, d_model)
        x = self.dropout(x)

        # 3. Quantum bidirectional SSM mixing
        for quantum_layer in self.quantum_hydra_layers:
            x_new = quantum_layer(x)
            x = x + x_new  # Residual connection

        # 4. Output
        x = self.output_norm(x)
        x = x.transpose(1, 2)  # (batch, d_model, n_chunks)
        x = self.pool(x).squeeze(-1)  # (batch, d_model)
        output = self.classifier(x)

        return output


# ================================================================================
# Quantum Attention Core for Classical -> Quantum Attention
# ================================================================================

class QuantumAttentionMixingCore(nn.Module):
    """
    Quantum Attention Core using LCU for global timestep mixing.

    This implements GLOBAL attention where all timesteps are mixed simultaneously
    via Linear Combination of Unitaries (LCU):
        |psi_mixed> = sum_t alpha_t U(theta_t)|psi>

    Key difference from SSM:
    - SSM: Sequential state evolution h[t] = f(h[t-1], x[t])
    - Attention: Global mixing of ALL timesteps at once

    Uses the same unified ansatz (5 params per qubit per layer) as SSM models.
    """

    def __init__(
        self,
        d_model: int,
        n_qubits: int = 6,
        n_layers: int = 2,
        n_chunks: int = 8,
        device: str = "cpu"
    ):
        super().__init__()

        self.d_model = d_model
        self.n_qubits = n_qubits
        self.n_layers = n_layers
        self.n_chunks = n_chunks
        self.torch_device = device

        # Output dimension: 3 * n_qubits (PauliX, PauliY, PauliZ)
        self.output_dim = 3 * n_qubits

        # Parameters per layer: 5 * n_qubits (RX, RY, RZ, CRX_fwd, CRX_bwd)
        self.n_params_per_layer = 5 * n_qubits
        self.n_circuit_params = self.n_params_per_layer * n_layers

        # Quantum device
        self.qdev = _get_quantum_device(n_qubits)

        # LCU mixing coefficients (complex for quantum superposition)
        self.mix_coeffs = nn.Parameter(
            torch.rand(n_chunks, dtype=torch.complex64)
        )

        # Projection from d_model to circuit parameters
        self.param_proj = nn.Linear(d_model, self.n_circuit_params)

        # Projection from quantum output to d_model
        self.output_proj = nn.Linear(self.output_dim, d_model)

        # QFF (Quantum Feed-Forward) parameters for final transformation
        self.qff_params = nn.Parameter(torch.randn(self.n_params_per_layer) * 0.1)

        self._build_qnodes()

    def _build_qnodes(self):
        """Build quantum circuits for attention mechanism with BATCHED support."""

        n_qubits = self.n_qubits
        n_layers = self.n_layers

        @qml.qnode(self.qdev, interface="torch", diff_method="backprop")
        def timestep_state_qnode(initial_state, params):
            """
            Apply unified ansatz circuit to transform quantum state.
            Supports BATCHED execution for efficiency.

            Args:
                initial_state: (2^n_qubits,) or (batch, 2^n_qubits) complex state
                params: (n_params,) or (batch, n_params) circuit parameters
            """
            qml.StatePrep(initial_state, wires=range(n_qubits))

            # Check if batched
            is_batched = params.ndim == 2

            # Unified ansatz
            param_idx = 0
            for _ in range(n_layers):
                # RX, RY, RZ rotations
                for i in range(n_qubits):
                    angle_x = params[:, param_idx] if is_batched else params[param_idx]
                    angle_y = params[:, param_idx + 1] if is_batched else params[param_idx + 1]
                    angle_z = params[:, param_idx + 2] if is_batched else params[param_idx + 2]
                    qml.RX(angle_x, wires=i)
                    qml.RY(angle_y, wires=i)
                    qml.RZ(angle_z, wires=i)
                    param_idx += 3

                # Forward CRX
                for i in range(n_qubits):
                    angle = params[:, param_idx] if is_batched else params[param_idx]
                    qml.CRX(angle, wires=[i, (i + 1) % n_qubits])
                    param_idx += 1

                # Backward CRX
                for i in range(n_qubits - 1, -1, -1):
                    angle = params[:, param_idx] if is_batched else params[param_idx]
                    qml.CRX(angle, wires=[i, (i - 1) % n_qubits])
                    param_idx += 1

            return qml.state()

        self.timestep_state_qnode = timestep_state_qnode

        @qml.qnode(self.qdev, interface="torch", diff_method="backprop")
        def qff_qnode(initial_state, params):
            """Final quantum feed-forward with multi-observable measurement."""
            qml.StatePrep(initial_state, wires=range(n_qubits))

            # Single layer unified ansatz (params are not batched for QFF)
            param_idx = 0
            for i in range(n_qubits):
                qml.RX(params[param_idx], wires=i)
                qml.RY(params[param_idx + 1], wires=i)
                qml.RZ(params[param_idx + 2], wires=i)
                param_idx += 3

            for i in range(n_qubits):
                qml.CRX(params[param_idx], wires=[i, (i + 1) % n_qubits])
                param_idx += 1

            for i in range(n_qubits - 1, -1, -1):
                qml.CRX(params[param_idx], wires=[i, (i - 1) % n_qubits])
                param_idx += 1

            # Multi-observable measurement
            observables = []
            for i in range(n_qubits):
                observables.append(qml.expval(qml.PauliX(i)))
            for i in range(n_qubits):
                observables.append(qml.expval(qml.PauliY(i)))
            for i in range(n_qubits):
                observables.append(qml.expval(qml.PauliZ(i)))

            return observables

        self.qff_qnode = qff_qnode

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass with global timestep attention via LCU.
        Uses VECTORIZED BATCHING for efficient quantum circuit execution.

        Args:
            x: (batch, n_chunks, d_model) chunk features

        Returns:
            output: (batch, d_model) attention-mixed features
        """
        batch_size, n_chunks, d_model = x.shape
        device = x.device
        state_dim = 2 ** self.n_qubits

        # Project to circuit parameters
        chunk_params = torch.sigmoid(self.param_proj(x))  # (batch, n_chunks, n_circuit_params)

        # Initialize |0>^n base state (same for all samples)
        base_state = torch.zeros(state_dim, dtype=torch.complex64, device=device)
        base_state[0] = 1.0

        # Flatten (batch × n_chunks) for vectorized QNode execution
        flat_params = chunk_params.reshape(batch_size * n_chunks, -1)  # (batch*n_chunks, n_circuit_params)
        repeated_base = base_state.unsqueeze(0).expand(batch_size * n_chunks, -1)  # (batch*n_chunks, 2^n_qubits)

        # Execute QNode ONCE with entire flattened batch
        evolved_states = self.timestep_state_qnode(repeated_base, flat_params)  # (batch*n_chunks, 2^n_qubits)

        # Reshape back to (batch, n_chunks, 2^n_qubits)
        evolved_states = evolved_states.reshape(batch_size, n_chunks, state_dim)

        # Normalize mixing coefficients
        coeffs = self.mix_coeffs[:n_chunks].to(device)
        coeffs_norm = coeffs / (torch.abs(coeffs).sum() + 1e-8)  # (n_chunks,)

        # Apply LCU: sum_t alpha_t |psi_t> for each batch sample
        # einsum: for each batch, sum over chunks weighted by coefficients
        mixed_states = torch.einsum('bti,t->bi', evolved_states.to(torch.complex64), coeffs_norm)  # (batch, 2^n_qubits)

        # Normalize mixed states
        norms = torch.linalg.vector_norm(mixed_states, dim=1, keepdim=True)
        mixed_states = mixed_states / (norms + 1e-9)

        # Apply QFF and measure for each batch sample
        # Note: QFF still needs per-sample execution due to StatePrep with different states
        batch_outputs = []
        for b in range(batch_size):
            expvals = self.qff_qnode(mixed_states[b], self.qff_params)
            q_out = torch.stack(expvals).float()
            batch_outputs.append(q_out)

        # Stack and project to d_model
        q_features = torch.stack(batch_outputs).to(device)  # (batch, 3*n_qubits)
        output = self.output_proj(q_features)  # (batch, d_model)

        return output


class ClassicalQuantumAttention(nn.Module):
    """
    Classical Feature Extraction + Quantum Attention Mixing.

    Architecture:
    1. Classical embedding (Linear + LayerNorm + Conv1D)
    2. Quantum Attention for mixing (LCU global timestep mixing)
    3. Classification head

    This is the inverse of QTSQuantumTransformer which uses:
    1. Classical feature extraction (QTSFeatureEncoder with Conv2D + GLU)
    2. Quantum Attention for mixing (same QuantumAttentionCore)

    Key difference from SSM models:
    - SSM: Sequential state evolution h[t] = f(h[t-1], x[t])
    - Attention: Global mixing of ALL timesteps via LCU
    """

    def __init__(
        self,
        n_qubits: int = 6,
        n_timesteps: int = 200,
        qlcu_layers: int = 2,
        feature_dim: int = 4,
        d_model: int = 64,
        d_state: int = 16,  # unused but kept for API consistency
        n_layers: int = 1,
        output_dim: int = 2,
        dropout: float = 0.1,
        chunk_size: int = 16,
        device: str = "cpu"
    ):
        super().__init__()

        self.n_qubits = n_qubits
        self.n_timesteps = n_timesteps
        self.feature_dim = feature_dim
        self.d_model = d_model
        self.chunk_size = chunk_size
        self.device = torch.device(device) if isinstance(device, str) else device

        # Calculate number of chunks
        self.n_chunks = (n_timesteps + chunk_size - 1) // chunk_size

        # 1. CLASSICAL Feature Extraction
        self.feature_extractor = ClassicalFeatureExtractor(
            n_channels=feature_dim,
            d_model=d_model,
            d_conv=4,
            dropout=dropout
        )

        # 2. Chunk aggregation
        self.chunk_attention = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.Tanh(),
            nn.Linear(d_model // 2, 1)
        )

        # 3. QUANTUM Attention Mixing
        self.quantum_attention = QuantumAttentionMixingCore(
            d_model=d_model,
            n_qubits=n_qubits,
            n_layers=qlcu_layers,
            n_chunks=self.n_chunks,
            device=device
        )

        # 4. Output layers
        self.output_norm = nn.LayerNorm(d_model)
        self.classifier = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, output_dim)
        )

        self.dropout = nn.Dropout(dropout)
        self.to(self.device)

    def _chunk_aggregate(self, x: torch.Tensor) -> torch.Tensor:
        """Aggregate timesteps within chunks using attention."""
        batch_size, seq_len, d_model = x.shape

        n_chunks = (seq_len + self.chunk_size - 1) // self.chunk_size

        if seq_len % self.chunk_size != 0:
            pad_len = self.chunk_size * n_chunks - seq_len
            x = F.pad(x, (0, 0, 0, pad_len))

        x_chunked = x.reshape(batch_size * n_chunks, self.chunk_size, d_model)

        attn_scores = self.chunk_attention(x_chunked)
        attn_weights = F.softmax(attn_scores, dim=1)
        chunk_summary = (attn_weights * x_chunked).sum(dim=1)

        chunks = chunk_summary.reshape(batch_size, n_chunks, d_model)

        return chunks

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Args:
            x: (batch, feature_dim, n_timesteps) or (batch, n_timesteps, feature_dim)

        Returns:
            output: (batch, output_dim)
        """
        # 1. Classical feature extraction
        x = self.feature_extractor(x)  # (batch, seq_len, d_model)

        # 2. Chunk aggregation
        x = self._chunk_aggregate(x)  # (batch, n_chunks, d_model)
        x = self.dropout(x)

        # 3. Quantum attention mixing (global)
        x = self.quantum_attention(x)  # (batch, d_model)

        # 4. Output
        x = self.output_norm(x)
        output = self.classifier(x)

        return output


# ================================================================================
# Testing
# ================================================================================

if __name__ == "__main__":
    print("=" * 80)
    print("Quantum Mixing SSM - Classical Features -> Quantum Mixing")
    print("=" * 80)

    device = "cpu"
    batch_size = 4
    n_timesteps = 64
    feature_dim = 4
    n_qubits = 4
    d_model = 32
    output_dim = 2

    # Test ClassicalFeatureExtractor
    print("\n[1] Testing ClassicalFeatureExtractor...")
    extractor = ClassicalFeatureExtractor(
        n_channels=feature_dim,
        d_model=d_model
    )
    x_input = torch.randn(batch_size, feature_dim, n_timesteps)
    features = extractor(x_input)
    print(f"    Input: {x_input.shape}")
    print(f"    Output: {features.shape}")

    # Test QuantumSSMCore
    print("\n[2] Testing QuantumSSMCore...")
    ssm_core = QuantumSSMCore(
        d_model=d_model,
        n_qubits=n_qubits,
        n_layers=2,
        device=device
    )
    y, h = ssm_core(features)
    print(f"    Input: {features.shape}")
    print(f"    Output: {y.shape}")
    print(f"    Hidden: {h.shape}")

    # Test ClassicalMambaQuantumSSM
    print("\n[3] Testing ClassicalMambaQuantumSSM (full model)...")
    model_mamba = ClassicalMambaQuantumSSM(
        n_qubits=n_qubits,
        n_timesteps=n_timesteps,
        qlcu_layers=2,
        feature_dim=feature_dim,
        d_model=d_model,
        n_layers=1,
        output_dim=output_dim,
        device=device
    )
    output = model_mamba(x_input)
    print(f"    Input: {x_input.shape}")
    print(f"    Output: {output.shape}")

    total_params = sum(p.numel() for p in model_mamba.parameters() if p.requires_grad)
    print(f"    Parameters: {total_params:,}")

    # Test ClassicalHydraQuantumSSM
    print("\n[4] Testing ClassicalHydraQuantumSSM (full model)...")
    model_hydra = ClassicalHydraQuantumSSM(
        n_qubits=n_qubits,
        n_timesteps=n_timesteps,
        qlcu_layers=2,
        feature_dim=feature_dim,
        d_model=d_model,
        n_layers=1,
        output_dim=output_dim,
        device=device
    )
    output_hydra = model_hydra(x_input)
    print(f"    Input: {x_input.shape}")
    print(f"    Output: {output_hydra.shape}")

    total_params = sum(p.numel() for p in model_hydra.parameters() if p.requires_grad)
    print(f"    Parameters: {total_params:,}")

    # Test gradient flow
    print("\n[5] Testing gradient flow...")
    model_mamba.train()
    optimizer = torch.optim.Adam(model_mamba.parameters(), lr=1e-3)
    criterion = nn.CrossEntropyLoss()

    output = model_mamba(x_input)
    loss = criterion(output, torch.randint(0, output_dim, (batch_size,)))
    loss.backward()
    optimizer.step()

    print(f"    Loss: {loss.item():.4f}")
    print(f"    Gradient flow: OK")

    print("\n" + "=" * 80)
    print("Testing complete!")
    print("=" * 80)
    print("\nKey Features:")
    print("  [ClassicalMambaQuantumSSM]")
    print("    - Classical embedding + Conv1D for feature extraction")
    print("    - Quantum circuits for SSM mixing")
    print("    - Input-dependent Delta modulates quantum gates")
    print("  [ClassicalHydraQuantumSSM]")
    print("    - Classical embedding + Conv1D for feature extraction")
    print("    - Quantum bidirectional SSM (forward + backward + diagonal)")
    print("    - Complex coefficient combination")
    print("=" * 80)
