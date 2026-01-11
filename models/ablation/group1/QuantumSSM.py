"""
Quantum SSM Models with Theoretically-Aligned Architectures

This module implements quantum versions of Mamba and Hydra with proper
state space mechanisms, replacing the outdated LSTM-style gating.

Key Design Principles:
- QuantumMambaSSM: Uses Mamba-style selective SSM with input-dependent Δ, B, C
- QuantumHydraSSM: Uses Hydra-style bidirectional SSM (forward + backward + diagonal)

References:
- Mamba: Gu & Dao (2024) "Mamba: Linear-Time Sequence Modeling with Selective State Spaces"
- Hydra: Hwang et al. (2024) "Hydra: Bidirectional State Space Models"

Author: Junghoon Park
Date: December 2024
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import pennylane as qml
import numpy as np
from typing import Optional, Tuple


# ================================================================================
# Quantum Feature Extraction (Preserved from original)
# ================================================================================

class QuantumFeatureExtractor(nn.Module):
    """
    Quantum circuit for feature extraction using variational quantum circuit.

    Implements a parameterized quantum circuit with:
    - Angle embedding for input features
    - Multiple layers of parameterized rotations and entanglement
    - Expectation value measurements for output

    Supports shared quantum device to avoid GPU resource contention.
    """

    def __init__(self, n_qubits: int, n_layers: int, device: str = "cpu", shared_qdev=None):
        super().__init__()

        self.n_qubits = n_qubits
        self.n_layers = n_layers
        # Unified ansatz: RX, RY, RZ (3) + Forward CRX (1) + Backward CRX (1) = 5 per qubit per layer
        self.n_params = n_qubits * n_layers * 5

        # Device setup for PennyLane
        self.quantum_device = device

        # Use shared quantum device if provided (avoids resource contention)
        if shared_qdev is not None:
            self.dev = shared_qdev
        else:
            # Use default.qubit for maximum stability and compatibility
            self.dev = qml.device("default.qubit", wires=n_qubits)

        # Build the quantum node with best differentiation method for device compatibility
        self.qnode = qml.QNode(self._circuit, self.dev, interface="torch", diff_method="best")

    def _circuit(self, params, inputs):
        """
        Variational quantum circuit with native batching support.

        UNIFIED ANSATZ (matching all other quantum models):
        - RY input embedding
        - Per layer: RX, RY, RZ (3 * n_qubits) + Forward CRX (n_qubits) + Backward CRX (n_qubits)
        - Multi-observable measurement: PauliX, PauliY, PauliZ

        Args:
            params: (batch, n_params) or (n_params,) circuit parameters
            inputs: (batch, input_dim) or (input_dim,) input features
        """
        is_batched = params.ndim == 2

        # Angle embedding via RY gates
        for i in range(self.n_qubits):
            idx = i % inputs.shape[-1]
            angle = inputs[:, idx] if is_batched else inputs[idx]
            qml.RY(angle, wires=i)

        # Variational layers with unified ansatz
        param_idx = 0
        for layer in range(self.n_layers):
            # Single-qubit rotations: RX, RY, RZ
            for i in range(self.n_qubits):
                angle_x = params[:, param_idx] if is_batched else params[param_idx]
                angle_y = params[:, param_idx + 1] if is_batched else params[param_idx + 1]
                angle_z = params[:, param_idx + 2] if is_batched else params[param_idx + 2]
                qml.RX(angle_x, wires=i)
                qml.RY(angle_y, wires=i)
                qml.RZ(angle_z, wires=i)
                param_idx += 3

            # Forward CRX entanglement (ring topology)
            for i in range(self.n_qubits):
                angle = params[:, param_idx] if is_batched else params[param_idx]
                qml.CRX(angle, wires=[i, (i + 1) % self.n_qubits])
                param_idx += 1

            # Backward CRX entanglement (ring topology)
            for i in range(self.n_qubits - 1, -1, -1):
                angle = params[:, param_idx] if is_batched else params[param_idx]
                qml.CRX(angle, wires=[i, (i - 1) % self.n_qubits])
                param_idx += 1

        # Multi-observable measurement: PauliX, PauliY, PauliZ on all qubits
        measurements = []
        for i in range(self.n_qubits):
            measurements.append(qml.expval(qml.PauliX(i)))
        for i in range(self.n_qubits):
            measurements.append(qml.expval(qml.PauliY(i)))
        for i in range(self.n_qubits):
            measurements.append(qml.expval(qml.PauliZ(i)))
        return measurements

    def forward(self, params: torch.Tensor, inputs: torch.Tensor) -> torch.Tensor:
        """
        Batched forward pass through quantum circuit.

        Args:
            params: (batch, n_params) circuit parameters
            inputs: (batch, input_dim) input features

        Returns:
            (batch, 3 * n_qubits) expectation values (PauliX, PauliY, PauliZ)
        """
        # Execute circuit with batched inputs (no for-loop!)
        measurements = self.qnode(params, inputs)
        # Stack measurements: list of (batch,) tensors -> (batch, 3 * n_qubits)
        return torch.stack(measurements, dim=-1)


class QuantumSuperpositionBranches(nn.Module):
    """
    Three-branch quantum superposition with trainable complex coefficients.

    Implements: |ψ⟩ = α|ψ₁⟩ + β|ψ₂⟩ + γ|ψ₃⟩

    Each branch uses a separate quantum circuit with independent parameters.

    Uses a SINGLE SHARED quantum device across all branches to avoid GPU
    resource contention that can cause hangs with multiple lightning.gpu devices.
    """

    def __init__(
        self,
        n_qubits: int,
        n_layers: int,
        feature_dim: int,
        device: str = "cpu"
    ):
        super().__init__()

        self.n_qubits = n_qubits
        self.feature_dim = feature_dim
        # Output dimension per branch: 3 * n_qubits (PauliX, PauliY, PauliZ)
        self.q_dim = 3 * n_qubits

        # Calculate parameters needed: unified ansatz = 5 * n_qubits per layer
        n_params = n_qubits * n_layers * 5

        # Parameter projections for each branch
        self.proj1 = nn.Linear(feature_dim, n_params)
        self.proj2 = nn.Linear(feature_dim, n_params)
        self.proj3 = nn.Linear(feature_dim, n_params)

        # Input projections to match qubit count
        self.input_proj = nn.Linear(feature_dim, n_qubits)

        # Create ONE SHARED quantum device for all branches
        # Use default.qubit for maximum stability and compatibility
        self.shared_qdev = qml.device("default.qubit", wires=n_qubits)

        # Quantum circuits for each branch sharing the SAME quantum device
        self.branch1 = QuantumFeatureExtractor(n_qubits, n_layers, device, shared_qdev=self.shared_qdev)
        self.branch2 = QuantumFeatureExtractor(n_qubits, n_layers, device, shared_qdev=self.shared_qdev)
        self.branch3 = QuantumFeatureExtractor(n_qubits, n_layers, device, shared_qdev=self.shared_qdev)

        # Trainable complex coefficients
        self.alpha_real = nn.Parameter(torch.rand(1) * 0.5 + 0.25)
        self.alpha_imag = nn.Parameter(torch.zeros(1))
        self.beta_real = nn.Parameter(torch.rand(1) * 0.5 + 0.25)
        self.beta_imag = nn.Parameter(torch.zeros(1))
        self.gamma_real = nn.Parameter(torch.rand(1) * 0.5 + 0.25)
        self.gamma_imag = nn.Parameter(torch.zeros(1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch, feature_dim) input features

        Returns:
            (batch, 9 * n_qubits) combined quantum features from 3 branches
        """
        # Project input to qubit dimension
        x_q = torch.tanh(self.input_proj(x)) * np.pi

        # Get parameters for each branch
        params1 = self.proj1(x)
        params2 = self.proj2(x)
        params3 = self.proj3(x)

        # Execute quantum circuits (outputs are on CPU from PennyLane)
        # Each branch outputs (batch, 3 * n_qubits) with unified ansatz
        out1 = self.branch1(params1, x_q)  # (batch, 3 * n_qubits)
        out2 = self.branch2(params2, x_q)  # (batch, 3 * n_qubits)
        out3 = self.branch3(params3, x_q)  # (batch, 3 * n_qubits)

        # Move quantum outputs to the same device as model parameters
        target_device = self.alpha_real.device
        out1 = out1.to(target_device)
        out2 = out2.to(target_device)
        out3 = out3.to(target_device)

        # Complex coefficients
        alpha = torch.complex(self.alpha_real, self.alpha_imag)
        beta = torch.complex(self.beta_real, self.beta_imag)
        gamma = torch.complex(self.gamma_real, self.gamma_imag)

        # Normalize coefficients (|α|² + |β|² + |γ|² = 1)
        norm = torch.sqrt(
            torch.abs(alpha)**2 + torch.abs(beta)**2 + torch.abs(gamma)**2 + 1e-8
        )
        alpha = alpha / norm
        beta = beta / norm
        gamma = gamma / norm

        # Combine via superposition (using magnitudes for real output)
        combined = torch.cat([
            torch.abs(alpha) * out1,
            torch.abs(beta) * out2,
            torch.abs(gamma) * out3
        ], dim=1)

        return combined


# ================================================================================
# Chunked Quantum Processor with Attention-Based Aggregation
# ================================================================================

class ChunkedQuantumProcessor(nn.Module):
    """
    Efficient chunked quantum processing - AGGREGATE FIRST, then quantum.

    Key insight: Instead of calling quantum circuits for every timestep (expensive!),
    we first aggregate timesteps classically within each chunk, THEN process only
    the aggregated chunk representations through quantum circuits.

    This reduces quantum calls from (batch * seq_len) to (batch * n_chunks):
    - Old: 32 * 200 = 6400 quantum calls per batch
    - New: 32 * 7 = 224 quantum calls per batch (28x faster!)

    Architecture:
    1. Divide sequence into chunks
    2. Aggregate timesteps within each chunk using attention (classical)
    3. Process chunk summaries through quantum circuits (much fewer calls!)
    4. Return chunk-level quantum features for SSM processing
    """

    def __init__(
        self,
        n_qubits: int,
        n_layers: int,
        feature_dim: int,
        hidden_dim: int,
        chunk_size: int = 32,
        device: str = "cpu"
    ):
        super().__init__()

        self.n_qubits = n_qubits
        self.feature_dim = feature_dim
        self.hidden_dim = hidden_dim
        self.chunk_size = chunk_size
        # Output from three-branch superposition: 3 branches × 3 measurements × n_qubits
        self.q_dim = 9 * n_qubits

        # Classical attention-based aggregation BEFORE quantum processing
        # This reduces the number of quantum calls dramatically
        self.chunk_attention = nn.Sequential(
            nn.Linear(feature_dim, feature_dim // 2),
            nn.Tanh(),
            nn.Linear(feature_dim // 2, 1)
        )

        # Quantum superposition branches - now only called once per chunk!
        self.quantum_branches = QuantumSuperpositionBranches(
            n_qubits=n_qubits,
            n_layers=n_layers,
            feature_dim=feature_dim,  # Takes aggregated chunk features
            device=device
        )

        # Project quantum features to hidden dimension
        self.output_proj = nn.Sequential(
            nn.Linear(self.q_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.SiLU()
        )

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Process sequence with classical aggregation first, then quantum.

        Args:
            x: (batch, seq_len, feature_dim) input sequence

        Returns:
            chunk_features: (batch, n_chunks, hidden_dim) quantum chunk features
            chunk_aggregated: (batch, n_chunks, feature_dim) pre-quantum aggregated features
        """
        batch_size, seq_len, _ = x.shape

        # Calculate number of chunks
        n_chunks = (seq_len + self.chunk_size - 1) // self.chunk_size

        # Pad sequence if needed
        if seq_len % self.chunk_size != 0:
            pad_len = self.chunk_size * n_chunks - seq_len
            x = F.pad(x, (0, 0, 0, pad_len))

        # Reshape to (batch * n_chunks, chunk_size, feature_dim)
        x_chunked = x.reshape(batch_size * n_chunks, self.chunk_size, self.feature_dim)

        # STEP 1: Classical attention-based aggregation within chunks
        # Compute attention scores for each timestep in the chunk
        attn_scores = self.chunk_attention(x_chunked)  # (B * n_chunks, chunk_size, 1)
        attn_weights = F.softmax(attn_scores, dim=1)

        # Weighted aggregation: (B * n_chunks, feature_dim)
        chunk_summary = (attn_weights * x_chunked).sum(dim=1)

        # STEP 2: Process aggregated chunks through quantum circuits
        # Now we only have (batch * n_chunks) quantum calls instead of (batch * seq_len)!
        q_features = self.quantum_branches(chunk_summary)  # (B * n_chunks, q_dim)

        # Project to hidden dimension (convert to float to match linear layer dtype)
        chunk_features = self.output_proj(q_features.float())  # (B * n_chunks, hidden_dim)

        # Reshape to (batch, n_chunks, hidden_dim)
        chunk_features = chunk_features.reshape(batch_size, n_chunks, self.hidden_dim)
        chunk_summary = chunk_summary.reshape(batch_size, n_chunks, self.feature_dim)

        return chunk_features, chunk_summary


# ================================================================================
# Mamba-Style Selective State Space
# ================================================================================

class MambaSelectiveSSM(nn.Module):
    """
    Mamba-style Selective State Space Model.

    Key innovation: Input-dependent Δ (time step), B (input matrix), C (output matrix)
    This allows the model to selectively propagate or forget information.

    Continuous SSM: dx/dt = Ax + Bu, y = Cx + Du
    Discretized:    x_k = Ā x_{k-1} + B̄ u_k, y_k = C x_k + D u_k
    where Ā = exp(ΔA), B̄ = (ΔA)^{-1}(exp(ΔA) - I) · ΔB ≈ ΔB

    Reference: Gu & Dao (2024), Section 3.2
    """

    def __init__(
        self,
        d_model: int,
        d_state: int = 16,
        dt_rank: str = "auto",
        dt_min: float = 0.001,
        dt_max: float = 0.1,
        dt_init: str = "random",
        dt_scale: float = 1.0,
        bias: bool = False,
    ):
        super().__init__()

        self.d_model = d_model
        self.d_state = d_state
        self.dt_rank = d_model // 16 if dt_rank == "auto" else dt_rank

        # Input projection: x -> (Δ_raw, B, C)
        # Δ_raw has dt_rank dimensions, B and C have d_state dimensions each
        self.x_proj = nn.Linear(d_model, self.dt_rank + d_state * 2, bias=False)

        # Δ projection: dt_rank -> d_model (applied after softplus)
        self.dt_proj = nn.Linear(self.dt_rank, d_model, bias=True)

        # Initialize Δ projection bias for desired time step range
        dt_init_std = self.dt_rank ** -0.5 * dt_scale
        if dt_init == "constant":
            nn.init.constant_(self.dt_proj.weight, dt_init_std)
        elif dt_init == "random":
            nn.init.uniform_(self.dt_proj.weight, -dt_init_std, dt_init_std)

        # Initialize bias to achieve time steps in [dt_min, dt_max]
        dt = torch.exp(
            torch.rand(d_model) * (np.log(dt_max) - np.log(dt_min)) + np.log(dt_min)
        )
        inv_dt = dt + torch.log(-torch.expm1(-dt))  # softplus^{-1}
        with torch.no_grad():
            self.dt_proj.bias.copy_(inv_dt)

        # A parameter (structured as log for numerical stability)
        # HiPPO-style initialization: A_ii = -(i+1)
        A = torch.arange(1, d_state + 1, dtype=torch.float32)
        self.A_log = nn.Parameter(torch.log(A))

        # D parameter (skip connection)
        self.D = nn.Parameter(torch.ones(d_model))

        # Output projection
        self.out_proj = nn.Linear(d_model, d_model, bias=bias)

    def forward(
        self,
        x: torch.Tensor,
        h: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Process a single timestep with selective SSM.

        Args:
            x: (batch, d_model) input at current timestep
            h: (batch, d_model, d_state) previous hidden state

        Returns:
            y: (batch, d_model) output
            h_new: (batch, d_model, d_state) new hidden state
        """
        batch_size = x.shape[0]

        # Initialize hidden state if needed
        if h is None:
            h = torch.zeros(batch_size, self.d_model, self.d_state, device=x.device)

        # Input-dependent projections: Δ, B, C
        x_proj = self.x_proj(x)  # (batch, dt_rank + 2*d_state)

        delta_raw = x_proj[:, :self.dt_rank]  # (batch, dt_rank)
        B = x_proj[:, self.dt_rank:self.dt_rank + self.d_state]  # (batch, d_state)
        C = x_proj[:, self.dt_rank + self.d_state:]  # (batch, d_state)

        # Compute Δ via projection and softplus
        delta = F.softplus(self.dt_proj(delta_raw))  # (batch, d_model)

        # Get A (always negative for stability)
        A = -torch.exp(self.A_log)  # (d_state,)

        # Discretization: Ā = exp(Δ * A)
        # delta: (batch, d_model), A: (d_state,)
        # We need A_bar: (batch, d_model, d_state)
        delta_A = delta.unsqueeze(-1) * A.unsqueeze(0).unsqueeze(0)  # (batch, d_model, d_state)
        A_bar = torch.exp(delta_A)

        # B̄ ≈ Δ * B (simplified discretization)
        # delta: (batch, d_model), B: (batch, d_state)
        B_bar = delta.unsqueeze(-1) * B.unsqueeze(1)  # (batch, d_model, d_state)

        # SSM recurrence: h_new = Ā * h + B̄ * x
        # h: (batch, d_model, d_state)
        # x: (batch, d_model) -> need to expand
        x_expanded = x.unsqueeze(-1)  # (batch, d_model, 1)
        h_new = A_bar * h + B_bar * x_expanded  # (batch, d_model, d_state)

        # Output: y = C * h + D * x
        # C: (batch, d_state), h_new: (batch, d_model, d_state)
        y = torch.einsum('bd,bmd->bm', C, h_new)  # (batch, d_model)
        y = y + self.D * x  # Skip connection

        # Output projection
        y = self.out_proj(y)

        return y, h_new


class MambaBlock(nn.Module):
    """
    Complete Mamba block with gated MLP and selective SSM.

    Architecture:
    1. Expand: d_model -> 2*d_model (via linear)
    2. Split into two paths: conv path and gate path
    3. Conv path: Conv1D -> SiLU -> SSM
    4. Gate path: SiLU activation
    5. Combine: conv_out * gate
    6. Project back: 2*d_model -> d_model
    """

    def __init__(
        self,
        d_model: int,
        d_state: int = 16,
        d_conv: int = 4,
        expand: int = 2,
        dt_rank: str = "auto",
        bias: bool = False,
    ):
        super().__init__()

        self.d_model = d_model
        self.d_inner = d_model * expand
        self.d_conv = d_conv

        # Input projection (expansion)
        self.in_proj = nn.Linear(d_model, self.d_inner * 2, bias=bias)

        # Convolution (for local context)
        self.conv1d = nn.Conv1d(
            self.d_inner,
            self.d_inner,
            kernel_size=d_conv,
            padding=d_conv - 1,
            groups=self.d_inner,
            bias=True
        )

        # Selective SSM
        self.ssm = MambaSelectiveSSM(
            d_model=self.d_inner,
            d_state=d_state,
            dt_rank=dt_rank,
            bias=bias
        )

        # Output projection
        self.out_proj = nn.Linear(self.d_inner, d_model, bias=bias)

        # Layer norm
        self.norm = nn.LayerNorm(d_model)

    def forward(
        self,
        x: torch.Tensor,
        h: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            x: (batch, seq_len, d_model) or (batch, d_model) for single step

        Returns:
            y: same shape as x
            h_new: hidden state
        """
        is_single_step = x.dim() == 2
        if is_single_step:
            x = x.unsqueeze(1)  # (batch, 1, d_model)

        batch_size, seq_len, _ = x.shape

        # Input projection
        x_proj = self.in_proj(x)  # (batch, seq_len, 2*d_inner)
        x_conv, x_gate = x_proj.chunk(2, dim=-1)  # Each: (batch, seq_len, d_inner)

        # Convolution path
        x_conv = x_conv.transpose(1, 2)  # (batch, d_inner, seq_len)
        x_conv = self.conv1d(x_conv)[:, :, :seq_len]  # Causal conv
        x_conv = x_conv.transpose(1, 2)  # (batch, seq_len, d_inner)
        x_conv = F.silu(x_conv)

        # SSM path (process sequentially)
        if h is None:
            h = torch.zeros(
                batch_size, self.d_inner, self.ssm.d_state,
                device=x.device
            )

        ssm_outputs = []
        for t in range(seq_len):
            y_t, h = self.ssm(x_conv[:, t, :], h)
            ssm_outputs.append(y_t)

        ssm_out = torch.stack(ssm_outputs, dim=1)  # (batch, seq_len, d_inner)

        # Gate path
        gate = F.silu(x_gate)

        # Combine and project
        y = ssm_out * gate
        y = self.out_proj(y)

        if is_single_step:
            y = y.squeeze(1)

        return y, h


# ================================================================================
# Hydra-Style Bidirectional State Space
# ================================================================================

class HydraBidirectionalSSM(nn.Module):
    """
    Hydra-style Bidirectional State Space Model.

    Key innovation: Combines forward SSM, backward SSM, and diagonal for
    bidirectional sequence modeling without attention's quadratic complexity.

    Output: y = α * SS_forward(x) + β * SS_backward(x) + γ * D(x)

    Reference: Hwang et al. (2024) "Hydra: Bidirectional State Space Models"
    """

    def __init__(
        self,
        d_model: int,
        d_state: int = 16,
        dt_rank: str = "auto",
        bias: bool = False,
    ):
        super().__init__()

        self.d_model = d_model
        self.d_state = d_state

        # Forward SSM
        self.forward_ssm = MambaSelectiveSSM(
            d_model=d_model,
            d_state=d_state,
            dt_rank=dt_rank,
            bias=bias
        )

        # Backward SSM (separate parameters)
        self.backward_ssm = MambaSelectiveSSM(
            d_model=d_model,
            d_state=d_state,
            dt_rank=dt_rank,
            bias=bias
        )

        # Diagonal (skip connection with learned scaling)
        self.diagonal = nn.Linear(d_model, d_model, bias=bias)

        # Learnable combination weights
        self.alpha = nn.Parameter(torch.tensor([1.0]))
        self.beta = nn.Parameter(torch.tensor([1.0]))
        self.gamma = nn.Parameter(torch.tensor([0.5]))

        # Output projection
        self.out_proj = nn.Linear(d_model, d_model, bias=bias)

        # Layer norm
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch, seq_len, d_model)

        Returns:
            y: (batch, seq_len, d_model)
        """
        batch_size, seq_len, _ = x.shape

        # Forward pass
        h_forward = None
        forward_outputs = []
        for t in range(seq_len):
            y_t, h_forward = self.forward_ssm(x[:, t, :], h_forward)
            forward_outputs.append(y_t)
        forward_out = torch.stack(forward_outputs, dim=1)

        # Backward pass (reverse sequence)
        x_reversed = torch.flip(x, dims=[1])
        h_backward = None
        backward_outputs = []
        for t in range(seq_len):
            y_t, h_backward = self.backward_ssm(x_reversed[:, t, :], h_backward)
            backward_outputs.append(y_t)
        backward_out = torch.flip(torch.stack(backward_outputs, dim=1), dims=[1])

        # Diagonal (element-wise transformation)
        diagonal_out = self.diagonal(x)

        # Normalize weights (move to same device as input)
        alpha = self.alpha.to(forward_out.device)
        beta = self.beta.to(forward_out.device)
        gamma = self.gamma.to(forward_out.device)

        total = torch.abs(alpha) + torch.abs(beta) + torch.abs(gamma) + 1e-8
        alpha_norm = alpha / total
        beta_norm = beta / total
        gamma_norm = gamma / total

        # Combine
        y = alpha_norm * forward_out + beta_norm * backward_out + gamma_norm * diagonal_out
        y = self.out_proj(y)

        return y


class HydraBlock(nn.Module):
    """
    Complete Hydra block with gated MLP and bidirectional SSM.

    Architecture similar to MambaBlock but with bidirectional processing.
    """

    def __init__(
        self,
        d_model: int,
        d_state: int = 16,
        d_conv: int = 4,
        expand: int = 2,
        dt_rank: str = "auto",
        bias: bool = False,
    ):
        super().__init__()

        self.d_model = d_model
        self.d_inner = d_model * expand

        # Input projection
        self.in_proj = nn.Linear(d_model, self.d_inner * 2, bias=bias)

        # Bidirectional SSM
        self.bidirectional_ssm = HydraBidirectionalSSM(
            d_model=self.d_inner,
            d_state=d_state,
            dt_rank=dt_rank,
            bias=bias
        )

        # Output projection
        self.out_proj = nn.Linear(self.d_inner, d_model, bias=bias)

        # Layer norm
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch, seq_len, d_model)

        Returns:
            y: (batch, seq_len, d_model)
        """
        # Input projection
        x_proj = self.in_proj(x)  # (batch, seq_len, 2*d_inner)
        x_ssm, x_gate = x_proj.chunk(2, dim=-1)

        # Bidirectional SSM path
        ssm_out = self.bidirectional_ssm(x_ssm)

        # Gate path
        gate = F.silu(x_gate)

        # Combine and project
        y = ssm_out * gate
        y = self.out_proj(y)

        return y


# ================================================================================
# Full Models: QuantumMambaSSM and QuantumHydraSSM
# ================================================================================

class QuantumMambaSSM(nn.Module):
    """
    Quantum Mamba with Selective State Space Model.

    Combines:
    - Chunked quantum processing with attention-based aggregation (FAST!)
    - Three-branch quantum superposition for feature extraction
    - Mamba-style selective SSM for sequence modeling
    - Input-dependent Δ, B, C for selective information propagation

    This is the theoretically-aligned version with efficient chunked processing.
    """

    def __init__(
        self,
        n_qubits: int = 4,
        n_timesteps: int = 160,
        qlcu_layers: int = 2,
        feature_dim: int = 64,
        d_model: int = 64,
        d_state: int = 16,
        n_layers: int = 2,
        output_dim: int = 2,
        dropout: float = 0.1,
        chunk_size: int = 32,
        device: str = "cpu"
    ):
        super().__init__()

        self.n_qubits = n_qubits
        self.n_timesteps = n_timesteps
        self.feature_dim = feature_dim
        self.d_model = d_model
        self.chunk_size = chunk_size
        self.device = torch.device(device) if isinstance(device, str) else device

        # Feature projection
        self.feature_proj = nn.Sequential(
            nn.Linear(feature_dim, d_model),
            nn.LayerNorm(d_model),
            nn.SiLU()
        )

        # Chunked quantum processor with attention-based aggregation
        self.quantum_processor = ChunkedQuantumProcessor(
            n_qubits=n_qubits,
            n_layers=qlcu_layers,
            feature_dim=d_model,
            hidden_dim=d_model,
            chunk_size=chunk_size,
            device=device
        )

        # Mamba blocks (selective SSM) - now operating on chunks!
        self.mamba_layers = nn.ModuleList([
            MambaBlock(
                d_model=d_model,
                d_state=d_state,
                d_conv=4,
                expand=2
            )
            for _ in range(n_layers)
        ])

        # Output layers
        self.output_norm = nn.LayerNorm(d_model)
        self.output_layer = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, output_dim)
        )

        self.dropout = nn.Dropout(dropout)
        self.to(self.device)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch, feature_dim) for 2D or
               (batch, feature_dim, n_timesteps) or (batch, n_timesteps, feature_dim) for 3D

        Returns:
            output: (batch, output_dim)
        """
        # Handle 2D input
        if x.dim() == 2:
            x = x.unsqueeze(1)  # (batch, 1, features)

        # Ensure (batch, n_timesteps, feature_dim)
        if x.dim() == 3 and x.shape[2] == self.n_timesteps:
            x = x.permute(0, 2, 1)

        batch_size, seq_len, _ = x.shape

        # Feature projection
        x = self.feature_proj(x)  # (batch, seq_len, d_model)

        # Chunked quantum processing with attention aggregation
        # This processes all timesteps in parallel within chunks, then aggregates
        chunk_features, _ = self.quantum_processor(x)  # (batch, n_chunks, d_model)

        # SSM now operates on chunk-level features (much faster!)
        x = chunk_features
        x = self.dropout(x)

        # Mamba layers (selective SSM) on chunks
        h = None
        for mamba_layer in self.mamba_layers:
            x_new, h = mamba_layer(x, h)
            x = x + x_new  # Residual connection

        # Global pooling and output
        x = self.output_norm(x)
        x_pooled = x.mean(dim=1)  # (batch, d_model)
        output = self.output_layer(x_pooled)

        return output


class QuantumHydraSSM(nn.Module):
    """
    Quantum Hydra with Bidirectional State Space Model.

    Combines:
    - Chunked quantum processing with attention-based aggregation (FAST!)
    - Three-branch quantum superposition for feature extraction
    - Hydra-style bidirectional SSM (forward + backward + diagonal)
    - Complex coefficients for quantum state combination

    This is the theoretically-aligned version with efficient chunked processing.
    """

    def __init__(
        self,
        n_qubits: int = 6,
        n_timesteps: int = 200,
        qlcu_layers: int = 2,
        feature_dim: int = 129,
        d_model: int = 64,
        d_state: int = 16,
        n_layers: int = 2,
        output_dim: int = 1,
        dropout: float = 0.1,
        chunk_size: int = 32,
        device: str = "cpu"
    ):
        super().__init__()

        self.n_qubits = n_qubits
        self.n_timesteps = n_timesteps
        self.feature_dim = feature_dim
        self.d_model = d_model
        self.chunk_size = chunk_size
        self.device = torch.device(device) if isinstance(device, str) else device

        # Feature projection
        self.feature_proj = nn.Sequential(
            nn.Linear(feature_dim, d_model),
            nn.LayerNorm(d_model),
            nn.SiLU()
        )

        # Three chunked quantum processors for Hydra's forward/backward/diagonal
        self.quantum_forward = ChunkedQuantumProcessor(
            n_qubits=n_qubits,
            n_layers=qlcu_layers,
            feature_dim=d_model,
            hidden_dim=d_model,
            chunk_size=chunk_size,
            device=device
        )

        self.quantum_backward = ChunkedQuantumProcessor(
            n_qubits=n_qubits,
            n_layers=qlcu_layers,
            feature_dim=d_model,
            hidden_dim=d_model,
            chunk_size=chunk_size,
            device=device
        )

        self.quantum_diagonal = ChunkedQuantumProcessor(
            n_qubits=n_qubits,
            n_layers=qlcu_layers,
            feature_dim=d_model,
            hidden_dim=d_model,
            chunk_size=chunk_size,
            device=device
        )

        # Hydra blocks (bidirectional SSM) - now operating on chunks!
        self.hydra_layers = nn.ModuleList([
            HydraBlock(
                d_model=d_model,
                d_state=d_state,
                d_conv=4,
                expand=2
            )
            for _ in range(n_layers)
        ])

        # Complex coefficients for combining forward/backward/diagonal
        self.alpha = nn.Parameter(torch.rand(1, dtype=torch.complex64))
        self.beta = nn.Parameter(torch.rand(1, dtype=torch.complex64))
        self.gamma = nn.Parameter(torch.rand(1, dtype=torch.complex64))

        # Output layers
        self.output_norm = nn.LayerNorm(d_model)
        self.output_layer = nn.Linear(d_model, output_dim)

        self.dropout = nn.Dropout(dropout)
        self.to(self.device)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch, feature_dim) for 2D or
               (batch, feature_dim, n_timesteps) or (batch, n_timesteps, feature_dim) for 3D

        Returns:
            output: (batch, output_dim)
        """
        batch_size = x.shape[0]

        # Handle 2D input
        if x.dim() == 2:
            x = x.unsqueeze(1)

        # Ensure (batch, n_timesteps, feature_dim)
        if x.dim() == 3 and x.shape[2] == self.n_timesteps:
            x = x.permute(0, 2, 1)

        seq_len = x.shape[1]

        # Feature projection
        x = self.feature_proj(x)  # (batch, seq_len, d_model)

        # Three-branch chunked quantum processing (Hydra-style)
        # Forward: process sequence normally
        q_forward, _ = self.quantum_forward(x)  # (batch, n_chunks, d_model)

        # Backward: process reversed sequence
        x_reversed = torch.flip(x, dims=[1])
        q_backward, _ = self.quantum_backward(x_reversed)  # (batch, n_chunks, d_model)
        q_backward = torch.flip(q_backward, dims=[1])  # Flip back

        # Diagonal: global processing (same as forward, but separate params)
        q_diagonal, _ = self.quantum_diagonal(x)  # (batch, n_chunks, d_model)

        # Combine quantum features with complex coefficients (move to same device as input)
        alpha = self.alpha.to(q_forward.device)
        beta = self.beta.to(q_forward.device)
        gamma = self.gamma.to(q_forward.device)

        norm = torch.sqrt(
            torch.abs(alpha)**2 + torch.abs(beta)**2 + torch.abs(gamma)**2 + 1e-8
        )
        alpha_norm = torch.abs(alpha / norm)
        beta_norm = torch.abs(beta / norm)
        gamma_norm = torch.abs(gamma / norm)

        # Weighted combination (using real magnitudes for gradient stability)
        x = alpha_norm * q_forward + beta_norm * q_backward + gamma_norm * q_diagonal
        x = self.dropout(x)

        # Hydra layers (bidirectional SSM) on chunks
        for hydra_layer in self.hydra_layers:
            x_new = hydra_layer(x)
            x = x + x_new  # Residual connection

        # Output pooling
        x = self.output_norm(x)
        x_pooled = x.mean(dim=1)  # (batch, d_model)
        output = self.output_layer(x_pooled)

        return output


# ================================================================================
# Testing
# ================================================================================

if __name__ == "__main__":
    print("=" * 80)
    print("Quantum SSM Models - Theoretically-Aligned Architectures")
    print("=" * 80)

    device = "cpu"
    batch_size = 4
    n_timesteps = 64
    feature_dim = 32
    n_qubits = 4
    output_dim = 2

    # Test MambaSelectiveSSM
    print("\n[1] Testing MambaSelectiveSSM...")
    ssm = MambaSelectiveSSM(d_model=32, d_state=16)
    x_step = torch.randn(batch_size, 32)
    y, h = ssm(x_step)
    print(f"    Input: {x_step.shape}")
    print(f"    Output: {y.shape}")
    print(f"    Hidden: {h.shape}")

    # Test MambaBlock
    print("\n[2] Testing MambaBlock...")
    mamba_block = MambaBlock(d_model=32, d_state=16)
    x_seq = torch.randn(batch_size, n_timesteps, 32)
    y_seq, _ = mamba_block(x_seq)
    print(f"    Input: {x_seq.shape}")
    print(f"    Output: {y_seq.shape}")

    # Test HydraBidirectionalSSM
    print("\n[3] Testing HydraBidirectionalSSM...")
    hydra_ssm = HydraBidirectionalSSM(d_model=32, d_state=16)
    y_bidirectional = hydra_ssm(x_seq)
    print(f"    Input: {x_seq.shape}")
    print(f"    Output: {y_bidirectional.shape}")

    # Test HydraBlock
    print("\n[4] Testing HydraBlock...")
    hydra_block = HydraBlock(d_model=32, d_state=16)
    y_hydra_block = hydra_block(x_seq)
    print(f"    Input: {x_seq.shape}")
    print(f"    Output: {y_hydra_block.shape}")

    # Test QuantumMambaSSM
    print("\n[5] Testing QuantumMambaSSM (full model)...")
    model_mamba = QuantumMambaSSM(
        n_qubits=n_qubits,
        n_timesteps=n_timesteps,
        qlcu_layers=2,
        feature_dim=feature_dim,
        d_model=32,
        d_state=16,
        n_layers=2,
        output_dim=output_dim,
        device=device
    )

    x_input = torch.randn(batch_size, feature_dim, n_timesteps)
    output = model_mamba(x_input)
    print(f"    Input: {x_input.shape}")
    print(f"    Output: {output.shape}")

    total_params = sum(p.numel() for p in model_mamba.parameters() if p.requires_grad)
    print(f"    Parameters: {total_params:,}")

    # Test QuantumHydraSSM
    print("\n[6] Testing QuantumHydraSSM (full model)...")
    model_hydra = QuantumHydraSSM(
        n_qubits=n_qubits,
        n_timesteps=n_timesteps,
        qlcu_layers=2,
        feature_dim=feature_dim,
        d_model=32,
        d_state=16,
        n_layers=2,
        output_dim=output_dim,
        device=device
    )

    output_hydra = model_hydra(x_input)
    print(f"    Input: {x_input.shape}")
    print(f"    Output: {output_hydra.shape}")

    total_params = sum(p.numel() for p in model_hydra.parameters() if p.requires_grad)
    print(f"    Parameters: {total_params:,}")

    # Test gradient flow
    print("\n[7] Testing gradient flow...")
    model_mamba.train()
    optimizer = torch.optim.Adam(model_mamba.parameters(), lr=1e-3)
    criterion = nn.CrossEntropyLoss()

    output = model_mamba(x_input)
    loss = criterion(output, torch.randint(0, output_dim, (batch_size,)))
    loss.backward()
    optimizer.step()

    print(f"    Loss: {loss.item():.4f}")
    print(f"    Gradient flow: OK")

    # Summary
    print("\n" + "=" * 80)
    print("Testing complete!")
    print("=" * 80)
    print("\nKey Features of New Architecture:")
    print("  [QuantumMambaSSM]")
    print("    - Three-branch quantum superposition")
    print("    - Mamba-style selective SSM (input-dependent Δ, B, C)")
    print("    - NOT LSTM gates (theoretically aligned)")
    print("  [QuantumHydraSSM]")
    print("    - Three-branch quantum superposition (forward/backward/diagonal)")
    print("    - Hydra-style bidirectional SSM")
    print("    - Complex coefficients (α, β, γ ∈ ℂ)")
    print("    - NOT LSTM gates (theoretically aligned)")
    print("=" * 80)
