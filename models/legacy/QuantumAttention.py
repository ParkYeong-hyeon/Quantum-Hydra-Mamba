"""
Quantum Attention Models

Implements quantum versions of Mamba and Hydra with self-attention mechanism
for selective information mixing, instead of LSTM gates (Gated) or SSM (SSM).

This completes the three-way comparison:
1. Gated: LSTM-style gates for selective forgetting (QuantumMambaGated, QuantumHydraGated)
2. SSM: State space models with selective scan (QuantumMambaSSM, QuantumHydraSSM)
3. Attention: Self-attention for global information mixing (QuantumMambaAttention, QuantumHydraAttention)

All models share:
- Three-branch quantum superposition for feature extraction
- Same input/output interface
- Same chunked processing for efficiency

Only the sequence mixing mechanism differs:
- Gated: h = f*h + i*c (forget/input gates)
- SSM: h = A*h + B*x, y = C*h + D*x (state space)
- Attention: y = softmax(QK^T/sqrt(d))V (self-attention)

Author: Junghoon Park
Date: December 2024
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import pennylane as qml
import numpy as np
from typing import Optional, Tuple
import math


# ================================================================================
# Import shared quantum components from QuantumGatedRecurrence
# ================================================================================

from models.core.gated.QuantumGatedRecurrence import (
    QuantumFeatureExtractor,
    QuantumSuperpositionBranches
)


# ================================================================================
# Multi-Head Self-Attention Module
# ================================================================================

class MultiHeadSelfAttention(nn.Module):
    """
    Standard multi-head self-attention mechanism.

    Computes: Attention(Q, K, V) = softmax(QK^T / sqrt(d_k)) V

    This provides global information mixing across all positions,
    unlike LSTM gates (local) or SSM (recurrent).
    """

    def __init__(
        self,
        d_model: int,
        n_heads: int = 4,
        dropout: float = 0.1,
        bias: bool = True
    ):
        super().__init__()

        assert d_model % n_heads == 0, "d_model must be divisible by n_heads"

        self.d_model = d_model
        self.n_heads = n_heads
        self.d_head = d_model // n_heads
        self.scale = math.sqrt(self.d_head)

        # Linear projections for Q, K, V
        self.q_proj = nn.Linear(d_model, d_model, bias=bias)
        self.k_proj = nn.Linear(d_model, d_model, bias=bias)
        self.v_proj = nn.Linear(d_model, d_model, bias=bias)

        # Output projection
        self.out_proj = nn.Linear(d_model, d_model, bias=bias)

        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Args:
            x: (batch, seq_len, d_model) input sequence
            mask: Optional attention mask

        Returns:
            y: (batch, seq_len, d_model) output sequence
        """
        batch_size, seq_len, _ = x.shape

        # Project to Q, K, V
        Q = self.q_proj(x)  # (batch, seq_len, d_model)
        K = self.k_proj(x)
        V = self.v_proj(x)

        # Reshape for multi-head attention
        # (batch, seq_len, d_model) -> (batch, n_heads, seq_len, d_head)
        Q = Q.view(batch_size, seq_len, self.n_heads, self.d_head).transpose(1, 2)
        K = K.view(batch_size, seq_len, self.n_heads, self.d_head).transpose(1, 2)
        V = V.view(batch_size, seq_len, self.n_heads, self.d_head).transpose(1, 2)

        # Compute attention scores
        # (batch, n_heads, seq_len, d_head) @ (batch, n_heads, d_head, seq_len)
        # -> (batch, n_heads, seq_len, seq_len)
        attn_scores = torch.matmul(Q, K.transpose(-2, -1)) / self.scale

        # Apply mask if provided
        if mask is not None:
            attn_scores = attn_scores.masked_fill(mask == 0, float('-inf'))

        # Softmax and dropout
        attn_weights = F.softmax(attn_scores, dim=-1)
        attn_weights = self.dropout(attn_weights)

        # Apply attention to values
        # (batch, n_heads, seq_len, seq_len) @ (batch, n_heads, seq_len, d_head)
        # -> (batch, n_heads, seq_len, d_head)
        attn_output = torch.matmul(attn_weights, V)

        # Reshape back
        # (batch, n_heads, seq_len, d_head) -> (batch, seq_len, d_model)
        attn_output = attn_output.transpose(1, 2).contiguous().view(batch_size, seq_len, self.d_model)

        # Output projection
        output = self.out_proj(attn_output)

        return output


# ================================================================================
# Attention Block (replaces MambaBlock/GatedRecurrence)
# ================================================================================

class AttentionBlock(nn.Module):
    """
    Attention block with gated MLP.

    Architecture:
    1. Multi-head self-attention
    2. Gated MLP (similar to Mamba/Hydra blocks)
    3. Layer normalization and residual connections
    """

    def __init__(
        self,
        d_model: int,
        n_heads: int = 4,
        expand: int = 2,
        dropout: float = 0.1,
        bias: bool = True
    ):
        super().__init__()

        self.d_model = d_model
        self.d_inner = d_model * expand

        # Layer norms
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)

        # Self-attention
        self.attention = MultiHeadSelfAttention(
            d_model=d_model,
            n_heads=n_heads,
            dropout=dropout,
            bias=bias
        )

        # Gated MLP (like Mamba/Hydra)
        self.in_proj = nn.Linear(d_model, self.d_inner * 2, bias=bias)
        self.out_proj = nn.Linear(self.d_inner, d_model, bias=bias)

        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch, seq_len, d_model)

        Returns:
            y: (batch, seq_len, d_model)
        """
        # Self-attention with residual
        x = x + self.dropout(self.attention(self.norm1(x)))

        # Gated MLP with residual
        x_norm = self.norm2(x)
        x_proj = self.in_proj(x_norm)
        x_gate, x_val = x_proj.chunk(2, dim=-1)

        # Gated activation (like Mamba/Hydra)
        x_mlp = F.silu(x_gate) * x_val
        x_mlp = self.out_proj(x_mlp)

        x = x + self.dropout(x_mlp)

        return x


# ================================================================================
# Chunked Attention with Quantum Superposition
# ================================================================================

class ChunkedAttentionSuperposition(nn.Module):
    """
    Chunked attention WITH true quantum superposition.

    Combines:
    1. Three-branch quantum superposition (like original models)
    2. Self-attention for selective information mixing
    3. Chunked processing for efficiency

    This is the attention-based equivalent of ChunkedGatedSuperposition.
    """

    def __init__(
        self,
        n_qubits: int = 4,
        qlcu_layers: int = 2,
        feature_dim: int = 64,
        hidden_dim: int = 64,
        n_heads: int = 4,
        chunk_size: int = 16,
        dropout: float = 0.1,
        device: str = "cpu"
    ):
        super().__init__()

        self.n_qubits = n_qubits
        self.hidden_dim = hidden_dim
        self.chunk_size = chunk_size
        self.device = torch.device(device) if isinstance(device, str) else device

        # Quantum feature dimension
        self.q_dim = 3 * n_qubits

        # Quantum superposition branches
        self.quantum_superposition = QuantumSuperpositionBranches(
            n_qubits, qlcu_layers, feature_dim, device
        )

        # Chunk aggregation
        self.chunk_agg = nn.Sequential(
            nn.Linear(self.q_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.SiLU()
        )

        # Self-attention for chunk-level sequence mixing
        self.chunk_attention = MultiHeadSelfAttention(
            d_model=hidden_dim,
            n_heads=n_heads,
            dropout=dropout
        )

        # Layer norm
        self.norm = nn.LayerNorm(hidden_dim)

        self.dropout = nn.Dropout(dropout)
        self.to(self.device)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Process sequence with chunked attention superposition.

        Args:
            x: (batch, seq_len, feature_dim)

        Returns:
            h: (batch, hidden_dim) final representation
            all_states: (batch, n_chunks, hidden_dim) all chunk states
        """
        batch_size, seq_len, feature_dim = x.shape

        n_chunks = (seq_len + self.chunk_size - 1) // self.chunk_size

        # Process each chunk through quantum superposition
        chunk_features_list = []

        for c in range(n_chunks):
            start = c * self.chunk_size
            end = min(start + self.chunk_size, seq_len)
            chunk = x[:, start:end, :]
            chunk_len = end - start

            # Flatten for parallel processing
            chunk_flat = chunk.reshape(batch_size * chunk_len, feature_dim)

            # Quantum superposition (three branches)
            q_features = self.quantum_superposition(chunk_flat)  # (B*chunk_len, q_dim)

            # Reshape and aggregate within chunk
            q_features = q_features.reshape(batch_size, chunk_len, self.q_dim)
            chunk_features = q_features.mean(dim=1)  # (batch, q_dim)
            chunk_features = self.chunk_agg(chunk_features)  # (batch, hidden_dim)
            chunk_features = self.dropout(chunk_features)

            chunk_features_list.append(chunk_features)

        # Stack all chunk features
        all_chunks = torch.stack(chunk_features_list, dim=1)  # (batch, n_chunks, hidden_dim)

        # Apply self-attention across chunks
        # This allows global information mixing (unlike LSTM's local gates or SSM's recurrence)
        attended = self.chunk_attention(self.norm(all_chunks))  # (batch, n_chunks, hidden_dim)
        all_states = all_chunks + attended  # Residual connection

        # Final state: mean pooling over attended chunks
        h = all_states.mean(dim=1)  # (batch, hidden_dim)

        return h, all_states


# ================================================================================
# Full Models with Attention
# ================================================================================

class QuantumMambaAttention(nn.Module):
    """
    Quantum Mamba with Self-Attention AND TRUE Quantum Superposition.

    Combines:
    - Three-branch quantum superposition (like original QuantumMamba)
    - Self-attention for global information mixing (instead of LSTM gates or SSM)
    - Chunked processing for efficiency

    This is the attention-based equivalent of QuantumMambaGated/QuantumMambaSSM.
    """

    def __init__(
        self,
        n_qubits: int = 4,
        n_timesteps: int = 160,
        qlcu_layers: int = 2,
        feature_dim: int = 64,
        hidden_dim: int = 64,
        output_dim: int = 2,
        n_heads: int = 4,
        n_layers: int = 1,
        chunk_size: int = 16,
        dropout: float = 0.1,
        device: str = "cpu"
    ):
        super().__init__()

        self.n_qubits = n_qubits
        self.n_timesteps = n_timesteps
        self.feature_dim = feature_dim
        self.hidden_dim = hidden_dim
        self.device = torch.device(device) if isinstance(device, str) else device

        # Feature projection
        self.feature_proj = nn.Linear(feature_dim, feature_dim)
        self.dropout = nn.Dropout(dropout)

        # Chunked attention with superposition
        self.attention_recurrence = ChunkedAttentionSuperposition(
            n_qubits=n_qubits,
            qlcu_layers=qlcu_layers,
            feature_dim=feature_dim,
            hidden_dim=hidden_dim,
            n_heads=n_heads,
            chunk_size=chunk_size,
            dropout=dropout,
            device=device
        )

        # Additional attention blocks for deeper processing
        self.attention_layers = nn.ModuleList([
            AttentionBlock(
                d_model=hidden_dim,
                n_heads=n_heads,
                expand=2,
                dropout=dropout
            )
            for _ in range(n_layers)
        ])

        # Output layer
        self.output_layer = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, output_dim)
        )

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
            x = x.unsqueeze(1)

        # Handle different input formats - ensure (batch_size, n_timesteps, feature_dim)
        if x.dim() == 3 and x.shape[1] != self.n_timesteps and x.shape[2] == self.n_timesteps:
            x = x.permute(0, 2, 1)

        # Feature projection
        x_proj = F.silu(self.feature_proj(self.dropout(x)))

        # Attention-based processing with quantum superposition
        h_final, all_states = self.attention_recurrence(x_proj)

        # Additional attention layers on chunk states
        x = all_states
        for attn_layer in self.attention_layers:
            x = attn_layer(x)

        # Pool and output
        h_final = x.mean(dim=1)
        output = self.output_layer(h_final)

        return output


class QuantumHydraAttention(nn.Module):
    """
    Quantum Hydra with Self-Attention AND TRUE Quantum Superposition.

    Three branches with bidirectional attention processing:
    - Branch 1: Forward with superposition + attention
    - Branch 2: Backward with superposition + attention
    - Branch 3: Global with superposition + attention

    Final output combines all branches via complex coefficients.

    This is the attention-based equivalent of QuantumHydraGated/QuantumHydraSSM.
    """

    def __init__(
        self,
        n_qubits: int = 6,
        n_timesteps: int = 200,
        qlcu_layers: int = 2,
        feature_dim: int = 129,
        hidden_dim: int = 64,
        output_dim: int = 1,
        n_heads: int = 4,
        n_layers: int = 1,
        chunk_size: int = 16,
        dropout: float = 0.1,
        device: str = "cpu"
    ):
        super().__init__()

        self.n_qubits = n_qubits
        self.n_timesteps = n_timesteps
        self.hidden_dim = hidden_dim
        self.device = torch.device(device) if isinstance(device, str) else device

        # Feature projection
        self.feature_proj = nn.Linear(feature_dim, feature_dim)

        # Forward and backward branches with attention + superposition
        self.branch_forward = ChunkedAttentionSuperposition(
            n_qubits=n_qubits,
            qlcu_layers=qlcu_layers,
            feature_dim=feature_dim,
            hidden_dim=hidden_dim,
            n_heads=n_heads,
            chunk_size=chunk_size,
            dropout=dropout,
            device=device
        )

        self.branch_backward = ChunkedAttentionSuperposition(
            n_qubits=n_qubits,
            qlcu_layers=qlcu_layers,
            feature_dim=feature_dim,
            hidden_dim=hidden_dim,
            n_heads=n_heads,
            chunk_size=chunk_size,
            dropout=dropout,
            device=device
        )

        # Global branch with superposition
        self.branch_global = QuantumSuperpositionBranches(
            n_qubits, qlcu_layers, feature_dim, device
        )
        self.global_proj = nn.Linear(3 * n_qubits, hidden_dim)

        # Additional attention layers
        self.attention_layers = nn.ModuleList([
            AttentionBlock(
                d_model=hidden_dim,
                n_heads=n_heads,
                expand=2,
                dropout=dropout
            )
            for _ in range(n_layers)
        ])

        # Complex coefficients for final combination
        self.alpha = nn.Parameter(torch.rand(1, dtype=torch.complex64))
        self.beta = nn.Parameter(torch.rand(1, dtype=torch.complex64))
        self.gamma = nn.Parameter(torch.rand(1, dtype=torch.complex64))

        # Output layer
        self.output_layer = nn.Linear(hidden_dim, output_dim)

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

        # Handle different input formats
        if x.dim() == 3 and x.shape[1] != self.n_timesteps and x.shape[2] == self.n_timesteps:
            x = x.permute(0, 2, 1)

        x_proj = self.feature_proj(self.dropout(x))

        # Forward branch
        h_forward, states_forward = self.branch_forward(x_proj)

        # Backward branch
        x_flipped = torch.flip(x_proj, dims=[1])
        h_backward, states_backward = self.branch_backward(x_flipped)

        # Global branch
        x_mean = x_proj.mean(dim=1)
        q_global = self.branch_global(x_mean)
        h_global = self.global_proj(q_global)

        # Combine with complex coefficients (move to same device as input)
        h_forward_c = h_forward.to(torch.complex64)
        h_backward_c = h_backward.to(torch.complex64)
        h_global_c = h_global.to(torch.complex64)

        alpha = self.alpha.to(h_forward_c.device)
        beta = self.beta.to(h_forward_c.device)
        gamma = self.gamma.to(h_forward_c.device)

        combined = alpha * h_forward_c + beta * h_backward_c + gamma * h_global_c

        # Normalize
        norm = torch.sqrt(torch.sum(torch.abs(combined) ** 2, dim=1, keepdim=True) + 1e-8)
        normalized = combined / norm

        output_features = torch.abs(normalized).float()

        # Additional attention processing
        # Reshape for attention layers: (batch, 1, hidden_dim)
        x = output_features.unsqueeze(1)
        for attn_layer in self.attention_layers:
            x = attn_layer(x)
        x = x.squeeze(1)

        output = self.output_layer(x)

        return output


# ================================================================================
# Testing
# ================================================================================

if __name__ == "__main__":
    print("=" * 80)
    print("Quantum Attention Models - Testing")
    print("=" * 80)

    device = "cpu"
    batch_size = 4
    n_timesteps = 64
    n_channels = 32
    n_qubits = 4
    output_dim = 2

    print("\n[1] Testing MultiHeadSelfAttention...")
    attention = MultiHeadSelfAttention(d_model=32, n_heads=4)
    x_seq = torch.randn(batch_size, n_timesteps, 32)
    attn_out = attention(x_seq)
    print(f"  Input: {x_seq.shape}")
    print(f"  Output: {attn_out.shape}")

    print("\n[2] Testing AttentionBlock...")
    attn_block = AttentionBlock(d_model=32, n_heads=4)
    block_out = attn_block(x_seq)
    print(f"  Input: {x_seq.shape}")
    print(f"  Output: {block_out.shape}")

    print("\n[3] Testing ChunkedAttentionSuperposition...")
    chunked = ChunkedAttentionSuperposition(
        n_qubits=n_qubits,
        qlcu_layers=2,
        feature_dim=n_channels,
        hidden_dim=32,
        n_heads=4,
        chunk_size=8,
        device=device
    )
    x_input = torch.randn(batch_size, n_timesteps, n_channels)
    h_final, all_states = chunked(x_input)
    print(f"  Input: {x_input.shape}")
    print(f"  Final state: {h_final.shape}")
    print(f"  All states: {all_states.shape}")

    print("\n[4] Testing QuantumMambaAttention (full model)...")
    model_mamba = QuantumMambaAttention(
        n_qubits=n_qubits,
        n_timesteps=n_timesteps,
        qlcu_layers=2,
        feature_dim=n_channels,
        hidden_dim=32,
        output_dim=output_dim,
        n_heads=4,
        chunk_size=8,
        device=device
    )

    x = torch.randn(batch_size, n_channels, n_timesteps)
    output = model_mamba(x)
    print(f"  Input: {x.shape}")
    print(f"  Output: {output.shape}")

    total_params = sum(p.numel() for p in model_mamba.parameters() if p.requires_grad)
    print(f"  Parameters: {total_params:,}")

    print("\n[5] Testing QuantumHydraAttention (full model)...")
    model_hydra = QuantumHydraAttention(
        n_qubits=n_qubits,
        n_timesteps=n_timesteps,
        qlcu_layers=2,
        feature_dim=n_channels,
        hidden_dim=32,
        output_dim=output_dim,
        n_heads=4,
        chunk_size=8,
        device=device
    )

    output = model_hydra(x)
    print(f"  Input: {x.shape}")
    print(f"  Output: {output.shape}")

    total_params = sum(p.numel() for p in model_hydra.parameters() if p.requires_grad)
    print(f"  Parameters: {total_params:,}")

    print("\n[6] Testing gradient flow...")
    model_mamba.train()
    optimizer = torch.optim.Adam(model_mamba.parameters(), lr=1e-3)
    criterion = nn.CrossEntropyLoss()

    output = model_mamba(x)
    loss = criterion(output, torch.randint(0, output_dim, (batch_size,)))
    loss.backward()
    optimizer.step()

    print(f"  Loss: {loss.item():.4f}")
    print(f"  Gradient flow: OK")

    print("\n" + "=" * 80)
    print("Testing complete!")
    print("=" * 80)
    print("\nKey Features:")
    print("  - TRUE quantum superposition (three branches)")
    print("  - Complex coefficients (alpha, beta, gamma)")
    print("  - Self-attention for global information mixing")
    print("  - Chunked parallel processing")
    print("  - Same interface as Gated and SSM versions")
    print("=" * 80)
    print("\nComparison of selective mechanisms:")
    print("  [Gated]     h = f*h + i*c          (LSTM-style gates)")
    print("  [SSM]       h = A*h + B*x          (State space)")
    print("  [Attention] y = softmax(QK^T/d)V   (Self-attention)")
    print("=" * 80)
