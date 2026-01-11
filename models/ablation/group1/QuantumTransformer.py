"""
Quantum Transformer Models with Full Sequence Self-Attention

This module implements quantum models with PROPER transformer-style attention:
- QuantumTransformer: Quantum feature extraction → Full transformer attention
- QuantumHydraTransformer: Quantum feature extraction → Bidirectional transformer attention

Key Design Principles:
- Uses the SAME quantum feature extraction as QuantumMambaSSM/QuantumHydraSSM
- Uses FULL SEQUENCE self-attention (not chunked attention like QuantumMambaAttention)
- Consistent with ClassicalTransformer for fair comparison

This ensures a clean ablation:
- QuantumMambaSSM vs QuantumTransformer: Same quantum features, SSM vs Attention mixing
- QuantumTransformer vs ClassicalTransformer: Quantum vs Classical features, same Attention mixing

Architecture Comparison:
┌─────────────────────┬────────────────────────────────────────────────────┐
│ Model               │ Architecture                                        │
├─────────────────────┼────────────────────────────────────────────────────┤
│ QuantumMambaSSM     │ Quantum Features → Classical SSM (Mamba)           │
│ QuantumHydraSSM     │ Quantum Features → Classical Bidirectional SSM     │
│ QuantumTransformer  │ Quantum Features → Full Transformer Attention ★    │
│ QuantumHydraTransf. │ Quantum Features → Bidirectional Transformer ★     │
│ ClassicalMamba      │ Classical Features → Classical SSM                 │
│ ClassicalHydra      │ Classical Features → Classical Bidirectional SSM   │
│ ClassicalTransformer│ Classical Features → Full Transformer Attention    │
│ ClassicalHydraTransf│ Classical Features → Bidirectional Transformer     │
└─────────────────────┴────────────────────────────────────────────────────┘

Author: Junghoon Park
Date: December 2024
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional, Tuple

# Import shared quantum components (same as QuantumSSM for consistency)
from models.core.gated.QuantumGatedRecurrence import (
    QuantumFeatureExtractor,       # Unified quantum circuit for feature extraction
    QuantumSuperpositionBranches   # Three-branch quantum superposition
)


# ================================================================================
# RMSNorm (consistent with other models)
# ================================================================================

class RMSNorm(nn.Module):
    """Root Mean Square Layer Normalization."""

    def __init__(self, d_model: int, eps: float = 1e-5):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(d_model))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        rms = torch.sqrt(torch.mean(x ** 2, dim=-1, keepdim=True) + self.eps)
        x_normalized = x / rms
        return self.weight * x_normalized


# ================================================================================
# Multi-Head Self-Attention (Full Sequence - NOT Chunked)
# ================================================================================

class MultiHeadSelfAttention(nn.Module):
    """
    Standard multi-head self-attention over the FULL SEQUENCE.

    This is the classical transformer attention mechanism:
    Attention(Q, K, V) = softmax(QK^T / sqrt(d_k)) V

    Unlike the chunked attention in QuantumMambaAttention, this attends
    to ALL positions in the sequence simultaneously.
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
        Full sequence self-attention.

        Args:
            x: (batch, seq_len, d_model) input sequence
            mask: Optional attention mask

        Returns:
            y: (batch, seq_len, d_model) output sequence
        """
        batch_size, seq_len, _ = x.shape

        # Project to Q, K, V
        Q = self.q_proj(x)
        K = self.k_proj(x)
        V = self.v_proj(x)

        # Reshape for multi-head attention
        Q = Q.view(batch_size, seq_len, self.n_heads, self.d_head).transpose(1, 2)
        K = K.view(batch_size, seq_len, self.n_heads, self.d_head).transpose(1, 2)
        V = V.view(batch_size, seq_len, self.n_heads, self.d_head).transpose(1, 2)

        # Compute attention scores over FULL sequence
        attn_scores = torch.matmul(Q, K.transpose(-2, -1)) / self.scale

        if mask is not None:
            attn_scores = attn_scores.masked_fill(mask == 0, float('-inf'))

        attn_weights = F.softmax(attn_scores, dim=-1)
        attn_weights = self.dropout(attn_weights)

        # Apply attention to values
        attn_output = torch.matmul(attn_weights, V)

        # Reshape back
        attn_output = attn_output.transpose(1, 2).contiguous().view(batch_size, seq_len, self.d_model)

        return self.out_proj(attn_output)


# ================================================================================
# Transformer Block (Pre-norm with Gated MLP)
# ================================================================================

class TransformerBlock(nn.Module):
    """
    Transformer block with self-attention and gated MLP.

    Architecture (consistent with ClassicalTransformer):
    1. RMSNorm + Multi-head self-attention + Residual
    2. RMSNorm + Gated MLP (SiLU activation) + Residual
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

        # Normalization layers
        self.norm1 = RMSNorm(d_model)
        self.norm2 = RMSNorm(d_model)

        # Self-attention
        self.attention = MultiHeadSelfAttention(
            d_model=d_model,
            n_heads=n_heads,
            dropout=dropout,
            bias=bias
        )

        # Gated MLP
        self.in_proj = nn.Linear(d_model, self.d_inner * 2, bias=bias)
        self.out_proj = nn.Linear(self.d_inner, d_model, bias=bias)

        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Self-attention with pre-norm and residual
        x = x + self.dropout(self.attention(self.norm1(x)))

        # Gated MLP with pre-norm and residual
        x_norm = self.norm2(x)
        x_proj = self.in_proj(x_norm)
        x_gate, x_val = x_proj.chunk(2, dim=-1)
        x_mlp = F.silu(x_gate) * x_val
        x_mlp = self.out_proj(x_mlp)

        x = x + self.dropout(x_mlp)

        return x


# ================================================================================
# Quantum Feature Processor (Chunked for Efficiency)
# ================================================================================

class QuantumFeatureProcessor(nn.Module):
    """
    Quantum feature extraction with chunked processing for efficiency.

    This is the quantum feature extraction component shared with QuantumMambaSSM.
    It processes the input sequence through quantum circuits, outputting
    quantum features that will be mixed by the transformer attention.

    Note: We use chunked quantum processing for efficiency (reducing quantum calls),
    but the MIXING is done with full transformer attention over all chunks.
    """

    def __init__(
        self,
        n_qubits: int,
        n_layers: int,
        feature_dim: int,
        hidden_dim: int,
        chunk_size: int = 16,
        device: str = "cpu"
    ):
        super().__init__()

        self.n_qubits = n_qubits
        self.feature_dim = feature_dim
        self.hidden_dim = hidden_dim
        self.chunk_size = chunk_size
        self.q_dim = 3 * n_qubits  # Output from quantum circuits

        # Classical attention-based aggregation within chunks
        self.chunk_attention = nn.Sequential(
            nn.Linear(feature_dim, feature_dim // 2),
            nn.Tanh(),
            nn.Linear(feature_dim // 2, 1)
        )

        # Quantum superposition branches
        self.quantum_branches = QuantumSuperpositionBranches(
            n_qubits=n_qubits,
            qlcu_layers=n_layers,  # QuantumGatedRecurrence uses qlcu_layers
            feature_dim=feature_dim,
            device=device
        )

        # Project quantum features to hidden dimension
        self.output_proj = nn.Sequential(
            nn.Linear(self.q_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.SiLU()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Process sequence through quantum feature extraction.

        Args:
            x: (batch, seq_len, feature_dim) input sequence

        Returns:
            chunk_features: (batch, n_chunks, hidden_dim) quantum features
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

        # Classical attention-based aggregation within chunks
        attn_scores = self.chunk_attention(x_chunked)
        attn_weights = F.softmax(attn_scores, dim=1)
        chunk_summary = (attn_weights * x_chunked).sum(dim=1)

        # Process through quantum circuits
        q_features = self.quantum_branches(chunk_summary)

        # Project to hidden dimension
        chunk_features = self.output_proj(q_features)

        # Reshape to (batch, n_chunks, hidden_dim)
        chunk_features = chunk_features.reshape(batch_size, n_chunks, self.hidden_dim)

        return chunk_features


# ================================================================================
# QuantumTransformer (Unidirectional - Full Sequence Attention)
# ================================================================================

class QuantumTransformer(nn.Module):
    """
    Quantum Feature Extraction + Full Transformer Self-Attention.

    This is the proper transformer-based model for fair comparison:
    - Uses SAME quantum feature extraction as QuantumMambaSSM
    - Uses FULL sequence self-attention (like ClassicalTransformer)

    Architecture:
        Input → Feature Projection → Quantum Features → Transformer Blocks → Classifier

    The key difference from QuantumMambaAttention:
    - QuantumMambaAttention: Attention only within/across chunks (local)
    - QuantumTransformer: Attention over ALL positions (global)
    """

    def __init__(
        self,
        n_qubits: int = 6,
        n_timesteps: int = 200,
        qlcu_layers: int = 2,
        feature_dim: int = 4,
        d_model: int = 64,
        d_state: int = 16,  # Not used, kept for interface consistency
        n_layers: int = 1,
        n_heads: int = 4,
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
        self.device = torch.device(device) if isinstance(device, str) else device

        # Feature projection (same as other quantum models)
        self.feature_proj = nn.Linear(feature_dim, feature_dim)
        self.dropout = nn.Dropout(dropout)

        # Quantum feature extraction (SAME as QuantumMambaSSM)
        self.quantum_processor = QuantumFeatureProcessor(
            n_qubits=n_qubits,
            n_layers=qlcu_layers,
            feature_dim=feature_dim,
            hidden_dim=d_model,
            chunk_size=chunk_size,
            device=device
        )

        # Positional encoding for transformer
        n_chunks = (n_timesteps + chunk_size - 1) // chunk_size
        self.pos_encoding = self._create_positional_encoding(n_chunks, d_model)

        # Transformer blocks for FULL sequence attention
        self.transformer_layers = nn.ModuleList([
            TransformerBlock(
                d_model=d_model,
                n_heads=n_heads,
                expand=2,
                dropout=dropout
            )
            for _ in range(n_layers)
        ])

        # Final normalization and classification
        self.final_norm = RMSNorm(d_model)
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.classifier = nn.Linear(d_model, output_dim)

        self.to(self.device)

    def _create_positional_encoding(self, max_len: int, d_model: int) -> torch.Tensor:
        position = torch.arange(max_len).unsqueeze(1).float()
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))

        pe = torch.zeros(1, max_len, d_model)
        pe[0, :, 0::2] = torch.sin(position * div_term)
        pe[0, :, 1::2] = torch.cos(position * div_term)

        return nn.Parameter(pe, requires_grad=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch, feature_dim, n_timesteps) or (batch, n_timesteps, feature_dim)

        Returns:
            output: (batch, output_dim)
        """
        # Handle input format
        if x.dim() == 2:
            x = x.unsqueeze(1)

        if x.dim() == 3 and x.shape[1] != self.n_timesteps and x.shape[2] == self.n_timesteps:
            x = x.permute(0, 2, 1)  # (B, C, T) -> (B, T, C)

        # Feature projection
        x = F.silu(self.feature_proj(self.dropout(x)))

        # Quantum feature extraction
        q_features = self.quantum_processor(x)  # (batch, n_chunks, d_model)

        # Add positional encoding
        seq_len = q_features.size(1)
        q_features = q_features + self.pos_encoding[:, :seq_len, :].to(q_features.device)

        # Full transformer attention over ALL chunks
        for transformer in self.transformer_layers:
            q_features = transformer(q_features)

        # Final normalization
        q_features = self.final_norm(q_features)

        # Pool and classify
        q_features = q_features.transpose(1, 2)  # (B, n_chunks, D) -> (B, D, n_chunks)
        pooled = self.pool(q_features).squeeze(-1)  # (B, D)
        output = self.classifier(pooled)

        return output


# ================================================================================
# QuantumHydraTransformer (Bidirectional - Full Sequence Attention)
# ================================================================================

class QuantumHydraTransformer(nn.Module):
    """
    Quantum Feature Extraction + Bidirectional Full Transformer Attention.

    Three branches (like Hydra):
    1. Forward branch: Processes sequence in forward direction
    2. Backward branch: Processes reversed sequence
    3. Global branch: Processes global (mean-pooled) representation

    All branches use FULL sequence self-attention.

    This is consistent with:
    - QuantumHydraSSM: Same quantum features, SSM mixing
    - ClassicalHydraTransformer: Classical features, same attention mixing
    """

    def __init__(
        self,
        n_qubits: int = 6,
        n_timesteps: int = 200,
        qlcu_layers: int = 2,
        feature_dim: int = 4,
        d_model: int = 64,
        d_state: int = 16,  # Not used, kept for interface consistency
        n_layers: int = 1,
        n_heads: int = 4,
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
        self.device = torch.device(device) if isinstance(device, str) else device

        # Feature projection
        self.feature_proj = nn.Linear(feature_dim, feature_dim)
        self.dropout = nn.Dropout(dropout)

        # Quantum feature extraction (shared across branches)
        self.quantum_processor = QuantumFeatureProcessor(
            n_qubits=n_qubits,
            n_layers=qlcu_layers,
            feature_dim=feature_dim,
            hidden_dim=d_model,
            chunk_size=chunk_size,
            device=device
        )

        # Positional encoding
        n_chunks = (n_timesteps + chunk_size - 1) // chunk_size
        self.pos_encoding = self._create_positional_encoding(n_chunks, d_model)

        # Forward branch transformers
        self.forward_layers = nn.ModuleList([
            TransformerBlock(d_model=d_model, n_heads=n_heads, expand=2, dropout=dropout)
            for _ in range(n_layers)
        ])

        # Backward branch transformers
        self.backward_layers = nn.ModuleList([
            TransformerBlock(d_model=d_model, n_heads=n_heads, expand=2, dropout=dropout)
            for _ in range(n_layers)
        ])

        # Global branch transformers
        self.global_layers = nn.ModuleList([
            TransformerBlock(d_model=d_model, n_heads=n_heads, expand=2, dropout=dropout)
            for _ in range(n_layers)
        ])

        # Final normalization for each branch
        self.forward_norm = RMSNorm(d_model)
        self.backward_norm = RMSNorm(d_model)
        self.global_norm = RMSNorm(d_model)

        # Learnable combination weights
        self.alpha = nn.Parameter(torch.ones(1) / 3)
        self.beta = nn.Parameter(torch.ones(1) / 3)
        self.gamma = nn.Parameter(torch.ones(1) / 3)

        # Pooling and classification
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.classifier = nn.Linear(d_model, output_dim)

        self.to(self.device)

    def _create_positional_encoding(self, max_len: int, d_model: int) -> torch.Tensor:
        position = torch.arange(max_len).unsqueeze(1).float()
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))

        pe = torch.zeros(1, max_len, d_model)
        pe[0, :, 0::2] = torch.sin(position * div_term)
        pe[0, :, 1::2] = torch.cos(position * div_term)

        return nn.Parameter(pe, requires_grad=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch, feature_dim, n_timesteps) or (batch, n_timesteps, feature_dim)

        Returns:
            output: (batch, output_dim)
        """
        # Handle input format
        if x.dim() == 2:
            x = x.unsqueeze(1)

        if x.dim() == 3 and x.shape[1] != self.n_timesteps and x.shape[2] == self.n_timesteps:
            x = x.permute(0, 2, 1)

        # Feature projection
        x_proj = F.silu(self.feature_proj(self.dropout(x)))

        # ===== Forward branch =====
        q_forward = self.quantum_processor(x_proj)
        seq_len = q_forward.size(1)
        pos_enc = self.pos_encoding[:, :seq_len, :].to(q_forward.device)
        q_forward = q_forward + pos_enc

        for layer in self.forward_layers:
            q_forward = layer(q_forward)
        q_forward = self.forward_norm(q_forward)
        h_forward = q_forward.transpose(1, 2)
        h_forward = self.pool(h_forward).squeeze(-1)

        # ===== Backward branch =====
        x_flipped = torch.flip(x_proj, dims=[1])
        q_backward = self.quantum_processor(x_flipped)
        q_backward = q_backward + pos_enc

        for layer in self.backward_layers:
            q_backward = layer(q_backward)
        q_backward = self.backward_norm(q_backward)
        h_backward = q_backward.transpose(1, 2)
        h_backward = self.pool(h_backward).squeeze(-1)

        # ===== Global branch =====
        x_global = x_proj.mean(dim=1, keepdim=True)  # Global average
        # Expand to match quantum processor input
        x_global_expanded = x_global.expand(-1, self.quantum_processor.chunk_size, -1)
        q_global = self.quantum_processor.quantum_branches(x_global.squeeze(1))
        q_global = self.quantum_processor.output_proj(q_global)  # (batch, d_model)

        # Process through global transformer (single token)
        q_global = q_global.unsqueeze(1)  # (batch, 1, d_model)
        for layer in self.global_layers:
            q_global = layer(q_global)
        q_global = self.global_norm(q_global)
        h_global = q_global.squeeze(1)

        # ===== Combine branches =====
        weights = F.softmax(torch.stack([self.alpha, self.beta, self.gamma]), dim=0)
        h_combined = weights[0] * h_forward + weights[1] * h_backward + weights[2] * h_global

        # Classification
        output = self.classifier(h_combined)

        return output


# ================================================================================
# Factory Functions
# ================================================================================

def create_quantum_transformer(
    n_qubits: int = 6,
    n_timesteps: int = 200,
    qlcu_layers: int = 2,
    feature_dim: int = 4,
    d_model: int = 64,
    d_state: int = 16,
    n_layers: int = 1,
    n_heads: int = 4,
    output_dim: int = 2,
    dropout: float = 0.1,
    chunk_size: int = 16,
    bidirectional: bool = False,
    device: str = "cpu"
):
    """Factory function to create Quantum Transformer models."""
    if bidirectional:
        return QuantumHydraTransformer(
            n_qubits=n_qubits,
            n_timesteps=n_timesteps,
            qlcu_layers=qlcu_layers,
            feature_dim=feature_dim,
            d_model=d_model,
            d_state=d_state,
            n_layers=n_layers,
            n_heads=n_heads,
            output_dim=output_dim,
            dropout=dropout,
            chunk_size=chunk_size,
            device=device
        )
    else:
        return QuantumTransformer(
            n_qubits=n_qubits,
            n_timesteps=n_timesteps,
            qlcu_layers=qlcu_layers,
            feature_dim=feature_dim,
            d_model=d_model,
            d_state=d_state,
            n_layers=n_layers,
            n_heads=n_heads,
            output_dim=output_dim,
            dropout=dropout,
            chunk_size=chunk_size,
            device=device
        )


# ================================================================================
# Testing
# ================================================================================

if __name__ == "__main__":
    print("=" * 80)
    print("Quantum Transformer Models - Testing")
    print("=" * 80)

    device = "cpu"
    batch_size = 4
    n_channels = 4  # DNA one-hot
    n_timesteps = 200
    n_qubits = 6
    qlcu_layers = 2
    d_model = 64
    n_layers = 1
    output_dim = 2

    print("\n[1] Testing QuantumTransformer...")
    model = QuantumTransformer(
        n_qubits=n_qubits,
        n_timesteps=n_timesteps,
        qlcu_layers=qlcu_layers,
        feature_dim=n_channels,
        d_model=d_model,
        n_layers=n_layers,
        output_dim=output_dim,
        device=device
    )

    x = torch.randn(batch_size, n_channels, n_timesteps)
    output = model(x)
    params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    print(f"  Input: {x.shape}")
    print(f"  Output: {output.shape}")
    print(f"  Parameters: {params:,}")

    print("\n[2] Testing QuantumHydraTransformer...")
    model_hydra = QuantumHydraTransformer(
        n_qubits=n_qubits,
        n_timesteps=n_timesteps,
        qlcu_layers=qlcu_layers,
        feature_dim=n_channels,
        d_model=d_model,
        n_layers=n_layers,
        output_dim=output_dim,
        device=device
    )

    output_hydra = model_hydra(x)
    params_hydra = sum(p.numel() for p in model_hydra.parameters() if p.requires_grad)

    print(f"  Input: {x.shape}")
    print(f"  Output: {output_hydra.shape}")
    print(f"  Parameters: {params_hydra:,}")

    print("\n[3] Testing gradient flow...")
    model.train()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.CrossEntropyLoss()

    output = model(x)
    loss = criterion(output, torch.randint(0, output_dim, (batch_size,)))
    loss.backward()
    optimizer.step()

    print(f"  Loss: {loss.item():.4f}")
    print(f"  Gradient flow: OK")

    print("\n" + "=" * 80)
    print("Testing complete!")
    print("=" * 80)
    print("\nKey Features:")
    print("  - Quantum feature extraction (same as QuantumMambaSSM)")
    print("  - FULL sequence self-attention (not chunked)")
    print("  - Consistent with ClassicalTransformer for fair comparison")
    print("  - Bidirectional version (QuantumHydraTransformer)")
    print("=" * 80)
