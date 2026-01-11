"""
Classical Self-Attention Transformer

Pure classical baseline with self-attention mixing mechanism.
This is the classical equivalent of:
- QuantumMambaAttention / QuantumHydraAttention (Quantum Features → Classical Attention)
- QTSQuantumTransformer (Classical Features → Quantum Attention)

Architecture:
    Input → Classical Embedding → Classical Self-Attention Blocks → Classification

This completes the ablation study grid:
┌─────────────────────┬──────────────────────┬─────────────────────────┐
│ Feature Extraction  │ Mixing Mechanism     │ Model                   │
├─────────────────────┼──────────────────────┼─────────────────────────┤
│ Classical           │ Classical SSM        │ TrueClassicalMamba      │
│ Classical           │ Classical Bidir SSM  │ TrueClassicalHydra      │
│ Classical           │ Classical Attention  │ ClassicalTransformer ★  │
│ Classical           │ Quantum SSM          │ ClassicalMambaQuantumSSM│
│ Classical           │ Quantum Bidir SSM    │ ClassicalHydraQuantumSSM│
│ Classical           │ Quantum Attention    │ ClassicalQuantumAttention│
│ Quantum             │ Classical SSM        │ QuantumMambaSSM         │
│ Quantum             │ Classical Bidir SSM  │ QuantumHydraSSM         │
│ Quantum             │ Classical Attention  │ QuantumMambaAttention   │
└─────────────────────┴──────────────────────┴─────────────────────────┘

Key design choices for fair comparison:
1. Same embedding layer as TrueClassicalMamba/TrueClassicalHydra
2. Same MultiHeadSelfAttention as QuantumAttention.py
3. Same hyperparameter interface (d_model, d_state, n_layers, dropout)
4. Same input/output format

Author: Junghoon Park
Date: December 2024
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional


# ================================================================================
# RMSNorm (same as TrueClassicalMamba for consistency)
# ================================================================================

class RMSNorm(nn.Module):
    """Root Mean Square Layer Normalization."""

    def __init__(self, d_model: int, eps: float = 1e-5):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(d_model))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch, seq_len, d_model)
        """
        rms = torch.sqrt(torch.mean(x ** 2, dim=-1, keepdim=True) + self.eps)
        x_normalized = x / rms
        return self.weight * x_normalized


# ================================================================================
# Multi-Head Self-Attention (consistent with QuantumAttention.py)
# ================================================================================

class MultiHeadSelfAttention(nn.Module):
    """
    Standard multi-head self-attention mechanism.

    Computes: Attention(Q, K, V) = softmax(QK^T / sqrt(d_k)) V

    This provides global information mixing across all positions,
    unlike SSM (recurrent) processing.

    Identical to the implementation in QuantumAttention.py for fair comparison.
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
# Transformer Block (similar structure to MambaBlock for fair comparison)
# ================================================================================

class TransformerBlock(nn.Module):
    """
    Transformer block with self-attention and gated MLP.

    Architecture (similar to MambaBlock for fair comparison):
    1. RMSNorm + Multi-head self-attention
    2. Residual connection
    3. RMSNorm + Gated MLP (SiLU activation, like Mamba)
    4. Residual connection

    This mirrors MambaBlock's structure but replaces SSM with attention.
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

        # Normalization layers (RMSNorm like MambaBlock)
        self.norm1 = RMSNorm(d_model)
        self.norm2 = RMSNorm(d_model)

        # Self-attention
        self.attention = MultiHeadSelfAttention(
            d_model=d_model,
            n_heads=n_heads,
            dropout=dropout,
            bias=bias
        )

        # Gated MLP (like MambaBlock's gating mechanism)
        # Projects to 2x expanded dimension: one for gate, one for value
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
        # Self-attention with pre-norm and residual
        x = x + self.dropout(self.attention(self.norm1(x)))

        # Gated MLP with pre-norm and residual
        x_norm = self.norm2(x)
        x_proj = self.in_proj(x_norm)
        x_gate, x_val = x_proj.chunk(2, dim=-1)

        # Gated activation (SiLU gating like Mamba)
        x_mlp = F.silu(x_gate) * x_val
        x_mlp = self.out_proj(x_mlp)

        x = x + self.dropout(x_mlp)

        return x


# ================================================================================
# Classical Transformer (Unidirectional - equivalent to ClassicalMamba)
# ================================================================================

class ClassicalTransformer(nn.Module):
    """
    Classical Self-Attention Transformer for sequence classification.

    This is the classical attention baseline, equivalent to:
    - TrueClassicalMamba (but with attention instead of SSM)
    - QuantumMambaAttention (but without quantum feature extraction)

    Architecture:
        Input → Embedding → TransformerBlocks → Pooling → Classifier

    Uses the same hyperparameter interface as TrueClassicalMamba for fair comparison.
    """

    def __init__(
        self,
        n_channels: int,
        n_timesteps: int,
        d_model: int = 64,
        d_state: int = 16,  # Not used, but kept for interface consistency with SSM models
        n_layers: int = 1,
        n_heads: int = 4,
        expand: int = 2,
        output_dim: int = 2,
        dropout: float = 0.1,
        device: str = "cpu",
    ):
        super().__init__()

        self.n_channels = n_channels
        self.n_timesteps = n_timesteps
        self.d_model = d_model
        self.d_state = d_state  # Stored for interface consistency
        self.n_layers = n_layers
        self.n_heads = n_heads
        self.device = device

        # Input embedding (same as TrueClassicalMamba)
        self.embedding = nn.Linear(n_channels, d_model)
        self.dropout = nn.Dropout(dropout)

        # Positional encoding (sinusoidal, standard for transformers)
        self.pos_encoding = self._create_positional_encoding(n_timesteps, d_model)

        # Transformer blocks
        self.layers = nn.ModuleList([
            TransformerBlock(
                d_model=d_model,
                n_heads=n_heads,
                expand=expand,
                dropout=dropout,
            )
            for _ in range(n_layers)
        ])

        # Final normalization
        self.final_norm = RMSNorm(d_model)

        # Pooling and classification (same as TrueClassicalMamba)
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.classifier = nn.Linear(d_model, output_dim)

        # Move to device
        self.to(device)

    def _create_positional_encoding(self, max_len: int, d_model: int) -> torch.Tensor:
        """Create sinusoidal positional encoding."""
        position = torch.arange(max_len).unsqueeze(1).float()
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))

        pe = torch.zeros(1, max_len, d_model)
        pe[0, :, 0::2] = torch.sin(position * div_term)
        pe[0, :, 1::2] = torch.cos(position * div_term)

        return nn.Parameter(pe, requires_grad=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch, n_channels, n_timesteps) for 3D inputs
               OR (batch, features) for 2D inputs

        Returns:
            output: (batch, output_dim)
        """
        # Handle both 2D and 3D inputs (same as TrueClassicalMamba)
        if x.dim() == 2:
            x = x.unsqueeze(1)
        elif x.dim() != 3:
            raise ValueError(f"Expected 2D or 3D input, got {x.dim()}D tensor")

        # (B, C, T) -> (B, T, C)
        x = x.transpose(1, 2)

        # Embedding
        x = self.embedding(x)
        x = self.dropout(x)

        # Add positional encoding
        seq_len = x.size(1)
        x = x + self.pos_encoding[:, :seq_len, :].to(x.device)

        # Transformer blocks
        for layer in self.layers:
            x = layer(x)

        # Final normalization
        x = self.final_norm(x)

        # Pool and classify (same as TrueClassicalMamba)
        x = x.transpose(1, 2)  # (B, T, D) -> (B, D, T)
        x = self.pool(x).squeeze(-1)  # (B, D)
        output = self.classifier(x)

        return output


# ================================================================================
# Bidirectional Classical Transformer (equivalent to ClassicalHydra)
# ================================================================================

class ClassicalHydraTransformer(nn.Module):
    """
    Bidirectional Classical Self-Attention Transformer.

    This is the classical bidirectional attention baseline, equivalent to:
    - TrueClassicalHydra (but with attention instead of SSM)
    - QuantumHydraAttention (but without quantum feature extraction)

    Architecture (three branches like Hydra):
        Branch 1: Forward Transformer
        Branch 2: Backward Transformer
        Branch 3: Global Transformer

        Output: Learned combination of all branches

    Uses the same hyperparameter interface as TrueClassicalHydra for fair comparison.
    """

    def __init__(
        self,
        n_channels: int,
        n_timesteps: int,
        d_model: int = 64,
        d_state: int = 16,  # Not used, but kept for interface consistency
        n_layers: int = 1,
        n_heads: int = 4,
        expand: int = 2,
        output_dim: int = 2,
        dropout: float = 0.1,
        device: str = "cpu",
    ):
        super().__init__()

        self.n_channels = n_channels
        self.n_timesteps = n_timesteps
        self.d_model = d_model
        self.d_state = d_state
        self.n_layers = n_layers
        self.n_heads = n_heads
        self.device = device

        # Input embedding (shared across branches)
        self.embedding = nn.Linear(n_channels, d_model)
        self.dropout = nn.Dropout(dropout)

        # Positional encoding
        self.pos_encoding = self._create_positional_encoding(n_timesteps, d_model)

        # Forward branch
        self.forward_layers = nn.ModuleList([
            TransformerBlock(d_model=d_model, n_heads=n_heads, expand=expand, dropout=dropout)
            for _ in range(n_layers)
        ])

        # Backward branch
        self.backward_layers = nn.ModuleList([
            TransformerBlock(d_model=d_model, n_heads=n_heads, expand=expand, dropout=dropout)
            for _ in range(n_layers)
        ])

        # Global branch (processes mean-pooled representation)
        self.global_layers = nn.ModuleList([
            TransformerBlock(d_model=d_model, n_heads=n_heads, expand=expand, dropout=dropout)
            for _ in range(n_layers)
        ])

        # Final normalization for each branch
        self.forward_norm = RMSNorm(d_model)
        self.backward_norm = RMSNorm(d_model)
        self.global_norm = RMSNorm(d_model)

        # Learnable combination weights (like Hydra's complex coefficients)
        self.alpha = nn.Parameter(torch.ones(1) / 3)
        self.beta = nn.Parameter(torch.ones(1) / 3)
        self.gamma = nn.Parameter(torch.ones(1) / 3)

        # Pooling and classification
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.classifier = nn.Linear(d_model, output_dim)

        # Move to device
        self.to(device)

    def _create_positional_encoding(self, max_len: int, d_model: int) -> torch.Tensor:
        """Create sinusoidal positional encoding."""
        position = torch.arange(max_len).unsqueeze(1).float()
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))

        pe = torch.zeros(1, max_len, d_model)
        pe[0, :, 0::2] = torch.sin(position * div_term)
        pe[0, :, 1::2] = torch.cos(position * div_term)

        return nn.Parameter(pe, requires_grad=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch, n_channels, n_timesteps) for 3D inputs
               OR (batch, features) for 2D inputs

        Returns:
            output: (batch, output_dim)
        """
        # Handle both 2D and 3D inputs
        if x.dim() == 2:
            x = x.unsqueeze(1)
        elif x.dim() != 3:
            raise ValueError(f"Expected 2D or 3D input, got {x.dim()}D tensor")

        # (B, C, T) -> (B, T, C)
        x = x.transpose(1, 2)

        # Embedding
        x = self.embedding(x)
        x = self.dropout(x)

        # Add positional encoding
        seq_len = x.size(1)
        pos_enc = self.pos_encoding[:, :seq_len, :].to(x.device)

        # ===== Forward branch =====
        x_forward = x + pos_enc
        for layer in self.forward_layers:
            x_forward = layer(x_forward)
        x_forward = self.forward_norm(x_forward)
        # Pool: (B, T, D) -> (B, D)
        h_forward = x_forward.transpose(1, 2)
        h_forward = self.pool(h_forward).squeeze(-1)

        # ===== Backward branch =====
        x_backward = torch.flip(x, dims=[1]) + pos_enc  # Flip sequence
        for layer in self.backward_layers:
            x_backward = layer(x_backward)
        x_backward = self.backward_norm(x_backward)
        h_backward = x_backward.transpose(1, 2)
        h_backward = self.pool(h_backward).squeeze(-1)

        # ===== Global branch =====
        # Use mean-pooled representation as a single "token"
        x_global = x.mean(dim=1, keepdim=True)  # (B, 1, D)
        x_global = x_global + pos_enc[:, :1, :]  # Single position encoding
        for layer in self.global_layers:
            x_global = layer(x_global)
        x_global = self.global_norm(x_global)
        h_global = x_global.squeeze(1)  # (B, D)

        # ===== Combine branches =====
        # Normalize weights to sum to 1 (softmax-like)
        weights = F.softmax(torch.stack([self.alpha, self.beta, self.gamma]), dim=0)

        h_combined = (
            weights[0] * h_forward +
            weights[1] * h_backward +
            weights[2] * h_global
        )

        # Classification
        output = self.classifier(h_combined)

        return output


# ================================================================================
# Factory function for easy model creation
# ================================================================================

def create_classical_transformer(
    n_channels: int,
    n_timesteps: int,
    d_model: int = 64,
    d_state: int = 16,
    n_layers: int = 1,
    n_heads: int = 4,
    expand: int = 2,
    output_dim: int = 2,
    dropout: float = 0.1,
    bidirectional: bool = False,
    device: str = "cpu",
):
    """
    Factory function to create Classical Transformer models.

    Args:
        n_channels: Number of input channels
        n_timesteps: Sequence length
        d_model: Model dimension
        d_state: State dimension (for interface consistency with SSM models)
        n_layers: Number of transformer blocks
        n_heads: Number of attention heads
        expand: MLP expansion factor
        output_dim: Number of output classes
        dropout: Dropout rate
        bidirectional: If True, creates ClassicalHydraTransformer
        device: Device for computation

    Returns:
        ClassicalTransformer or ClassicalHydraTransformer
    """
    if bidirectional:
        return ClassicalHydraTransformer(
            n_channels=n_channels,
            n_timesteps=n_timesteps,
            d_model=d_model,
            d_state=d_state,
            n_layers=n_layers,
            n_heads=n_heads,
            expand=expand,
            output_dim=output_dim,
            dropout=dropout,
            device=device,
        )
    else:
        return ClassicalTransformer(
            n_channels=n_channels,
            n_timesteps=n_timesteps,
            d_model=d_model,
            d_state=d_state,
            n_layers=n_layers,
            n_heads=n_heads,
            expand=expand,
            output_dim=output_dim,
            dropout=dropout,
            device=device,
        )


# ================================================================================
# Testing
# ================================================================================

if __name__ == "__main__":
    print("=" * 80)
    print("Classical Self-Attention Transformer - Testing")
    print("=" * 80)

    device = "cpu"
    batch_size = 4
    n_channels = 4  # DNA one-hot encoding
    n_timesteps = 200
    d_model = 64
    d_state = 16
    n_layers = 1
    n_heads = 4
    output_dim = 2

    print("\n[1] Testing MultiHeadSelfAttention...")
    attention = MultiHeadSelfAttention(d_model=d_model, n_heads=n_heads)
    x_seq = torch.randn(batch_size, n_timesteps, d_model)
    attn_out = attention(x_seq)
    print(f"  Input: {x_seq.shape}")
    print(f"  Output: {attn_out.shape}")
    assert attn_out.shape == x_seq.shape, "Attention output shape mismatch!"

    print("\n[2] Testing TransformerBlock...")
    block = TransformerBlock(d_model=d_model, n_heads=n_heads)
    block_out = block(x_seq)
    print(f"  Input: {x_seq.shape}")
    print(f"  Output: {block_out.shape}")
    assert block_out.shape == x_seq.shape, "Block output shape mismatch!"

    print("\n[3] Testing ClassicalTransformer (full model)...")
    model = ClassicalTransformer(
        n_channels=n_channels,
        n_timesteps=n_timesteps,
        d_model=d_model,
        d_state=d_state,
        n_layers=n_layers,
        n_heads=n_heads,
        output_dim=output_dim,
        device=device
    )

    # Test with 3D input: (B, C, T)
    x_3d = torch.randn(batch_size, n_channels, n_timesteps)
    output = model(x_3d)
    print(f"  Input (3D): {x_3d.shape}")
    print(f"  Output: {output.shape}")
    assert output.shape == (batch_size, output_dim), "Model output shape mismatch!"

    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Parameters: {total_params:,}")

    print("\n[4] Testing ClassicalHydraTransformer (bidirectional)...")
    model_hydra = ClassicalHydraTransformer(
        n_channels=n_channels,
        n_timesteps=n_timesteps,
        d_model=d_model,
        d_state=d_state,
        n_layers=n_layers,
        n_heads=n_heads,
        output_dim=output_dim,
        device=device
    )

    output_hydra = model_hydra(x_3d)
    print(f"  Input (3D): {x_3d.shape}")
    print(f"  Output: {output_hydra.shape}")
    assert output_hydra.shape == (batch_size, output_dim), "Hydra output shape mismatch!"

    total_params_hydra = sum(p.numel() for p in model_hydra.parameters() if p.requires_grad)
    print(f"  Parameters: {total_params_hydra:,}")

    print("\n[5] Testing gradient flow...")
    model.train()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.CrossEntropyLoss()

    output = model(x_3d)
    loss = criterion(output, torch.randint(0, output_dim, (batch_size,)))
    loss.backward()
    optimizer.step()

    print(f"  Loss: {loss.item():.4f}")
    print(f"  Gradient flow: OK")

    print("\n[6] Comparing parameter counts with SSM models...")
    # Import TrueClassicalMamba for comparison
    try:
        from TrueClassicalMamba import TrueClassicalMamba
        mamba = TrueClassicalMamba(
            n_channels=n_channels,
            n_timesteps=n_timesteps,
            d_model=d_model,
            d_state=d_state,
            n_layers=n_layers,
            output_dim=output_dim,
            device=device
        )
        mamba_params = sum(p.numel() for p in mamba.parameters() if p.requires_grad)
        print(f"  ClassicalTransformer: {total_params:,}")
        print(f"  TrueClassicalMamba:   {mamba_params:,}")
    except ImportError:
        print(f"  ClassicalTransformer: {total_params:,}")
        print("  (TrueClassicalMamba not available for comparison)")

    print("\n" + "=" * 80)
    print("Testing complete!")
    print("=" * 80)
    print("\nKey Features:")
    print("  - Pure classical self-attention (no quantum components)")
    print("  - Same interface as TrueClassicalMamba/TrueClassicalHydra")
    print("  - Gated MLP like Mamba (SiLU activation)")
    print("  - RMSNorm for normalization")
    print("  - Sinusoidal positional encoding")
    print("  - Bidirectional version (ClassicalHydraTransformer)")
    print("=" * 80)
    print("\nComparison of mixing mechanisms:")
    print("  [SSM]       h = A*h + B*x              (State space, sequential)")
    print("  [Attention] y = softmax(QK^T/sqrt(d))V (Self-attention, global)")
    print("=" * 80)
