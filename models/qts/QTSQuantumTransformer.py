"""
QTS + Quantum Transformer (Ablation Study Baseline)

Classical Feature Extraction (QTSFeatureEncoder) + Quantum Attention (QSVT/LCU)

This model is the BASELINE for the ablation study, using:
1. QTSFeatureEncoder: IDENTICAL classical encoder as SSM models
2. QuantumAttentionCore: Original QuantumTSTransformer's quantum attention mechanism

Architecture:
    Input → QTSFeatureEncoder (CLASSICAL) → QuantumAttentionCore (QUANTUM) → Output

CRITICAL: The encoder is IDENTICAL to QTSQuantumHydraSSMAdvanced and
QTSQuantumMambaSSMAdvanced, ensuring fair comparison where ONLY the quantum
mixing mechanism differs:
- QTSQuantumTransformer: Quantum Attention (QSVT + LCU timestep mixing)
- QTSQuantumHydraSSMAdvanced: Quantum Bidirectional SSM (QSVT + LCU combination)
- QTSQuantumMambaSSMAdvanced: Quantum Selective SSM (QSVT + input-dependent Δ)

Key Differences in Mixing Mechanism:
-----------------------------------
| Model | Mixing Type | Key Feature |
|-------|-------------|-------------|
| Transformer | Global Attention | LCU mixes ALL timesteps simultaneously |
| Hydra SSM | Bidirectional | Forward + Backward sequential passes + LCU |
| Mamba SSM | Selective | Input-dependent Δ controls forgetting |

The Transformer uses GLOBAL attention where each timestep influences
the final output through learned LCU coefficients, while SSMs use
SEQUENTIAL processing with state evolution.

Author: Junghoon Park
Date: December 2024
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Optional, Dict, Any

# Import shared components
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.core.encoders.qts_encoder import QTSFeatureEncoder, create_qts_encoder
from models.core.quantum_cores.quantum_attention_core import QuantumAttentionCore


class QTSQuantumTransformer(nn.Module):
    """
    QTS Encoder + Quantum Attention (Transformer)

    This is the BASELINE model for ablation studies.

    Components:
    - encoder: QTSFeatureEncoder (CLASSICAL, identical to SSM models)
    - quantum_attention: QuantumAttentionCore (QUANTUM attention via QSVT + LCU)
    - output_ff: Final feed-forward layer for classification

    The model uses:
    - Classical Conv2d + GLU for feature extraction (shared with SSM models)
    - Quantum attention for sequence mixing (global timestep mixing)
    - QSVT polynomial for quantum state transformation
    - LCU for combining information across ALL timesteps simultaneously

    Quantum Attention Mechanism:
    1. Each timestep t has parameters θ_t from the encoder
    2. sim14 circuit transforms quantum state: U(θ_t)|ψ⟩
    3. LCU combines timesteps: |ψ_mixed⟩ = Σ_t α_t U(θ_t)|ψ⟩
    4. QSVT polynomial: P(LCU)|0⟩
    5. Multi-observable measurement: ⟨X⟩, ⟨Y⟩, ⟨Z⟩

    This is fundamentally different from SSM:
    - Attention: Global mixing (all timesteps at once)
    - SSM: Sequential evolution h[t] = f(h[t-1], x[t])

    Args:
        feature_dim: Input feature dimension
        n_timesteps: Sequence length
        num_classes: Number of output classes
        n_qubits: Number of qubits for quantum circuit
        n_layers: Number of sim14 variational layers
        qsvt_degree: Degree of QSVT polynomial
        projection_type: Encoder projection type ('Conv2d_GLU' recommended)
        dropout: Dropout rate
        device: Device for computation

    Example:
        model = QTSQuantumTransformer(
            feature_dim=4,
            n_timesteps=200,
            num_classes=2,
            n_qubits=4,
            device='cuda'
        )
        x = torch.randn(32, 4, 200)  # (batch, features, seq_len)
        logits = model(x)  # (batch, num_classes)
    """

    def __init__(
        self,
        feature_dim: int,
        n_timesteps: int,
        num_classes: int,
        n_qubits: int = 4,
        n_layers: int = 2,
        qsvt_degree: int = 3,
        projection_type: str = 'Conv2d_GLU',
        dropout: float = 0.1,
        device: str = "cpu"
    ):
        super().__init__()

        self.feature_dim = feature_dim
        self.n_timesteps = n_timesteps
        self.num_classes = num_classes
        self.n_qubits = n_qubits
        self.n_layers = n_layers
        self.qsvt_degree = qsvt_degree
        self.device_str = device

        # Number of rotations per timestep: 4 × n_qubits × n_layers (sim14)
        self.n_rots = 4 * n_qubits * n_layers

        # Output dimension from quantum attention: 3 × n_qubits
        self.quantum_output_dim = 3 * n_qubits

        # ============================================
        # CLASSICAL: QTS Feature Encoder
        # IDENTICAL to SSM models for fair comparison
        # Output: (batch, n_timesteps, n_qubits) angles in [0, π]
        # ============================================
        self.encoder = create_qts_encoder(
            feature_dim=feature_dim,
            n_timesteps=n_timesteps,
            n_qubits=n_qubits,
            projection_type=projection_type,
            dropout=dropout
        )

        # ============================================
        # ADDITIONAL: Projection to sim14 parameters
        # Encoder outputs n_qubits angles, but sim14 needs n_rots parameters
        # This matches original QuantumTSTransformer's architecture
        # ============================================
        self.param_projection = nn.Sequential(
            nn.Linear(n_qubits, self.n_rots),
            nn.Sigmoid()  # Scale to [0, 1] for circuit parameters
        )

        # ============================================
        # QUANTUM: Attention Core (QSVT + LCU)
        # This is where quantum attention happens
        # ============================================
        self.quantum_attention = QuantumAttentionCore(
            n_qubits=n_qubits,
            n_timesteps=n_timesteps,
            n_layers=n_layers,
            qsvt_degree=qsvt_degree,
            device=device
        )

        # ============================================
        # OUTPUT: Feed-Forward for Classification
        # Input: 3 × n_qubits from multi-observable measurement
        # ============================================
        self.output_ff = nn.Linear(self.quantum_output_dim, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Flow:
        1. Classical encoder extracts features → angles [0, π]
        2. Project to sim14 parameters → [0, 1]
        3. Quantum attention processes ALL timesteps via QSVT + LCU
        4. Multi-observable measurement → classification

        Args:
            x: (batch, feature_dim, n_timesteps) or (batch, n_timesteps, feature_dim)

        Returns:
            (batch, num_classes) classification logits
        """
        # ============================================
        # STEP 1: CLASSICAL Feature Extraction
        # QTSFeatureEncoder: Conv2d → GLU → Sigmoid(·π)
        # Output: angles in [0, π]
        # ============================================
        angles = self.encoder(x)  # (batch, n_timesteps, n_qubits)

        # ============================================
        # STEP 2: Project to sim14 parameters
        # Expand from n_qubits to n_rots = 4 × n_qubits × n_layers
        # ============================================
        timestep_params = self.param_projection(angles)  # (batch, n_timesteps, n_rots)

        # ============================================
        # STEP 3: QUANTUM Attention
        # QuantumAttentionCore: QSVT + LCU timestep mixing
        #
        # Key mechanism:
        # 1. Each timestep t: U(θ_t)|ψ⟩ via sim14 circuit
        # 2. LCU: |ψ_mixed⟩ = Σ_t α_t U(θ_t)|ψ⟩
        # 3. QSVT: P(LCU)|0⟩
        # 4. Measure: ⟨X⟩, ⟨Y⟩, ⟨Z⟩
        #
        # This is GLOBAL ATTENTION - all timesteps mixed simultaneously
        # Unlike SSM which processes sequentially
        # ============================================
        quantum_out = self.quantum_attention(timestep_params)  # (batch, 3*n_qubits)

        # ============================================
        # STEP 4: Classification
        # ============================================
        logits = self.output_ff(quantum_out)  # (batch, num_classes)

        return logits

    def get_param_count(self) -> Dict[str, int]:
        """Return parameter counts by component."""
        encoder_params = sum(p.numel() for p in self.encoder.parameters())
        projection_params = sum(p.numel() for p in self.param_projection.parameters())
        quantum_params = sum(p.numel() for p in self.quantum_attention.parameters())
        output_params = sum(p.numel() for p in self.output_ff.parameters())

        return {
            'encoder (CLASSICAL)': encoder_params,
            'param_projection (classical)': projection_params,
            'quantum_attention (QUANTUM)': quantum_params,
            'output_ff (classical)': output_params,
            'total': encoder_params + projection_params + quantum_params + output_params
        }

    def get_config(self) -> Dict[str, Any]:
        """Return model configuration."""
        return {
            'model_type': 'QTSQuantumTransformer',
            'feature_dim': self.feature_dim,
            'n_timesteps': self.n_timesteps,
            'num_classes': self.num_classes,
            'n_qubits': self.n_qubits,
            'n_layers': self.n_layers,
            'encoder_type': self.encoder.projection_type,
            'qsvt_degree': self.qsvt_degree,
            'mixing_mechanism': 'QSVT + LCU global attention',
            'measurement': 'multi-observable (X,Y,Z)',
            'quantum_output_dim': self.quantum_output_dim,
        }

    def __repr__(self) -> str:
        params = self.get_param_count()
        return (
            f"QTSQuantumTransformer(\n"
            f"  feature_dim={self.feature_dim},\n"
            f"  n_timesteps={self.n_timesteps},\n"
            f"  num_classes={self.num_classes},\n"
            f"  n_qubits={self.n_qubits},\n"
            f"  n_layers={self.n_layers},\n"
            f"  encoder={self.encoder.projection_type},\n"
            f"  mixing='QSVT + LCU global attention',\n"
            f"  measurement='multi-observable (X,Y,Z)',\n"
            f"  params={{\n"
            f"    encoder (CLASSICAL): {params['encoder (CLASSICAL)']:,},\n"
            f"    param_projection: {params['param_projection (classical)']:,},\n"
            f"    quantum_attention (QUANTUM): {params['quantum_attention (QUANTUM)']:,},\n"
            f"    output_ff: {params['output_ff (classical)']:,},\n"
            f"    total: {params['total']:,}\n"
            f"  }}\n"
            f")"
        )


def create_qts_quantum_transformer(
    feature_dim: int,
    n_timesteps: int,
    num_classes: int,
    n_qubits: int = 4,
    n_layers: int = 2,
    qsvt_degree: int = 3,
    projection_type: str = 'Conv2d_GLU',
    dropout: float = 0.1,
    device: str = "cpu"
) -> QTSQuantumTransformer:
    """
    Convenience function to create QTS + Quantum Transformer model.

    This is the BASELINE model for ablation studies.

    Args:
        feature_dim: Input feature dimension
        n_timesteps: Sequence length
        num_classes: Number of output classes
        n_qubits: Number of qubits
        n_layers: sim14 variational circuit layers
        qsvt_degree: QSVT polynomial degree
        projection_type: Encoder projection type
        dropout: Dropout rate
        device: Computation device

    Returns:
        QTSQuantumTransformer model instance
    """
    return QTSQuantumTransformer(
        feature_dim=feature_dim,
        n_timesteps=n_timesteps,
        num_classes=num_classes,
        n_qubits=n_qubits,
        n_layers=n_layers,
        qsvt_degree=qsvt_degree,
        projection_type=projection_type,
        dropout=dropout,
        device=device
    )


# Export
__all__ = ['QTSQuantumTransformer', 'create_qts_quantum_transformer']
