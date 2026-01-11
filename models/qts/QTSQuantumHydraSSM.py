"""
QTS + Quantum Hydra SSM (Model C1a)

Classical Feature Extraction (QTSTransformer Encoder) + Quantum Hydra SSM (QSVT/LCU)

This model combines:
1. QTSFeatureEncoder: Classical feature extraction IDENTICAL to QTSTransformer
2. QuantumHydraSSMCore: Quantum bidirectional SSM with QSVT selective forgetting

Architecture:
    Input → QTSFeatureEncoder (CLASSICAL) → QuantumHydraSSMCore (QUANTUM) → Output

The encoder is shared with QTSTransformer for controlled ablation studies.
Only the mixing mechanism differs:
- QTSTransformer: Quantum Attention
- This model: Quantum Hydra SSM (bidirectional QSVT)

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
from models.core.quantum_cores.quantum_hydra_ssm_core import QuantumHydraSSMCore


class SequencePooling(nn.Module):
    """
    Attention-weighted sequence pooling.

    Learns which timesteps are most important for classification
    and computes a weighted average.
    """

    def __init__(self, d_model: int):
        super().__init__()
        self.attention = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.Tanh(),
            nn.Linear(d_model // 2, 1)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch, seq_len, d_model)
        Returns:
            (batch, d_model) pooled representation
        """
        # Compute attention scores
        attn_scores = self.attention(x)  # (batch, seq_len, 1)
        attn_weights = torch.softmax(attn_scores, dim=1)

        # Weighted sum
        pooled = (attn_weights * x).sum(dim=1)  # (batch, d_model)
        return pooled


class QTSQuantumHydraSSM(nn.Module):
    """
    QTS Encoder + Quantum Hydra SSM

    This is Model C1a in the ablation study framework.

    Components:
    - encoder: QTSFeatureEncoder (CLASSICAL, identical to QTSTransformer)
    - quantum_ssm: QuantumHydraSSMCore (QUANTUM selective forgetting via QSVT)
    - output layers: Projection, pooling, and classification head

    The model uses:
    - Classical Conv2d + GLU for feature extraction (proven effective)
    - Quantum bidirectional SSM for sequence mixing (novel contribution)
    - QSVT polynomial for state transitions (quantum selective forgetting)
    - LCU for combining forward/backward passes (quantum combination)

    Comparison with QTSTransformer:
    - Same encoder (QTSFeatureEncoder)
    - Different mixing: SSM vs Attention
    - Both use QSVT but for different purposes

    Args:
        feature_dim: Input feature dimension (e.g., 4 for genomic one-hot)
        n_timesteps: Sequence length (e.g., 200 for genomic sequences)
        num_classes: Number of output classes
        n_qubits: Number of qubits for quantum circuit
        n_layers: Number of variational layers in quantum circuit
        qsvt_degree: Degree of QSVT polynomial
        projection_type: Encoder projection type ('Linear', 'Conv2d', 'Conv2d_GLU')
        dropout: Dropout rate
        device: Device for computation

    Example:
        model = QTSQuantumHydraSSM(
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
        self.device_str = device

        # ============================================
        # CLASSICAL: QTS Feature Encoder
        # IDENTICAL to QTSTransformer encoder
        # This ensures controlled comparison
        # ============================================
        self.encoder = create_qts_encoder(
            feature_dim=feature_dim,
            n_timesteps=n_timesteps,
            n_qubits=n_qubits,
            projection_type=projection_type,
            dropout=dropout
        )

        # ============================================
        # QUANTUM: Hydra SSM Core
        # Bidirectional quantum SSM with QSVT
        # This is where quantum selective forgetting happens
        # ============================================
        self.quantum_ssm = QuantumHydraSSMCore(
            n_qubits=n_qubits,
            n_layers=n_layers,
            qsvt_degree=qsvt_degree,
            device=device
        )

        # ============================================
        # OUTPUT LAYERS
        # Project quantum output and classify
        # ============================================
        # Project quantum output (n_qubits) to larger dimension
        self.output_proj = nn.Linear(n_qubits, n_qubits * 3)
        self.layer_norm = nn.LayerNorm(n_qubits * 3)

        # Sequence pooling
        self.pooling = SequencePooling(n_qubits * 3)

        # Classification head
        self.classifier = nn.Sequential(
            nn.Linear(n_qubits * 3, n_qubits * 3),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(n_qubits * 3, num_classes)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Flow:
        1. Classical encoder extracts features and converts to angles
        2. Quantum Hydra SSM processes sequence bidirectionally
        3. Residual connection adds input information
        4. Output projection, pooling, and classification

        Args:
            x: (batch, feature_dim, n_timesteps) or (batch, n_timesteps, feature_dim)

        Returns:
            (batch, num_classes) classification logits
        """
        # ============================================
        # STEP 1: CLASSICAL Feature Extraction
        # QTSFeatureEncoder: Conv2d → GLU → Sigmoid(·π)
        # Output: angles in [0, π] for quantum encoding
        # ============================================
        angles = self.encoder(x)  # (batch, seq_len, n_qubits)

        # ============================================
        # STEP 2: QUANTUM Bidirectional SSM
        # QuantumHydraSSMCore: QSVT + LCU
        # - Forward pass: left-to-right
        # - Backward pass: right-to-left
        # - LCU combination
        # ============================================
        quantum_out = self.quantum_ssm(angles)  # (batch, seq_len, n_qubits)

        # ============================================
        # STEP 3: Residual Connection
        # Helps gradient flow and preserves input info
        # ============================================
        residual = quantum_out + angles

        # ============================================
        # STEP 4: Output Processing
        # Project → Normalize → Pool → Classify
        # ============================================
        projected = self.output_proj(residual)  # (batch, seq_len, n_qubits*3)
        projected = self.layer_norm(projected)

        pooled = self.pooling(projected)  # (batch, n_qubits*3)

        logits = self.classifier(pooled)  # (batch, num_classes)

        return logits

    def get_param_count(self) -> Dict[str, int]:
        """Return parameter counts by component."""
        encoder_params = sum(p.numel() for p in self.encoder.parameters())
        quantum_params = sum(p.numel() for p in self.quantum_ssm.parameters())
        output_params = sum(p.numel() for p in self.output_proj.parameters())
        output_params += sum(p.numel() for p in self.layer_norm.parameters())
        output_params += sum(p.numel() for p in self.pooling.parameters())
        output_params += sum(p.numel() for p in self.classifier.parameters())

        return {
            'encoder (CLASSICAL)': encoder_params,
            'quantum_ssm (QUANTUM)': quantum_params,
            'output (classical)': output_params,
            'total': encoder_params + quantum_params + output_params
        }

    def get_config(self) -> Dict[str, Any]:
        """Return model configuration."""
        return {
            'model_type': 'QTSQuantumHydraSSM',
            'feature_dim': self.feature_dim,
            'n_timesteps': self.n_timesteps,
            'num_classes': self.num_classes,
            'n_qubits': self.n_qubits,
            'encoder_type': self.encoder.projection_type,
            'quantum_layers': self.quantum_ssm.n_layers,
            'qsvt_degree': self.quantum_ssm.qsvt_degree,
        }

    def __repr__(self) -> str:
        params = self.get_param_count()
        return (
            f"QTSQuantumHydraSSM(\n"
            f"  feature_dim={self.feature_dim},\n"
            f"  n_timesteps={self.n_timesteps},\n"
            f"  num_classes={self.num_classes},\n"
            f"  n_qubits={self.n_qubits},\n"
            f"  encoder={self.encoder.projection_type},\n"
            f"  params={{\n"
            f"    encoder (CLASSICAL): {params['encoder (CLASSICAL)']:,},\n"
            f"    quantum_ssm (QUANTUM): {params['quantum_ssm (QUANTUM)']:,},\n"
            f"    output: {params['output (classical)']:,},\n"
            f"    total: {params['total']:,}\n"
            f"  }}\n"
            f")"
        )


def create_qts_quantum_hydra_ssm(
    feature_dim: int,
    n_timesteps: int,
    num_classes: int,
    n_qubits: int = 4,
    n_layers: int = 2,
    qsvt_degree: int = 3,
    projection_type: str = 'Conv2d_GLU',
    dropout: float = 0.1,
    device: str = "cpu"
) -> QTSQuantumHydraSSM:
    """
    Convenience function to create QTS + Quantum Hydra SSM model.

    This is Model C1a in the ablation study.

    Args:
        feature_dim: Input feature dimension
        n_timesteps: Sequence length
        num_classes: Number of output classes
        n_qubits: Number of qubits
        n_layers: Variational circuit layers
        qsvt_degree: QSVT polynomial degree
        projection_type: Encoder projection type
        dropout: Dropout rate
        device: Computation device

    Returns:
        QTSQuantumHydraSSM model instance
    """
    return QTSQuantumHydraSSM(
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
__all__ = ['QTSQuantumHydraSSM', 'create_qts_quantum_hydra_ssm', 'SequencePooling']
