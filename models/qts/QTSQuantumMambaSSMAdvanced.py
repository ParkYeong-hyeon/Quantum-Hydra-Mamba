"""
QTS + Advanced Quantum Mamba SSM (Model C1b-Advanced)

Classical Feature Extraction (QTSTransformer Encoder) + Advanced Quantum Mamba SSM

ADVANCED VERSION with:
1. sim14 ansatz from Sim et al. (2019) - proven higher expressivity
2. Multi-observable measurement (PauliX, PauliY, PauliZ) - 3x more information
3. INPUT-DEPENDENT Δ for selective QSVT modulation
4. Consistent with QTSTransformer quantum circuit design

This model combines:
1. QTSFeatureEncoder: Classical feature extraction IDENTICAL to QTSTransformer
2. QuantumMambaSSMCoreAdvanced: Advanced quantum selective SSM with input-dependent Δ

Architecture:
    Input → QTSFeatureEncoder (CLASSICAL) → QuantumMambaSSMCoreAdvanced (QUANTUM) → Output

Key Difference from Hydra-Advanced:
- Hydra-Advanced: Bidirectional with fixed A, B, C and LCU combination
- Mamba-Advanced: Unidirectional with INPUT-DEPENDENT Δ, B, C (selective mechanism)

The selective mechanism (Mamba-style) allows the model to:
- High Δ: Remember new information strongly (input-dependent)
- Low Δ: Forget/maintain existing state (input-dependent)
- All controlled by the input content itself

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
from models.core.quantum_cores.quantum_mamba_ssm_core_advanced import QuantumMambaSSMCoreAdvanced


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
        attn_scores = self.attention(x)
        attn_weights = torch.softmax(attn_scores, dim=1)
        pooled = (attn_weights * x).sum(dim=1)
        return pooled


class QTSQuantumMambaSSMAdvanced(nn.Module):
    """
    QTS Encoder + Advanced Quantum Mamba SSM

    This is Model C1b-Advanced in the ablation study framework.

    Advanced Features:
    - sim14 ansatz: RY → CRX(reverse) → RY → CRX(circular) per layer
    - Multi-observable: PauliX, Y, Z measurements (3 × n_qubits outputs)
    - Input-dependent Δ: Selective QSVT modulation
    - Δ-scaled input injection: Selective information flow

    Components:
    - encoder: QTSFeatureEncoder (CLASSICAL, identical to QTSTransformer)
    - quantum_ssm: QuantumMambaSSMCoreAdvanced (QUANTUM selective with sim14)
    - output layers: Projection, pooling, and classification head

    The model uses:
    - Classical Conv2d + GLU for feature extraction
    - Advanced quantum selective SSM for sequence mixing
    - sim14 variational ansatz (consistent with QTSTransformer)
    - INPUT-DEPENDENT Δ for Mamba-style selectivity
    - Multi-Pauli measurement for richer output

    Comparison with other models:
    - QTSTransformer: Same encoder, Quantum Attention mixing
    - QTSQuantumHydraSSMAdvanced: Same encoder, Bidirectional sim14 QSVT + LCU
    - This model: Same encoder, Selective sim14 QSVT with input-dependent Δ

    Args:
        feature_dim: Input feature dimension
        n_timesteps: Sequence length
        num_classes: Number of output classes
        n_qubits: Number of qubits for quantum circuit
        n_layers: Number of sim14 variational layers
        qsvt_degree: Degree of QSVT polynomial
        dt_rank: Rank of Δ projection ("auto" = n_qubits // 2)
        projection_type: Encoder projection type
        dropout: Dropout rate
        device: Device for computation

    Example:
        model = QTSQuantumMambaSSMAdvanced(
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
        dt_rank: str = "auto",
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

        # Output dimension from quantum SSM: 3 × n_qubits
        self.quantum_output_dim = 3 * n_qubits

        # ============================================
        # CLASSICAL: QTS Feature Encoder
        # IDENTICAL to QTSTransformer encoder
        # ============================================
        self.encoder = create_qts_encoder(
            feature_dim=feature_dim,
            n_timesteps=n_timesteps,
            n_qubits=n_qubits,
            projection_type=projection_type,
            dropout=dropout
        )

        # ============================================
        # QUANTUM: Advanced Mamba SSM Core (SELECTIVE)
        # sim14 ansatz + multi-observable + input-dependent Δ
        # ============================================
        self.quantum_ssm = QuantumMambaSSMCoreAdvanced(
            n_qubits=n_qubits,
            n_layers=n_layers,
            qsvt_degree=qsvt_degree,
            dt_rank=dt_rank,
            device=device
        )

        # ============================================
        # OUTPUT LAYERS
        # Adjusted for 3 × n_qubits quantum output
        # ============================================
        hidden_dim = self.quantum_output_dim * 2  # Expand for expressivity
        self.output_proj = nn.Linear(self.quantum_output_dim, hidden_dim)
        self.layer_norm = nn.LayerNorm(hidden_dim)

        # Sequence pooling
        self.pooling = SequencePooling(hidden_dim)

        # Classification head
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes)
        )

        # Residual projection (match dimensions)
        self.residual_proj = nn.Linear(n_qubits, self.quantum_output_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Flow:
        1. Classical encoder extracts features and converts to angles
        2. Advanced Quantum Mamba SSM processes sequence with selective Δ
        3. Residual connection adds input information
        4. Output projection, pooling, and classification

        The selective mechanism (Mamba-style):
        - Δ[t] = f(x[t]) controls transformation strength at each timestep
        - High Δ[t] → strong quantum transformation → remember input
        - Low Δ[t] → weak quantum transformation → maintain state
        - Multi-observable extracts rich information from quantum state

        Args:
            x: (batch, feature_dim, n_timesteps) or (batch, n_timesteps, feature_dim)

        Returns:
            (batch, num_classes) classification logits
        """
        # ============================================
        # STEP 1: CLASSICAL Feature Extraction
        # QTSFeatureEncoder: Conv2d → GLU → Sigmoid(·π)
        # ============================================
        angles = self.encoder(x)  # (batch, seq_len, n_qubits)

        # ============================================
        # STEP 2: ADVANCED QUANTUM Selective SSM
        # QuantumMambaSSMCoreAdvanced: sim14 + QSVT + input-dependent Δ
        # Output: (batch, seq_len, 3 * n_qubits)
        # ============================================
        quantum_out = self.quantum_ssm(angles)  # (batch, seq_len, 3*n_qubits)

        # ============================================
        # STEP 3: Residual Connection
        # Project angles to match quantum output dimension
        # ============================================
        angles_expanded = self.residual_proj(angles)  # (batch, seq_len, 3*n_qubits)
        residual = quantum_out + angles_expanded

        # ============================================
        # STEP 4: Output Processing
        # Project → Normalize → Pool → Classify
        # ============================================
        projected = self.output_proj(residual)  # (batch, seq_len, hidden_dim)
        projected = self.layer_norm(projected)

        pooled = self.pooling(projected)  # (batch, hidden_dim)

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
        output_params += sum(p.numel() for p in self.residual_proj.parameters())

        return {
            'encoder (CLASSICAL)': encoder_params,
            'quantum_ssm (QUANTUM SELECTIVE ADVANCED)': quantum_params,
            'output (classical)': output_params,
            'total': encoder_params + quantum_params + output_params
        }

    def get_config(self) -> Dict[str, Any]:
        """Return model configuration."""
        return {
            'model_type': 'QTSQuantumMambaSSMAdvanced',
            'feature_dim': self.feature_dim,
            'n_timesteps': self.n_timesteps,
            'num_classes': self.num_classes,
            'n_qubits': self.n_qubits,
            'encoder_type': self.encoder.projection_type,
            'quantum_layers': self.quantum_ssm.n_layers,
            'qsvt_degree': self.quantum_ssm.qsvt_degree,
            'dt_rank': self.quantum_ssm.dt_rank,
            'ansatz': 'sim14',
            'measurement': 'multi-observable (X,Y,Z)',
            'selective_mechanism': 'input-dependent Δ',
            'quantum_output_dim': self.quantum_output_dim,
        }

    def __repr__(self) -> str:
        params = self.get_param_count()
        return (
            f"QTSQuantumMambaSSMAdvanced(\n"
            f"  feature_dim={self.feature_dim},\n"
            f"  n_timesteps={self.n_timesteps},\n"
            f"  num_classes={self.num_classes},\n"
            f"  n_qubits={self.n_qubits},\n"
            f"  encoder={self.encoder.projection_type},\n"
            f"  ansatz='sim14',\n"
            f"  measurement='multi-observable (X,Y,Z)',\n"
            f"  selective_mechanism='input-dependent Δ',\n"
            f"  dt_rank={self.quantum_ssm.dt_rank},\n"
            f"  quantum_output_dim={self.quantum_output_dim},\n"
            f"  params={{\n"
            f"    encoder (CLASSICAL): {params['encoder (CLASSICAL)']:,},\n"
            f"    quantum_ssm (QUANTUM SELECTIVE ADVANCED): {params['quantum_ssm (QUANTUM SELECTIVE ADVANCED)']:,},\n"
            f"    output: {params['output (classical)']:,},\n"
            f"    total: {params['total']:,}\n"
            f"  }}\n"
            f")"
        )


def create_qts_quantum_mamba_ssm_advanced(
    feature_dim: int,
    n_timesteps: int,
    num_classes: int,
    n_qubits: int = 4,
    n_layers: int = 2,
    qsvt_degree: int = 3,
    dt_rank: str = "auto",
    projection_type: str = 'Conv2d_GLU',
    dropout: float = 0.1,
    device: str = "cpu"
) -> QTSQuantumMambaSSMAdvanced:
    """
    Convenience function to create Advanced QTS + Quantum Mamba SSM model.

    This is Model C1b-Advanced in the ablation study.

    Args:
        feature_dim: Input feature dimension
        n_timesteps: Sequence length
        num_classes: Number of output classes
        n_qubits: Number of qubits
        n_layers: sim14 variational circuit layers
        qsvt_degree: QSVT polynomial degree
        dt_rank: Rank of Δ projection ("auto" = n_qubits // 2)
        projection_type: Encoder projection type
        dropout: Dropout rate
        device: Computation device

    Returns:
        QTSQuantumMambaSSMAdvanced model instance
    """
    return QTSQuantumMambaSSMAdvanced(
        feature_dim=feature_dim,
        n_timesteps=n_timesteps,
        num_classes=num_classes,
        n_qubits=n_qubits,
        n_layers=n_layers,
        qsvt_degree=qsvt_degree,
        dt_rank=dt_rank,
        projection_type=projection_type,
        dropout=dropout,
        device=device
    )


# Export
__all__ = ['QTSQuantumMambaSSMAdvanced', 'create_qts_quantum_mamba_ssm_advanced', 'SequencePooling']
