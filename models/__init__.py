"""
Quantum Hydra/Mamba/Transformer Models Package

This package contains models for the ABLATION STUDY comparing three quantum
mixing mechanisms with IDENTICAL classical feature extraction:

MODELS FOR ABLATION STUDY:
--------------------------
1. QTSQuantumTransformer (BASELINE): Quantum Attention (QSVT + LCU global mixing)
2. QTSQuantumHydraSSMAdvanced: Quantum Bidirectional SSM (QSVT + LCU combination)
3. QTSQuantumMambaSSMAdvanced: Quantum Selective SSM (QSVT + input-dependent Δ)

All three models share the IDENTICAL QTSFeatureEncoder (Conv2d + GLU).
The ONLY difference is the quantum mixing mechanism.

Mixing Mechanism Comparison:
----------------------------
| Model | Mixing Type | Key Feature |
|-------|-------------|-------------|
| Transformer | Global Attention | LCU mixes ALL timesteps simultaneously |
| Hydra SSM | Bidirectional | Forward + Backward sequential passes + LCU |
| Mamba SSM | Selective | Input-dependent Δ controls forgetting |

Basic vs Advanced Models:
-------------------------
- Basic: RX,RY,RZ + linear CNOT, PauliZ only measurement
- Advanced: sim14 ansatz, PauliX,Y,Z multi-observable (RECOMMENDED)

Usage for Ablation Study:
    from models import (
        QTSQuantumTransformer,           # Baseline: Quantum Attention
        QTSQuantumHydraSSMAdvanced,      # Quantum Bidirectional SSM
        QTSQuantumMambaSSMAdvanced,      # Quantum Selective SSM
    )

    # All models use identical encoder for fair comparison
    transformer = QTSQuantumTransformer(feature_dim=4, n_timesteps=200, num_classes=2)
    hydra = QTSQuantumHydraSSMAdvanced(feature_dim=4, n_timesteps=200, num_classes=2)
    mamba = QTSQuantumMambaSSMAdvanced(feature_dim=4, n_timesteps=200, num_classes=2)

Author: Junghoon Park
Date: December 2024
"""

# ============================================
# Classical Encoder (shared across ALL models)
# ============================================
from models.qts_encoder import (
    QTSFeatureEncoder,
    Conv2dFeatureExtractor,
    GatedFeedForward,
    Conv2dGLUPreprocessor,
    create_qts_encoder,
)

# ============================================
# Quantum Cores
# ============================================
# Basic SSM Cores
from models.quantum_hydra_ssm_core import QuantumHydraSSMCore
from models.quantum_mamba_ssm_core import QuantumMambaSSMCore

# Advanced SSM Cores (sim14 + multi-observable)
from models.quantum_hydra_ssm_core_advanced import QuantumHydraSSMCoreAdvanced
from models.quantum_mamba_ssm_core_advanced import QuantumMambaSSMCoreAdvanced

# Quantum Attention Core (for Transformer)
from models.quantum_attention_core import QuantumAttentionCore, sim14_circuit

# ============================================
# Basic Complete Models
# ============================================
from models.QTSQuantumHydraSSM import (
    QTSQuantumHydraSSM,
    create_qts_quantum_hydra_ssm,
)
from models.QTSQuantumMambaSSM import (
    QTSQuantumMambaSSM,
    create_qts_quantum_mamba_ssm,
)

# ============================================
# Advanced Complete Models (RECOMMENDED for Ablation)
# ============================================
from models.QTSQuantumHydraSSMAdvanced import (
    QTSQuantumHydraSSMAdvanced,
    create_qts_quantum_hydra_ssm_advanced,
)
from models.QTSQuantumMambaSSMAdvanced import (
    QTSQuantumMambaSSMAdvanced,
    create_qts_quantum_mamba_ssm_advanced,
)

# ============================================
# Quantum Transformer (BASELINE for Ablation)
# ============================================
from models.QTSQuantumTransformer import (
    QTSQuantumTransformer,
    create_qts_quantum_transformer,
)

# ============================================
# Classical Features -> Quantum Mixing Models
# (Inverse architecture for ablation study)
# ============================================
from models.QuantumMixingSSM import (
    ClassicalMambaQuantumSSM,
    ClassicalHydraQuantumSSM,
    ClassicalQuantumAttention,
    ClassicalFeatureExtractor,
    QuantumSSMCore,
    QuantumBidirectionalSSMCore,
    QuantumAttentionMixingCore,
)

# ============================================
# Classical Self-Attention Transformer
# (Pure classical baseline with attention)
# ============================================
from models.ClassicalTransformer import (
    ClassicalTransformer,
    ClassicalHydraTransformer,
    create_classical_transformer,
)

# ============================================
# Quantum Features -> Full Transformer Attention
# (Proper transformer attention, not chunked)
# ============================================
from models.QuantumTransformer import (
    QuantumTransformer,
    QuantumHydraTransformer,
    create_quantum_transformer,
)

# ============================================
# Quantum Features -> Classical SSM Mixing
# (Completes Group 1 of the 2x2x3 ablation study)
# ============================================
from models.QuantumSSM import (
    QuantumMambaSSM,
    QuantumHydraSSM,
)

# ============================================
# QSVT+LCU SSM Models (True Quantum Mixing)
# Classical Features -> QSVT+LCU Quantum Mixing
# ============================================
from models.QuantumQSVTSSM import (
    QSVTLCUCore,
    QuantumQSVTMambaSSM,
    QuantumQSVTHydraSSM,
)

# ============================================
# End-to-End Quantum Models
# (Quantum Features -> Quantum Mixing -> Single Measurement)
# NO intermediate measurements - true quantum coherence
# ============================================
from models.QuantumE2E import (
    QuantumMambaE2E,
    QuantumHydraE2E,
    QuantumTransformerE2E,
    create_quantum_mamba_e2e,
    create_quantum_hydra_e2e,
    create_quantum_transformer_e2e,
)

# Export all public symbols
__all__ = [
    # ============================================
    # Encoder (shared by all)
    # ============================================
    'QTSFeatureEncoder',
    'Conv2dFeatureExtractor',
    'GatedFeedForward',
    'Conv2dGLUPreprocessor',
    'create_qts_encoder',

    # ============================================
    # Quantum Cores
    # ============================================
    # Basic SSM Cores
    'QuantumHydraSSMCore',
    'QuantumMambaSSMCore',
    # Advanced SSM Cores
    'QuantumHydraSSMCoreAdvanced',
    'QuantumMambaSSMCoreAdvanced',
    # Attention Core
    'QuantumAttentionCore',
    'sim14_circuit',

    # ============================================
    # Basic Complete Models
    # ============================================
    'QTSQuantumHydraSSM',
    'QTSQuantumMambaSSM',
    'create_qts_quantum_hydra_ssm',
    'create_qts_quantum_mamba_ssm',

    # ============================================
    # Advanced Complete Models (RECOMMENDED)
    # ============================================
    'QTSQuantumHydraSSMAdvanced',
    'QTSQuantumMambaSSMAdvanced',
    'create_qts_quantum_hydra_ssm_advanced',
    'create_qts_quantum_mamba_ssm_advanced',

    # ============================================
    # Quantum Transformer (BASELINE)
    # ============================================
    'QTSQuantumTransformer',
    'create_qts_quantum_transformer',

    # ============================================
    # Classical Features -> Quantum Mixing Models
    # ============================================
    'ClassicalMambaQuantumSSM',
    'ClassicalHydraQuantumSSM',
    'ClassicalQuantumAttention',
    'ClassicalFeatureExtractor',
    'QuantumSSMCore',
    'QuantumBidirectionalSSMCore',
    'QuantumAttentionMixingCore',

    # ============================================
    # Classical Self-Attention Transformer
    # ============================================
    'ClassicalTransformer',
    'ClassicalHydraTransformer',
    'create_classical_transformer',

    # ============================================
    # Quantum Features -> Full Transformer Attention
    # ============================================
    'QuantumTransformer',
    'QuantumHydraTransformer',
    'create_quantum_transformer',

    # ============================================
    # Quantum Features -> Classical SSM Mixing
    # ============================================
    'QuantumMambaSSM',
    'QuantumHydraSSM',

    # ============================================
    # QSVT+LCU SSM Models (True Quantum Mixing)
    # ============================================
    'QSVTLCUCore',
    'QuantumQSVTMambaSSM',
    'QuantumQSVTHydraSSM',

    # ============================================
    # End-to-End Quantum Models (Quantum -> Quantum)
    # ============================================
    'QuantumMambaE2E',
    'QuantumHydraE2E',
    'QuantumTransformerE2E',
    'create_quantum_mamba_e2e',
    'create_quantum_hydra_e2e',
    'create_quantum_transformer_e2e',
]

# Version
__version__ = '3.0.0'

# ============================================
# Quick Reference for Ablation Study
# ============================================
ABLATION_MODELS = {
    'transformer': QTSQuantumTransformer,      # Baseline: Global Attention
    'hydra': QTSQuantumHydraSSMAdvanced,       # Bidirectional SSM
    'mamba': QTSQuantumMambaSSMAdvanced,       # Selective SSM
}

ABLATION_FACTORIES = {
    'transformer': create_qts_quantum_transformer,
    'hydra': create_qts_quantum_hydra_ssm_advanced,
    'mamba': create_qts_quantum_mamba_ssm_advanced,
}
