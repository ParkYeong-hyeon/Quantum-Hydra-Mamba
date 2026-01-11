"""
Quantum Hydra/Mamba/Transformer Models Package

This package contains models for the ABLATION STUDY comparing quantum architectures
organized in a hierarchical structure:

- core/: Reusable components (encoders, quantum cores, gated modules)
- ablation/: Ablation study models organized by groups (Group 1-4)
- qts/: QTS wrapper models for initial experiments
- legacy/: Previous versions (reference only)

For backward compatibility, all models can still be imported from models.*
"""

# ============================================
# Core Components (re-exported for convenience)
# ============================================
from models.core import (
    # Encoders
    QTSFeatureEncoder,
    Conv2dFeatureExtractor,
    GatedFeedForward,
    Conv2dGLUPreprocessor,
    create_qts_encoder,
    # Quantum Cores
    QuantumAttentionCore,
    unified_ansatz_circuit,
    sim14_circuit,
    QuantumHydraSSMCore,
    QuantumMambaSSMCore,
    QuantumHydraSSMCoreAdvanced,
    QuantumMambaSSMCoreAdvanced,
    # Gated
    QuantumFeatureExtractor,
    QuantumStateProcessor,
    QuantumSuperpositionBranches,
    ChunkedGatedSuperposition,
    QuantumMambaGated,
    QuantumHydraGated,
)

# ============================================
# QTS Wrapper Models (Initial Ablation Study)
# ============================================
from models.qts import (
    QTSQuantumTransformer,
    create_qts_quantum_transformer,
    QTSQuantumHydraSSM,
    create_qts_quantum_hydra_ssm,
    QTSQuantumHydraSSMAdvanced,
    create_qts_quantum_hydra_ssm_advanced,
    QTSQuantumMambaSSM,
    create_qts_quantum_mamba_ssm,
    QTSQuantumMambaSSMAdvanced,
    create_qts_quantum_mamba_ssm_advanced,
)

# ============================================
# Ablation Study Models (2×2×3 Factorial Design)
# ============================================
from models.ablation import (
    # Group 1: Quantum Features → Classical Mixing
    QuantumTransformer,
    QuantumHydraTransformer,
    create_quantum_transformer,
    QuantumMambaSSM,
    QuantumHydraSSM,
    # Group 2: Classical Features → Quantum Mixing
    ClassicalMambaQuantumSSM,
    ClassicalHydraQuantumSSM,
    ClassicalQuantumAttention,
    ClassicalFeatureExtractor,
    QuantumSSMCore,
    QuantumBidirectionalSSMCore,
    QuantumAttentionMixingCore,
    QuantumMambaHydraSSM,
    QuantumHydraHydraSSM,
    QuantumHydraSSMCore,
    QuantumHydraSSMBidirectional,
    # Group 3: Classical Baseline
    ClassicalTransformer,
    ClassicalHydraTransformer,
    create_classical_transformer,
    TrueClassicalMamba,
    TrueClassicalHydra,
    # Group 4: E2E Quantum
    QuantumMambaE2E,
    QuantumHydraE2E,
    QuantumTransformerE2E,
    create_quantum_mamba_e2e,
    create_quantum_hydra_e2e,
    create_quantum_transformer_e2e,
    QuantumMambaE2E_Superposition,
    QuantumHydraE2E_Superposition,
    QuantumE2ESuperpositionCore,
    create_quantum_mamba_e2e_superposition,
    create_quantum_hydra_e2e_superposition,
)

# ============================================
# Export all public symbols
# ============================================
__all__ = [
    # Core Components
    'QTSFeatureEncoder',
    'Conv2dFeatureExtractor',
    'GatedFeedForward',
    'Conv2dGLUPreprocessor',
    'create_qts_encoder',
    'QuantumAttentionCore',
    'unified_ansatz_circuit',
    'sim14_circuit',
    'QuantumHydraSSMCore',
    'QuantumMambaSSMCore',
    'QuantumHydraSSMCoreAdvanced',
    'QuantumMambaSSMCoreAdvanced',
    'QuantumFeatureExtractor',
    'QuantumStateProcessor',
    'QuantumSuperpositionBranches',
    'ChunkedGatedSuperposition',
    'QuantumMambaGated',
    'QuantumHydraGated',
    # QTS Wrapper Models
    'QTSQuantumTransformer',
    'create_qts_quantum_transformer',
    'QTSQuantumHydraSSM',
    'create_qts_quantum_hydra_ssm',
    'QTSQuantumHydraSSMAdvanced',
    'create_qts_quantum_hydra_ssm_advanced',
    'QTSQuantumMambaSSM',
    'create_qts_quantum_mamba_ssm',
    'QTSQuantumMambaSSMAdvanced',
    'create_qts_quantum_mamba_ssm_advanced',
    # Ablation Study Models - Group 1
    'QuantumTransformer',
    'QuantumHydraTransformer',
    'create_quantum_transformer',
    'QuantumMambaSSM',
    'QuantumHydraSSM',
    # Ablation Study Models - Group 2
    'ClassicalMambaQuantumSSM',
    'ClassicalHydraQuantumSSM',
    'ClassicalQuantumAttention',
    'ClassicalFeatureExtractor',
    'QuantumSSMCore',
    'QuantumBidirectionalSSMCore',
    'QuantumAttentionMixingCore',
    'QuantumMambaHydraSSM',
    'QuantumHydraHydraSSM',
    'QuantumHydraSSMCore',
    'QuantumHydraSSMBidirectional',
    # Ablation Study Models - Group 3
    'ClassicalTransformer',
    'ClassicalHydraTransformer',
    'create_classical_transformer',
    'TrueClassicalMamba',
    'TrueClassicalHydra',
    # Ablation Study Models - Group 4
    'QuantumMambaE2E',
    'QuantumHydraE2E',
    'QuantumTransformerE2E',
    'create_quantum_mamba_e2e',
    'create_quantum_hydra_e2e',
    'create_quantum_transformer_e2e',
    'QuantumMambaE2E_Superposition',
    'QuantumHydraE2E_Superposition',
    'QuantumE2ESuperpositionCore',
    'create_quantum_mamba_e2e_superposition',
    'create_quantum_hydra_e2e_superposition',
]

# Version
__version__ = '4.0.0'

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
