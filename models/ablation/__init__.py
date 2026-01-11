"""
Ablation Study Models - 2×2×3 Factorial Design

This package contains all models for the ablation study organized by groups:
- group1: Quantum Features → Classical Mixing (1a, 1b, 1c)
- group2: Classical Features → Quantum Mixing (2a-2e)
- group3: Classical Features → Classical Mixing (3a, 3b, 3c) - Baseline
- group4: Quantum Features → Quantum Mixing (4a-4e) - E2E Quantum
"""

# Group 1: Quantum Features → Classical Mixing
from models.ablation.group1.QuantumTransformer import QuantumTransformer, QuantumHydraTransformer, create_quantum_transformer
from models.ablation.group1.QuantumSSM import QuantumMambaSSM, QuantumHydraSSM

# Group 2: Classical Features → Quantum Mixing
from models.ablation.group2.QuantumMixingSSM import (
    ClassicalMambaQuantumSSM,
    ClassicalHydraQuantumSSM,
    ClassicalQuantumAttention,
    ClassicalFeatureExtractor,
    QuantumSSMCore,
    QuantumBidirectionalSSMCore,
    QuantumAttentionMixingCore,
)
from models.ablation.group2.QuantumHydraSSM import (
    QuantumMambaHydraSSM,
    QuantumHydraHydraSSM,
    QuantumHydraSSMCore,
    QuantumHydraSSMBidirectional,
)

# Group 3: Classical Baseline
from models.ablation.group3.ClassicalTransformer import ClassicalTransformer, ClassicalHydraTransformer, create_classical_transformer
from models.ablation.group3.TrueClassicalMamba import TrueClassicalMamba
from models.ablation.group3.TrueClassicalHydra import TrueClassicalHydra

# Group 4: E2E Quantum
from models.ablation.group4.QuantumE2E import (
    QuantumMambaE2E,
    QuantumHydraE2E,
    QuantumTransformerE2E,
    create_quantum_mamba_e2e,
    create_quantum_hydra_e2e,
    create_quantum_transformer_e2e,
)
from models.ablation.group4.QuantumE2E_Superposition import (
    QuantumMambaE2E_Superposition,
    QuantumHydraE2E_Superposition,
    QuantumE2ESuperpositionCore,
    create_quantum_mamba_e2e_superposition,
    create_quantum_hydra_e2e_superposition,
)

__all__ = [
    # Group 1
    'QuantumTransformer',
    'QuantumHydraTransformer',
    'create_quantum_transformer',
    'QuantumMambaSSM',
    'QuantumHydraSSM',
    # Group 2
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
    # Group 3
    'ClassicalTransformer',
    'ClassicalHydraTransformer',
    'create_classical_transformer',
    'TrueClassicalMamba',
    'TrueClassicalHydra',
    # Group 4
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
