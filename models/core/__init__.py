"""
Core Components for Quantum Models

This package contains reusable components:
- encoders: Feature extraction modules
- quantum_cores: Quantum circuit cores for SSM, Attention, etc.
- gated: Gated recurrence modules
"""

from models.core.encoders.qts_encoder import (
    QTSFeatureEncoder,
    Conv2dFeatureExtractor,
    GatedFeedForward,
    Conv2dGLUPreprocessor,
    create_qts_encoder,
)

from models.core.quantum_cores.quantum_attention_core import (
    QuantumAttentionCore,
    unified_ansatz_circuit,
    sim14_circuit,
)

from models.core.quantum_cores.quantum_hydra_ssm_core import QuantumHydraSSMCore
from models.core.quantum_cores.quantum_mamba_ssm_core import QuantumMambaSSMCore
from models.core.quantum_cores.quantum_hydra_ssm_core_advanced import QuantumHydraSSMCoreAdvanced
from models.core.quantum_cores.quantum_mamba_ssm_core_advanced import QuantumMambaSSMCoreAdvanced

from models.core.gated.QuantumGatedRecurrence import (
    QuantumFeatureExtractor,
    QuantumStateProcessor,
    QuantumSuperpositionBranches,
    ChunkedGatedSuperposition,
    QuantumMambaGated,
    QuantumHydraGated,
)

__all__ = [
    # Encoders
    'QTSFeatureEncoder',
    'Conv2dFeatureExtractor',
    'GatedFeedForward',
    'Conv2dGLUPreprocessor',
    'create_qts_encoder',
    # Quantum Cores
    'QuantumAttentionCore',
    'unified_ansatz_circuit',
    'sim14_circuit',
    'QuantumHydraSSMCore',
    'QuantumMambaSSMCore',
    'QuantumHydraSSMCoreAdvanced',
    'QuantumMambaSSMCoreAdvanced',
    # Gated
    'QuantumFeatureExtractor',
    'QuantumStateProcessor',
    'QuantumSuperpositionBranches',
    'ChunkedGatedSuperposition',
    'QuantumMambaGated',
    'QuantumHydraGated',
]
