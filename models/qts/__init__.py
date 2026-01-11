"""
QTS Wrapper Models - Initial Ablation Study

These models use the same QTSFeatureEncoder and compare different quantum mixing mechanisms:
- QTSQuantumTransformer: Quantum Attention (QSVT + LCU global mixing)
- QTSQuantumHydraSSM: Quantum Bidirectional SSM
- QTSQuantumMambaSSM: Quantum Selective SSM
"""

from models.qts.QTSQuantumTransformer import QTSQuantumTransformer, create_qts_quantum_transformer
from models.qts.QTSQuantumHydraSSM import QTSQuantumHydraSSM, create_qts_quantum_hydra_ssm
from models.qts.QTSQuantumHydraSSMAdvanced import QTSQuantumHydraSSMAdvanced, create_qts_quantum_hydra_ssm_advanced
from models.qts.QTSQuantumMambaSSM import QTSQuantumMambaSSM, create_qts_quantum_mamba_ssm
from models.qts.QTSQuantumMambaSSMAdvanced import QTSQuantumMambaSSMAdvanced, create_qts_quantum_mamba_ssm_advanced

__all__ = [
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
]
