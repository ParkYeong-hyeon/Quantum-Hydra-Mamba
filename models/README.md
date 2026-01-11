# Models Directory Structure

## Overview

This directory contains all model implementations organized in a hierarchical structure following the 2×2×3 factorial design of the ablation study.

## Directory Structure

```
models/
├── core/                    # Reusable components
│   ├── encoders/           # Feature extraction modules
│   ├── quantum_cores/      # Quantum circuit cores
│   └── gated/              # Gated recurrence modules
│
├── ablation/               # Ablation Study Models (2×2×3 Factorial Design)
│   ├── group1/             # Quantum Features → Classical Mixing (1a, 1b, 1c)
│   ├── group2/             # Classical Features → Quantum Mixing (2a-2e)
│   ├── group3/             # Classical Baseline (3a, 3b, 3c)
│   └── group4/             # E2E Quantum (4a-4e)
│
├── qts/                    # QTS Wrapper Models (Initial experiments)
└── legacy/                 # Previous versions (Reference only)
```

## Import Usage

### Backward Compatibility

All models can still be imported using the old paths for backward compatibility:

```python
# Old import paths still work
from models.QuantumTransformer import QuantumTransformer
from models.QuantumSSM import QuantumMambaSSM
from models.ClassicalTransformer import ClassicalTransformer
```

### New Recommended Import Paths

```python
# Core components
from models.core import QTSFeatureEncoder, QuantumAttentionCore
from models.core.encoders import create_qts_encoder
from models.core.quantum_cores import QuantumHydraSSMCore

# Ablation study models
from models.ablation import QuantumTransformer, QuantumMambaSSM
from models.ablation.group1 import QuantumTransformer, QuantumMambaSSM
from models.ablation.group2 import ClassicalMambaQuantumSSM
from models.ablation.group3 import ClassicalTransformer, TrueClassicalMamba
from models.ablation.group4 import QuantumMambaE2E

# QTS wrapper models
from models.qts import QTSQuantumTransformer, QTSQuantumHydraSSMAdvanced

# Or import directly from models (re-exported)
from models import QuantumTransformer, QuantumMambaSSM, ClassicalTransformer
```

## Model Groups

### Group 1: Quantum Features → Classical Mixing
- **1a**: `QuantumTransformer` - Quantum feature extraction → Classical Transformer attention
- **1b**: `QuantumMambaSSM` - Quantum feature → Classical Mamba SSM
- **1c**: `QuantumHydraSSM` - Quantum feature → Classical Hydra SSM

### Group 2: Classical Features → Quantum Mixing
- **2a**: `ClassicalQuantumAttention` - Classical feature → Quantum attention
- **2b**: `ClassicalMambaQuantumSSM` - Classical feature → Quantum Mamba SSM
- **2c**: `ClassicalHydraQuantumSSM` - Classical feature → Quantum Hydra SSM
- **2d**: `QuantumMambaHydraSSM` - Classical feature → True superposition Mamba
- **2e**: `QuantumHydraHydraSSM` - Classical feature → True superposition Hydra

### Group 3: Classical Baseline
- **3a**: `ClassicalTransformer` - Full classical Transformer
- **3b**: `TrueClassicalMamba` - Full classical Mamba
- **3c**: `TrueClassicalHydra` - Full classical Hydra

### Group 4: E2E Quantum
- **4a**: `QuantumTransformerE2E` - End-to-end quantum Transformer
- **4b**: `QuantumMambaE2E` - End-to-end quantum Mamba
- **4c**: `QuantumHydraE2E` - End-to-end quantum Hydra
- **4d**: `QuantumMambaE2E_Superposition` - E2E + True superposition Mamba
- **4e**: `QuantumHydraE2E_Superposition` - E2E + True superposition Hydra

## Core Components

### Encoders
- `QTSFeatureEncoder`: Unified classical feature encoder (Conv2d + GLU)
- `Conv2dFeatureExtractor`: 2D CNN for spatio-temporal patterns
- `GatedFeedForward`: GLU-based non-linear transformation

### Quantum Cores
- `QuantumAttentionCore`: Quantum attention circuit (QSVT + LCU)
- `QuantumHydraSSMCore`: Quantum bidirectional SSM core
- `QuantumMambaSSMCore`: Quantum selective SSM core
- Advanced versions with sim14 ansatz and multi-observable measurements

### Gated Modules
- `QuantumFeatureExtractor`: Variational quantum circuit for feature extraction
- `QuantumSuperpositionBranches`: Three-branch quantum superposition
- `QuantumMambaGated`: LSTM-style quantum gating for Mamba
- `QuantumHydraGated`: LSTM-style quantum gating for Hydra

## QTS Wrapper Models

These models use the same `QTSFeatureEncoder` and compare different quantum mixing mechanisms:
- `QTSQuantumTransformer`: Quantum Attention (baseline)
- `QTSQuantumHydraSSM`: Quantum Bidirectional SSM
- `QTSQuantumMambaSSM`: Quantum Selective SSM

Advanced versions use sim14 ansatz and multi-observable measurements.

## Legacy Models

Previous versions kept for reference:
- `QuantumHydra.py`, `QuantumMamba.py`: Original implementations
- `QuantumHydraHybrid.py`, `QuantumMambaHybrid.py`: Hybrid variants
- `QuantumMambaLite.py`: Lightweight versions
- `QuantumAttention.py`: Attention utilities

These are not used in the current ablation study.

## Notes

- All import paths maintain backward compatibility
- New code should use the hierarchical import paths for clarity
- Core components are designed to be reusable across different model architectures
- Each group's `__init__.py` exports all models in that group
