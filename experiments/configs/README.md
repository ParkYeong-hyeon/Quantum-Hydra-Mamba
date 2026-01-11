# Experiment Configuration Files

This directory contains YAML configuration files for all experiments in the Quantum-Hydra-Mamba project. The configurations are organized into **Legacy** and **Ablation** categories to reflect the evolution of the project.

## Directory Structure

```
experiments/configs/
├── base.yaml                    # Common settings (SLURM, environment, training defaults)
├── ablation/                    # Ablation Study Experiments (2×2×3 Factorial Design)
│   ├── synthetic.yaml          # Synthetic benchmark experiments (432 experiments)
│   └── eeg.yaml                 # EEG ablation study (126 experiments)
└── legacy/                      # Legacy/Initial Experiments
    ├── eeg.yaml                 # Legacy models on EEG (30 experiments)
    ├── dna.yaml                 # Legacy models on DNA (30 experiments)
    ├── mnist.yaml               # Legacy models on MNIST (30 experiments)
    ├── forrelation.yaml         # Forrelation quantum advantage testing (144 experiments)
    ├── gated.yaml               # Gated model experiments
    ├── genomic.yaml             # Genomic sequence experiments
    ├── glue.yaml                # GLUE benchmark experiments
    └── full_comparison.yaml     # Full model comparison experiments
```

## Configuration Categories

### Ablation Study (`ablation/`)

The ablation study follows a **2×2×3 factorial design** to systematically isolate the contributions of quantum components:

- **Feature Extraction**: Quantum vs Classical
- **Mixing Mechanism**: Attention vs Mamba SSM vs Hydra SSM
- **Architecture Type**: Hybrid vs End-to-End

**Models (14 total)**:
- Group 1: Quantum Features → Classical Mixing (1a, 1b, 1c)
- Group 2: Classical Features → Quantum Mixing (2a, 2b, 2c, 2d, 2e)
- Group 3: Classical Baselines (3a, 3b, 3c)
- Group 4: E2E Quantum (4a, 4b, 4c, 4d, 4e)

**Experiments**:
- `synthetic.yaml`: 12 models × 3 tasks × 4 seq_lens × 3 seeds = 432 experiments
- `eeg.yaml`: 14 models × 3 sampling_freqs × 3 seeds = 126 experiments

**Job Scripts Location**: `jobs/synthetic/`, `jobs/ablation_eeg/`

### Legacy Experiments (`legacy/`)

Initial experiments with the original model architectures before the systematic ablation study:

**Models**:
- `quantum_hydra`, `quantum_mamba`: Original quantum models
- `quantum_hydra_hybrid`, `quantum_mamba_hybrid`: Hybrid variants
- `quantum_mamba_gated`, `quantum_hydra_gated`: Gated variants
- `classical_hydra`, `classical_mamba`: Classical baselines

**Experiments**:
- `eeg.yaml`: 6 models × 5 seeds = 30 experiments
- `dna.yaml`: 6 models × 5 seeds = 30 experiments
- `mnist.yaml`: 6 models × 5 seeds = 30 experiments
- `gated.yaml`: Gated models on multiple datasets
- `genomic.yaml`: Genomic sequence experiments
- `glue.yaml`: GLUE benchmark (9 tasks)
- `full_comparison.yaml`: Full model comparison

**Job Scripts Location**: `job_scripts/`

## Configuration File Format

Each YAML file follows this general structure:

```yaml
# Extends base configuration
_extends: ../base.yaml

# SLURM Configuration (overrides base)
slurm:
  account: "m4807_g"
  constraint: "gpu&hbm80g"
  time: "24:00:00"
  # ... other SLURM settings

# Model Configurations
models:
  "1a":
    name: "QuantumTransformer"
    group: 1
    description: "..."

# Training Hyperparameters
training:
  n_qubits: 6
  n_epochs: 100
  # ... other hyperparameters

# Experiment Combinations
experiments:
  # Define specific experiment combinations
```

## Key Differences: Legacy vs Ablation

| Aspect | Legacy | Ablation |
|--------|--------|----------|
| **Model Naming** | `quantum_hydra`, `quantum_mamba` | Model IDs: `1a`, `1b`, `1c`, etc. |
| **Design** | Ad-hoc experiments | Systematic 2×2×3 factorial design |
| **Purpose** | Initial exploration | Controlled ablation study |
| **Job Location** | `job_scripts/` | `jobs/` |
| **Model Groups** | Not organized | Organized into 4 groups |

## Usage

### Reading Configuration Files

```python
import yaml
from pathlib import Path

# Load an ablation study config
config_path = Path("experiments/configs/ablation/synthetic.yaml")
with open(config_path) as f:
    config = yaml.safe_load(f)

# Load a legacy config
legacy_path = Path("experiments/configs/legacy/eeg.yaml")
with open(legacy_path) as f:
    legacy_config = yaml.safe_load(f)
```

### Generating Job Scripts

You can use these YAML files to generate SLURM job scripts programmatically. The job generation scripts in `scripts/job_generation/` can be updated to read from these YAML files.

## File Descriptions

### `base.yaml`
Common settings shared across all experiments:
- SLURM defaults (account, constraint, resources)
- Environment setup (conda, paths, environment variables)
- Common training hyperparameters
- Output and data directory paths

### Ablation Study Configs

#### `ablation/synthetic.yaml`
Synthetic benchmark experiments:
- **Models**: 12 models (Groups 1-4, excluding some variants)
- **Tasks**: forrelation, adding_problem, selective_copy
- **Sequence Lengths**: Task-specific (50-1000)
- **Seeds**: [2024, 2025, 2026]
- **Total**: 432 experiments

#### `ablation/eeg.yaml`
EEG ablation study:
- **Models**: 14 models (all groups)
- **Sampling Frequencies**: [40, 80, 160] Hz
- **Seeds**: [2024, 2025, 2026]
- **Total**: 126 experiments

### Legacy Configs

#### `legacy/eeg.yaml`, `legacy/dna.yaml`, `legacy/mnist.yaml`
Initial experiments with legacy models:
- **Models**: 6 models (quantum_hydra, quantum_mamba, hybrid variants, classical baselines)
- **Seeds**: [2024, 2025, 2026, 2027, 2028]
- **Total**: 30 experiments per dataset

#### `legacy/forrelation.yaml`
Forrelation quantum advantage testing:
- **Models**: 6 models (legacy models + baselines)
- **Datasets**: Phase 3 (4 datasets) + Phase 4 (4 datasets) = 8 datasets
- **Seeds**: [2024, 2025, 2026]
- **Total**: 6 models × 8 datasets × 3 seeds = 144 experiments

#### `legacy/gated.yaml`
Gated model experiments:
- **Models**: QuantumMambaGated, QuantumHydraGated
- **Datasets**: EEG, DNA, Genomic
- **Array Jobs**: 2 models × 5 seeds = 10 jobs per dataset

#### `legacy/genomic.yaml`
Genomic sequence classification:
- **Models**: SSM models, gated models
- **Datasets**: Multiple genomic benchmark datasets

#### `legacy/glue.yaml`
GLUE benchmark experiments:
- **Tasks**: 9 GLUE tasks (CoLA, SST-2, MRPC, etc.)
- **Models**: All available models

#### `legacy/full_comparison.yaml`
Full model comparison:
- **Models**: 6 models (SSM, Gated, Classical)
- **Datasets**: 3 smallest genomic datasets
- **Fair Comparison**: Same hyperparameters for all models

## Best Practices

1. **Inheritance**: All experiment configs extend `base.yaml` for common settings
2. **Documentation**: Each model/dataset includes a `description` field
3. **Consistency**: Use consistent naming conventions within each category
4. **Version Control**: Keep configs in Git for reproducibility
5. **Separation**: Keep legacy and ablation configs clearly separated

## Notes

- The `_extends` field uses relative paths (`../base.yaml`) to reference the base config
- All paths are relative to the project root unless specified as absolute
- Time estimates are approximate and may vary based on hardware and dataset size
- Legacy experiments use model names (e.g., `quantum_hydra`), while ablation uses model IDs (e.g., `1a`)
