# Scripts Directory

This directory contains all scripts for the Quantum Hydra/Mamba project, organized by functionality.

## Directory Structure

```
scripts/
├── training/              # Model training execution scripts
├── job_generation/        # SLURM job script generators
├── aggregation/           # Result aggregation and statistics
├── analysis/              # Experiment analysis and diagnostics
├── evaluation/            # Model evaluation and testing
├── specialized/           # Task-specific wrappers
└── generate_all_synthetic_datasets.py  # Synthetic dataset generation
```

## Organization

### Training (`training/`)
Scripts for running model training experiments:
- `run_ablation_eeg.py` - Main EEG ablation study training
- `run_synthetic_benchmark.py` - Synthetic benchmark execution
- `run_single_model_*.py` - Single model training (EEG, DNA, MNIST, Forrelation)
- `run_full_comparison.py` - Full model comparison
- `run_gated_*.py` - Gated model training
- `run_glue.py` - GLUE benchmark training
- `run_ssm_genomic_comparison.py` - SSM genomic comparison

### Job Generation (`job_generation/`)
Scripts for generating SLURM job scripts:
- `generate_ablation_eeg_jobs.py` - Ablation study job generation
- `generate_synthetic_jobs.py` - Synthetic benchmark job generation
- `generate_*_job_scripts.py` - Dataset-specific job script generators

### Data Generation (scripts root)
Scripts for generating datasets:
- `generate_all_synthetic_datasets.py` - Synthetic dataset generation (Forrelation, Adding Problem, Selective Copy)

### Aggregation (`aggregation/`)
Scripts for aggregating experiment results:
- `aggregate_ablation_results.py` - Ablation study result aggregation
- `aggregate_synthetic_results.py` - Synthetic benchmark aggregation
- `aggregate_*_results.py` - Dataset-specific result aggregation

### Analysis (`analysis/`)
Scripts for analyzing experiments:
- `analyze_all_experiments.py` - Comprehensive experiment log analysis

### Evaluation (`evaluation/`)
Scripts for model evaluation and testing:
- `evaluate_checkpoint.py` - Checkpoint evaluation
- `test_*_with_quantum_models.py` - Dataset compatibility testing

### Specialized (`specialized/`)
Task-specific wrappers and utilities:
- `QuantumHydraGLUE.py` - GLUE benchmark wrapper

## Usage

### Direct Execution (Recommended)

Use the new organized paths:

```bash
# Training
python scripts/training/run_ablation_eeg.py --model-id 1a --sampling-freq 80

# Job generation
python scripts/job_generation/generate_ablation_eeg_jobs.py

# Data generation
python scripts/generate_all_synthetic_datasets.py --data-dir ./data/synthetic_benchmarks

# Aggregation
python scripts/aggregation/aggregate_ablation_results.py
```

### Important Notes

**File Organization**: All actual script files are organized in subdirectories (`training/`, `job_generation/`, `aggregation/`, etc.). The scripts root directory only contains `__init__.py` and `README.md`.

**Job Script Generation**: When generating new job scripts using the `job_generation/` scripts, they will automatically use the new organized paths:
- `scripts/training/run_*.py` for training scripts
- `scripts/aggregation/aggregate_*.py` for aggregation scripts
- `scripts/job_generation/generate_*.py` for job generation scripts

### Module Imports

Scripts can also be imported as modules (though this is less common):

```python
from scripts.training.run_ablation_eeg import train_model
from scripts.job_generation.generate_ablation_eeg_jobs import generate_jobs
```

## Note on Existing Job Scripts

**Existing job scripts** in `job_scripts/` may still use old paths like `scripts/run_*.py`. These will need to be updated manually or regenerated using the `job_generation/` scripts, which now generate job scripts with the correct new paths.

To update existing job scripts, change:
- `scripts/run_*.py` → `scripts/training/run_*.py`
- `scripts/aggregate_*.py` → `scripts/aggregation/aggregate_*.py`
- `scripts/generate_*.py` → `scripts/job_generation/generate_*.py`
