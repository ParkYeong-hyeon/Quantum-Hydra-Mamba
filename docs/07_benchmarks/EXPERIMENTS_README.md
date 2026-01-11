# Tier 1 Experiments - Comprehensive Job Scripts

This directory contains all job scripts for running Tier 1 experiments from EXPERIMENTAL_PLAN_README.md.

**Status**: ✅ Using correct conda environment: `./conda-envs/qml_env`

---

## 📋 Overview

All experiments follow **Option 1: Individual Job Scripts** approach:
- **30 job scripts per experiment** (6 models × 5 seeds)
- **Total: 90 job scripts** across 3 Tier 1 experiments
- Each job runs **1 model** with **1 seed** (guaranteed < 48 hours)
- Maximum parallelism and fault tolerance

---

## ✅ Experiment 1.1: EEG Motor Imagery Classification

**Dataset**: PhysioNet EEG Motor Imagery
**Task**: Binary classification (left/right hand movement)
**Status**: ✅ **READY TO RUN**

### Files Created

#### Training Script
- `run_single_model_eeg.py` - Main training script for single (model, seed) pair

#### Job Scripts (30 total)
- `eeg_job_scripts/eeg_quantum_hydra_seed{2024,2025,2026,2027,2028}.sh`
- `eeg_job_scripts/eeg_quantum_hydra_hybrid_seed{2024,2025,2026,2027,2028}.sh`
- `eeg_job_scripts/eeg_quantum_mamba_seed{2024,2025,2026,2027,2028}.sh`
- `eeg_job_scripts/eeg_quantum_mamba_hybrid_seed{2024,2025,2026,2027,2028}.sh`
- `eeg_job_scripts/eeg_classical_hydra_seed{2024,2025,2026,2027,2028}.sh`
- `eeg_job_scripts/eeg_classical_mamba_seed{2024,2025,2026,2027,2028}.sh`

#### Master Submission
- `eeg_job_scripts/submit_all_eeg_jobs.sh` - Submit all 30 jobs in parallel

#### Aggregation
- `aggregate_eeg_results.py` - Compute mean ± std across seeds
- Generates: CSV, LaTeX tables, comparison plots

### Hyperparameters
```python
n_qubits = 6
qlcu_layers = 2
d_model = 128
d_state = 16
n_epochs = 50
batch_size = 16
learning_rate = 1e-3
sample_size = 10
sampling_freq = 100 Hz
```

### Usage
```bash
# Submit all 30 EEG jobs
bash job_scripts/eeg/submit_all_eeg_jobs.sh

# Or submit individual job
bash job_scripts/eeg/eeg_quantum_hydra_seed2024.sh

# Monitor progress
tail -f results/eeg_results/logs/*.log

# After completion, aggregate results
python scripts/aggregate_eeg_results.py
```

---

## ✅ Experiment 1.2: MNIST Image Classification

**Dataset**: MNIST
**Task**: 10-class digit classification
**Status**: ✅ **READY TO RUN**

### Files Created

#### Training Script
- `run_single_model_mnist.py` - Main training script for single (model, seed) pair

#### Job Scripts (30 total)
- `mnist_job_scripts/mnist_quantum_hydra_seed{2024,2025,2026,2027,2028}.sh`
- `mnist_job_scripts/mnist_quantum_hydra_hybrid_seed{2024,2025,2026,2027,2028}.sh`
- `mnist_job_scripts/mnist_quantum_mamba_seed{2024,2025,2026,2027,2028}.sh`
- `mnist_job_scripts/mnist_quantum_mamba_hybrid_seed{2024,2025,2026,2027,2028}.sh`
- `mnist_job_scripts/mnist_classical_hydra_seed{2024,2025,2026,2027,2028}.sh`
- `mnist_job_scripts/mnist_classical_mamba_seed{2024,2025,2026,2027,2028}.sh`

#### Master Submission
- `mnist_job_scripts/submit_all_mnist_jobs.sh` - Submit all 30 jobs in parallel

#### Aggregation
- `aggregate_mnist_results.py` - Compute mean ± std across seeds
- Generates: CSV, LaTeX tables, comparison plots

### Hyperparameters
```python
n_qubits = 6
qlcu_layers = 2
d_model = 128
d_state = 16
n_epochs = 50
batch_size = 16
learning_rate = 1e-3
n_train = 1000      # Training samples
n_valtest = 500     # Val + Test samples
```

### Usage
```bash
# Submit all 30 MNIST jobs
bash job_scripts/submit_all_mnist_jobs.sh

# Or submit individual job
bash job_scripts/mnist_quantum_hydra_seed2024.sh

# Monitor progress
tail -f results/mnist_results/logs/*.log

# After completion, aggregate results
python scripts/aggregate_mnist_results.py
```

---

## ⏳ Experiment 1.3: DNA Sequence Classification

**Dataset**: UCI DNA Promoter/Splice Junction
**Task**: Binary/multi-class sequence classification
**Status**: ⏳ **PENDING** (requires DNA data loader)

### TODO
1. Create `Load_DNA_Sequences.py` data loader
2. Download UCI promoter/splice junction datasets
3. Create `run_single_model_dna.py` training script
4. Generate 30 job scripts (6 models × 5 seeds)
5. Create `aggregate_dna_results.py` aggregation script

---

## 📊 Models Under Test (6 total)

### Quantum Models (4)
1. **Quantum Hydra (Superposition)** - Option A
   - Files: `QuantumHydra.py`, `QuantumHydraTS`
   - Superposition: |ψ⟩ = α|ψ_shift⟩ + β|ψ_flip⟩ + γ|ψ_diag⟩

2. **Quantum Hydra (Hybrid)** - Option B
   - Files: `QuantumHydraHybrid.py`, `QuantumHydraHybridTS`
   - Classical combination: y = w₁·y₁ + w₂·y₂ + w₃·y₃

3. **Quantum Mamba (Superposition)** - Option A
   - Files: `QuantumMamba.py`, `QuantumMambaTS`
   - Superposition: |ψ⟩ = α|ψ_ssm⟩ + β|ψ_gate⟩ + γ|ψ_skip⟩

4. **Quantum Mamba (Hybrid)** - Option B
   - Files: `QuantumMambaHybrid.py`, `QuantumMambaHybridTS`
   - Classical combination: y = w₁·y_ssm + w₂·y_gate + w₃·y_skip

### Classical Baselines (2)
5. **Classical Hydra**
   - File: `TrueClassicalHydra.py`
   - Faithful implementation of Hwang et al. (2024)

6. **Classical Mamba**
   - File: `TrueClassicalMamba.py`
   - Faithful implementation of Gu & Dao (2024)

---

## 🔧 Generator Scripts

All job scripts are auto-generated using Python generators:

```bash
# EEG job scripts
python scripts/generate_eeg_job_scripts.py

# MNIST job scripts
python scripts/generate_mnist_job_scripts.py

# DNA job scripts (when ready)
python scripts/generate_dna_job_scripts.py
```

**Benefits of Generators:**
- ✅ Consistent conda environment across all scripts
- ✅ Easy to modify hyperparameters globally
- ✅ Automatic master submission script creation
- ✅ Reproducible and version-controlled

---

## 📈 Results Aggregation

Each experiment has a dedicated aggregation script that:
1. Loads results JSON from all seeds
2. Computes **mean ± std** for all metrics
3. Performs **statistical significance tests** (paired t-test)
4. Generates **CSV and LaTeX tables**
5. Creates **comparison plots** (accuracy, AUC, F1, training time, parameter efficiency)

### Output Files (per experiment)
- `{experiment}_aggregated_results.csv` - Summary table
- `{experiment}_aggregated_results.tex` - LaTeX table
- `{experiment}_comparison_plots.pdf` - Visualizations

---

## 🔬 Statistical Protocol

From EXPERIMENTAL_PLAN_README.md (lines 527-539):

- **Seeds**: 2024, 2025, 2026, 2027, 2028 (5 total)
- **Metrics**: Test Accuracy, Test AUC, Test F1, Training Time
- **Statistics**: Mean ± Std, 95% Confidence Intervals
- **Significance**: Paired t-test (p < 0.05)
- **Correction**: Bonferroni for multiple comparisons

---

## 📂 Directory Structure

```
quantum_hydra_mamba/
├── docs/                           # Documentation (this file is here)
│   └── EXPERIMENTS_README.md
│
├── scripts/                        # Training & aggregation scripts
│   ├── run_single_model_eeg.py
│   ├── run_single_model_mnist.py
│   ├── run_single_model_dna.py
│   ├── run_gated_models.py
│   ├── run_gated_genomic.py
│   ├── generate_eeg_job_scripts.py
│   ├── generate_mnist_job_scripts.py
│   ├── generate_dna_job_scripts.py
│   ├── aggregate_eeg_results.py
│   ├── aggregate_mnist_results.py
│   └── aggregate_dna_results.py
│
├── job_scripts/                    # SLURM job scripts
│   ├── eeg/                        # 30 EEG job scripts
│   │   ├── eeg_quantum_hydra_seed2024.sh
│   │   ├── ... (29 more)
│   │   └── submit_all_eeg_jobs.sh
│   ├── mnist/                      # 30 MNIST job scripts
│   ├── dna/                        # 30 DNA job scripts
│   └── gated/                      # Gated model job scripts
│
└── results/                        # Experiment outputs
    ├── eeg_results/
    │   ├── logs/
    │   ├── {model}_seed{seed}_results.json
    │   └── {model}_seed{seed}_model.pt
    ├── mnist_results/
    ├── dna_results/
    ├── genomic/
    └── gated/
```

---

## ⚙️ Environment Configuration

**Conda Environment**: `./conda-envs/qml_env` (VERIFIED ✅)

All job scripts use:
```bash
source activate ./conda-envs/qml_env
```

**Key Packages** (from conda environment):
- PennyLane (quantum circuits)
- PyTorch (neural networks)
- Qiskit (IBM quantum backend)
- Nevergrad (gradient-free optimization)
- NumPy, Pandas, Matplotlib, Scikit-Learn

---

## 🚀 Quick Start Guide

### Step 1: Run Experiments

```bash
# Option A: Run ALL experiments in parallel (60 jobs total)
bash job_scripts/eeg/submit_all_eeg_jobs.sh
bash job_scripts/submit_all_mnist_jobs.sh

# Option B: Run one experiment at a time
bash job_scripts/eeg/submit_all_eeg_jobs.sh
# Wait for completion...
bash job_scripts/submit_all_mnist_jobs.sh
```

### Step 2: Monitor Progress

```bash
# Watch all EEG logs
tail -f results/eeg_results/logs/*.log

# Watch specific model
tail -f results/eeg_results/logs/eeg_quantum_hydra_*.log

# Count completed jobs
ls results/eeg_results/*.json | wc -l
# Should eventually show 30 (6 models × 5 seeds)
```

### Step 3: Aggregate Results

```bash
# After all EEG jobs complete
python scripts/aggregate_eeg_results.py

# After all MNIST jobs complete
python scripts/aggregate_mnist_results.py
```

### Step 4: Analyze Results

```bash
# View summary tables
cat results/eeg_results/eeg_aggregated_results.csv
cat results/mnist_results/mnist_aggregated_results.csv

# View plots (on local machine with PDF viewer)
open results/eeg_results/eeg_comparison_plots.pdf
open results/mnist_results/mnist_comparison_plots.pdf
```

---

## ⏱️ Estimated Runtime

### Per Job (single model, single seed)
- **Quantum models**: 1-4 hours (depends on qubits, circuit depth)
- **Classical models**: 30-60 minutes

### Total Runtime (if run sequentially)
- **EEG (30 jobs)**: ~60-120 hours
- **MNIST (30 jobs)**: ~40-80 hours

### Recommended Approach
**Run in parallel** with available GPU resources:
- 6 GPUs → ~6-12 hours per experiment
- 12 GPUs → ~3-6 hours per experiment

---

## 🛠️ Troubleshooting

### Missing conda environment
```bash
# Check if environment exists
conda env list | grep qml_env

# If missing, check CLAUDE.md for installation instructions
```

### Job failures
```bash
# Check error logs
cat results/eeg_results/logs/eeg_quantum_hydra_seed2024.sh.log

# Common issues:
# 1. Missing packages → check conda env
# 2. CUDA out of memory → reduce batch size
# 3. Missing data files → check data loaders
```

### Incomplete results
```bash
# Count expected vs actual results
echo "Expected: 30 (6 models × 5 seeds)"
echo "Actual: $(ls results/eeg_results/*_results.json 2>/dev/null | wc -l)"

# Re-run missing jobs individually
bash job_scripts/eeg/eeg_quantum_hydra_seed2024.sh
```

---

## 📝 Next Steps

1. ✅ **Complete DNA experiment** (Experiment 1.3)
   - Create DNA data loader
   - Generate job scripts
   - Run experiments

2. ⏳ **Hyperparameter sweeps** (from EXPERIMENTAL_PLAN_README.md lines 468-469)
   - Vary qubits: 4, 6, 8
   - Compare performance vs model size

3. ⏳ **Cross-experiment analysis**
   - Compare models across all 3 modalities (EEG, MNIST, DNA)
   - Identify where quantum models excel

4. ⏳ **Publication preparation**
   - Create main results table (6 models × 3 tasks)
   - Write results section
   - Prepare supplementary materials

---

## 📚 References

- **EXPERIMENTAL_PLAN_README.md** - Full experimental design
- **CLAUDE.md** - Codebase overview and development guide
- **compare_all_models.py** - Original 4-model comparison (extended to 6 models)

---

**Last Updated**: 2025-11-08
**Author**: Junghoon Park
**Status**: 2/3 Tier 1 Experiments Ready
