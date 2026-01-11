# Synthetic Benchmark Experimental Protocol

## Long-Range Sequence Learning Evaluation for Quantum-Classical Hybrid Models

**Version:** 1.0
**Date:** January 2026
**Authors:** Research Team
**Status:** Protocol Design Complete

---

## Table of Contents

1. [Introduction and Motivation](#1-introduction-and-motivation)
2. [Task Descriptions](#2-task-descriptions)
3. [Rationale for Task Selection](#3-rationale-for-task-selection)
4. [Models Under Evaluation](#4-models-under-evaluation)
5. [Experimental Design](#5-experimental-design)
6. [Step-by-Step Procedures](#6-step-by-step-procedures)
7. [Evaluation Metrics](#7-evaluation-metrics)
8. [Expected Outcomes and Hypotheses](#8-expected-outcomes-and-hypotheses)
9. [Analysis Plan](#9-analysis-plan)
10. [Computational Requirements](#10-computational-requirements)
11. [Appendix: Dataloader Usage](#appendix-dataloader-usage)

---

## 1. Introduction and Motivation

### 1.1 Background

The EEG ablation study (see `ABLATION_EEG_RESULTS_SUMMARY.md`) revealed that:
- Model performance differences between top performers (e.g., 1c vs 2e) fall within confidence intervals
- Statistical significance requires more controlled experiments
- Real-world biomedical data contains confounding factors that obscure model capability differences

### 1.2 Why Synthetic Benchmarks?

Synthetic tasks provide:
1. **Ground Truth**: Perfect labels enable precise capability measurement
2. **Controlled Difficulty**: Adjustable sequence lengths to probe dependency distances
3. **Interpretable Failures**: Clear task structure reveals where models fail
4. **Reproducibility**: Deterministic generation with fixed seeds
5. **Scalability**: Generate unlimited data without collection costs

### 1.3 Research Questions

| # | Question | Addressed By |
|---|----------|--------------|
| RQ1 | Which models best capture long-range dependencies? | All tasks at L=500+ |
| RQ2 | Does quantum feature extraction improve sequence memory? | 1a-1c vs 3a-3c |
| RQ3 | Does true quantum superposition help with sequence learning? | 2d/2e vs 2b/2c |
| RQ4 | At what sequence length do models fail? | Length scaling experiments |
| RQ5 | Do quantum models show advantage on quantum-inspired tasks? | Forrelation task |

---

## 2. Task Descriptions

### 2.1 Forrelation Task (Classification)

**Origin:** Aaronson & Ambainis (2015) - Quantum algorithm for distinguishing forrelated vs random functions

**Task Definition:**
- **Input:** Two binary functions f, g encoded as sequences
- **Output:** Binary classification (forrelated=1, random=0)
- **Forrelation Condition:** Functions satisfy quantum correlation property

**Mathematical Formulation:**
```
Forrelation(f, g) = (1/2^n) * Σ_x Σ_y (-1)^(f(x) + g(y) + x·y)
```

**Input Format:**
- Shape: `(batch, 2, seq_len)` where seq_len = 2 * 2^n_bits
- Channel 0: Function f values
- Channel 1: Function g values

**Why This Task:**
- Quantum computers can solve this exponentially faster than classical
- Tests whether quantum circuits can detect quantum correlations
- Provides potential quantum advantage signal

**Baseline Performance:**
- Random guess: 50% accuracy
- Well-tuned classical model: ~70-80% accuracy

---

### 2.2 Adding Problem (Regression)

**Origin:** Hochreiter & Schmidhuber (1997) - LSTM paper

**Task Definition:**
- **Input:** Sequence of random values in [0, 1] with exactly two markers
- **Output:** Sum of the two marked values (scalar in [0, 2])
- **Challenge:** Markers placed at sequence extremes to maximize dependency distance

**Mathematical Formulation:**
```
Given: values = [v_0, v_1, ..., v_{L-1}], markers = [m_0, m_1, ..., m_{L-1}]
Where: m_i ∈ {0, 1}, Σm_i = 2
Output: y = Σ(v_i * m_i) = v_{pos1} + v_{pos2}
```

**Marker Placement Strategies:**
| Strategy | Description | Dependency Distance |
|----------|-------------|---------------------|
| `extremes` | First 10% + Last 10% | ~80% of sequence length |
| `halves` | First half + Second half | ~50% of sequence length |
| `uniform` | Random positions | Variable |

**Input Format:**
- Shape: `(batch, 2, seq_len)`
- Channel 0: Random values [0, 1]
- Channel 1: Binary markers (exactly two 1s)
- Target: `(batch, 1)` sum of marked values

**Why This Task:**
- Classic benchmark for sequence memory
- Adjustable difficulty via sequence length
- Clear failure mode: MSE approaching baseline indicates memory failure

**Baseline Performance:**
- Predict mean (1.0): MSE ~ 0.167
- Perfect model: MSE ~ 0.0

---

### 2.3 Selective Copy Task (Sequence-to-Sequence Regression)

**Task Definition:**
- **Input:** Sequence of random tokens with sparse markers
- **Output:** Only the marked tokens in their original order
- **Challenge:** Remember multiple marked elements across long distances

**Mathematical Formulation:**
```
Given: tokens = [t_0, t_1, ..., t_{L-1}], markers = [m_0, m_1, ..., m_{L-1}]
Where: t_i ∈ [0, 1], m_i ∈ {0, 1}, Σm_i = K (fixed number of markers)
Output: y = [t_{pos_1}, t_{pos_2}, ..., t_{pos_K}] where pos_1 < pos_2 < ... < pos_K
```

**Marker Placement Strategies:**
| Strategy | Description | Pattern |
|----------|-------------|---------|
| `uniform` | Random positions | Sparse, unpredictable |
| `spread` | Evenly distributed with jitter | Regular spacing |
| `clustered` | Grouped in clusters | Tests local vs global attention |

**Input Format:**
- Shape: `(batch, 2, seq_len)`
- Channel 0: Random token values [0, 1]
- Channel 1: Binary markers (K markers)
- Target: `(batch, num_markers)` marked tokens in order

**Why This Task:**
- Tests both memory AND selective attention
- Multiple dependency relationships (not just two)
- Relevant for SSM's selective mechanism (Delta in Mamba)
- More challenging than Adding Problem

**Baseline Performance:**
- Predict mean (0.5): MSE ~ 0.083
- Perfect model: MSE ~ 0.0

---

## 3. Rationale for Task Selection

### 3.1 Complementary Capabilities Tested

| Capability | Forrelation | Adding Problem | Selective Copy |
|------------|-------------|----------------|----------------|
| Long-range memory | Medium | High | Very High |
| Selective attention | Low | Medium | Very High |
| Quantum correlation detection | High | Low | Low |
| Multi-element tracking | Low | Low (2 elements) | High (8+ elements) |
| Sequence-to-scalar | Yes | Yes | No |
| Sequence-to-sequence | No | No | Yes |

### 3.2 Task Difficulty Progression

```
Forrelation (Easiest) → Adding Problem → Selective Copy (Hardest)
       ↓                      ↓                    ↓
  Classification         Regression          Multi-output Regression
  (binary output)      (scalar output)      (sequence output)
```

### 3.3 Why These Three Tasks Together

1. **Forrelation**: Tests quantum-specific advantages (if any exist)
2. **Adding Problem**: Tests fundamental long-range memory
3. **Selective Copy**: Tests the full capability required for real-world sequence modeling

**Key Insight:** A model that excels at Selective Copy likely has the capabilities needed for complex real-world tasks like EEG classification.

---

## 4. Models Under Evaluation

### 4.1 Complete Model List (from Ablation Study)

| ID | Model Name | Feature Extraction | Sequence Mixing | Architecture |
|----|-----------|-------------------|-----------------|--------------|
| **Group 1: Quantum Feature + Classical Mixing** |
| 1a | QuantumTransformer | Quantum | Classical | Transformer |
| 1b | QuantumMambaSSM | Quantum | Classical | Mamba |
| 1c | QuantumHydraSSM | Quantum | Classical | Hydra |
| **Group 2: Classical Feature + Quantum Mixing** |
| 2a | ClassicalQuantumAttention | Classical | Quantum | Transformer |
| 2b | ClassicalMambaQuantumSSM | Classical | Quantum | Mamba |
| 2c | ClassicalHydraQuantumSSM | Classical | Quantum | Hydra |
| 2d | QuantumMambaHydraSSM | Classical | Quantum (Superposition) | Mamba + Hydra |
| 2e | QuantumHydraHydraSSM | Classical | Quantum (Superposition) | Hydra + Hydra |
| **Group 3: Classical Baselines** |
| 3a | ClassicalTransformer | Classical | Classical | Transformer |
| 3b | TrueClassicalMamba | Classical | Classical | Mamba |
| 3c | TrueClassicalHydra | Classical | Classical | Hydra |
| **Group 4: End-to-End Quantum** |
| 4a | QuantumTransformerE2E | Quantum | Quantum | Transformer |
| 4b | QuantumMambaE2E | Quantum | Quantum | Mamba |
| 4c | QuantumHydraE2E | Quantum | Quantum | Hydra |
| 4d | QuantumMambaE2E_Superposition | Quantum | Quantum (Superposition) | Mamba |
| 4e | QuantumHydraE2E_Superposition | Quantum | Quantum (Superposition) | Hydra |

### 4.2 Key Model Comparisons

| Comparison | Models | Tests |
|------------|--------|-------|
| Quantum vs Classical Features | 1a-1c vs 3a-3c | Does quantum encoding help? |
| Quantum vs Classical Mixing | 2a-2e vs 3a-3c | Does quantum state evolution help? |
| True Superposition Effect | 2d/2e vs 2b/2c | Does true |ψ⟩ = α|ψ₁⟩ + β|ψ₂⟩ + γ|ψ₃⟩ help? |
| E2E Quantum Effect | 4a-4e vs 1a-1c | Is full quantum better than hybrid? |
| Architecture Effect | *a vs *b vs *c | Transformer vs Mamba vs Hydra |

### 4.3 Model Subset Selection

For computational efficiency, consider running in phases:

**Phase 1 (Core Comparison):**
- 1c (QuantumHydraSSM) - Best hybrid on EEG
- 2e (QuantumHydraHydraSSM) - True superposition
- 3b (TrueClassicalMamba) - Best classical on EEG
- 3c (TrueClassicalHydra) - Classical Hydra baseline

**Phase 2 (Extended Comparison):**
- Add remaining Group 1 and Group 3 models

**Phase 3 (Full Comparison):**
- All 16 models

---

## 5. Experimental Design

### 5.1 Dataset Generation Parameters

#### Forrelation
| Parameter | Values |
|-----------|--------|
| n_bits | 6 |
| seq_len | 50, 100, 200 |
| num_samples | 5000 per length |
| balance | 50% forrelated, 50% random |

#### Adding Problem
| Parameter | Values |
|-----------|--------|
| seq_len | 100, 200, 500, 1000 |
| num_samples | 5000 per length |
| marker_strategy | "extremes" (default) |
| seed | 2024 |

#### Selective Copy
| Parameter | Values |
|-----------|--------|
| seq_len | 100, 200, 500, 1000 |
| num_markers | 8 |
| num_samples | 5000 per length |
| marker_strategy | "uniform" (default) |
| seed | 2024 |

### 5.2 Training Configuration

| Parameter | Value | Notes |
|-----------|-------|-------|
| Qubits | 6 | Match EEG ablation |
| Layers | 2 | Match EEG ablation |
| d_model | 128 | Match EEG ablation |
| d_state | 16 | Match EEG ablation |
| Epochs | 100 | Longer for convergence |
| Batch Size | 32 | Standard |
| Learning Rate | 0.001 | Adam optimizer |
| Weight Decay | 0.0001 | Regularization |
| Scheduler | Cosine Annealing | With warm restarts |
| Early Stopping | Patience=20 | On validation loss |
| Seeds | 2024, 2025, 2026 | 3 seeds for statistics |

### 5.3 Data Splits

| Split | Ratio | Purpose |
|-------|-------|---------|
| Training | 80% | Model training |
| Validation | 10% | Hyperparameter selection, early stopping |
| Test | 10% | Final evaluation (never used during training) |

### 5.4 Experimental Matrix

**Total Experiments per Task:**
- Models: 16
- Sequence Lengths: 4
- Seeds: 3
- **Total: 192 experiments per task**

**Grand Total: 576 experiments (3 tasks × 192)**

---

## 6. Step-by-Step Procedures

### Step 1: Environment Setup

```bash
# Activate conda environment
conda activate ./conda-envs/qml_eeg

# Navigate to project directory
cd /pscratch/sd/j/junghoon/quantum_hydra_mamba

# Verify dependencies
python -c "import torch; import pennylane; print('Setup OK')"
```

### Step 2: Generate Datasets

```bash
# Create data directory
mkdir -p data/synthetic_benchmarks

# Generate Forrelation datasets
cd data_loaders
for seq_len in 50 100 200; do
    python generate_forrelation_dataset.py \
        --num_pairs=5000 \
        --n_bits=6 \
        --seq_len=${seq_len} \
        --output_dir=../data/synthetic_benchmarks/forrelation
done

# Generate Adding Problem datasets
for seq_len in 100 200 500 1000; do
    python generate_adding_problem.py \
        --num_samples=5000 \
        --seq_len=${seq_len} \
        --marker_strategy=extremes \
        --filename=../data/synthetic_benchmarks/adding_problem/adding_L${seq_len}_seed2024.pt \
        --seed=2024 \
        --verify
done

# Generate Selective Copy datasets
for seq_len in 100 200 500 1000; do
    python generate_selective_copy.py \
        --num_samples=5000 \
        --seq_len=${seq_len} \
        --num_markers=8 \
        --marker_strategy=uniform \
        --filename=../data/synthetic_benchmarks/selective_copy/selective_copy_L${seq_len}_M8_seed2024.pt \
        --seed=2024 \
        --verify
done
```

### Step 3: Verify Generated Datasets

```bash
# Check dataset statistics
python -c "
import torch
for task in ['forrelation', 'adding_problem', 'selective_copy']:
    print(f'\\n=== {task.upper()} ===')
    # Load and verify each dataset
"
```

### Step 4: Create Training Script

Create `train_synthetic_benchmark.py`:
```python
# See Appendix for full training script template
```

### Step 5: Submit SLURM Jobs

```bash
# Phase 1: Core comparison (4 models × 4 lengths × 3 seeds × 3 tasks = 144 jobs)
sbatch scripts/synthetic_benchmark_phase1.sh

# Monitor progress
squeue -u $USER
```

### Step 6: Run Experiments

For each model, task, sequence length, and seed:
1. Load appropriate dataset
2. Initialize model with matching configuration
3. Train with early stopping
4. Save best model checkpoint
5. Evaluate on test set
6. Log all metrics

### Step 7: Collect Results

```bash
# Aggregate results
python scripts/aggregate_synthetic_results.py \
    --results_dir=results/synthetic_benchmarks \
    --output=results/synthetic_benchmark_summary.csv
```

### Step 8: Analyze Results

```bash
# Generate analysis plots and tables
python scripts/analyze_synthetic_benchmarks.py \
    --input=results/synthetic_benchmark_summary.csv \
    --output_dir=results/figures
```

---

## 7. Evaluation Metrics

### 7.1 Forrelation (Classification)

| Metric | Formula | Target |
|--------|---------|--------|
| **Accuracy** | TP + TN / Total | > 80% |
| **ROC-AUC** | Area under ROC curve | > 0.85 |
| **F1 Score** | 2 × (P × R) / (P + R) | > 0.80 |

### 7.2 Adding Problem (Regression)

| Metric | Formula | Target |
|--------|---------|--------|
| **MSE** | (1/N) Σ(y - ŷ)² | < 0.01 |
| **MAE** | (1/N) Σ|y - ŷ| | < 0.05 |
| **R² Score** | 1 - SS_res/SS_tot | > 0.95 |
| **Relative to Baseline** | MSE / 0.167 | < 0.1 |

### 7.3 Selective Copy (Multi-Output Regression)

| Metric | Formula | Target |
|--------|---------|--------|
| **MSE** | (1/NK) ΣΣ(y_ik - ŷ_ik)² | < 0.01 |
| **MAE** | (1/NK) ΣΣ|y_ik - ŷ_ik| | < 0.05 |
| **Position Accuracy** | % positions within tolerance (0.1) | > 90% |
| **Sequence Accuracy** | % sequences all correct | > 70% |

### 7.4 Efficiency Metrics (All Tasks)

| Metric | Unit | Calculation |
|--------|------|-------------|
| Training Time | hours | Wall clock time |
| Convergence Epoch | epoch | First epoch reaching 95% of final performance |
| Parameters | count | Model parameter count |
| FLOPs | operations | Forward pass complexity |

---

## 8. Expected Outcomes and Hypotheses

### 8.1 Primary Hypotheses

| # | Hypothesis | Test |
|---|-----------|------|
| H1 | Group 1 models (quantum feature extraction) will outperform Group 3 (classical) on Forrelation | Compare 1a-1c vs 3a-3c accuracy |
| H2 | True superposition models (2d, 2e) will show more consistent performance across sequence lengths | Compare variance of 2d/2e vs 2b/2c |
| H3 | All models will show performance degradation as sequence length increases | Plot performance vs seq_len |
| H4 | SSM architectures (Mamba, Hydra) will outperform Transformers at longer sequences | Compare *b/*c vs *a at L=1000 |
| H5 | Quantum models will not catastrophically fail on synthetic tasks (unlike 2b/2c on EEG at 80Hz) | Check for ~50% performance |

### 8.2 Expected Performance Ranges

#### Forrelation
| Model Group | Expected Accuracy |
|-------------|-------------------|
| Group 1 (Q-Feat + C-Mix) | 75-85% |
| Group 2 (C-Feat + Q-Mix) | 65-80% |
| Group 3 (Classical) | 70-80% |
| Group 4 (E2E Quantum) | 70-85% |

#### Adding Problem (L=200)
| Model Group | Expected MSE |
|-------------|--------------|
| All models (if working) | < 0.02 |
| Failing models | > 0.10 |
| Baseline | 0.167 |

#### Selective Copy (L=200, M=8)
| Model Group | Expected MSE |
|-------------|--------------|
| All models (if working) | < 0.03 |
| Failing models | > 0.06 |
| Baseline | 0.083 |

### 8.3 Failure Mode Predictions

| Model | Task | Predicted Failure Mode |
|-------|------|------------------------|
| 2b, 2c | All | Possible instability (observed on EEG) |
| All | Selective Copy L=1000 | Potential performance degradation |
| 4a-4e | Adding Problem L=1000 | May show earlier degradation than Group 1 |

---

## 9. Analysis Plan

### 9.1 Primary Analyses

#### Analysis 1: Per-Task Performance Comparison
- Create ranked tables for each task (like EEG results)
- Statistical tests: ANOVA + post-hoc Tukey HSD
- Effect sizes: Cohen's d for pairwise comparisons

#### Analysis 2: Sequence Length Scaling
- Plot performance vs sequence length for each model
- Fit degradation curves: performance = a × log(L) + b
- Identify "critical length" where performance drops below threshold

#### Analysis 3: Model Group Comparison
- Aggregate by group (1, 2, 3, 4)
- Compare group means with confidence intervals
- Test if quantum components provide advantage

### 9.2 Secondary Analyses

#### Analysis 4: Consistency Analysis
- Calculate coefficient of variation (CV) across seeds
- Identify most/least stable models
- Compare CV between quantum and classical models

#### Analysis 5: Efficiency-Performance Trade-off
- Plot Pareto frontier: performance vs training time
- Identify efficient models (good performance, low cost)

#### Analysis 6: Architecture Deep Dive
- Compare Transformer vs Mamba vs Hydra within each group
- Identify which architecture best suits each task

### 9.3 Visualization Plan

| Figure | Type | Shows |
|--------|------|-------|
| Fig 1 | Heatmap | Performance matrix (models × tasks × lengths) |
| Fig 2 | Line plot | Performance degradation curves |
| Fig 3 | Bar chart | Model rankings with error bars |
| Fig 4 | Scatter | Accuracy vs Training time |
| Fig 5 | Box plot | Performance distribution by group |
| Fig 6 | Radar | Multi-dimensional model comparison |

### 9.4 Statistical Tests

| Comparison | Test | Significance Level |
|------------|------|-------------------|
| Overall model differences | One-way ANOVA | α = 0.05 |
| Pairwise model comparisons | Tukey HSD | Family-wise α = 0.05 |
| Group-level comparisons | Two-sample t-test | α = 0.05 with Bonferroni correction |
| Correlation with sequence length | Spearman's ρ | α = 0.05 |

---

## 10. Computational Requirements

### 10.1 Estimated Training Times (per experiment)

| Model Group | Time per Experiment | Total (Phase 1) | Total (All) |
|-------------|---------------------|-----------------|-------------|
| Group 1 | 0.3-0.5 hours | 14.4-24 hours | 43.2-72 hours |
| Group 2 | 2-10 hours | 96-480 hours | 288-1440 hours |
| Group 3 | 0.1-0.3 hours | 4.8-14.4 hours | 14.4-43.2 hours |
| Group 4 | 0.1-0.2 hours | 4.8-9.6 hours | 14.4-28.8 hours |
| Group 4 (Super) | 5-15 hours | 240-720 hours | 720-2160 hours |

**Phase 1 Total (4 models):** ~20-50 GPU hours
**Full Study Total (16 models):** ~1000-4000 GPU hours

### 10.2 Storage Requirements

| Item | Size |
|------|------|
| Datasets (all lengths) | ~2 GB |
| Model checkpoints (16 models × 4 lengths × 3 seeds × 3 tasks) | ~50 GB |
| Results and logs | ~1 GB |
| **Total** | **~55 GB** |

### 10.3 SLURM Resource Allocation

```bash
#SBATCH --account=m4138_g
#SBATCH --constraint=gpu&hbm80g
#SBATCH --qos=shared
#SBATCH --time=48:00:00
#SBATCH --nodes=1
#SBATCH --gpus=1
#SBATCH --cpus-per-task=32
```

---

## Appendix: Dataloader Usage

### A.1 Forrelation Dataloader

```python
from data_loaders.forrelation_dataloader import get_forrelation_dataloader

train_loader, test_loader, params = get_forrelation_dataloader(
    dataset_path="data/synthetic_benchmarks/forrelation/forrelation_L100.pt",
    batch_size=32,
    shuffle=True
)

# Input shape: (batch, 2, seq_len)
# Labels: (batch,) binary {0, 1}
```

### A.2 Adding Problem Dataloader

```python
from data_loaders.adding_problem_dataloader import get_adding_problem_dataloader

train_loader, val_loader, test_loader, params = get_adding_problem_dataloader(
    dataset_path="data/synthetic_benchmarks/adding_problem/adding_L200_seed2024.pt",
    batch_size=32,
    train_ratio=0.8,
    val_ratio=0.1,
    seed=2024
)

# Input shape: (batch, 2, seq_len)
# Channel 0: values [0, 1]
# Channel 1: markers (binary)
# Target shape: (batch, 1)
```

### A.3 Selective Copy Dataloader

```python
from data_loaders.selective_copy_dataloader import get_selective_copy_dataloader

train_loader, val_loader, test_loader, params = get_selective_copy_dataloader(
    dataset_path="data/synthetic_benchmarks/selective_copy/selective_copy_L200_M8_seed2024.pt",
    batch_size=32,
    train_ratio=0.8,
    val_ratio=0.1,
    seed=2024
)

# Input shape: (batch, 2, seq_len)
# Channel 0: tokens [0, 1]
# Channel 1: markers (binary)
# Target shape: (batch, num_markers)
```

### A.4 Selective Copy Metrics

```python
from data_loaders.selective_copy_dataloader import compute_selective_copy_metrics

metrics = compute_selective_copy_metrics(predictions, targets, tolerance=0.1)
# Returns: {'mse': float, 'mae': float, 'position_accuracy': float, 'sequence_accuracy': float}
```

---

## Document History

| Version | Date | Changes |
|---------|------|---------|
| 1.0 | Jan 2026 | Initial protocol design |

---

*This protocol is designed to provide rigorous, reproducible evaluation of quantum-classical hybrid models on controlled synthetic benchmarks for long-range sequence learning.*
