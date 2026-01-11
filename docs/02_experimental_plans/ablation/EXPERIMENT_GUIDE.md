# Experimental Guide: Quantum vs Classical State-Space Models

> Comprehensive guide for running experiments comparing Quantum Hydra, Quantum Mamba, and their classical baselines

**Last Updated**: November 2025

---

## 📋 Table of Contents

1. [Overview](#overview)
2. [Models Under Investigation](#models-under-investigation)
3. [Datasets](#datasets)
4. [Research Questions](#research-questions)
5. [Running Experiments](#running-experiments)
6. [Expected Results](#expected-results)
7. [Analysis](#analysis)

---

## 🎯 Overview

This guide outlines experiments to validate **4 quantum state-space models** against **2 classical baselines** across multiple data modalities.

### The Six Models

| Model | Type | File | Description |
|-------|------|------|-------------|
| **Quantum Hydra (Super)** | Quantum A | `QuantumHydra.py` | Superposition with complex coefficients |
| **Quantum Hydra (Hybrid)** | Quantum B | `QuantumHydraHybrid.py` | Classical combination with real weights |
| **Quantum Mamba (Super)** | Quantum A | `QuantumMamba.py` | Superposition with complex coefficients |
| **Quantum Mamba (Hybrid)** | Quantum B | `QuantumMambaHybrid.py` | Classical combination with real weights |
| **True Classical Hydra** | Classical | `TrueClassicalHydra.py` | Baseline (Hwang et al., 2024) |
| **True Classical Mamba** | Classical | `TrueClassicalMamba.py` | Baseline (Gu & Dao, 2024) |

---

## 🧬 Models Under Investigation

### Quantum Models (4 total)

#### 1. Quantum Hydra (Superposition) - Option A
**Architecture:**
```
|ψ⟩ = α|ψ_shift⟩ + β|ψ_flip⟩ + γ|ψ_diag⟩
where α, β, γ ∈ ℂ
```

**Key Features:**
- Quantum superposition before measurement
- Complex coefficients
- Potential for quantum interference

**Hypothesis:** Quantum interference captures non-classical correlations in time-series

---

#### 2. Quantum Hydra (Hybrid) - Option B
**Architecture:**
```
y = w₁·y₁ + w₂·y₂ + w₃·y₃
where w₁, w₂, w₃ ∈ ℝ
```

**Key Features:**
- Three independent quantum circuits
- Classical weighted combination
- Real-valued weights
- Faithful to classical Hydra semantics

**Hypothesis:** Classical combination more robust to quantum noise

---

#### 3. Quantum Mamba (Superposition) - Option A
**Architecture:**
```
|ψ⟩ = α|ψ_ssm⟩ + β|ψ_gate⟩ + γ|ψ_skip⟩
where α, β, γ ∈ ℂ
```

**Key Features:**
- Selective SSM via quantum circuits
- Input-dependent B, C, dt parameters
- Quantum superposition of SSM + gating + skip

**Hypothesis:** Quantum selective SSM better at content-based reasoning

---

#### 4. Quantum Mamba (Hybrid) - Option B
**Architecture:**
```
y = w₁·y_ssm + w₂·y_gate + w₃·y_skip
where w₁, w₂, w₃ ∈ ℝ
```

**Key Features:**
- Three independent quantum paths
- Classical combination
- Interpretable branch contributions

**Hypothesis:** Hybrid approach more stable and interpretable

---

### Classical Baselines (2 total)

#### 5. True Classical Hydra
**Architecture:** Faithful implementation of Hwang et al. (2024)
- Bidirectional processing
- Semi-separable matrix operations
- Shift, flip, diagonal branches

**Purpose:** Fair classical baseline for Quantum Hydra models

---

#### 6. True Classical Mamba
**Architecture:** Faithful implementation of Gu & Dao (2024)
- Selective SSM with input-dependent parameters
- RMSNorm
- Gated MLP blocks

**Purpose:** SOTA classical SSM baseline for comparison

---

## 📊 Datasets

### 1. EEG Motor Imagery Classification ⭐⭐⭐

**Dataset:** PhysioNet Motor Imagery

**Details:**
- **Task:** Binary classification (left/right hand movement)
- **Channels:** 64 EEG channels
- **Sequence Length:** 160 timesteps
- **Classes:** 2 (left hand, right hand)
- **Samples:** 10-20 subjects

**Why Essential:**
✅ Real-world biomedical application
✅ Rich temporal patterns (SSMs excel here)
✅ Manageable scale for quantum circuits
✅ Tests selective SSM mechanisms

**Run experiments:**
```bash
# Single model
python scripts/run_single_model_eeg.py \
    --model-name quantum_hydra \
    --seed 2024 \
    --device cuda

# All 6 models
bash scripts/run_all_eeg_experiments.sh
```

**Expected Outcome:** Quantum models competitive (75-85% accuracy)

---

### 2. MNIST Image Classification ⭐⭐⭐

**Dataset:** MNIST (28×28 grayscale images)

**Details:**
- **Task:** 10-class digit recognition
- **Image Size:** 28×28 grayscale (784 pixels)
- **Training Samples:** 500-1,000 (subset for quantum)
- **Test Samples:** 250-500

**Why Essential:**
✅ Standard ML benchmark
✅ Easy comparison with literature
✅ Small enough for quantum encoding
✅ Well-understood baselines

**Data Encoding:**
- Flatten 28×28 → 784 pixels
- Downsample or PCA to fit qubit constraints
- Amplitude encoding: |ψ⟩ = Σᵢ xᵢ|i⟩

**Run experiments:**
```bash
# Single model
python scripts/run_single_model_mnist.py \
    --model-name quantum_mamba \
    --seed 2024 \
    --device cuda

# All 6 models
bash scripts/run_all_mnist_experiments.sh
```

**Expected Outcome:** Quantum models 85-90% accuracy (classical 95-98%)

---

### 3. DNA Sequence Classification ⭐⭐⭐

**Dataset:** DNA Promoter Recognition (UCI ML Repository)

**Details:**
- **Task:** Binary classification (promoter vs non-promoter)
- **Sequence Length:** 57 nucleotides
- **Alphabet:** {A, C, G, T} → 4 symbols
- **Samples:** 106 promoters vs 106 non-promoters

**Why EXCELLENT for Quantum:**
✅ Gu & Dao tested Mamba on DNA (long sequences)
✅ Natural quantum encoding: A→|00⟩, C→|01⟩, G→|10⟩, T→|11⟩
✅ Limited alphabet (2 qubits per position)
✅ Sequential structure (perfect for SSMs)
✅ Short sequences feasible for quantum

**Data Encoding:**
```python
# One-hot encoding per nucleotide
A → [1, 0, 0, 0]
C → [0, 1, 0, 0]
G → [0, 0, 1, 0]
T → [0, 0, 0, 1]
```

**Run experiments:**
```bash
# Single model
python scripts/run_single_model_dna.py \
    --model-name classical_mamba \
    --encoding onehot \
    --seed 2024 \
    --device cuda

# All 6 models
bash scripts/run_all_dna_experiments.sh
```

**Expected Outcome:** Quantum models may **EXCEED** classical (natural encoding advantage)

---

### 4. Forrelation (Quantum Advantage Test) ⭐⭐⭐

**Dataset:** Sequential Forrelation (generated)

**Details:**
- **Task:** Binary classification (high vs low forrelation)
- **Sequence Length:** Variable (20, 40, 80, 160)
- **n_bits:** 6 or 8
- **Purpose:** Test proven quantum advantage (Aaronson & Ambainis, 2015)

**Why CRITICAL:**
✅ Proven quantum advantage exists
✅ Tests sample efficiency (fewer samples needed)
✅ Tests scaling with problem complexity
✅ Designed specifically for quantum computers

**Generate datasets:**
```bash
bash scripts/generate_forrelation_datasets.sh
```

**Run experiments:**
```bash
# All models on all Forrelation datasets
bash scripts/run_all_forrelation_experiments.sh

# Analyze quantum advantage
python scripts/aggregate_forrelation_results.py
```

**Expected Outcome:**
- **Silver Standard**: Quantum models need 30-50% **fewer samples** to reach 90% accuracy
- **Gold Standard**: Advantage **increases** with problem complexity

---

## 🔬 Research Questions

### RQ1: Can quantum models be competitive with classical baselines?

**Hypothesis:** Quantum Hydra/Mamba achieve ≥95% of classical performance on small-scale tasks

**Tests:**
- All datasets (EEG, MNIST, DNA, Forrelation)
- Compare test accuracy across all 6 models
- Statistical significance tests

**Success Criteria:**
- Quantum models within 2-3% accuracy of classical
- At least one task where quantum ≥ classical

---

### RQ2: Is quantum advantage architecture-specific or general?

**Hypothesis:** Quantum advantages are general across SSM architectures

**Tests:**
- Compare Quantum Hydra vs Quantum Mamba
- Performance across different data modalities
- Analyze which architecture benefits more from quantization

**Success Criteria:**
- Both Quantum Hydra and Quantum Mamba outperform or match baselines
- If only one excels → architecture-specific advantage

---

### RQ3: Does quantum superposition help?

**Hypothesis:** Quantum superposition (Option A) provides performance benefits over classical combination (Option B)

**Tests:**
- Option A vs Option B for both Hydra and Mamba
- Across all tasks
- Analyze when superposition helps vs when hybrid is better

**Success Criteria:**
- Option A > Option B on at least 2 out of 4 tasks
- Identify task characteristics where superposition helps

---

### RQ4: Where do quantum SSMs excel?

**Hypothesis:** Quantum models excel on tasks with natural quantum encoding (e.g., DNA) and pattern detection

**Tests:**
- Performance breakdown by data modality
- Analyze: Time-series vs Images vs Sequences

**Success Criteria:**
- Identify sweet spot for quantum SSMs
- DNA and EEG likely candidates

---

## 🚀 Running Experiments

### Quick Start

**1. Install dependencies:**
```bash
pip install -r requirements.txt
```

**2. Run a single experiment:**
```bash
python scripts/run_single_model_eeg.py \
    --model-name quantum_hydra \
    --n-qubits 6 \
    --n-epochs 50 \
    --batch-size 32 \
    --seed 2024 \
    --device cuda
```

**3. Run all experiments for one dataset:**
```bash
# EEG experiments (18 total: 6 models × 3 seeds)
bash scripts/run_all_eeg_experiments.sh

# MNIST experiments
bash scripts/run_all_mnist_experiments.sh

# DNA experiments
bash scripts/run_all_dna_experiments.sh
```

**4. Forrelation experiments:**
```bash
# Step 1: Generate datasets
bash scripts/generate_forrelation_datasets.sh

# Step 2: Run experiments (144 total: 6 models × 8 datasets × 3 seeds)
bash scripts/run_all_forrelation_experiments.sh

# Step 3: Analyze quantum advantage
python scripts/aggregate_forrelation_results.py
```

---

### Hyperparameters

#### Quantum Models
```python
n_qubits = 6              # Number of qubits
qlcu_layers = 2           # Circuit depth
d_model = 128             # Hidden dimension
batch_size = 32
learning_rate = 1e-3
optimizer = "Adam"
n_epochs = 50
```

#### Classical Models
```python
d_model = 128
d_state = 16
dropout = 0.1
batch_size = 32
learning_rate = 1e-3
n_epochs = 50
```

---

### Metrics

#### Performance Metrics
| Metric | Purpose |
|--------|---------|
| **Test Accuracy** | Primary performance measure |
| **Test AUC** | Robust to class imbalance |
| **Test F1 Score** | Balanced precision/recall |

#### Efficiency Metrics
| Metric | Purpose |
|--------|---------|
| **Training Time** | Computational cost |
| **Model Parameters** | Model size |
| **Convergence Epochs** | Training efficiency |

---

## 📈 Expected Results

### Scenario 1: Performance Comparison

**Expected:**
- **EEG**: Quantum ≈ Classical (within 2-3% accuracy)
- **MNIST**: Quantum 85-90%, Classical 95-98%
- **DNA**: Quantum possibly > Classical (natural encoding advantage)
- **Forrelation**: Quantum demonstrates sample efficiency

**Implication:** Quantum SSMs are viable for small-scale tasks

---

### Scenario 2: Parameter Efficiency

**Expected:**
- Quantum models: ~1,500-3,000 parameters
- Classical models: ~5,000-8,000 parameters
- Similar performance with 50-70% fewer parameters

**Implication:** Quantum models suitable for edge deployment, parameter-constrained settings

---

### Scenario 3: Superposition Benefits

**Expected:**
- Option A > Option B on DNA sequences (pattern detection)
- Option B > Option A on images (more stable, interpretable)
- Mixed results on EEG

**Implication:** Design choice depends on task characteristics

---

### Scenario 4: Architecture Comparison

**Expected:**
- Quantum Mamba slightly better on sequences (DNA)
- Quantum Hydra competitive on time-series (EEG)
- Both comparable on images

**Implication:** Architecture choice matters, but both are viable quantum SSM designs

---

## 📊 Analysis

### Aggregate Results

After running experiments, aggregate results:

```bash
# EEG results
python scripts/aggregate_eeg_results.py
# → Generates: eeg_results/eeg_all_results.csv

# MNIST results
python scripts/aggregate_mnist_results.py
# → Generates: mnist_results/mnist_all_results.csv

# DNA results
python scripts/aggregate_dna_results.py
# → Generates: dna_results/dna_all_results.csv

# Forrelation (Quantum Advantage Analysis)
python scripts/aggregate_forrelation_results.py
# → Generates:
#   - forrelation_all_results.csv
#   - forrelation_sample_efficiency.png
#   - forrelation_heatmap.png
```

### Statistical Analysis

**Multiple Runs:**
- Minimum 3 runs per configuration (different seeds: 2024, 2025, 2026)
- Report mean ± standard deviation

**Statistical Tests:**
- Paired t-test for model comparisons
- Confidence intervals (95%)

---

## 📚 References

### Key Papers

1. **Gu, A., & Dao, T. (2024).** Mamba: Linear-Time Sequence Modeling with Selective State Spaces.
   - https://arxiv.org/html/2312.00752v2

2. **Hwang, W., Kim, M., Zhang, X., & Song, H. (2024).** Hydra: Bidirectional State Space Models Through Generalized Matrix Mixers.
   - https://arxiv.org/pdf/2407.09941

3. **Aaronson, S., & Ambainis, A. (2015).** Forrelation: A Problem that Optimally Separates Quantum from Classical Computing.
   - https://arxiv.org/abs/1411.5729

---

## ✅ Success Criteria

### Minimum Viable Results

**Required:**
- ✅ All 6 models implemented and tested
- ✅ At least 3 out of 4 datasets completed
- ✅ At least one task where quantum ≥ classical
- ✅ Ablation: Superposition vs Hybrid

**Highly Desired:**
- ⭐ All 4 datasets completed
- ⭐ Quantum models within 5% of classical on all tasks
- ⭐ DNA or Forrelation task shows quantum advantage
- ⭐ Parameter efficiency clearly demonstrated

---

**Author:** Junghoon Park
**Last Updated:** November 2025
**License:** MIT
