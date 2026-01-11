# Quantum Hydra & Quantum Mamba: Time-Series Modeling with Quantum Circuits

> Quantum implementations of state-space models (Hydra & Mamba) for time-series classification tasks

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c.svg)](https://pytorch.org/)
[![PennyLane](https://img.shields.io/badge/PennyLane-0.33+-green.svg)](https://pennylane.ai/)

---

## 📋 Overview

This repository implements **quantum versions of state-space models** based on two recent architectures:

- **Quantum Hydra** (based on Hwang et al., 2024)
- **Quantum Mamba** (based on Gu & Dao, 2024)

Each quantum model comes in **two variants**:
1. **Superposition**: Uses quantum interference (α|ψ₁⟩ + β|ψ₂⟩ + γ|ψ₃⟩)
2. **Hybrid**: Classical combination of quantum circuits (w₁·y₁ + w₂·y₂ + w₃·y₃)

Plus **two classical baselines** for comparison.

**Total: 8 models** for comprehensive comparison (6 quantum + 2 classical).

---

## 🎯 Quick Start

> **⚡ Very busy? See [QUICK_START.md](QUICK_START.md) for 30-second guide with just the commands!**

### 1. Installation

```bash
# Clone the repository
git clone https://github.com/JHPark9090/Quantum-Hydra-Mamba.git
cd Quantum-Hydra-Mamba

# Install dependencies
pip install -r requirements.txt
```

### 2. Download Data

The repository includes data loaders for the following datasets:

- **PhysioNet EEG**: Motor imagery classification (automatic download via MNE)
- **MNIST**: Image classification (automatic download via PyTorch)
- **DNA Sequences**: Promoter recognition (automatic download via UCI ML)
- **Forrelation**: Quantum advantage test (generate with provided script)

### 3. Run Your First Experiment

```bash
# Test a single model on EEG data
python scripts/run_single_model_eeg.py \
    --model-name quantum_hydra \
    --n-qubits 6 \
    --n-epochs 50 \
    --batch-size 32 \
    --seed 2024 \
    --device cuda

# Run all 8 models on MNIST (sequential execution)
bash scripts/run_all_mnist_experiments.sh

# Generate Forrelation datasets for quantum advantage testing
bash scripts/generate_forrelation_datasets.sh

# Run Forrelation experiments (144 experiments total)
bash scripts/run_all_forrelation_experiments.sh
```

---

## 📂 Repository Structure

```
quantum-hydra-mamba/
├── models/                          # Model implementations
│   ├── QuantumHydra.py             # Quantum Hydra (superposition)
│   ├── QuantumHydraHybrid.py       # Quantum Hydra (hybrid)
│   ├── QuantumMamba.py             # Quantum Mamba (superposition)
│   ├── QuantumMambaHybrid.py       # Quantum Mamba (hybrid)
│   ├── QuantumMambaLite.py         # Quantum Mamba Lite (superposition, 62% fewer params)
│   ├── QuantumMambaHybridLite.py   # Quantum Mamba Hybrid Lite (62% fewer params)
│   ├── TrueClassicalHydra.py       # Classical Hydra baseline
│   └── TrueClassicalMamba.py       # Classical Mamba baseline
│
├── datasets/                        # Data loading utilities
│   ├── Load_Image_Datasets.py      # MNIST, Fashion-MNIST loaders
│   ├── Load_PhysioNet_EEG_NoPrompt.py  # EEG data loader
│   ├── Load_DNA_Sequences.py       # DNA promoter data loader
│   ├── generate_forrelation_dataset.py  # Forrelation dataset generator
│   └── forrelation_dataloader.py   # Forrelation data loader
│
├── experiments/                     # Training scripts
│   ├── run_single_model_eeg.py     # EEG experiments
│   ├── run_single_model_mnist.py   # MNIST experiments
│   ├── run_single_model_dna.py     # DNA experiments
│   ├── run_single_model_forrelation.py  # Forrelation experiments
│   ├── aggregate_eeg_results.py    # Aggregate EEG results
│   ├── aggregate_mnist_results.py  # Aggregate MNIST results
│   ├── aggregate_dna_results.py    # Aggregate DNA results
│   └── aggregate_forrelation_results.py  # Analyze quantum advantage
│
├── scripts/                         # Convenience scripts
│   ├── run_all_eeg_experiments.sh
│   ├── run_all_mnist_experiments.sh
│   ├── run_all_dna_experiments.sh
│   ├── generate_forrelation_datasets.sh
│   └── run_all_forrelation_experiments.sh
│
├── docs/                            # Documentation
│   ├── README.md                    # Documentation index
│   ├── EXPERIMENT_GUIDE.md          # Complete experiment guide
│   ├── QUANTUM_HYDRA_GUIDE.md       # Quantum Hydra detailed guide
│   ├── QUANTUM_MAMBA_GUIDE.md       # Quantum Mamba detailed guide
│   ├── TIMING_AND_METRICS.md        # Timing and metrics tracking
│   ├── FORRELATION_README.md        # Forrelation experiment overview
│   ├── Forrelation_Experiment_Rationale.md
│   ├── Forrelation_Dataset_Usage_Guide.md
│   └── Quantum_Advantage_Test_Plan.md
│
├── QUICK_START.md                   # 30-second command guide
├── UPDATE_NOTES.md                  # November 2025 update changelog
├── REPOSITORY_STATUS.md             # Repository status summary
├── requirements.txt                 # Python dependencies
├── .gitignore                       # Git ignore file
└── README.md                        # This file
```

---

## 🧬 The Eight Models

### Quantum Models (6 total)

| Model | File | Description | Key Feature |
|-------|------|-------------|-------------|
| **Quantum Hydra (Superposition)** | `QuantumHydra.py` | Quantum superposition of 3 branches | Complex coefficients α, β, γ ∈ ℂ |
| **Quantum Hydra (Hybrid)** | `QuantumHydraHybrid.py` | Classical combination of 3 branches | Real weights w₁, w₂, w₃ ∈ ℝ |
| **Quantum Mamba (Superposition)** | `QuantumMamba.py` | Quantum superposition of SSM/gate/skip | Complex coefficients α, β, γ ∈ ℂ |
| **Quantum Mamba (Hybrid)** | `QuantumMambaHybrid.py` | Classical combination of SSM/gate/skip | Real weights w₁, w₂, w₃ ∈ ℝ |
| **Quantum Mamba Lite (Superposition)** | `QuantumMambaLite.py` | Lightweight variant matching Hydra architecture | Timestep loop, 62% fewer params |
| **Quantum Mamba Hybrid Lite** | `QuantumMambaHybridLite.py` | Lightweight hybrid variant | Timestep loop, 62% fewer params |

### Classical Baselines (2 total)

| Model | File | Description |
|-------|------|-------------|
| **True Classical Hydra** | `TrueClassicalHydra.py` | Faithful implementation of Hwang et al. (2024) |
| **True Classical Mamba** | `TrueClassicalMamba.py` | Faithful implementation of Gu & Dao (2024) |

**Note on Automated Scripts:**
- The repository contains **8 model implementations** (listed above)
- Automated batch scripts (`scripts/run_all_*.sh`) now run **all 8 models** including Lite variants
- For EEG and Forrelation: All 8 models are tested (time-series data)
- For MNIST and DNA: Only 6 models are tested (Lite variants are time-series specific)

---

## 🧪 Experiments

The repository includes **four types of experiments**:

### 1. EEG Motor Imagery Classification

**Dataset**: PhysioNet Motor Imagery
**Task**: Binary classification (left/right hand movement)
**Why important**: Real-world biomedical application, tests temporal pattern detection

**Run experiments**:
```bash
# Single model
python scripts/run_single_model_eeg.py \
    --model-name quantum_hydra \
    --seed 2024

# All 6 core models (sequential)
bash scripts/run_all_eeg_experiments.sh
```

**Expected results**: Quantum models should achieve 75-85% accuracy

---

### 2. MNIST Image Classification

**Dataset**: MNIST (28×28 grayscale images)
**Task**: 10-class digit recognition
**Why important**: Standard ML benchmark, tests spatial pattern recognition

**Run experiments**:
```bash
# Single model
python scripts/run_single_model_mnist.py \
    --model-name quantum_mamba \
    --seed 2024

# All 6 core models
bash scripts/run_all_mnist_experiments.sh
```

**Expected results**: Quantum models 85-90% accuracy (classical 95-98%)

---

### 3. DNA Sequence Classification

**Dataset**: DNA Promoter Recognition (UCI ML Repository)
**Task**: Binary classification (promoter vs non-promoter)
**Why important**: Natural quantum encoding (A/C/G/T → 2 qubits), tests sequence modeling

**Run experiments**:
```bash
# Single model
python scripts/run_single_model_dna.py \
    --model-name classical_mamba \
    --encoding onehot \
    --seed 2024

# All 6 core models
bash scripts/run_all_dna_experiments.sh
```

**Expected results**: Quantum models may **outperform** classical (natural encoding advantage)

---

### 4. Forrelation (Quantum Advantage Test)

**Dataset**: Sequential Forrelation (generated)
**Task**: Binary classification (high vs low forrelation)
**Why important**: Proven quantum advantage (Aaronson & Ambainis, 2015)

**Run experiments**:
```bash
# Step 1: Generate datasets
bash scripts/generate_forrelation_datasets.sh

# Step 2: Run experiments (144 total: 6 core models × 8 datasets × 3 seeds)
bash scripts/run_all_forrelation_experiments.sh

# Step 3: Analyze quantum advantage
python scripts/aggregate_forrelation_results.py
```

**Expected results**:
- **Silver Standard**: Quantum models need 30-50% **fewer samples** to reach 90% accuracy
- **Gold Standard**: Advantage **increases** with problem complexity

See `docs/FORRELATION_README.md` for complete details.

---

## 🔧 Usage Examples

### Basic Usage

```python
import torch
from QuantumHydra import QuantumHydraTS
from QuantumMamba import QuantumMambaTS
from QuantumMambaLite import QuantumMambaTS_Lite
from TrueClassicalHydra import TrueClassicalHydra

# EEG data: (batch_size, channels, timesteps)
eeg_data = torch.randn(16, 64, 160)

# Quantum Hydra (Superposition)
model1 = QuantumHydraTS(
    n_qubits=6,
    n_timesteps=160,
    qlcu_layers=2,
    feature_dim=64,
    output_dim=2,
    device="cuda"  # or "cpu"
)
output1 = model1(eeg_data)  # (16, 2)

# Quantum Mamba (Hybrid)
model2 = QuantumMambaTS(
    n_qubits=6,
    n_timesteps=160,
    qlcu_layers=2,
    feature_dim=64,
    output_dim=2,
    device="cuda"  # or "cpu"
)
output2 = model2(eeg_data)  # (16, 2)

# Quantum Mamba Lite (Lightweight variant)
model3 = QuantumMambaTS_Lite(
    n_qubits=6,
    n_timesteps=160,
    qlcu_layers=2,
    feature_dim=64,
    output_dim=2,
    device="cuda"  # 62% fewer parameters than QuantumMambaTS
)
output3 = model3(eeg_data)  # (16, 2)

# Classical Hydra Baseline
model4 = TrueClassicalHydra(
    n_channels=64,
    n_timesteps=160,
    d_model=128,
    d_state=16,
    output_dim=2,
    device="cuda"  # Device parameter added
)
output4 = model4(eeg_data)  # (16, 2)
```

**Note on Input Shapes:**
- **3D inputs** (EEG): `(batch, channels, timesteps)` - e.g., (16, 64, 160)
- **2D inputs** (DNA, MNIST): `(batch, features)` - e.g., (16, 228)
- All models automatically handle both formats

**Note on Device Parameter:**
- All models now support `device` parameter: `"cpu"` or `"cuda"`
- Models automatically move all parameters to the specified device
- Default is `"cpu"` for backward compatibility

### Training Loop

```python
import torch.nn as nn
import torch.optim as optim

# Setup
model = QuantumHydraTS(n_qubits=6, output_dim=2, device="cuda")
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=1e-3)

# Training
for epoch in range(num_epochs):
    for x_batch, y_batch in train_loader:
        x_batch = x_batch.to("cuda")
        y_batch = y_batch.to("cuda")

        optimizer.zero_grad()
        outputs = model(x_batch)
        loss = criterion(outputs, y_batch)
        loss.backward()
        optimizer.step()
```

---

## 📊 Analyzing Results

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

---

## ⚙️ Configuration

### Hyperparameters

Default hyperparameters for each dataset:

**EEG (PhysioNet)**:
```python
n_qubits = 6
qlcu_layers = 2
d_model = 128
d_state = 16
n_epochs = 50
batch_size = 32
learning_rate = 1e-3
sample_size = 10  # Number of subjects
```

**MNIST**:
```python
n_qubits = 6
qlcu_layers = 2
d_model = 128
n_epochs = 50
batch_size = 32
learning_rate = 1e-3
n_train = 500
n_valtest = 250
```

**DNA**:
```python
n_qubits = 6
qlcu_layers = 2
d_model = 128
n_epochs = 50
batch_size = 32
learning_rate = 1e-3
encoding = "onehot"  # or "integer"
```

**Forrelation**:
```python
n_qubits = 6
qlcu_layers = 2
d_model = 128
n_epochs = 100
batch_size = 32
learning_rate = 1e-3
early_stopping_patience = 15
```

---

## 🔬 Research Questions

This repository enables you to answer:

1. **Do quantum models outperform classical baselines?**
   - Compare accuracy across all models (6 core + 2 Lite variants)
   - Test on 4 different datasets

2. **Is quantum advantage architecture-specific?**
   - Quantum Hydra vs Quantum Mamba
   - Which architecture benefits more from quantization?

3. **Does quantum superposition help?**
   - Superposition (Option A) vs Hybrid (Option B)
   - When does quantum interference provide benefits?

4. **Where do quantum models excel?**
   - EEG, MNIST, DNA, or Forrelation?
   - Task characteristics that favor quantum models

5. **Is there genuine quantum advantage?**
   - Forrelation experiments test this rigorously
   - Sample efficiency and scaling analysis

---

## 📈 Expected Performance

### Performance Tiers

| Dataset | Quantum Models | Classical Models | Gap |
|---------|----------------|------------------|-----|
| **EEG** | 75-85% | 80-85% | Competitive |
| **MNIST** | 85-90% | 95-98% | Classical better |
| **DNA** | 70-80% | 65-75% | Quantum better? |
| **Forrelation** | High with fewer samples | High with more samples | **Quantum advantage** |

### Parameter Efficiency

- **Quantum Lite models**: ~1,500-2,000 parameters (most efficient)
- **Quantum standard models**: ~2,500-5,000 parameters
- **Classical models**: ~5,000-8,000 parameters
- **Ratio**: Quantum Lite uses **70-75% fewer parameters** than classical
- **Quantum Mamba Lite vs Standard**: 62% reduction (removing Conv1d)

---

## 🛠️ Hardware Requirements

### Minimum Requirements
- **GPU**: NVIDIA GPU with 4GB+ VRAM (e.g., GTX 1650)
- **RAM**: 8GB
- **Storage**: 5GB

### Recommended
- **GPU**: NVIDIA GPU with 8GB+ VRAM (e.g., RTX 3060)
- **RAM**: 16GB
- **Storage**: 10GB

### Running on CPU
All scripts support CPU execution (remove `--device cuda` flag or use `--device cpu`), but will be **much slower** (~10-20× slower).

### GPU Acceleration for Quantum Circuits

**Optional quantum circuit GPU acceleration** is available via PennyLane-Lightning-GPU:

```bash
pip install pennylane-lightning-gpu
```

When installed, quantum models automatically use `lightning.gpu` device when:
- `device="cuda"` is specified
- CUDA is available
- `pennylane-lightning-gpu` is installed

**Benefits:**
- Additional speedup for quantum circuit evaluation (beyond PyTorch GPU usage)
- Especially beneficial for larger qubit counts (8+ qubits)
- Graceful fallback to CPU quantum simulation if not available

**Note:** This is optional. Models work fine with standard PyTorch GPU acceleration alone.

---

## 📝 Citation

If you use this code in your research, please cite:

```bibtex
@software{quantum_hydra_mamba_2025,
  author = {Park, Junghoon},
  title = {Quantum Hydra \& Quantum Mamba: Quantum State-Space Models for Time-Series},
  year = {2025},
  url = {https://github.com/JHPark9090/Quantum-Hydra-Mamba}
}
```

### Related Papers

1. **Hwang et al. (2024)** - Hydra: Bidirectional State Space Models
   - https://arxiv.org/pdf/2407.09941

2. **Gu & Dao (2024)** - Mamba: Linear-Time Sequence Modeling
   - https://arxiv.org/html/2312.00752v2

3. **Aaronson & Ambainis (2015)** - Forrelation: Quantum Advantage
   - https://arxiv.org/abs/1411.5729

---

## 🤝 Contributing

Contributions are welcome! Please:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

---

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

---

## 🙏 Acknowledgments

- **PennyLane** for quantum circuit simulation
- **PyTorch** for deep learning framework
- **PhysioNet** for EEG datasets
- **UCI ML Repository** for DNA datasets

---

## ⚠️ Troubleshooting

### Common Issues

**NaN Loss During Training:**
- Gradient clipping is already implemented in `run_single_model_eeg.py`
- If you encounter NaN in custom training loops, add:
  ```python
  torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
  ```

**CUDA Out of Memory:**
- Reduce batch size: `--batch-size 16` (default is 32)
- Use smaller model: `--n-qubits 4` instead of 6

**Slow Training on CPU:**
- Expected: CPU is 10-20× slower than GPU
- Solution: Use `--device cuda` if GPU available

---

## 📧 Contact

For questions or issues:
- **GitHub Issues**: [Open an issue](https://github.com/JHPark9090/Quantum-Hydra-Mamba/issues)

---

**Last Updated**: November 2025
**Status**: Research code (experimental)
**Python Version**: 3.8+
**Tested On**: Ubuntu 20.04, CUDA 11.8
