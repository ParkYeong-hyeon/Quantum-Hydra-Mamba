# EEG Datasets Setup Guide

Complete guide for setting up PhysioNet, SEED, and FACED EEG datasets for Quantum Hydra/Mamba/Transformer models.

---

## Table of Contents

1. [Quick Start](#1-quick-start)
2. [Package Installation](#2-package-installation)
3. [Dataset Downloads](#3-dataset-downloads)
4. [Directory Structure](#4-directory-structure)
5. [Verification](#5-verification)
6. [Troubleshooting](#6-troubleshooting)

---

## 1. Quick Start

```bash
# 1. Activate your conda environment
conda activate ./conda-envs/qml_eeg

# 2. Install TorchEEG
pip install torcheeg

# 3. Install MNE for PhysioNet (if not already installed)
pip install mne

# 4. Download datasets (see Section 3)

# 5. Test the loaders
cd /pscratch/sd/j/junghoon/quantum_hydra_mamba
python -c "from data_loaders.eeg_datasets import print_comparison_table; print_comparison_table()"
```

---

## 2. Package Installation

### 2.1 Using pip (Recommended)

```bash
# Activate your environment
conda activate ./conda-envs/qml_eeg

# Install TorchEEG and dependencies
pip install torcheeg>=1.1.0

# Install MNE for PhysioNet
pip install mne>=1.0.0

# Install LMDB for efficient caching
pip install lmdb>=1.3.0
```

### 2.2 Using conda

```bash
conda activate ./conda-envs/qml_eeg
conda install -c conda-forge torcheeg
conda install -c conda-forge mne
```

### 2.3 Verify Installation

```python
# Test imports
import torcheeg
print(f"TorchEEG version: {torcheeg.__version__}")

import mne
print(f"MNE version: {mne.__version__}")

from torcheeg.datasets import SEEDDataset, FACEDDataset
print("TorchEEG datasets imported successfully!")
```

---

## 3. Dataset Downloads

### 3.1 PhysioNet Motor Imagery (Auto-download)

**No manual download required!** MNE automatically downloads the data.

```python
# The data will be downloaded to ./PhysioNet_EEG/ on first use
from data_loaders.eeg_datasets import load_eeg_dataset

train_loader, val_loader, test_loader, input_dim, num_classes = load_eeg_dataset(
    dataset_name='physionet',
    seed=2024,
    batch_size=32
)
```

**Dataset Info:**
- Size: ~2 GB
- Location: Auto-downloaded to `./PhysioNet_EEG/`
- Format: EDF files

---

### 3.2 SEED Dataset (Manual Download Required)

**Step 1: Request Access**
1. Go to: https://bcmi.sjtu.edu.cn/home/seed/index.html
2. Fill out the data request form
3. Wait for approval email (usually 1-3 business days)

**Step 2: Download**
1. After approval, download `Preprocessed_EEG.zip`
2. File size: ~1.5 GB

**Step 3: Extract**
```bash
# Create directory
mkdir -p /pscratch/sd/j/junghoon/quantum_hydra_mamba/Preprocessed_EEG

# Extract (adjust path to your download location)
unzip Preprocessed_EEG.zip -d /pscratch/sd/j/junghoon/quantum_hydra_mamba/

# Verify structure
ls /pscratch/sd/j/junghoon/quantum_hydra_mamba/Preprocessed_EEG/
# Should show: 1_20131027.mat, 1_20131030.mat, ... etc.
```

**Expected Directory Structure:**
```
Preprocessed_EEG/
├── 1_20131027.mat      # Subject 1, Session 1
├── 1_20131030.mat      # Subject 1, Session 2
├── 1_20131107.mat      # Subject 1, Session 3
├── 2_20140404.mat      # Subject 2, Session 1
├── ...
├── 15_20140603.mat     # Subject 15, Session 3
└── label.mat           # Labels
```

**Dataset Details:**
- 15 subjects × 3 sessions = 45 recordings
- 62 EEG channels at 200 Hz
- 15 trials per session (film clips)
- Labels: positive (1), neutral (0), negative (-1)

---

### 3.3 FACED Dataset (Manual Download Required)

**Step 1: Create Synapse Account**
1. Go to: https://www.synapse.org/
2. Click "Register" and create an account
3. Verify your email

**Step 2: Request Access**
1. Go to: https://www.synapse.org/#!Synapse:syn50614194/files/
2. Click "Request Access"
3. Agree to data use terms
4. Wait for approval (usually automatic)

**Step 3: Download**
1. Navigate to: https://www.synapse.org/#!Synapse:syn50615881
2. Download `Processed_data.zip`
3. File size: ~8 GB

**Step 4: Extract**
```bash
# Create directory
mkdir -p /pscratch/sd/j/junghoon/quantum_hydra_mamba/Processed_data

# Extract (adjust path to your download location)
unzip Processed_data.zip -d /pscratch/sd/j/junghoon/quantum_hydra_mamba/

# Verify structure
ls /pscratch/sd/j/junghoon/quantum_hydra_mamba/Processed_data/
```

**Expected Directory Structure:**
```
Processed_data/
├── sub001/
│   ├── video01.mat
│   ├── video02.mat
│   └── ...
├── sub002/
│   └── ...
├── ...
└── sub123/
    └── ...
```

**Dataset Details:**
- 123 subjects
- 30 EEG channels at 250 Hz
- 28 video clips per subject
- Labels: 9 discrete emotions + valence (positive/neutral/negative)

---

## 4. Directory Structure

After setup, your directory should look like:

```
/pscratch/sd/j/junghoon/quantum_hydra_mamba/
├── data_loaders/
│   ├── eeg_datasets.py              # Unified loader
│   ├── Load_PhysioNet_EEG_NoPrompt.py
│   ├── Load_SEED_EEG.py
│   └── Load_FACED_EEG.py
├── PhysioNet_EEG/                    # Auto-downloaded
│   └── files/
│       └── eegmmidb/
│           └── 1.0.0/
│               ├── S001/
│               └── ...
├── Preprocessed_EEG/                 # SEED - Manual download
│   ├── 1_20131027.mat
│   └── ...
├── Processed_data/                   # FACED - Manual download
│   ├── sub001/
│   └── ...
├── seed_io/                          # TorchEEG cache (auto-created)
├── faced_io/                         # TorchEEG cache (auto-created)
└── requirements_eeg_datasets.txt
```

---

## 5. Verification

### 5.1 Test All Loaders

```python
import torch
from data_loaders.eeg_datasets import load_eeg_dataset, print_comparison_table

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Print dataset comparison
print_comparison_table()

# Test each dataset
datasets_to_test = ['physionet', 'seed', 'faced_valence', 'faced_emotion']

for dataset_name in datasets_to_test:
    print(f"\n{'='*60}")
    print(f"Testing: {dataset_name}")
    print('='*60)

    try:
        train_loader, val_loader, test_loader, input_dim, num_classes = load_eeg_dataset(
            dataset_name=dataset_name,
            seed=2024,
            device=device,
            batch_size=32
        )

        # Get one batch
        for X, y in train_loader:
            print(f"✓ Success! X shape: {X.shape}, y shape: {y.shape}")
            print(f"  Labels range: {y.min().item()} to {y.max().item()}")
            break

    except FileNotFoundError as e:
        print(f"✗ Dataset not found: {e}")
    except Exception as e:
        print(f"✗ Error: {e}")
```

### 5.2 Quick Test Script

Save this as `test_eeg_datasets.py`:

```python
#!/usr/bin/env python
"""Quick test for EEG dataset loaders."""

import sys
sys.path.insert(0, '/pscratch/sd/j/junghoon/quantum_hydra_mamba')

import torch
from data_loaders.eeg_datasets import load_eeg_dataset, DATASET_INFO

def test_dataset(name, device):
    print(f"\nTesting {name}...")
    try:
        loaders = load_eeg_dataset(name, seed=2024, device=device, batch_size=16)
        train_loader = loaders[0]
        for X, y in train_loader:
            print(f"  ✓ X: {X.shape}, y: {y.shape}")
            return True
    except Exception as e:
        print(f"  ✗ Error: {e}")
        return False

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    results = {}
    for name in DATASET_INFO.keys():
        results[name] = test_dataset(name, device)

    print("\n" + "="*40)
    print("Summary:")
    for name, success in results.items():
        status = "✓ OK" if success else "✗ FAILED"
        print(f"  {name}: {status}")
```

Run with:
```bash
python test_eeg_datasets.py
```

---

## 6. Troubleshooting

### 6.1 TorchEEG Import Error

```
ImportError: No module named 'torcheeg'
```

**Solution:**
```bash
pip install torcheeg
```

### 6.2 LMDB Error

```
ModuleNotFoundError: No module named 'lmdb'
```

**Solution:**
```bash
pip install lmdb
```

### 6.3 MNE PhysioNet Download Hangs

**Problem:** Interactive prompt waiting for input

**Solution:** Use `update_path=True` in the loader (already fixed in our loaders)

### 6.4 SEED/FACED File Not Found

```
FileNotFoundError: [Errno 2] No such file or directory: './Preprocessed_EEG'
```

**Solution:** Specify the correct root_path:
```python
train_loader, _, _, _, _ = load_eeg_dataset(
    dataset_name='seed',
    root_path='/path/to/your/Preprocessed_EEG',
    seed=2024,
    batch_size=32
)
```

### 6.5 CUDA Out of Memory

**Solution:** Reduce batch_size or use CPU for data loading:
```python
train_loader, _, _, _, _ = load_eeg_dataset(
    dataset_name='seed',
    device=torch.device('cpu'),  # Load to CPU first
    batch_size=16  # Smaller batch
)
```

### 6.6 TorchEEG Cache Issues

**Problem:** Corrupted cache files

**Solution:** Delete the IO cache and re-run:
```bash
rm -rf ./seed_io
rm -rf ./faced_io
rm -rf ./faced_io_emotion
```

---

## Dataset Citations

If you use these datasets, please cite:

### PhysioNet Motor Imagery
```bibtex
@article{schalk2004bci2000,
  title={BCI2000: a general-purpose brain-computer interface (BCI) system},
  author={Schalk, Gerwin and McFarland, Dennis J and Hinterberger, Thilo and Birbaumer, Niels and Wolpaw, Jonathan R},
  journal={IEEE Transactions on biomedical engineering},
  volume={51},
  number={6},
  pages={1034--1043},
  year={2004}
}
```

### SEED
```bibtex
@article{zheng2015investigating,
  title={Investigating critical frequency bands and channels for EEG-based emotion recognition with deep neural networks},
  author={Zheng, Wei-Long and Lu, Bao-Liang},
  journal={IEEE Transactions on Autonomous Mental Development},
  volume={7},
  number={3},
  pages={162--175},
  year={2015}
}
```

### FACED
```bibtex
@article{faced2023,
  title={FACED: A Fine-grained Affective Computing EEG Dataset},
  author={Tsinghua Laboratory of Brain and Intelligence},
  year={2023},
  note={Available at: https://www.synapse.org/\#!Synapse:syn50614194}
}
```

### TorchEEG
```bibtex
@article{zhang2024torcheeg,
  title={TorchEEG: A PyTorch-based Toolbox for EEG Signal Analysis},
  author={Zhang, Zhi and others},
  journal={Expert Systems with Applications},
  year={2024}
}
```

---

## Summary

| Dataset | Download | Size | Setup Time |
|---------|----------|------|------------|
| PhysioNet | Auto | ~2 GB | ~5 min |
| SEED | Manual (requires approval) | ~1.5 GB | 1-3 days |
| FACED | Manual (Synapse account) | ~8 GB | ~1 hour |

**Quick Install:**
```bash
pip install torcheeg mne lmdb
```

**Quick Test:**
```python
from data_loaders.eeg_datasets import load_eeg_dataset
loaders = load_eeg_dataset('physionet', seed=2024, batch_size=32)
```

---

*Last updated: December 2024*
