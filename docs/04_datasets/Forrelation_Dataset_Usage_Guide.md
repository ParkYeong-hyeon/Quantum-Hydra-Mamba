# Usage Guide: BQP-Complete Forrelation Dataset (V2)

This guide provides step-by-step instructions for generating and using the BQP-complete Forrelation dataset for training sequence models.

## Overview

The Forrelation dataset tests whether models can detect hidden Fourier correlations between two Boolean functions - a task where quantum algorithms have a proven exponential advantage over classical algorithms.

### Data Pipeline Files

| File | Purpose |
|------|---------|
| `generate_forrelation_dataset_v2.py` | Generates BQP-complete Forrelation dataset using Gaussian rounding |
| `forrelation_dataloader.py` | Loads dataset into PyTorch DataLoaders |

### Version History

| Version | Method | Status |
|---------|--------|--------|
| V1 | Delta functions (BUGGY - data leakage) | Deprecated |
| **V2** | **Gaussian rounding (Theorem 6)** | **Current** |

---

## The Gaussian Rounding Method (V2)

### Why V2 is Correct

The V2 implementation follows **Theorem 6** from Aaronson & Ambainis (2015):

**For FORRELATED pairs:**
```python
f_real = np.random.normal(0, 1, size=N)      # Gaussian random
g_real = walsh_hadamard(f_real) / sqrt(N)    # Fourier transform
f = sign(f_real)                              # Round to {-1, +1}
g = sign(g_real)                              # Round to {-1, +1}
# Expected Phi(f,g) = 2/pi ≈ 0.637
```

**For UNFORRELATED pairs:**
```python
f_real = np.random.normal(0, 1, size=N)      # Independent Gaussian
g_real = np.random.normal(0, 1, size=N)      # Independent Gaussian
f = sign(f_real)                              # Round to {-1, +1}
g = sign(g_real)                              # Round to {-1, +1}
# Expected |Phi(f,g)| ≈ 0
```

### Key Property: No Data Leakage

In **both** classes:
- f has ~50% positive values (looks random)
- g has ~50% positive values (looks random)
- Individual statistics are **indistinguishable**

The only difference is the hidden Fourier correlation - exactly what makes this BQP-complete!

---

## Step 1: Generate the Dataset

### Basic Usage

```bash
cd data_loaders/
python generate_forrelation_dataset_v2.py
```

### Custom Parameters

```bash
python generate_forrelation_dataset_v2.py \
    --num_pairs 5000 \
    --n_bits 6 \
    --seq_len 100 \
    --seed 2024 \
    --filename forrelation_v2.pt
```

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--num_pairs` | 5000 | Number of (f, g) function pairs |
| `--n_bits` | 6 | Input bits (N = 2^n_bits = 64) |
| `--seq_len` | 100 | Sequence length per pair |
| `--seed` | None | Random seed for reproducibility |
| `--filename` | forrelation_v2.pt | Output file path |

### Run Tests

```bash
python generate_forrelation_dataset_v2.py --test
```

Expected output:
```
Testing Forrelation Computation
========================================
Forrelated pair: Φ = 0.625 (expected > 0.3)
Unforrelated pair: |Φ| = 0.125 (expected < 0.15)

Data leakage check:
  Forrelated f: 56.25% positive
  Unforrelated f: 50.00% positive
  Both should be ~50%

[PASS] Forrelation computation tests passed
```

---

## Step 2: Use in Training

### Loading the Dataset

```python
from data_loaders.forrelation_dataloader import get_forrelation_dataloader

train_loader, test_loader, params = get_forrelation_dataloader(
    dataset_path="forrelation_v2.pt",
    batch_size=32,
    shuffle=True
)

print(f"Channels: {params['num_channels']}")  # 14 for n_bits=6
print(f"Sequence length: {params['seq_len']}")
```

### Example Training Script

```python
import torch
import torch.nn as nn
from data_loaders.forrelation_dataloader import get_forrelation_dataloader

# Load data
train_loader, test_loader, params = get_forrelation_dataloader(
    dataset_path="data/synthetic_benchmarks/forrelation/forrelation_L100_seed2024.pt",
    batch_size=32
)

# Model expects: (batch, channels, timesteps)
# Dataset provides: (batch, channels=14, timesteps=seq_len)

# Your model here
model = YourModel(
    n_channels=params['num_channels'],  # 14
    n_timesteps=params['seq_len'],       # 100
    output_dim=2                          # Binary classification
)

# Training loop
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

for epoch in range(100):
    model.train()
    for sequences, labels in train_loader:
        optimizer.zero_grad()
        outputs = model(sequences)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
```

---

## Data Format

### Input Features (14 channels for n_bits=6)

Each timestep contains:
```
[x_0, x_1, x_2, x_3, x_4, x_5, f(x), y_0, y_1, y_2, y_3, y_4, y_5, g(y)]
```

| Channels | Description |
|----------|-------------|
| 0-5 | Binary representation of x (6 bits) |
| 6 | f(x) value in {-1, +1} |
| 7-12 | Binary representation of y (6 bits) |
| 13 | g(y) value in {-1, +1} |

### Tensor Shapes

| Stage | Shape |
|-------|-------|
| Stored on disk | (num_pairs, seq_len, num_channels) |
| After dataloader | (batch, num_channels, seq_len) |

---

## Integration with Synthetic Benchmarks

The `run_synthetic_benchmark.py` script automatically uses V2:

```bash
python scripts/run_synthetic_benchmark.py \
    --model_id 1a \
    --task forrelation \
    --seq_len 100 \
    --seed 2024
```

If the dataset doesn't exist, it will be generated automatically using the V2 generator.

---

## Interpreting Results

### Expected Performance

| Model Type | Expected Accuracy | Reasoning |
|------------|------------------|-----------|
| Classical (Group 3) | ~50% (baseline) | Cannot efficiently detect Fourier correlation |
| Quantum (Groups 1,2,4) | > 50%? | May leverage quantum-inspired Fourier mechanisms |

### What Success Looks Like

If quantum models show advantage:
- Accuracy significantly above 50% baseline
- Higher accuracy with shorter sequences (sample efficiency)
- Performance gap widens with larger n_bits

### What Failure Looks Like

If quantum models show no advantage:
- All models perform similarly (~50-60%)
- Suggests quantum components don't help with this task
- May indicate limitations of the quantum-classical hybrid approach

---

## Troubleshooting

### "100% accuracy" Warning

If any model achieves 100% accuracy, this indicates a bug:
- The V1 dataset had data leakage (delta functions)
- Make sure you're using V2 (check for `version: 2` in params)
- Delete old datasets and regenerate

### Verifying Dataset Version

```python
import torch
data = torch.load("forrelation_dataset.pt")
print(data['params'].get('version', 'V1 (old)'))
# Should print: 2
```

### Regenerating Dataset

```bash
# Delete old dataset
rm data/synthetic_benchmarks/forrelation/*.pt

# New datasets will be generated automatically on next run
```

---

## References

1. **Aaronson & Ambainis (2015)**: "Forrelation: A Problem that Optimally Separates Quantum from Classical Computing"
   - Paper: https://www.scottaaronson.com/papers/for.pdf
   - Theorem 6: Gaussian rounding method
   - Theorem 5: BQP-completeness

2. **Implementation**: `data_loaders/generate_forrelation_dataset_v2.py`
