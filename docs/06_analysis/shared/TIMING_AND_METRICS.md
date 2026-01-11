# Timing and Performance Metrics

This document explains how experiment timing information is recorded and how to access it.

---

## ✅ Summary

**YES**, all experiment scripts in `/pscratch/sd/j/junghoon/quantum_hydra_mamba/scripts/` **DO record timing information** for each experiment.

---

## 📊 What Timing Information is Recorded?

### 1. **Total Training Time**
Every experiment records the total time from start to finish of training.

**Recorded as:**
- `training_time` (in seconds) - saved in results JSON file
- Also printed as: `Total Time: XX.XXs (X.XX min)`

**Example output:**
```
Total Time: 3245.67s (54.09 min)
```

---

### 2. **Per-Epoch Time**
Each epoch's duration is tracked and displayed during training.

**Displayed as:**
```
Epoch 5/50 (12.34s) ✓ | Train Loss: 0.4532, Train Acc: 0.8234 | Val Loss: 0.3421, Val Acc: 0.8567
```

The `(12.34s)` shows how long that epoch took.

---

### 3. **Timestamp**
Each experiment records when it was run.

**Recorded as:**
- `timestamp` - formatted as `YYYYMMDD_HHMMSS`

**Example:**
```json
"timestamp": "20241113_143052"
```

---

## 📁 Where is Timing Information Saved?

### JSON Results Files

Each experiment saves a JSON file with complete timing information:

**File naming convention:**
```
{model_name}_seed{seed}_results.json
```

**Examples:**
- `quantum_hydra_seed2024_results.json`
- `quantum_mamba_seed2025_results.json`
- `classical_hydra_seed2026_results.json`

**File locations:**
- **EEG**: `eeg_results/quantum_hydra_seed2024_results.json`
- **MNIST**: `mnist_results/quantum_mamba_seed2025_results.json`
- **DNA**: `dna_results/classical_mamba_seed2026_results.json`
- **Forrelation**: `forrelation_results/quantum_hydra_L20_seed2024_results.json`

---

## 🔍 JSON File Structure

Each results JSON file contains:

```json
{
  "model_name": "quantum_hydra",
  "seed": 2024,
  "n_params": 2451,
  "hyperparameters": {
    "n_qubits": 6,
    "qlcu_layers": 2,
    "d_model": 128,
    "n_epochs": 50,
    "batch_size": 32,
    "learning_rate": 0.001
  },
  "history": {
    "train_loss": [0.693, 0.542, 0.421, ...],
    "train_acc": [0.523, 0.672, 0.745, ...],
    "val_loss": [0.651, 0.498, 0.387, ...],
    "val_acc": [0.567, 0.698, 0.776, ...],
    "epochs": [1, 2, 3, ...]
  },
  "best_val_acc": 0.8234,
  "test_acc": 0.8156,
  "test_auc": 0.8923,
  "test_f1": 0.8067,
  "test_cm": [[45, 5], [6, 44]],
  "training_time": 3245.67,      ← TOTAL TIME IN SECONDS
  "timestamp": "20241113_143052"  ← WHEN IT RAN
}
```

---

## 💻 How to Access Timing Information

### Method 1: Read JSON Directly

```python
import json

# Load results
with open('eeg_results/quantum_hydra_seed2024_results.json', 'r') as f:
    results = json.load(f)

# Get timing information
training_time = results['training_time']  # in seconds
training_time_min = training_time / 60     # in minutes
timestamp = results['timestamp']

print(f"Model: {results['model_name']}")
print(f"Seed: {results['seed']}")
print(f"Training Time: {training_time:.2f}s ({training_time_min:.2f} min)")
print(f"Test Accuracy: {results['test_acc']:.4f}")
print(f"Timestamp: {timestamp}")
```

**Output:**
```
Model: quantum_hydra
Seed: 2024
Training Time: 3245.67s (54.09 min)
Test Accuracy: 0.8156
Timestamp: 20241113_143052
```

---

### Method 2: Use Aggregation Scripts

The aggregation scripts automatically extract timing information:

```bash
# Aggregate EEG results (includes timing)
python scripts/aggregate_eeg_results.py
```

**This creates:** `eeg_results/eeg_all_results.csv`

**CSV contains:**
```csv
model_name,seed,test_acc,test_auc,test_f1,n_params,training_time,timestamp
quantum_hydra,2024,0.8156,0.8923,0.8067,2451,3245.67,20241113_143052
quantum_hydra,2025,0.8234,0.8998,0.8123,2451,3198.45,20241113_150234
quantum_mamba,2024,0.7989,0.8756,0.7845,2567,3567.89,20241113_153421
...
```

---

### Method 3: Load and Analyze Multiple Experiments

```python
import json
import pandas as pd
from pathlib import Path

# Load all EEG results
results_dir = Path('eeg_results')
all_results = []

for json_file in results_dir.glob('*_results.json'):
    with open(json_file, 'r') as f:
        data = json.load(f)
        all_results.append({
            'model': data['model_name'],
            'seed': data['seed'],
            'test_acc': data['test_acc'],
            'training_time_min': data['training_time'] / 60,
            'n_params': data['n_params']
        })

# Create DataFrame
df = pd.DataFrame(all_results)

# Analyze timing by model
timing_by_model = df.groupby('model')['training_time_min'].agg(['mean', 'std'])
print("\nTraining Time by Model (minutes):")
print(timing_by_model)

# Compare quantum vs classical
print("\nParameter Efficiency:")
print(df[['model', 'n_params', 'training_time_min', 'test_acc']])
```

**Output:**
```
Training Time by Model (minutes):
                        mean        std
model
quantum_hydra          54.32       2.14
quantum_hydra_hybrid   58.67       3.21
quantum_mamba          59.78       2.89
quantum_mamba_hybrid   62.34       3.56
classical_hydra        45.23       1.87
classical_mamba        48.91       2.34

Parameter Efficiency:
                model  n_params  training_time_min  test_acc
0       quantum_hydra      2451              54.32    0.8156
1  quantum_hydra_hybrid      2789              58.67    0.8234
2       quantum_mamba      2567              59.78    0.7989
...
```

---

## 📊 Example: Complete Timing Analysis

Here's a complete script to analyze timing across all experiments:

```python
import json
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

def load_all_results(results_dir):
    """Load all JSON results from a directory."""
    all_results = []

    for json_file in Path(results_dir).glob('*_results.json'):
        with open(json_file, 'r') as f:
            data = json.load(f)
            all_results.append(data)

    return all_results

def analyze_timing(results_dir, dataset_name):
    """Analyze timing information for a dataset."""
    results = load_all_results(results_dir)

    # Convert to DataFrame
    df_data = []
    for r in results:
        df_data.append({
            'model': r['model_name'],
            'seed': r['seed'],
            'test_acc': r['test_acc'],
            'training_time_min': r['training_time'] / 60,
            'n_params': r['n_params'],
            'timestamp': r['timestamp']
        })

    df = pd.DataFrame(df_data)

    # Summary statistics
    print(f"\n{'='*60}")
    print(f"Timing Analysis for {dataset_name}")
    print(f"{'='*60}\n")

    summary = df.groupby('model').agg({
        'training_time_min': ['mean', 'std', 'min', 'max'],
        'test_acc': ['mean', 'std'],
        'n_params': 'first'
    }).round(2)

    print(summary)

    # Time vs Accuracy scatter plot
    plt.figure(figsize=(10, 6))
    for model in df['model'].unique():
        model_data = df[df['model'] == model]
        plt.scatter(model_data['training_time_min'],
                   model_data['test_acc'],
                   label=model, s=100, alpha=0.7)

    plt.xlabel('Training Time (minutes)')
    plt.ylabel('Test Accuracy')
    plt.title(f'Training Time vs Accuracy - {dataset_name}')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig(f'{results_dir}/time_vs_accuracy.png', dpi=300, bbox_inches='tight')
    print(f"\nPlot saved to: {results_dir}/time_vs_accuracy.png")

    return df

# Analyze all datasets
eeg_df = analyze_timing('eeg_results', 'EEG')
mnist_df = analyze_timing('mnist_results', 'MNIST')
dna_df = analyze_timing('dna_results', 'DNA')
```

---

## 🎯 Key Findings You Can Extract

### 1. **Which model trains fastest?**
```python
fastest_model = df.groupby('model')['training_time_min'].mean().idxmin()
print(f"Fastest model: {fastest_model}")
```

### 2. **Time per accuracy point**
```python
df['time_per_acc'] = df['training_time_min'] / df['test_acc']
efficiency = df.groupby('model')['time_per_acc'].mean().sort_values()
print("Time efficiency (lower is better):")
print(efficiency)
```

### 3. **Quantum vs Classical timing**
```python
quantum_models = df[df['model'].str.contains('quantum')]
classical_models = df[df['model'].str.contains('classical')]

print(f"Quantum avg time: {quantum_models['training_time_min'].mean():.2f} min")
print(f"Classical avg time: {classical_models['training_time_min'].mean():.2f} min")
```

### 4. **Parameter efficiency**
```python
df['params_per_acc'] = df['n_params'] / df['test_acc']
efficiency = df.groupby('model')[['n_params', 'test_acc', 'training_time_min']].mean()
print("\nModel Efficiency:")
print(efficiency)
```

---

## 📈 Expected Timing Results

Based on typical hardware (RTX 3060, 8GB VRAM):

| Model | Dataset | Training Time (avg) | Notes |
|-------|---------|---------------------|-------|
| Quantum Hydra | EEG | ~50-60 min | 50 epochs |
| Quantum Mamba | EEG | ~55-65 min | 50 epochs |
| Classical Hydra | EEG | ~40-50 min | 50 epochs (fewer params) |
| Classical Mamba | EEG | ~45-55 min | 50 epochs |
| Quantum Hydra | MNIST | ~30-40 min | 50 epochs, smaller dataset |
| Quantum Mamba | MNIST | ~35-45 min | 50 epochs |
| Quantum Models | DNA | ~20-30 min | 50 epochs, very small dataset |
| Quantum Models | Forrelation | ~25-35 min | 100 epochs |

**Note:** Times vary based on:
- GPU model and VRAM
- Batch size
- Number of qubits
- Circuit depth (qlcu_layers)
- Early stopping (may finish before 50 epochs)

---

## 🔧 Customizing Timing Output

If you want more detailed timing (e.g., per-batch timing), you can modify the experiment scripts:

```python
# Add to train_epoch() function
batch_times = []
for batch_x, batch_y in train_loader:
    batch_start = time.time()

    # ... training code ...

    batch_time = time.time() - batch_start
    batch_times.append(batch_time)

avg_batch_time = np.mean(batch_times)
return avg_loss, accuracy, avg_batch_time  # Add batch_time to return
```

---

## ✅ Summary

**All experiment scripts record complete timing information:**

### `/pscratch/sd/j/junghoon/quantum_hydra_mamba/scripts/`
✅ Records `training_time` in results JSON
✅ Records per-epoch time during training
✅ Records timestamp for each run
✅ All training scripts track timing
✅ Aggregation scripts include timing in CSV output

**You can access timing for:**
- Each model (quantum_hydra, quantum_mamba, etc.)
- Each seed (2024, 2025, 2026)
- Each dataset (EEG, MNIST, DNA, Forrelation, Genomic)
- Each experiment configuration

**File format:** JSON files saved in `results/{dataset}/` directories

---

**Last Updated:** November 2025
