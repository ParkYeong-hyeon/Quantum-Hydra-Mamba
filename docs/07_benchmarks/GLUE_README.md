# Quantum Hydra on GLUE Benchmark

This directory contains code for evaluating **QuantumHydraGated** and **QuantumMambaGated** on the GLUE (General Language Understanding Evaluation) benchmark.

## Overview

We adapt the quantum state-space models (originally designed for EEG/genomic data) for natural language understanding tasks by adding:
- Token embedding layer with positional encoding
- Sentence pair handling for tasks like MRPC, QQP, RTE
- Task-specific output heads (classification/regression)

## Quick Start

### 1. Test Run (Quick Verification)

```bash
# Test with mini dataset (1000 train samples)
python run_glue.py --task=sst2 --model=quantum_hydra --mini --n-epochs=5
```

### 2. Full Single Task

```bash
# Train all models on SST-2
python run_glue.py --task=sst2 --model=all --n-epochs=50
```

### 3. Submit SLURM Jobs

```bash
# Submit all recommended tasks
bash run_all_glue.sh
```

## Files

| File | Description |
|------|-------------|
| `Load_GLUE.py` | Data loader for all 9 GLUE tasks |
| `QuantumHydraGLUE.py` | Model wrappers for GLUE |
| `run_glue.py` | Main training script |
| `run_glue_sst2.sh` | SLURM script for SST-2 |
| `run_glue_cola.sh` | SLURM script for CoLA |
| `run_glue_mrpc.sh` | SLURM script for MRPC |
| `run_all_glue.sh` | Submit all experiments |

## Models

### QuantumHydraGLUE (Recommended)
- **Architecture**: Bidirectional quantum processing (forward + backward + global)
- **Gating**: Three-gate LSTM-style (forget, input, output)
- **Quantum**: Three-branch superposition with complex coefficients
- **Tested**: Works on all sequence lengths

### QuantumMambaGLUE
- **Architecture**: Unidirectional quantum processing
- **Gating**: Three-gate LSTM-style
- **Limitation**: May have stability issues on very long sequences (>1000)

### LSTMBaseline
- **Architecture**: Bidirectional LSTM
- **Purpose**: Fair classical comparison

## GLUE Tasks

| Task | Type | Samples | Metric | Recommended |
|------|------|---------|--------|-------------|
| **CoLA** | Single | 8.5k | Matthews Corr | ✅ Start here |
| **SST-2** | Single | 67k | Accuracy | ✅ Start here |
| **MRPC** | Pair | 3.7k | F1 | ✅ Good test |
| **QQP** | Pair | 364k | F1 | Later |
| **STS-B** | Pair | 5.7k | Pearson | Regression |
| **MNLI** | Pair | 393k | Accuracy | Large |
| **QNLI** | Pair | 105k | Accuracy | Medium |
| **RTE** | Pair | 2.5k | Accuracy | Small |
| **WNLI** | Pair | 634 | Accuracy | Very small |

## Hyperparameters

### Default Configuration

```python
{
    'embedding_dim': 128,      # Token embedding size
    'hidden_dim': 64,          # Hidden layer size
    'n_qubits': 6,             # Quantum circuit qubits
    'qlcu_layers': 2,          # Quantum circuit depth
    'chunk_size': 16,          # Chunked processing size
    'max_length': 128,         # Maximum sequence length
    'batch_size': 32,          # Training batch size
    'learning_rate': 2e-4,     # AdamW learning rate
    'weight_decay': 0.01,      # L2 regularization
    'n_epochs': 50,            # Training epochs
    'patience': 10,            # Early stopping patience
}
```

### Tuning Recommendations

| Parameter | Range | Effect |
|-----------|-------|--------|
| `n_qubits` | 4-8 | More qubits = more expressiveness |
| `qlcu_layers` | 1-3 | More layers = deeper circuits |
| `embedding_dim` | 64-256 | Higher = better text representation |
| `hidden_dim` | 32-128 | Higher = more capacity |
| `learning_rate` | 1e-4 to 5e-4 | Higher may converge faster |

## Expected Results

Based on preliminary estimates:

| Model | SST-2 | CoLA | MRPC |
|-------|-------|------|------|
| QuantumHydraGLUE | ~75-80% | ~0.35-0.45 MCC | ~70-80% F1 |
| QuantumMambaGLUE | ~73-78% | ~0.30-0.40 MCC | ~68-78% F1 |
| LSTM Baseline | ~80-85% | ~0.40-0.50 MCC | ~75-85% F1 |
| BERT-base | ~93% | ~0.60 MCC | ~88% F1 |

**Note**: Quantum models are NOT expected to beat BERT. The goal is to demonstrate:
1. Quantum SSMs can process NLP tasks
2. Competitive with simple baselines (LSTM)
3. Parameter efficiency advantages

## Usage Examples

### Command Line

```bash
# Single model, single task
python run_glue.py --task=cola --model=quantum_hydra --n-epochs=50

# All models comparison
python run_glue.py --task=sst2 --model=all --n-epochs=50

# Custom hyperparameters
python run_glue.py \
    --task=mrpc \
    --model=quantum_hydra \
    --n-qubits=8 \
    --qlcu-layers=3 \
    --embedding-dim=256 \
    --hidden-dim=128 \
    --n-epochs=100

# Quick test (mini dataset)
python run_glue.py --task=sst2 --model=quantum_hydra --mini --n-epochs=5
```

### Python API

```python
from Load_GLUE import load_glue_task
from QuantumHydraGLUE import QuantumHydraGLUE

# Load data
train_loader, val_loader, test_loader, metadata = load_glue_task(
    task_name='sst2',
    max_length=128,
    batch_size=32
)

# Create model
model = QuantumHydraGLUE(
    vocab_size=metadata['vocab_size'],
    embedding_dim=128,
    n_qubits=6,
    num_labels=metadata['num_labels'],
    device='cuda'
)

# Training loop
for batch in train_loader:
    outputs = model(
        input_ids=batch['input_ids'].to('cuda'),
        attention_mask=batch['attention_mask'].to('cuda'),
        labels=batch['labels'].to('cuda')
    )
    loss = outputs['loss']
    loss.backward()
    ...
```

## Output Structure

```
glue_results/
├── sst2/
│   ├── quantum_hydra_sst2_seed2024_best.pt
│   ├── quantum_hydra_sst2_seed2024_results.json
│   ├── quantum_mamba_sst2_seed2024_best.pt
│   ├── quantum_mamba_sst2_seed2024_results.json
│   ├── lstm_baseline_sst2_seed2024_best.pt
│   ├── lstm_baseline_sst2_seed2024_results.json
│   └── all_models_sst2_seed2024_summary.json
├── cola/
│   └── ...
└── mrpc/
    └── ...
```

## Requirements

```
torch>=2.0
transformers>=4.30
datasets>=2.14
scikit-learn>=1.0
scipy>=1.10
pennylane>=0.35
tqdm
numpy
```

Install with:
```bash
pip install datasets transformers
```

## Troubleshooting

### CUDA Out of Memory
```bash
# Reduce batch size
python run_glue.py --task=sst2 --batch-size=16

# Or reduce sequence length
python run_glue.py --task=sst2 --max-length=64
```

### Slow Training
```bash
# Reduce quantum circuit complexity
python run_glue.py --task=sst2 --n-qubits=4 --qlcu-layers=1
```

### Import Errors
```bash
# Make sure you're in the right directory
cd /pscratch/sd/j/junghoon/QuantERA
conda activate /pscratch/sd/j/junghoon/conda-envs/qml_eeg

# Test imports
python -c "from Load_GLUE import load_glue_task; print('OK')"
python -c "from QuantumHydraGLUE import QuantumHydraGLUE; print('OK')"
```

