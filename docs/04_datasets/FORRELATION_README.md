# Forrelation Experiments: Testing for Quantum Advantage

This directory contains all scripts and tools for running the **Sequential Forrelation** experiments, designed to test for quantum advantage in the Quantum Hydra and Quantum Mamba models.

## Overview

The Forrelation problem provides a **theoretically proven quantum advantage** (Aaronson & Ambainis, 2015). We adapt this to a machine learning task:

- **Task**: Predict if two Boolean functions f and g have high or low forrelation
- **Input**: A sequence of samples (x, f(x), y, g(y))
- **Hypothesis**: Quantum models should achieve high accuracy with **fewer samples** than classical models

## Test Standards

Following `Quantum_Advantage_Test_Plan.md`, we test three standards:

| Standard | Test | Evidence |
|----------|------|----------|
| **Bronze** | Peak performance on fixed dataset | Quantum > Classical accuracy |
| **Silver** | Sample efficiency across varying L | Quantum needs fewer samples to reach target accuracy |
| **Gold** | Scaling with problem complexity | Advantage increases with n_bits |

## Directory Structure

```
quantum_hydra_mamba/
├── scripts/                               # Training & analysis scripts
│   ├── run_single_model_forrelation.py
│   ├── generate_forrelation_datasets_all.sh
│   ├── generate_forrelation_job_scripts.py
│   └── aggregate_forrelation_results.py
│
├── job_scripts/forrelation/               # SLURM job scripts
│   └── submit_forrelation_experiments.sh
│
├── results/forrelation_results/           # Outputs (JSON, models, plots)
│   └── logs/
│
└── data/forrelation_data/                 # Datasets
    ├── tuning_dataset.pt                  # Phase 1: Hyperparameter tuning
    ├── challenge_dataset.pt               # Phase 2: Bronze standard
    ├── forrelation_L20.pt                 # Phase 3: Sample efficiency (n=6)
    ├── forrelation_L40.pt
    ├── forrelation_L80.pt
    ├── forrelation_L160.pt
    ├── forrelation_n8_L40.pt              # Phase 4: Scaling (n=8)
    ├── forrelation_n8_L80.pt
    ├── forrelation_n8_L160.pt
    └── forrelation_n8_L320.pt
```

## Quick Start

### Step 1: Generate Datasets

Generate all datasets for the test plan:

```bash
bash scripts/generate_forrelation_datasets_all.sh
```

This creates:
- **Tuning dataset**: 2000 pairs, n_bits=6, L=80
- **Challenge dataset**: 5000 pairs, n_bits=7, L=120
- **Phase 3 datasets** (Silver Standard): n_bits=6, L ∈ {20, 40, 80, 160}
- **Phase 4 datasets** (Gold Standard): n_bits=8, L ∈ {40, 80, 160, 320}

**Time**: ~10-30 minutes depending on system

### Step 2: Generate Job Scripts

Create SLURM job submission scripts for all models and datasets:

```bash
python scripts/generate_forrelation_job_scripts.py
```

This generates **144 job scripts** (6 models × 8 datasets × 3 seeds):
- quantum_hydra
- quantum_hydra_hybrid
- quantum_mamba
- quantum_mamba_hybrid
- classical_hydra
- classical_mamba

### Step 3: Submit Experiments

Submit all jobs to SLURM:

```bash
bash job_scripts/submit_forrelation_experiments.sh
```

**Resources per job**:
- 1 GPU (80GB HBM)
- 32 CPU cores
- 48 hour time limit
- Account: m4138_g

**Monitor jobs**:
```bash
watch -n 10 'squeue -u $USER | grep forr_'
```

**Estimated total time**: 2-4 days (depends on queue availability)

### Step 4: Analyze Results

After experiments complete, aggregate and analyze results:

```bash
python scripts/aggregate_forrelation_results.py
```

**Outputs**:
- `forrelation_all_results.csv`: Complete results table
- `forrelation_sample_efficiency.png`: Accuracy vs sequence length plots
- `forrelation_heatmap.png`: Model comparison heatmap

## Understanding the Results

### Sample Efficiency (Silver Standard)

The key plot shows **Accuracy vs Sequence Length** for n_bits=6:

```
Accuracy
1.0 |                    Q----Q----Q (Quantum)
    |                   /
0.9 |                  /
    |                 /      C----C----C (Classical)
0.8 |                /      /
    |               /      /
0.7 |              /      /
    |             /      /
0.6 |        Q---Q      /
    |       /          /
0.5 |  Q---Q      C---C
    +----------------------------------------> Sequence Length
      20   40   80   160
```

**Silver Standard is met if**:
- Quantum models reach 90% accuracy at L_quantum
- Classical models reach 90% accuracy at L_classical
- L_quantum << L_classical (quantum needs significantly fewer samples)

### Scaling (Gold Standard)

Compare the performance gap at different complexities:

| n_bits | Quantum Acc | Classical Acc | Advantage |
|--------|-------------|---------------|-----------|
| 6      | 0.92        | 0.85          | +0.07     |
| 8      | 0.88        | 0.75          | +0.13     |

**Gold Standard is met if**:
- Silver Standard is met
- The advantage **increases** as n_bits increases
- This shows quantum models scale better to harder problems

## Dataset Details

### Phase 3: Sample Efficiency (n_bits=6)

| Dataset | n_bits | Seq Length | Domain Size | Samples per Function |
|---------|--------|------------|-------------|----------------------|
| L20     | 6      | 20         | 2^6 = 64    | 3000                 |
| L40     | 6      | 40         | 2^6 = 64    | 3000                 |
| L80     | 6      | 80         | 2^6 = 64    | 3000                 |
| L160    | 6      | 160        | 2^6 = 64    | 3000                 |

### Phase 4: Scaling (n_bits=8)

| Dataset  | n_bits | Seq Length | Domain Size | Samples per Function |
|----------|--------|------------|-------------|----------------------|
| n8_L40   | 8      | 40         | 2^8 = 256   | 3000                 |
| n8_L80   | 8      | 80         | 2^8 = 256   | 3000                 |
| n8_L160  | 8      | 160        | 2^8 = 256   | 3000                 |
| n8_L320  | 8      | 320        | 2^8 = 256   | 3000                 |

## Custom Experiments

### Generate a Single Dataset

```bash
python generate_forrelation_dataset.py \
    --num_pairs 2000 \
    --n_bits 7 \
    --seq_len 100 \
    --filename forrelation_data/custom_dataset.pt
```

### Train a Single Model

```bash
python scripts/run_single_model_forrelation.py \
    --model-name quantum_mamba \
    --dataset-path forrelation_data/forrelation_L80.pt \
    --n-qubits 6 \
    --qlcu-layers 2 \
    --n-epochs 100 \
    --batch-size 32 \
    --lr 1e-3 \
    --seed 2024 \
    --output-dir results/forrelation_results \
    --device cuda
```

### Hyperparameter Tuning

For Phase 1 (finding optimal hyperparameters):

```bash
# Generate tuning dataset
python generate_forrelation_dataset.py \
    --num_pairs 2000 \
    --n_bits 6 \
    --seq_len 80 \
    --filename forrelation_data/tuning_dataset.pt

# Train with different hyperparameters
for lr in 1e-4 1e-3 1e-2; do
    for d_model in 64 128 256; do
        python scripts/run_single_model_forrelation.py \
            --model-name quantum_mamba \
            --dataset-path forrelation_data/tuning_dataset.pt \
            --lr $lr \
            --d-model $d_model \
            --seed 2024
    done
done
```

## Expected Results

Based on the theoretical foundation:

### Quantum Models
- Should reach high accuracy (>90%) with **shorter sequences**
- Expected advantage: 30-50% fewer samples than classical
- Should scale better to larger n_bits

### Classical Models
- Require longer sequences to achieve similar accuracy
- Performance degrades more rapidly as n_bits increases
- Still competitive on easier tasks (n_bits=6, long sequences)

## Interpreting Quantum Advantage

### Strong Evidence (Gold Standard)
✓ Quantum models consistently outperform classical
✓ Require significantly fewer samples (>30% reduction)
✓ Advantage increases with problem complexity
✓ Results consistent across multiple seeds

### Moderate Evidence (Silver Standard)
✓ Quantum models require fewer samples
✓ Statistically significant difference
~ Advantage stable but doesn't increase with complexity

### Weak Evidence (Bronze Standard)
✓ Quantum models achieve higher peak accuracy
~ Only on specific datasets
~ Small or inconsistent advantage

## Troubleshooting

### Dataset generation fails
**Error**: Memory error during dataset generation

**Solution**: Reduce `--num_pairs` or `--n_bits`:
```bash
python generate_forrelation_dataset.py --num_pairs 1000 --n_bits 6
```

### Training crashes with CUDA OOM
**Error**: CUDA out of memory

**Solution**: Reduce batch size:
```bash
python scripts/run_single_model_forrelation.py --batch-size 16
```

### No quantum advantage observed
**Possible causes**:
1. Insufficient training epochs (increase `--n-epochs`)
2. Poor hyperparameters (run Phase 1 tuning)
3. Task too easy/hard (adjust n_bits and seq_len)
4. Model architecture limitations

## References

1. **Aaronson, S., & Ambainis, A. (2015)**. "Forrelation: A problem that optimally separates quantum from classical computing." STOC 2015. [arXiv:1411.5729](https://arxiv.org/abs/1411.5729)

2. **Forrelation_Experiment_Rationale.md**: Detailed motivation and experimental design

3. **Quantum_Advantage_Test_Plan.md**: Step-by-step testing framework

4. **Forrelation_Dataset_Usage_Guide.md**: Dataset format and usage

## Contact

For questions or issues, refer to the main project documentation or check the logs in `results/forrelation_results/logs/`.
