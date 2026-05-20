# Synthetic Benchmark Experiments Progress Report

**Updated:** 2026-01-16 12:05 KST
**Project:** Quantum Hydra Mamba Ablation Study

---

## Executive Summary

| Task | Completed | Total | Progress |
|------|-----------|-------|----------|
| Forrelation | 103 | 108 | **95%** |
| Adding Problem | 19 | 72 | 26% |
| Selective Copy | 0 | 72 | 0% |
| **Total** | **122** | **252** | **48%** |

---

## Currently Running Experiments (8 jobs)

### Forrelation Batch (finishing)
| Model | Task | Seq Len | Seed | Status |
|-------|------|---------|------|--------|
| 4e | forrelation | L=100 | 2025 | Running |
| 4e | forrelation | L=100 | 2026 | Running |
| 4e | forrelation | L=200 | 2024 | Running |

### Adding Problem Batch (main)
| Model | Task | Seq Len | Seed | Status |
|-------|------|---------|------|--------|
| 3b | adding_problem | L=500 | 2024 | Running |
| 3b | adding_problem | L=500 | 2025 | Running |
| 3b | adding_problem | L=1000 | 2024 | Running |

### Adding Problem + Selective Copy Batch (2d/2e)
| Model | Task | Seq Len | Seed | Status |
|-------|------|---------|------|--------|
| 2d | adding_problem | L=100 | 2024 | Running |
| 2d | adding_problem | L=100 | 2025 | Running |

---

## Forrelation Results (BQP-Complete Classification)

**Baseline Accuracy:** 0.5000 (random guessing)
**Status:** 103/108 completed (95%)

### Results by Model

| Model | Name | Group | Count | Avg Acc | Best Acc | vs Baseline |
|-------|------|-------|-------|---------|----------|-------------|
| 1a | QuantumTransformer | Q→C | 9/9 ✅ | 0.5197 | 0.5350 | +3.9% |
| 1b | QuantumMambaSSM | Q→C | 9/9 ✅ | 0.5240 | 0.5390 | +4.8% |
| 1c | QuantumHydraSSM | Q→C | 9/9 ✅ | 0.5253 | 0.5540 | +5.1% |
| 2a | ClassicalQuantumAttention | C→Q | 9/9 ✅ | 0.5189 | 0.5360 | +3.8% |
| 2d | QuantumMambaHydraSSM | C→Q(S) | 9/9 ✅ | **0.5286** | 0.5410 | **+5.7%** |
| 2e | QuantumHydraHydraSSM | C→Q(S) | 9/9 ✅ | 0.5261 | 0.5430 | +5.2% |
| 3a | ClassicalTransformer | C→C | 9/9 ✅ | 0.5278 | 0.5450 | +5.6% |
| 3b | TrueClassicalMamba | C→C | 9/9 ✅ | 0.5248 | 0.5430 | +5.0% |
| 3c | TrueClassicalHydra | C→C | 9/9 ✅ | 0.5263 | 0.5510 | +5.3% |
| 4a | QuantumTransformerE2E | Q→Q | 9/9 ✅ | 0.5262 | 0.5420 | +5.2% |
| 4d | QuantumMambaE2E_Super | Q(S)→Q(S) | 9/9 ✅ | 0.5247 | 0.5360 | +4.9% |
| 4e | QuantumHydraE2E_Super | Q(S)→Q(S) | 4/9 🔄 | 0.5235 | 0.5350 | +4.7% |

**Legend:** Q=Quantum, C=Classical, S=Superposition, →=Features to Mixing

### Top Performers (Forrelation)
1. **2d** (QuantumMambaHydraSSM) - Avg: 0.5286, Best: 0.5410
2. **3a** (ClassicalTransformer) - Avg: 0.5278, Best: 0.5450
3. **3c** (TrueClassicalHydra) - Avg: 0.5263, Best: 0.5510

### Model 4e Progress (In Progress)

| Seq Len | Seed 2024 | Seed 2025 | Seed 2026 |
|---------|-----------|-----------|-----------|
| L=50 | 0.5240 ✅ | 0.5350 ✅ | 0.5110 ✅ |
| L=100 | 0.5240 ✅ | Running | Running |
| L=200 | Running | Pending | Pending |

---

## Adding Problem Results (Long-Range Regression)

**Baseline MSE:** 0.167 (predicting mean)
**Status:** 19/72 completed (26%)

### Results by Model

| Model | Name | Count | Avg MSE | Best MSE | Improvement |
|-------|------|-------|---------|----------|-------------|
| 3a | ClassicalTransformer | 12/12 ✅ | 0.000219 | 0.000084 | 99.87% |
| 3b | TrueClassicalMamba | 7/12 🔄 | 0.000826 | 0.000078 | 99.50% |
| 3c | TrueClassicalHydra | 0/12 ⏳ | - | - | - |
| 4a | QuantumTransformerE2E | 0/12 ⏳ | - | - | - |
| 4d | QuantumMambaE2E_Super | 0/12 ⏳ | - | - | - |
| 4e | QuantumHydraE2E_Super | 0/12 ⏳ | - | - | - |
| 2d | QuantumMambaHydraSSM | 0/12 🔄 | - | - | - |
| 2e | QuantumHydraHydraSSM | 0/12 ⏳ | - | - | - |

### Detailed Results - Model 3a (ClassicalTransformer) ✅ Complete

| Seq Len | Seed 2024 | Seed 2025 | Seed 2026 | Avg |
|---------|-----------|-----------|-----------|-----|
| L=100 | 0.000177 | 0.000215 | 0.000159 | 0.000184 |
| L=200 | 0.000147 | 0.000364 | 0.000084 | 0.000198 |
| L=500 | 0.000210 | 0.000222 | 0.000381 | 0.000271 |
| L=1000 | 0.000322 | 0.000155 | 0.000194 | 0.000224 |

### Detailed Results - Model 3b (TrueClassicalMamba) - In Progress

| Seq Len | Seed 2024 | Seed 2025 | Seed 2026 | Avg |
|---------|-----------|-----------|-----------|-----|
| L=100 | 0.000078 ✅ | 0.000150 ✅ | 0.000357 ✅ | 0.000195 |
| L=200 | 0.000624 ✅ | 0.000610 ✅ | 0.000570 ✅ | 0.000601 |
| L=500 | Running | Running | 0.003394 ✅ | - |
| L=1000 | Running | Pending | Pending | - |

---

## Selective Copy Results (Multi-Output Regression)

**Baseline MSE:** 0.083 (predicting 0.5)
**Status:** 0/72 completed (0%)

| Model | Name | Count | Status |
|-------|------|-------|--------|
| 3a | ClassicalTransformer | 0/12 | Pending |
| 3b | TrueClassicalMamba | 0/12 | Pending |
| 3c | TrueClassicalHydra | 0/12 | Pending |
| 4a | QuantumTransformerE2E | 0/12 | Pending |
| 4d | QuantumMambaE2E_Super | 0/12 | Pending |
| 4e | QuantumHydraE2E_Super | 0/12 | Pending |
| 2d | QuantumMambaHydraSSM | 0/12 | Queued (in 2d/2e batch) |
| 2e | QuantumHydraHydraSSM | 0/12 | Queued (in 2d/2e batch) |

**Note:** Selective Copy experiments will begin after Adding Problem completes for each batch.

---

## Active Batches

### Batch 1: Forrelation (Started Jan 8)
- **Models:** All 12 models
- **Status:** 103/108 completed (95%)
- **Remaining:** 4e L=100 (2 seeds), 4e L=200 (3 seeds)
- **Log:** `jobs/logs/forrelation_batch_20260108_172357.log`

### Batch 2: Adding Problem + Selective Copy (Started Jan 13)
- **Models:** 3a, 3b, 3c, 4a, 4d, 4e
- **Status:** 19/144 completed (13%)
- **Current:** 3b adding_problem L=500-L=1000
- **Log:** `jobs/logs/adding_selective_batch_20260113_021957.log`

### Batch 3: 2d/2e Adding Problem + Selective Copy (Started Jan 16)
- **Models:** 2d, 2e
- **Status:** 0/48 completed (just started)
- **Total experiments:** 48 (2 models × 2 tasks × 4 lengths × 3 seeds)
- **Parallel:** 2 jobs
- **Current:** 2d adding_problem L=100

---

## Experiment Configuration

### Models Tested (12 total)

| Group | ID | Model Name | Features | Mixing | Params |
|-------|-----|------------|----------|--------|--------|
| 1 | 1a | QuantumTransformer | Quantum | Classical | 338K |
| 1 | 1b | QuantumMambaSSM | Quantum | Classical | 423K |
| 1 | 1c | QuantumHydraSSM | Quantum | Classical | 912K |
| 2 | 2a | ClassicalQuantumAttention | Classical | Quantum | 37K |
| 2 | 2d | QuantumMambaHydraSSM | Classical | Quantum (Super) | 48K |
| 2 | 2e | QuantumHydraHydraSSM | Classical | Quantum (Super) | 73K |
| 3 | 3a | ClassicalTransformer | Classical | Classical | 358K |
| 3 | 3b | TrueClassicalMamba | Classical | Classical | 236K |
| 3 | 3c | TrueClassicalHydra | Classical | Classical | 234K |
| 4 | 4a | QuantumTransformerE2E | Quantum | Quantum | 41K |
| 4 | 4d | QuantumMambaE2E_Superposition | Quantum (Super) | Quantum (Super) | 58K |
| 4 | 4e | QuantumHydraE2E_Superposition | Quantum (Super) | Quantum (Super) | 84K |

### Hyperparameters

| Parameter | Value |
|-----------|-------|
| n_qubits | 6 |
| n_layers | 2 |
| d_model | 128 |
| d_state | 16 |
| n_epochs | 100 |
| batch_size | 32 |
| learning_rate | 0.001 |
| weight_decay | 0.0001 |
| early_stopping | 20 epochs |

### Task Configurations

| Task | Type | Seq Lengths | Seeds | Samples |
|------|------|-------------|-------|---------|
| Forrelation | Classification | 50, 100, 200 | 2024, 2025, 2026 | 5000 |
| Adding Problem | Regression | 100, 200, 500, 1000 | 2024, 2025, 2026 | 5000 |
| Selective Copy | Regression | 100, 200, 500, 1000 | 2024, 2025, 2026 | 5000 |

---

## Monitoring Commands

```bash
# Watch specific batch logs
tail -f /home/connectome/justin/quantum_hydra_mamba/jobs/logs/forrelation_batch_20260108_172357.log
tail -f /home/connectome/justin/quantum_hydra_mamba/jobs/logs/adding_selective_batch_20260113_021957.log

# Check GPU utilization
watch -n 5 nvidia-smi

# List all running experiments
ps aux | grep run_synthetic | grep -v grep

# Count completed results
ls results/synthetic_benchmarks/synthetic_*_forrelation_*.json | wc -l
ls results/synthetic_benchmarks/synthetic_*_adding_problem_*.json | wc -l
ls results/synthetic_benchmarks/synthetic_*_selective_copy_*.json | wc -l
```

---

## Next Steps

1. 🔄 Complete Forrelation experiments (5 remaining for model 4e)
2. 🔄 Complete Adding Problem experiments (53 remaining in main batch + 24 in 2d/2e batch)
3. ⏳ Run all Selective Copy experiments (72 total)
4. ⏳ Aggregate final results and generate visualizations
5. ⏳ Statistical analysis across seeds

---

*Report updated automatically by Claude Code - 2026-01-16 12:05 KST*
