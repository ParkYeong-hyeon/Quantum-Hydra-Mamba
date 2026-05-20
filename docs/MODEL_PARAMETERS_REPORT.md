# Model Parameters Report

## Quantum Hydra/Mamba Architecture Comparison

**Generated:** January 20, 2026
**Experimental Setting:** Selective Copy Task Configuration

---

## Experimental Configuration

All models were instantiated with identical hyperparameters matching the Selective Copy benchmark settings:

| Parameter | Value |
|-----------|-------|
| n_qubits | 6 |
| n_layers (qlcu_layers) | 2 |
| d_model | 128 |
| d_state | 16 |
| feature_dim (n_channels) | 10 |
| n_timesteps | 100 |
| output_dim | 10 |
| dropout | 0.1 |

---

## Parameter Counts by Model

### Group 1: Quantum Feature + Classical Mixing (Q→C)

| Model ID | Architecture Name | Trainable Parameters |
|----------|-------------------|---------------------|
| 1a | QuantumFeatureMambaSSM | 169,833 |
| 1b | QuantumFeatureMambaHydraSSM | 422,633 |
| 1c | QuantumFeatureHydraHydraSSM | 910,868 |

**Group 1 Statistics:**
- Mean: 501,111 parameters
- Min: 169,833 (1a)
- Max: 910,868 (1c)
- Range: 741,035

### Group 2: Classical Feature + Quantum Mixing (C→Q)

| Model ID | Architecture Name | Trainable Parameters |
|----------|-------------------|---------------------|
| 2a | ClassicalFeatureQuantumMixingMambaSSM | 36,714 |
| 2d | QuantumMambaStatevectorReconstruction | 46,507 |
| 2e | QuantumHydraHydraStatevectorReconstruction | 72,037 |
| 2f | QuantumHydraStatevectorReconstruction (Mamba-Hydra) | 46,507 |

**Group 2 Statistics:**
- Mean: 50,441 parameters
- Min: 36,714 (2a)
- Max: 72,037 (2e)
- Range: 35,323

### Group 3: Classical Feature + Classical Mixing (C→C)

| Model ID | Architecture Name | Trainable Parameters |
|----------|-------------------|---------------------|
| 3a | ClassicalTransformer | 332,040 |
| 3b | TrueClassicalMamba | 234,760 |
| 3c | TrueClassicalHydra | 232,776 |

**Group 3 Statistics:**
- Mean: 266,525 parameters
- Min: 232,776 (3c)
- Max: 332,040 (3a)
- Range: 99,264

---

## Cross-Group Comparison

| Group | Feature Type | Mixing Type | Avg Parameters | Min | Max |
|-------|--------------|-------------|----------------|-----|-----|
| Group 1 | Quantum | Classical | 501,111 | 169,833 | 910,868 |
| Group 2 | Classical | Quantum | 50,441 | 36,714 | 72,037 |
| Group 3 | Classical | Classical | 266,525 | 232,776 | 332,040 |

### Parameter Efficiency Ranking (Smallest to Largest)

| Rank | Model | Parameters | Group |
|------|-------|------------|-------|
| 1 | 2a | 36,714 | Group 2 |
| 2 | 2d | 46,507 | Group 2 |
| 3 | 2f | 46,507 | Group 2 |
| 4 | 2e | 72,037 | Group 2 |
| 5 | 1a | 169,833 | Group 1 |
| 6 | 3c | 232,776 | Group 3 |
| 7 | 3b | 234,760 | Group 3 |
| 8 | 3a | 332,040 | Group 3 |
| 9 | 1b | 422,633 | Group 1 |
| 10 | 1c | 910,868 | Group 1 |

---

## Key Insights

### 1. Group 2 is Most Parameter-Efficient
- Group 2 models average only **50K parameters**, ~10x smaller than Group 1
- The quantum mixing component adds minimal overhead to classical features
- Models 2d and 2f have identical parameter counts (46,507) despite different architectures

### 2. Quantum Features Increase Model Size
- Group 1 (quantum features) averages **501K parameters**
- This is ~5x larger than Group 3 (pure classical)
- The quantum feature encoding circuit requires substantial classical post-processing

### 3. Hydra Architecture Impact
- Within Group 1: 1a (Mamba) → 1b (Mamba+Hydra) → 1c (Hydra+Hydra)
- Each Hydra layer adds significant parameters (~400K per Hydra stage)
- The multi-head gating mechanism of Hydra contributes to parameter growth

### 4. Classical Baselines (Group 3)
- Transformer (3a) is largest at 332K (attention mechanism overhead)
- Mamba (3b) and Hydra (3c) are similar at ~233K
- These represent well-established SSM architectures

---

## Architecture Notes

### Quantum Components
- **Quantum Feature Encoding (Group 1):** Uses variational quantum circuits to encode classical inputs into quantum states
- **Quantum Mixing (Group 2):** Applies quantum operations to mix classical features
- **Statevector Reconstruction:** Uses classical matrix computation ⟨ψ|O|ψ⟩ instead of circuit-based measurement

### Classical Components
- **Mamba:** Selective state space model with input-dependent dynamics
- **Hydra:** Multi-head gated state space architecture
- **Transformer:** Self-attention mechanism for sequence modeling

---

## Experimental Context

This parameter analysis supports the synthetic benchmark experiments:
- **Tasks:** Adding Problem, Selective Copy, Forrelation
- **Sequence Lengths:** L = 100, 200, 500, 1000
- **Seeds:** 2024, 2025, 2026 (3 runs per configuration)
- **Hardware:** DGX Spark (NVIDIA GPU) and NERSC (Perlmutter)

For detailed performance results, see: `SYNTHETIC_BENCHMARK_COMBINED_RESULTS.md`
