# Computational Complexity Comparison: Classical Hydra vs QuantumHydraSSM vs QSVT/LCU QuantumHydra

This document provides a comprehensive comparison of three SSM architectures in terms of computational complexity, trainability, performance, scalability, and runtime on classical GPUs.

## Overview

| Model | Type | Description |
|-------|------|-------------|
| **Classical Hydra** | Pure Classical | Bidirectional SSM (Hwang et al., 2024) |
| **Current QuantumHydraSSM** | Hybrid | Quantum feature extraction + Classical SSM |
| **QSVT/LCU QuantumHydra** | Full Quantum | Quantum features + Quantum SSM via QSVT |

---

## Notation

| Symbol | Meaning |
|--------|---------|
| L | Sequence length |
| d | Model dimension (d_model) |
| N | State dimension (d_state) |
| q | Number of qubits |
| l | Number of quantum layers |
| k | Polynomial degree (QSVT) |
| C | Chunk size |

---

## 1. Classical Hydra

### Architecture
Pure classical SSM with bidirectional processing based on Hwang et al. (2024).

### Computational Complexity

| Operation | Complexity |
|-----------|------------|
| Forward SSM scan | O(L · d · N) |
| Backward SSM scan | O(L · d · N) |
| Selective projections (Δ, B, C) | O(L · d · N) |
| **Total per forward pass** | **O(L · d · N)** |

### Example Calculation
- Typical values: L=200, d=64, N=16
- FLOPs: ~200,000 per sample

### Key Operations
```
SSM Recurrence:
  x[t] = A · x[t-1] + B · u[t]    # State update
  y[t] = C · x[t] + D · u[t]      # Output
```

---

## 2. Current QuantumHydraSSM (Hybrid)

### Architecture
Quantum feature extraction with 3-branch superposition, followed by classical SSM.

### Computational Complexity

| Operation | Complexity |
|-----------|------------|
| Classical chunk aggregation | O(L · d) |
| Quantum circuit execution (per chunk) | O(L/C · 2^q · l · q) simulated |
| Classical SSM on chunks | O(L/C · d · N) |
| **Total per forward pass** | **O(L · d + L/C · 2^q · l · q)** |

### Example Calculation
- Typical values: L=200, d=64, q=4, l=2, C=32
- Classical FLOPs: ~50,000
- Quantum circuit calls: 7 (one per chunk)

### Key Insight
Chunking dramatically reduces quantum calls:
- **Without chunking**: 200 quantum calls per sequence
- **With chunking (C=32)**: 7 quantum calls per sequence
- **Speedup**: ~28x fewer quantum circuit evaluations

### Quantum Feature Extraction
```
|ψ⟩ = α|ψ₁⟩ + β|ψ₂⟩ + γ|ψ₃⟩  (3-branch superposition)

Encoding: Angle encoding via RY rotations
  for i in range(n_qubits):
      qml.RY(angle[i], wires=i)
```

---

## 3. QSVT/LCU QuantumHydra (Hypothetical)

### Architecture
Full quantum pipeline using Quantum Singular Value Transformation (QSVT) and Linear Combination of Unitaries (LCU) for SSM operations.

### Computational Complexity

| Operation | Complexity |
|-----------|------------|
| Angle encoding | O(L · q) |
| QSVT polynomial (degree k) | O(k · 2^q) simulated |
| LCU state mixing | O(L · 2^q) simulated |
| State readout (tomography) | O(M · 2^q) where M = measurement shots |
| **Total per forward pass** | **O(L · 2^q · k)** simulated |

### Example Calculation
- Typical values: L=200, q=4, k=3
- Simulated operations: ~10,000
- **Critical overhead**: State tomography required for classical readout

### QSVT/LCU Operations
```
QSVT: Polynomial transformation P(A) of block-encoded matrix A
  |ψ_out⟩ = P(A)|ψ_in⟩

LCU: Linear combination of unitaries
  |ψ_mixed⟩ = Σ_t c_t U_t|ψ⟩
```

---

## Complexity Summary Table

| Model | Time Complexity | Space Complexity |
|-------|-----------------|------------------|
| Classical Hydra | O(L · d · N) | O(d · N) |
| Current QuantumHydraSSM | O(L · d + L/C · 2^q) | O(d · N + 3 · q · l) |
| QSVT/LCU QuantumHydra | O(L · 2^q · k) | O(2^q + k) |

### Numerical Example (d=64, L=200, q=4, N=16, C=32, k=3)

| Model | Approximate FLOPs | Quantum Calls |
|-------|-------------------|---------------|
| Classical Hydra | ~200K | 0 |
| Current QuantumHydraSSM | ~50K | 7 |
| QSVT/LCU QuantumHydra | ~10K* | 200+ |

*Plus significant state tomography overhead

---

## Advantages and Disadvantages

### Classical Hydra

| Aspect | Advantages | Disadvantages |
|--------|------------|---------------|
| **Trainability** | Excellent - well-understood gradients, stable optimization | N/A |
| **Performance** | Strong baseline, proven on long sequences | May miss quantum correlations |
| **Scalability** | Linear in L, scales to 10K+ sequences easily | No quantum advantage possible |
| **GPU Runtime** | Fastest (~1ms per batch) | Classical limitations |

### Current QuantumHydraSSM (Hybrid)

| Aspect | Advantages | Disadvantages |
|--------|------------|---------------|
| **Trainability** | Good - chunking provides smooth gradients | Quantum-classical interface can cause gradient issues |
| **Performance** | Can capture quantum correlations in features | Feature extraction disconnected from SSM |
| **Scalability** | Chunking keeps quantum calls manageable | Limited by simulator's exponential scaling |
| **GPU Runtime** | Moderate (~10-50ms per batch) | Requires quantum simulator overhead |

**Additional Advantages:**
- Works today on classical GPUs
- Practical for NISQ-era experiments
- Proven on Genomic Benchmarks, EEG data

**Additional Disadvantages:**
- Hybrid design limits quantum depth
- Classical SSM may bottleneck quantum expressivity

### QSVT/LCU QuantumHydra (Hypothetical)

| Aspect | Advantages | Disadvantages |
|--------|------------|---------------|
| **Trainability** | Can express complex transformations | Deep circuits cause barren plateaus |
| **Performance** | Theoretically richer quantum dynamics | Unproven on real ML tasks |
| **Scalability** | Polynomial in sequence length (on quantum HW) | Exponential simulation cost with qubits |
| **GPU Runtime** | N/A | Very slow (~100-500ms per batch) |

**Additional Advantages:**
- End-to-end quantum processing
- May show advantage on quantum hardware
- Mathematically elegant (polynomial transformations)

**Additional Disadvantages:**
- QSVT polynomials hard to optimize
- State tomography creates classical bottleneck
- Not practical for NISQ era
- Requires q ≤ 20 on classical GPUs

---

## Runtime Estimates on Classical GPU (NVIDIA A100)

### Per-Batch Runtime (Batch Size = 32)

| Model | L=200 | L=500 | L=1000 |
|-------|-------|-------|--------|
| Classical Hydra | ~1-2 ms | ~3-5 ms | ~5-10 ms |
| Current QuantumHydraSSM | ~20-50 ms | ~30-80 ms | ~50-100 ms |
| QSVT/LCU QuantumHydra | ~200-500 ms | ~500-1000 ms | ~1-2 sec |

### Per-Epoch Runtime (1000 samples, Batch Size = 32)

| Model | L=200 | L=1000 |
|-------|-------|--------|
| Classical Hydra | ~30-60 ms | ~150-300 ms |
| Current QuantumHydraSSM | ~0.6-1.5 sec | ~1.5-3 sec |
| QSVT/LCU QuantumHydra | ~6-15 sec | ~30-60 sec |

---

## Encoding Comparison

All models in this comparison use **angle encoding** (RY rotations):

| Encoding | Method | Qubits Needed | Circuit Depth | Gradient Behavior |
|----------|--------|---------------|---------------|-------------------|
| **Angle Encoding** (Used) | RY(θ) rotations | O(n) for n features | Shallow | Smooth, trainable |
| Amplitude Encoding | StatePrep | O(log n) for n features | Deep, O(2^q) | Can have barren plateaus |

### Why Angle Encoding?

1. **Trainability**: Provides smooth gradients for variational parameter optimization
2. **Circuit Depth**: Keeps circuits shallow (single layer of rotations)
3. **Compatibility**: Works well with lightning.kokkos backend
4. **Interpretability**: Each qubit processes one feature directly

---

## Recommendations

### For Current Experiments (Classical GPU Simulation)

1. **Primary Model**: Use **Current QuantumHydraSSM**
   - Best balance of quantum expressivity and practical runtime
   - Proven on Genomic Benchmarks, EEG, and other datasets

2. **Baseline**: Use **Classical Hydra**
   - Strong classical baseline for fair comparison
   - Validates quantum advantage claims

3. **Optional**: **QSVT/LCU QuantumHydra**
   - Only for theoretical completeness
   - Or if targeting future quantum hardware

### When QSVT/LCU Becomes Advantageous

The QSVT/LCU approach becomes practical when:
- Running on actual **quantum hardware** (not simulators)
- Problem has inherent **quantum structure** (e.g., Forrelation)
- Sequence length is **short enough** to avoid barren plateaus
- **Fault-tolerant quantum computers** become available

---

## References

1. **Gu, A., & Dao, T. (2024).** Mamba: Linear-Time Sequence Modeling with Selective State Spaces. [arXiv:2312.00752](https://arxiv.org/abs/2312.00752)

2. **Hwang, W., Lahoti, V., Dao, T., & Gu, A. (2024).** Hydra: Bidirectional State Space Models Through Generalized Matrix Mixers. [arXiv:2407.09941](https://arxiv.org/abs/2407.09941)

3. **Gilyen, A., Su, Y., Low, G. H., & Wiebe, N. (2019).** Quantum singular value transformation and beyond: exponential improvements for quantum matrix arithmetics. [arXiv:1806.01838](https://arxiv.org/abs/1806.01838)

4. **Park, J. et al. (2025).** Resting-state fMRI Analysis using Quantum Time-Series Transformer. (Internal reference)

---

## Document Information

- **Author**: Junghoon Park
- **Created**: December 2024
- **Last Updated**: December 2024
- **Related Files**:
  - `models/QuantumSSM.py` - Current QuantumHydraSSM implementation
  - `models/TrueClassicalHydra.py` - Classical Hydra implementation
  - `QTSTransformer.py` - QSVT/LCU reference implementation
