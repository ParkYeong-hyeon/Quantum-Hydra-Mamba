# Four-Model Architecture Comparison: Quantum and Classical SSM Variants

This document provides a comprehensive comparison of four SSM (State Space Model) architectures, analyzing computational complexity, trainability, performance, scalability, and runtime on classical GPUs.

## Architecture Overview

| Model | Feature Extraction | Sequence Mixing (SSM) | Key Characteristic |
|-------|-------------------|----------------------|-------------------|
| **1. Classical Hydra** | Classical (Linear) | Classical SSM | Pure classical baseline |
| **2. Current QuantumHydraSSM** | Quantum (3-branch VQC) | Classical SSM | Quantum features → Classical dynamics |
| **3. End-to-End Quantum (QSVT/LCU)** | Quantum (VQC) | Quantum (QSVT/LCU) | Full quantum pipeline |
| **4. Classical Features + Quantum SSM** | Classical (CNN/MLP) | Quantum (QSVT/LCU) | Classical features → Quantum dynamics |

---

## Visual Architecture Diagrams

### Model 1: Classical Hydra
```
Input → [Linear Projection] → [Bidirectional SSM] → [Output Head] → Output
              Classical            Classical           Classical
```

### Model 2: Current QuantumHydraSSM
```
Input → [Chunk Aggregation] → [3-Branch VQC] → [Classical SSM] → [Output Head] → Output
            Classical           QUANTUM           Classical        Classical
```

### Model 3: End-to-End Quantum (QSVT/LCU)
```
Input → [Angle Encoding] → [VQC Features] → [QSVT/LCU SSM] → [Measurement] → Output
            QUANTUM           QUANTUM           QUANTUM          QUANTUM
```

### Model 4: Classical Features + Quantum SSM (NEW)
```
Input → [CNN/MLP Encoder] → [Dim Reduction] → [QSVT/LCU SSM] → [Output Head] → Output
            Classical          Classical          QUANTUM         Classical
```

---

## Notation

| Symbol | Meaning | Typical Value |
|--------|---------|---------------|
| L | Sequence length | 200 |
| d | Model dimension | 64 |
| N | State dimension | 16 |
| q | Number of qubits | 4-8 |
| l | Quantum layers | 2 |
| k | QSVT polynomial degree | 3-5 |
| C | Chunk size | 32 |
| d_hidden | Classical hidden dim | 128-256 |

---

## 1. Computational Complexity

### Theoretical Complexity

| Model | Feature Extraction | SSM/Mixing | Readout | **Total** |
|-------|-------------------|------------|---------|-----------|
| **Classical Hydra** | O(L·d·d) | O(L·d·N) | O(d) | **O(L·d·N)** |
| **Current QuantumHydraSSM** | O(L/C · 2^q · l) | O(L/C · d · N) | O(q) | **O(L·d + L/C·2^q)** |
| **End-to-End Quantum** | O(L/C · 2^q · l) | O(L/C · k · 2^q) | O(L/C · 2^q) | **O(L/C · k · 2^q)** |
| **Classical + Quantum SSM** | O(L·d·d_hidden) | O(L · k · 2^q) | O(L · 2^q) | **O(L·d·d_hidden + L·k·2^q)** |

### Numerical Estimates

**Parameters:** L=200, d=64, q=4, k=3, C=32, d_hidden=128, N=16

| Model | Feature FLOPs | SSM FLOPs | Quantum Calls | **Total Ops** |
|-------|--------------|-----------|---------------|---------------|
| **Classical Hydra** | ~820K | ~200K | 0 | **~1M** |
| **Current QuantumHydraSSM** | ~50K classical | ~12K | 7 | **~62K + 7 QC** |
| **End-to-End Quantum** | ~10K | ~5K | 14 | **~15K + 14 QC** |
| **Classical + Quantum SSM** | ~1.6M | ~10K | 200 | **~1.6M + 200 QC** |

### Key Observations

1. **Classical Hydra**: Most FLOPs but all GPU-optimized matrix operations
2. **Current QuantumHydraSSM**: Chunking reduces quantum calls to L/C (e.g., 200 → 7)
3. **End-to-End Quantum**: Fewest classical ops but quantum simulation dominates
4. **Classical + Quantum SSM**: Most quantum calls (one per timestep) but best feature quality

---

## 2. Trainability Analysis

### Summary Scores

| Model | Gradient Flow | Barren Plateaus | Optimization Difficulty | **Score** |
|-------|--------------|-----------------|------------------------|-----------|
| **Classical Hydra** | Excellent | None | Easy | ⭐⭐⭐⭐⭐ |
| **Current QuantumHydraSSM** | Good | Mild | Moderate | ⭐⭐⭐⭐ |
| **End-to-End Quantum** | Poor | Severe | Very Hard | ⭐⭐ |
| **Classical + Quantum SSM** | Mixed | Moderate | Moderate-Hard | ⭐⭐⭐ |

### Detailed Analysis

#### Classical Hydra
- **Strengths:**
  - Well-understood backpropagation
  - Stable gradients throughout network
  - Standard optimizers work well (Adam, AdamW)
- **Weaknesses:**
  - None significant for training

#### Current QuantumHydraSSM
- **Strengths:**
  - Classical SSM provides stable gradient backbone
  - Chunking reduces circuit depth → mitigates barren plateaus
  - Quantum gradients via parameter-shift rule are well-defined
- **Weaknesses:**
  - Quantum-classical interface can introduce gradient discontinuities
  - Requires careful learning rate tuning for quantum parameters

#### End-to-End Quantum (QSVT/LCU)
- **Strengths:**
  - QSVT polynomials have structured form
  - Theoretically rich expressivity
- **Weaknesses:**
  - Deep quantum circuits prone to barren plateaus
  - QSVT polynomial coefficients hard to optimize jointly
  - May require quantum-aware optimizers (SPSA, natural gradient)
  - Gradient signal can vanish exponentially with circuit depth

#### Classical Features + Quantum SSM
- **Strengths:**
  - Classical encoder has excellent, stable gradients
  - Quantum SSM gradients are localized to dynamics
  - **Key advantage**: Classical encoder can pretrain independently
  - QSVT structure provides some gradient signal
  - Transfer learning possible from pretrained models
- **Weaknesses:**
  - Quantum SSM still has optimization challenges
  - Many quantum calls may slow training iterations

---

## 3. Performance Potential

### Expected Performance

| Model | Feature Quality | Temporal Modeling | Expressivity | **Expected Accuracy** |
|-------|----------------|-------------------|--------------|----------------------|
| **Classical Hydra** | High | Excellent | Classical only | 70-85% |
| **Current QuantumHydraSSM** | Moderate | Excellent | Hybrid | 65-80% |
| **End-to-End Quantum** | Low | Quantum dynamics | Full quantum | 50-70%* |
| **Classical + Quantum SSM** | High | Quantum dynamics | Hybrid (best) | **70-85%+** |

*End-to-End Quantum may underperform due to training difficulties

### Performance Analysis by Model

#### Classical Hydra
- Proven strong baseline on sequence tasks
- Benefits from decades of deep learning optimization research
- Reliable and predictable performance

#### Current QuantumHydraSSM
- Quantum features may capture correlations classical methods miss
- Limited by small qubit count (4-8) for feature extraction
- Classical SSM is the computational workhorse
- Good balance of quantum expressivity and practical training

#### End-to-End Quantum (QSVT/LCU)
- Theoretical potential for quantum advantage
- Practical performance limited by training difficulties
- May excel on quantum-native problems (e.g., Forrelation)
- Best suited for problems with inherent quantum structure

#### Classical Features + Quantum SSM
- **Best feature quality** from mature classical encoders
- Quantum SSM naturally models state evolution (h_t = Ah_{t-1} + Bx_t)
- Can leverage pretrained encoders (ImageNet, BERT, etc.)
- Quantum resources focused where they matter most (temporal dynamics)
- **Most promising for demonstrating quantum utility**

---

## 4. Scalability

### Scaling with Sequence Length (L)

| Model | L=200 | L=500 | L=1000 | L=5000 | **Scaling Behavior** |
|-------|-------|-------|--------|--------|---------------------|
| **Classical Hydra** | Fast | Fast | Moderate | Slow | O(L) - linear |
| **Current QuantumHydraSSM** | Moderate | Moderate | Moderate | Moderate | O(L/C) quantum calls |
| **End-to-End Quantum** | Slow | Very Slow | Impractical | Impractical | O(L/C) but heavy per-call |
| **Classical + Quantum SSM** | Moderate | Slow | Very Slow | Impractical | O(L) quantum calls |

### Scaling with Number of Qubits (q)

| Model | q=4 | q=8 | q=12 | q=16 | **Scaling Behavior** |
|-------|-----|-----|------|------|---------------------|
| **Classical Hydra** | N/A | N/A | N/A | N/A | No quantum component |
| **Current QuantumHydraSSM** | Fast | Moderate | Slow | Very Slow | O(2^q) simulation |
| **End-to-End Quantum** | Moderate | Slow | Very Slow | Impractical | O(2^q · k) simulation |
| **Classical + Quantum SSM** | Moderate | Slow | Very Slow | Impractical | O(L · 2^q) simulation |

### Scalability Insights

**Classical + Quantum SSM Bottleneck:**
- Requires quantum circuit evaluation at **every timestep**
- For L=200: 200 quantum circuit evaluations per sample per forward pass
- **Potential mitigation**: Chunked Quantum SSM variant
  - Process chunks of timesteps together
  - Trade temporal resolution for speed

**Practical Qubit Limits on Classical GPU:**
| Qubits | State Vector Size | Simulation Feasibility |
|--------|------------------|----------------------|
| 4 | 16 | Trivial |
| 8 | 256 | Easy |
| 12 | 4,096 | Moderate |
| 16 | 65,536 | Slow |
| 20 | 1,048,576 | Very Slow |
| 24+ | 16M+ | Impractical |

---

## 5. Running Time on Classical GPU

### Hardware Assumptions
- **GPU**: NVIDIA A100 (80GB HBM)
- **Quantum Simulator**: PennyLane with lightning.qubit/lightning.kokkos
- **Batch Size**: 32
- **Sequence Length**: 200 (default)

### Per-Batch Runtime

| Model | Forward Pass | Backward Pass | **Total/Batch** |
|-------|-------------|---------------|-----------------|
| **Classical Hydra** | ~1 ms | ~2 ms | **~3 ms** |
| **Current QuantumHydraSSM** | ~15 ms | ~30 ms | **~45 ms** |
| **End-to-End Quantum** | ~80 ms | ~200 ms | **~280 ms** |
| **Classical + Quantum SSM** | ~150 ms | ~400 ms | **~550 ms** |

### Per-Epoch Runtime (1000 samples, Batch=32 → ~31 batches)

| Model | L=200 | L=500 | L=1000 |
|-------|-------|-------|--------|
| **Classical Hydra** | ~0.1 sec | ~0.2 sec | ~0.4 sec |
| **Current QuantumHydraSSM** | ~1.4 sec | ~2.5 sec | ~4 sec |
| **End-to-End Quantum** | ~9 sec | ~20 sec | ~40 sec |
| **Classical + Quantum SSM** | ~17 sec | ~40 sec | ~80 sec |

### Full Training Time (30 epochs, 1000 samples)

| Model | L=200 | L=500 | L=1000 |
|-------|-------|-------|--------|
| **Classical Hydra** | ~3 sec | ~6 sec | ~12 sec |
| **Current QuantumHydraSSM** | ~42 sec | ~75 sec | ~2 min |
| **End-to-End Quantum** | ~4.5 min | ~10 min | ~20 min |
| **Classical + Quantum SSM** | ~8.5 min | ~20 min | ~40 min |

### Runtime Scaling Visualization

```
Training Time (30 epochs, L=200)
─────────────────────────────────────────────────────
Classical Hydra:        ██ 3 sec
Current QuantumHydraSSM: ████████████ 42 sec
End-to-End Quantum:     ██████████████████████████████████████████ 4.5 min
Classical + Quantum SSM: ████████████████████████████████████████████████████████████████████████████ 8.5 min
```

---

## 6. Summary Comparison Matrix

| Aspect | Classical Hydra | Current QuantumHydraSSM | End-to-End Quantum | Classical + Quantum SSM |
|--------|----------------|------------------------|-------------------|------------------------|
| **Complexity** | O(L·d·N) | O(L·d + L/C·2^q) | O(L/C·k·2^q) | O(L·d·d_h + L·k·2^q) |
| **Trainability** | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐ | ⭐⭐ | ⭐⭐⭐ |
| **Performance Potential** | ⭐⭐⭐⭐ | ⭐⭐⭐⭐ | ⭐⭐⭐ | ⭐⭐⭐⭐⭐ |
| **Scalability (L)** | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐ | ⭐⭐ | ⭐⭐⭐ |
| **Scalability (q)** | N/A | ⭐⭐⭐ | ⭐⭐ | ⭐⭐ |
| **GPU Runtime** | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐ | ⭐⭐ | ⭐⭐ |
| **Quantum Advantage Potential** | None | Possible | Theoretical | **Most Promising** |
| **Practical Today** | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ | ⭐⭐ | ⭐⭐⭐ |

---

## 7. Recommendations

### For Current Experiments on Classical GPUs

#### Primary Choice: Current QuantumHydraSSM
- Best balance of quantum expressivity and practical runtime
- Proven on Genomic Benchmarks, EEG, and other datasets
- Manageable training time (~1-2 minutes per experiment)

#### Baseline: Classical Hydra
- Essential for fair comparison and validating quantum claims
- Fast iteration for hyperparameter tuning
- Strong performance benchmark

### For Future Research

#### Most Promising: Classical Features + Quantum SSM
**Why this architecture deserves attention:**

1. **Plays to Strengths**
   - Classical deep learning excels at feature extraction
   - Quantum dynamics naturally model state evolution
   - Each component does what it's best at

2. **Natural Mathematical Fit**
   - SSM recurrence: `h_t = A·h_{t-1} + B·x_t`
   - Quantum evolution: `|ψ(t)⟩ = e^{-iHt}|ψ(0)⟩`
   - Matrix exponentials have known quantum speedups via QSVT

3. **Practical Advantages**
   - Can use pretrained classical encoders (transfer learning)
   - Classical features are high-quality from the start
   - Quantum resources focused on the computationally "hard" part

4. **Path to Quantum Advantage**
   - Clearest case for where quantum helps (dynamics, not features)
   - Can benchmark classical vs quantum SSM directly
   - Scales naturally to quantum hardware

### Not Recommended for Current Work

#### End-to-End Quantum (QSVT/LCU)
- Training too difficult (barren plateaus)
- Too slow on classical simulators
- **Reserve for:**
  - Actual quantum hardware
  - Quantum-native problems (Forrelation, quantum chemistry)
  - Theoretical/proof-of-concept studies

---

## 8. Future Directions

### Chunked Classical + Quantum SSM

To address the O(L) quantum calls bottleneck:

```
Input → [Classical Encoder] → [Chunk into groups of C timesteps]
                                        │
                                        ▼
                              [Quantum SSM per chunk]
                                        │
                                        ▼
                              [Interpolate/Upsample]
                                        │
                                        ▼
                                    Output
```

**Benefits:**
- Reduces quantum calls from L to L/C
- Maintains quantum dynamics modeling
- Trades some temporal resolution for speed

### Hybrid Training Strategy

1. **Phase 1**: Pretrain classical encoder on task (fast)
2. **Phase 2**: Freeze encoder, train quantum SSM (focused)
3. **Phase 3**: Fine-tune end-to-end with small learning rate

### Quantum Hardware Transition

When quantum hardware becomes available:
- Classical + Quantum SSM architecture transfers directly
- Classical encoder runs on GPU, quantum SSM on QPU
- Measurement overhead reduced on real quantum hardware

---

## 9. Encoding Method Note

All quantum models in this comparison use **angle encoding** (RY rotations):

```python
# Angle encoding implementation
for i in range(n_qubits):
    qml.RY(input_features[i], wires=i)
```

**Why Angle Encoding:**
| Property | Angle Encoding | Amplitude Encoding |
|----------|---------------|-------------------|
| Qubits needed | O(n) | O(log n) |
| Circuit depth | O(1) | O(2^n) |
| Gradient behavior | Smooth | Can have plateaus |
| Implementation | Simple rotations | Complex StatePrep |

Angle encoding is chosen for trainability and practical circuit depth, despite requiring more qubits.

---

## References

1. **Gu, A., & Dao, T. (2024).** Mamba: Linear-Time Sequence Modeling with Selective State Spaces. [arXiv:2312.00752](https://arxiv.org/abs/2312.00752)

2. **Hwang, W., Lahoti, V., Dao, T., & Gu, A. (2024).** Hydra: Bidirectional State Space Models Through Generalized Matrix Mixers. [arXiv:2407.09941](https://arxiv.org/abs/2407.09941)

3. **Gilyen, A., Su, Y., Low, G. H., & Wiebe, N. (2019).** Quantum singular value transformation and beyond: exponential improvements for quantum matrix arithmetics. [arXiv:1806.01838](https://arxiv.org/abs/1806.01838)

4. **McClean, J. R., et al. (2018).** Barren plateaus in quantum neural network training landscapes. Nature Communications, 9(1), 4812.

5. **Park, J. et al. (2025).** Resting-state fMRI Analysis using Quantum Time-Series Transformer. (Internal reference)

---

## Document Information

- **Author**: Junghoon Park
- **Created**: December 2024
- **Last Updated**: December 2024
- **Related Documents**:
  - `COMPUTATIONAL_COMPLEXITY_COMPARISON.md` - Three-model comparison
  - `QUANTUM_SSM_README.md` - QuantumSSM implementation details
- **Related Code Files**:
  - `models/QuantumSSM.py` - Current QuantumHydraSSM
  - `models/TrueClassicalHydra.py` - Classical Hydra baseline
  - `QTSTransformer.py` - QSVT/LCU reference implementation
