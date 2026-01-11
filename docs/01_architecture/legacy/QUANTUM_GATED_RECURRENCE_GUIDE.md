# Quantum Gated Recurrence: QuantumHydraGated and QuantumMambaGated

**Date**: November 24, 2025
**Version**: 1.0

---

## Table of Contents

1. [Overview](#overview)
2. [Processing Pipeline: From Input to Prediction](#processing-pipeline)
3. [Comparison with Classical Models](#comparison-with-classical-models)
4. [Quantum Components: Superposition and Entanglement](#quantum-components)
5. [Advantages and Limitations](#advantages-and-limitations)
6. [Technical Specifications](#technical-specifications)
7. [Experimental Results](#experimental-results)

---

## 1. Overview

QuantumHydraGated and QuantumMambaGated are hybrid quantum-classical neural network architectures designed for sequential data processing. They combine:

- **Three-branch quantum superposition** for rich feature extraction
- **LSTM-style gating** for selective memory management
- **Chunked processing** for efficient long sequence handling

These models extend the original QuantumHydra/QuantumMamba architectures by adding recurrent processing capabilities essential for learning long-range dependencies.

### Key Innovations

| Feature | QuantumMambaGated | QuantumHydraGated |
|---------|-------------------|-------------------|
| **Processing Direction** | Unidirectional (forward) | Bidirectional (forward + backward + global) |
| **Branches** | 3 quantum superposition branches | 3 × 2 branches (forward/backward each have 3) + 3 global |
| **Gating** | LSTM-style (forget, input, output) | LSTM-style per direction |
| **Final Combination** | Direct output | Complex coefficient combination |

---

## 2. Processing Pipeline: From Input to Prediction

### QuantumMambaGated Pipeline

```
Input: x ∈ ℝ^(batch × timesteps × features)
         ↓
┌─────────────────────────────────────┐
│ 1. Feature Projection               │
│    x_proj = SiLU(Linear(Dropout(x)))│
│    • Mixes all input channels       │
│    • Enables cross-channel learning │
└─────────────────────────────────────┘
         ↓
┌─────────────────────────────────────┐
│ 2. Chunked Processing               │
│    for each chunk c in chunks:      │
│    ┌───────────────────────────┐    │
│    │ 2a. Quantum Superposition │    │
│    │     • Branch 1: QLCU #1   │    │
│    │     • Branch 2: QLCU #2   │    │
│    │     • Branch 3: QLCU #3   │    │
│    │     Combined with α,β,γ   │    │
│    └───────────────────────────┘    │
│              ↓                      │
│    ┌───────────────────────────┐    │
│    │ 2b. Chunk Aggregation     │    │
│    │     mean → Linear → LN    │    │
│    └───────────────────────────┘    │
│              ↓                      │
│    ┌───────────────────────────┐    │
│    │ 2c. Gated State Update    │    │
│    │     f = σ(W_f[h, chunk])  │    │
│    │     i = σ(W_i[h, chunk])  │    │
│    │     o = σ(W_o[h, chunk])  │    │
│    │     c̃ = tanh(W_c[h,chunk])│    │
│    │     h = o * tanh(f*h+i*c̃) │    │
│    └───────────────────────────┘    │
└─────────────────────────────────────┘
         ↓
┌─────────────────────────────────────┐
│ 3. Output Layer                     │
│    output = Linear(ReLU(Linear(h))) │
└─────────────────────────────────────┘
         ↓
Output: predictions ∈ ℝ^(batch × output_dim)
```

### QuantumHydraGated Pipeline

```
Input: x ∈ ℝ^(batch × timesteps × features)
         ↓
┌─────────────────────────────────────┐
│ 1. Feature Projection               │
│    x_proj = Linear(Dropout(x))      │
└─────────────────────────────────────┘
         ↓
    ┌────┴────┬─────────┐
    ↓         ↓         ↓
┌───────┐ ┌───────┐ ┌───────┐
│Forward│ │Backward│ │Global │
│Branch │ │Branch │ │Branch │
└───┬───┘ └───┬───┘ └───┬───┘
    │         │         │
    │  Chunked│  Chunked│  Mean →
    │  Gated  │  Gated  │  Quantum
    │  Super- │  Super- │  Super-
    │  position│ position│ position
    │         │         │
    ↓         ↓         ↓
  h_fwd    h_bwd     h_global
    │         │         │
    └────┬────┴─────────┘
         ↓
┌─────────────────────────────────────┐
│ 2. Complex Coefficient Combination  │
│    combined = α*h_fwd + β*h_bwd     │
│               + γ*h_global          │
│    normalized = combined/||combined││
│    output_features = |normalized|   │
└─────────────────────────────────────┘
         ↓
┌─────────────────────────────────────┐
│ 3. Output Layer                     │
│    output = Linear(output_features) │
└─────────────────────────────────────┘
         ↓
Output: predictions ∈ ℝ^(batch × output_dim)
```

### Quantum Superposition Branch Detail

Each `QuantumSuperpositionBranches` module:

```python
# Three independent quantum circuits
params1 = proj1(x)  # Linear(feature_dim → n_params)
params2 = proj2(x)
params3 = proj3(x)

# Quantum feature extraction per branch
m1 = QuantumFeatureExtractor(params1)  # → 3*n_qubits measurements
m2 = QuantumFeatureExtractor(params2)
m3 = QuantumFeatureExtractor(params3)

# Complex coefficient combination
α = (α_real + i*α_imag) / norm
β = (β_real + i*β_imag) / norm
γ = (γ_real + i*γ_imag) / norm

# Superposition in measurement space
combined = α*m1 + β*m2 + γ*m3
output = |combined|  # Magnitude
```

---

## 3. Comparison with Classical Models

### Classical Mamba (Gu & Dao, 2024)

```python
# Selective State Space Model
B = Linear(x)       # Input-dependent
C = Linear(x)       # Input-dependent
dt = softplus(Linear(x))

# State evolution (parallel scan)
A = exp(-dt * A_base)
h[t] = A * h[t-1] + B * x[t]
y[t] = C * h[t] + D * x[t]

# Gating
output = y * silu(gate)
```

**Key Properties:**
- O(n) linear complexity via parallel scan
- Input-dependent selectivity (B, C, dt)
- Continuous-time discretization

### QuantumMambaGated Differences

| Aspect | Classical Mamba | QuantumMambaGated |
|--------|-----------------|-------------------|
| **State Representation** | Real vectors | Quantum measurements |
| **Selectivity** | A, B, C matrices | Quantum circuit parameters |
| **State Evolution** | Linear recurrence | LSTM-style gating |
| **Feature Space** | d_model dimensions | 2^n_qubits Hilbert space |
| **Parallelism** | Parallel scan O(log T) | Chunked O(T/chunk_size) |

### Classical Hydra (Hwang et al., 2024)

```python
# Bidirectional with Quasi-Separable
QS(X) = shift(SS(X)) + flip(shift(SS(flip(X)))) + DX

# Semi-Separable matrix (like Mamba's A)
SS = structured state-space transformation

# Final output
Y = W · QS(X)
```

**Key Properties:**
- Bidirectional processing
- Quasiseparable matrix structure
- Shift and flip operations

### QuantumHydraGated Differences

| Aspect | Classical Hydra | QuantumHydraGated |
|--------|-----------------|-------------------|
| **Branches** | shift+SS, flip+shift+SS+flip, D | Forward gated, Backward gated, Global |
| **Combination** | Element-wise addition | Complex coefficient superposition |
| **SS Matrix** | Quasiseparable | Quantum circuit (QLCU) |
| **Bidirectionality** | Flip operation | Separate backward branch with reversed input |

### Long Sequence Processing Comparison

| Model | Complexity | Memory | Forgetting | Dependencies |
|-------|------------|--------|------------|--------------|
| Classical Mamba | O(n log n) | O(n) | Input-dependent A | Long-range |
| Classical Hydra | O(n log n) | O(n) | Implicit | Bidirectional |
| **QuantumMambaGated** | O(n × Q) | O(n/c) | LSTM gates | Chunked recurrence |
| **QuantumHydraGated** | O(n × Q) | O(n/c) | LSTM gates | Bidirectional chunked |

Where:
- Q = quantum circuit evaluation cost
- c = chunk_size

---

## 4. Quantum Components: Superposition and Entanglement

### Quantum Feature Extractor Circuit

```
     ┌──────────── Layer 1 ────────────┐   ┌──── Layer 2 ────┐
     │                                 │   │                 │
|0⟩──┤ RY(θ₁) ─ CRX(θ₅) ─ RY(θ₉) ─ CRX ├─┬─┤ ...             ├── ⟨X⟩, ⟨Y⟩, ⟨Z⟩
|0⟩──┤ RY(θ₂) ─ CRX(θ₆) ─ RY(θ₁₀)─ CRX ├─┤─┤                 ├── ⟨X⟩, ⟨Y⟩, ⟨Z⟩
|0⟩──┤ RY(θ₃) ─ CRX(θ₇) ─ RY(θ₁₁)─ CRX ├─┤─┤                 ├── ⟨X⟩, ⟨Y⟩, ⟨Z⟩
...  │                                 │   │                 │
```

**Circuit Structure per Layer:**
1. **RY rotations** on each qubit (single-qubit gates)
2. **CRX entangling** gates in ring topology (forward direction)
3. **RY rotations** again
4. **CRX entangling** gates in reverse direction

**Parameters per layer:** 4 × n_qubits

### How Superposition is Used

**Measurement-Based Superposition:**

Instead of maintaining true quantum superposition until final measurement (which breaks gradients), we use:

```python
# Get classical measurements from each quantum branch
m1 = branch1(params1)  # ⟨ψ₁|O|ψ₁⟩
m2 = branch2(params2)  # ⟨ψ₂|O|ψ₂⟩
m3 = branch3(params3)  # ⟨ψ₃|O|ψ₃⟩

# Combine in complex space (inspired by quantum superposition)
combined = α*m1 + β*m2 + γ*m3
output = |combined|
```

**Why this approach:**
- Maintains gradient flow (backprop through measurements)
- Trainable complex coefficients mimic superposition weights
- Magnitude extraction analogous to measurement probability

**Quantum Superposition Analogy:**
```
True superposition:     |ψ⟩ = α|ψ₁⟩ + β|ψ₂⟩ + γ|ψ₃⟩
                        Measurement: ⟨ψ|O|ψ⟩

Our approach:           m = α⟨ψ₁|O|ψ₁⟩ + β⟨ψ₂|O|ψ₂⟩ + γ⟨ψ₃|O|ψ₃⟩
                        Output: |m|
```

### How Entanglement is Used

**Entanglement via CRX Gates:**

```python
# Ring entanglement (forward)
for i in range(n_qubits):
    qml.CRX(angle, wires=[i, (i+1) % n_qubits])

# Ring entanglement (backward)
for i in range(n_qubits-1, -1, -1):
    qml.CRX(angle, wires=[i, (i-1) % n_qubits])
```

**Purpose of Entanglement:**
1. **Correlation capture**: CRX creates correlations between adjacent qubits
2. **Information mixing**: Data encoded in different qubits interacts
3. **Expressivity**: Entangled states access larger regions of Hilbert space
4. **Non-classical features**: May capture patterns classical models cannot

**Entanglement Structure:**
```
Qubit 0 ←→ Qubit 1 ←→ Qubit 2 ←→ ... ←→ Qubit n-1
    ↑                                        ↓
    └────────────────────────────────────────┘
                  (Ring topology)
```

### Observables and Measurements

```python
observables = [qml.PauliX(i) for i in range(n_qubits)] + \
              [qml.PauliY(i) for i in range(n_qubits)] + \
              [qml.PauliZ(i) for i in range(n_qubits)]
```

**Output dimension:** 3 × n_qubits

**Physical interpretation:**
- **PauliX**: Measures superposition in X basis (|+⟩, |−⟩)
- **PauliY**: Measures superposition with phase
- **PauliZ**: Measures computational basis (|0⟩, |1⟩)

Together, these provide a complete characterization of the quantum state's local properties.

---

## 5. Advantages and Limitations

### Advantages

#### 1. Expressivity from Quantum Circuits

- **Hilbert space dimension**: 2^n_qubits (exponential in qubits)
- **Entanglement**: Captures non-local correlations
- **Complex amplitudes**: Richer representation than real vectors

```
6 qubits → 64-dimensional Hilbert space
But only 48 parameters (4 × 6 × 2 layers)
```

#### 2. Selective Memory via Gating

- **Forget gate**: Learns to discard irrelevant past information
- **Input gate**: Controls new information integration
- **Output gate**: Regulates state output

```python
# Example: Long sequence with important ending
Early tokens: forget_gate ≈ 0.1 (forget)
Final tokens: forget_gate ≈ 0.9 (remember)
```

#### 3. Efficient Long Sequence Handling

- **Chunked processing**: O(T/chunk_size) sequential steps instead of O(T)
- **Parallel within chunk**: All timesteps in chunk processed simultaneously
- **Memory efficient**: State compressed between chunks

#### 4. Bidirectional Context (HydraGated)

- **Forward branch**: Causal dependencies
- **Backward branch**: Future context
- **Global branch**: Overall sequence summary

#### 5. Learnable Superposition Weights

- **Complex coefficients**: α, β, γ ∈ ℂ
- **Adaptive combination**: Model learns optimal branch weighting
- **Interference-like effects**: Phase relationships matter

### Limitations

#### 1. Simulation Overhead

```python
# Quantum circuit simulation cost
n_qubits = 6
state_dim = 2^6 = 64
matrix_ops = O(64 × 64) per gate
```

**On classical hardware**: Exponential scaling with n_qubits
**Practical limit**: ~12-16 qubits for simulation

#### 2. No True Quantum Advantage (Yet)

- **Simulated on classical computers**: No exponential speedup
- **Hybrid approach**: Quantum-inspired, not quantum-native
- **Measurement-based superposition**: Loses some quantum properties

#### 3. Chunking Approximation

- **Temporal resolution**: Information within chunk is averaged
- **Boundary effects**: Sharp transitions at chunk boundaries
- **Trade-off**: Larger chunks = faster but less granular

#### 4. Training Complexity

- **Barren plateaus**: Gradient vanishing in quantum circuits
- **Complex optimization**: α, β, γ ∈ ℂ harder than real weights
- **Hyperparameter sensitivity**: chunk_size, n_qubits, qlcu_layers

#### 5. Hardware Constraints

- **NISQ noise**: Real quantum hardware has decoherence
- **Gate errors**: Imperfect quantum operations
- **Connectivity**: Limited qubit connections on real devices

### Comparison Summary

| Aspect | Advantage | Limitation |
|--------|-----------|------------|
| **Expressivity** | Exponential Hilbert space | Simulation cost |
| **Memory** | Selective gating | Chunking approximation |
| **Speed** | Parallel within chunks | Sequential across chunks |
| **Learning** | Gradient-based | Barren plateaus possible |
| **Hardware** | Works on classical | No quantum speedup yet |

---

## 6. Technical Specifications

### Hyperparameters

| Parameter | Typical Range | Description |
|-----------|---------------|-------------|
| `n_qubits` | 4-8 | Number of qubits in circuit |
| `qlcu_layers` | 1-3 | Depth of quantum circuit |
| `hidden_dim` | 32-128 | Hidden state dimension |
| `chunk_size` | 8-64 | Timesteps per chunk |
| `dropout` | 0.1-0.3 | Regularization |

### Parameter Counts

```python
# QuantumMambaGated
params = feature_proj(feature_dim²) +
         3 × branch_proj(feature_dim × n_params) +
         3 × quantum_base_params(n_params) +
         chunk_agg(3*n_qubits × hidden_dim) +
         gates(4 × 2*hidden_dim × hidden_dim) +
         output_layer(2 × hidden_dim × hidden_dim)

# Example: feature_dim=64, n_qubits=6, qlcu_layers=2, hidden_dim=64
# ≈ 52,000 parameters

# QuantumHydraGated: ~2× more due to bidirectional branches
# ≈ 102,000 parameters
```

### Memory Requirements

```python
# Per forward pass
state_memory = batch_size × n_chunks × hidden_dim
quantum_memory = batch_size × chunk_size × 3 × n_qubits
gate_memory = batch_size × hidden_dim × 4

# Example: batch=32, seq_len=500, chunk_size=32, hidden_dim=64
# n_chunks = 16
# Total ≈ 32 × 16 × 64 + 32 × 32 × 18 + 32 × 64 × 4
#       ≈ 32KB + 18KB + 8KB ≈ 58KB per sample
```

---

## 7. Experimental Results

### EEG Classification (64 channels, 249 timesteps)

| Model | Accuracy | AUC | Parameters |
|-------|----------|-----|------------|
| QuantumMambaGated | 69.4% | 77.9% | 52K |
| QuantumHydraGated | 70.2% | 78.7% | 103K |

**Improvement over non-gated versions**: From ~50% (random) to ~70%

### Genomic Classification (4 channels, 500-4776 bp)

**Human Enhancers Cohn (500 bp):**

| Model | Accuracy | AUC |
|-------|----------|-----|
| QuantumMambaGated | 72.3% | 74.6% |
| QuantumHydraGated | 72.2% | 75.0% |

**Mouse Enhancers (4776 bp):**

| Model | Accuracy | AUC | Note |
|-------|----------|-----|------|
| QuantumMambaGated | 50.0% | 49.2% | Not learning |
| QuantumHydraGated | 76%+ | 80%+ | Still training |

**Observation**: HydraGated's bidirectional processing helps with longer sequences.

### DNA Classification (57 nucleotides)

| Model | Accuracy | AUC |
|-------|----------|-----|
| QuantumMambaGated | 74.4% | 86.5% |
| QuantumHydraGated | 86.7% | 95.5% |

---

## References

1. **Gu, A., & Dao, T. (2024)**. Mamba: Linear-Time Sequence Modeling with Selective State Spaces. *arXiv:2312.00752*

2. **Hwang, W., et al. (2024)**. Hydra: Bidirectional State Space Models Through Generalized Matrix Mixers. *arXiv:2407.09941*

3. **PennyLane Documentation**. Quantum Machine Learning. https://pennylane.ai/

---

**Implementation**: `/pscratch/sd/j/junghoon/QuantumGatedRecurrence.py`
**Status**: Production-ready for research
**Last Updated**: November 24, 2025
