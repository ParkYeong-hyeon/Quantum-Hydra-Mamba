# Quantum Hydra: Two Approaches to Quantum State-Space Models

This document provides a comprehensive comparison of two quantum implementations of the Hydra state-space model, inspired by classical deep learning architectures (Hwang et al., 2024).

## Table of Contents
1. [Background: Classical Hydra](#background-classical-hydra)
2. [Option A: Quantum Superposition](#option-a-quantum-superposition)
3. [Option B: Hybrid Classical-Quantum](#option-b-hybrid-classical-quantum)
4. [Mathematical Comparison](#mathematical-comparison)
5. [Implementation Comparison](#implementation-comparison)
6. [Usage Examples](#usage-examples)
7. [When to Use Which](#when-to-use-which)
8. [References](#references)

---

## Background: Classical Hydra

Classical Hydra (Hwang et al., 2024) is a state-space model with the following operations:

### **Classical Hydra Equation**

```
QS(X) = shift(SS(X)) + flip(shift(SS(flip(X)))) + DX
Y = W · QS(X)
```

Where:
- `X ∈ ℝ^(L×d)`: Input sequence (L = length, d = dimension)
- `SS`: Semi-separable matrix operation
- `shift`: Circular shift operation
- `flip`: Sequence reversal operation
- `D`: Diagonal matrix
- `W`: Weight matrix
- `+`: Classical vector addition

**Key Property**: Classical Hydra combines three branches via **element-wise addition** of vectors.

---

## Option A: Quantum Superposition

### **Mathematical Formulation**

Option A creates a quantum superposition of three branches:

```
|ψ⟩ = α|ψ₁⟩ + β|ψ₂⟩ + γ|ψ₃⟩
```

Where each branch is:

```
|ψ₁⟩ = Qshift(QLCU|X⟩)
|ψ₂⟩ = Qflip(Qshift(QLCU(Qflip|X⟩)))
|ψ₃⟩ = QD|X⟩
```

And:
- `α, β, γ ∈ ℂ`: Trainable complex coefficients
- `QLCU`: Quantum Linear Combination of Unitaries (simulates SS matrix)
- `Qshift`: Quantum cyclic shift via SWAP gates
- `Qflip`: Quantum reversal via SWAP gates
- `QD`: Quantum diagonal operation (single-qubit rotations)

**Normalization**:
```
|ψ_norm⟩ = |ψ⟩ / ||ψ||
```

**Measurement**:
```
Y_i = ⟨ψ_norm|M_i|ψ_norm⟩  for i = 1, 2, ..., 3n
```

Where `M_i ∈ {X_j, Y_j, Z_j}` are Pauli observables on qubits j = 0, 1, ..., n-1.

**Final Output**:
```
Y = W_out · [Y_1, Y_2, ..., Y_{3n}]^T
```

### **Key Properties**

**✅ Advantages:**
- **Quantum interference**: Branches can interfere constructively/destructively
- **Exponential state space**: Can represent 2^n states simultaneously
- **Potential quantum advantage**: May capture correlations classical models cannot
- **Fewer measurements**: Single measurement on combined state

**❌ Disadvantages:**
- **Different from classical**: Not equivalent to classical Hydra's addition
- **Harder to interpret**: Quantum correlations obscure individual branch contributions
- **Sensitive to decoherence**: Superposition fragile to environmental noise
- **Complex coefficients**: Requires careful optimization of α, β, γ ∈ ℂ

### **Python Implementation**

```python
from QuantumHydra import QuantumHydraTS

# Create Option A model
model_a = QuantumHydraTS(
    n_qubits=6,              # Number of qubits
    n_timesteps=160,         # EEG timesteps
    qlcu_layers=2,           # QLCU circuit depth
    feature_dim=64,          # EEG channels
    output_dim=2,            # Binary classification
    device="cuda"
)

# Forward pass
import torch
eeg_data = torch.randn(16, 64, 160)  # (batch, channels, timesteps)
predictions = model_a(eeg_data)  # (16, 2)

# Check learned complex coefficients
print(f"Alpha: {model_a.quantum_hydra.alpha.data}")   # Complex number
print(f"Beta: {model_a.quantum_hydra.beta.data}")     # Complex number
print(f"Gamma: {model_a.quantum_hydra.gamma.data}")   # Complex number
```

---

## Option B: Hybrid Classical-Quantum

### **Mathematical Formulation**

Option B computes each branch independently, then combines classically:

```
Step 1: Compute quantum branches
  |ψ₁⟩ = Qshift(QLCU|X⟩)
  |ψ₂⟩ = Qflip(Qshift(QLCU(Qflip|X⟩)))
  |ψ₃⟩ = QD|X⟩

Step 2: Measure each branch separately
  y₁ = [⟨ψ₁|M_i|ψ₁⟩]_{i=1}^{3n} ∈ ℝ^{3n}
  y₂ = [⟨ψ₂|M_i|ψ₂⟩]_{i=1}^{3n} ∈ ℝ^{3n}
  y₃ = [⟨ψ₃|M_i|ψ₃⟩]_{i=1}^{3n} ∈ ℝ^{3n}

Step 3: Classical weighted combination
  y_combined = w₁·y₁ + w₂·y₂ + w₃·y₃

Step 4: Output layer
  Y = W_out · y_combined
```

Where:
- `w₁, w₂, w₃ ∈ ℝ₊`: Trainable real-valued weights
- `+`: Classical vector addition (element-wise)

### **Key Properties**

**✅ Advantages:**
- **Faithful to classical Hydra**: Preserves classical addition semantics
- **Interpretable**: Can analyze each branch's contribution independently
- **Robust to noise**: Each branch measured separately (no interference loss)
- **Real-valued weights**: Easier optimization (w₁, w₂, w₃ ∈ ℝ)
- **Ablation-friendly**: Can disable branches to study importance

**❌ Disadvantages:**
- **No quantum interference**: Loses potential quantum advantage
- **More measurements**: Requires three separate quantum circuit executions
- **Higher computational cost**: Each branch computed independently
- **No entanglement across branches**: Branches don't share quantum information

### **Python Implementation**

```python
from QuantumHydraHybrid import QuantumHydraHybridTS

# Create Option B model
model_b = QuantumHydraHybridTS(
    n_qubits=6,
    n_timesteps=160,
    qlcu_layers=2,
    feature_dim=64,
    output_dim=2,
    device="cuda"
)

# Forward pass
eeg_data = torch.randn(16, 64, 160)
predictions = model_b(eeg_data)

# Check learned real-valued weights
print(f"w1: {model_b.quantum_hydra.w1.data.item():.4f}")
print(f"w2: {model_b.quantum_hydra.w2.data.item():.4f}")
print(f"w3: {model_b.quantum_hydra.w3.data.item():.4f}")

# Analyze branch contributions
contributions = model_b.quantum_hydra.get_branch_contributions(eeg_data[:2])
print(f"Branch 1 output shape: {contributions['branch1'].shape}")
print(f"Branch 2 output shape: {contributions['branch2'].shape}")
print(f"Branch 3 output shape: {contributions['branch3'].shape}")
print(f"Normalized weights: {contributions['weights']}")
```

---

## Mathematical Comparison

### **Core Difference**

| Aspect | Option A (Quantum) | Option B (Hybrid) |
|--------|-------------------|-------------------|
| **Branch Combination** | Quantum superposition before measurement | Classical addition after measurement |
| **Equation** | `\|ψ⟩ = α\|ψ₁⟩ + β\|ψ₂⟩ + γ\|ψ₃⟩` | `y = w₁·y₁ + w₂·y₂ + w₃·y₃` |
| **State Space** | Single combined quantum state | Three independent measured vectors |
| **Coefficients** | Complex (α, β, γ ∈ ℂ) | Real (w₁, w₂, w₃ ∈ ℝ) |
| **Measurement** | Once (after superposition) | Three times (per branch) |

---

## When to Use Which

### **Use Option A (Quantum Superposition) when:**

1. **Exploring quantum advantage**: You want to investigate if quantum interference helps
2. **Theoretical research**: Studying fundamental quantum ML capabilities
3. **Small-scale proof-of-concept**: Testing on limited qubits (4-8)
4. **Pattern recognition**: Data may benefit from quantum correlations (e.g., certain EEG patterns)
5. **Fewer measurements preferred**: Cost of quantum circuit execution is high

**Example scenarios:**
- Academic research on quantum ML theory
- Benchmark comparisons for quantum vs classical
- Exploring non-classical computation paradigms

### **Use Option B (Hybrid) when:**

1. **Faithful quantum translation**: You want quantum version of classical Hydra
2. **Interpretability matters**: Need to understand individual branch contributions
3. **Ablation studies**: Want to analyze which branches are important
4. **Production systems**: Robustness to noise is critical
5. **Classical baselines**: Comparing directly to classical Hydra algorithm

**Example scenarios:**
- Medical diagnosis systems (interpretability required)
- Comparing quantum vs classical Hydra fairly
- Understanding which Hydra operations (shift/flip/diagonal) matter most
- NISQ hardware deployment (noise robustness)

### **Quick Decision Matrix**

| Criterion | Option A | Option B |
|-----------|----------|----------|
| **Quantum advantage seeking** | ✅ Better | ❌ Limited |
| **Interpretability** | ❌ Harder | ✅ Easier |
| **Faithful to classical Hydra** | ❌ Different | ✅ Faithful |
| **Noise robustness** | ❌ Fragile | ✅ Robust |
| **Computational cost** | ✅ Lower | ❌ Higher |
| **Research novelty** | ✅ High | ⚠️ Moderate |
| **Production readiness** | ❌ Experimental | ✅ Practical |

---

## References

1. **Hwang et al. (2024)** - "Hydra: Bidirectional State Space Models Through Generalized Matrix Mixers"
   - Original classical Hydra architecture
   - https://arxiv.org/pdf/2407.09941

2. **Gu & Dao (2024)** - "Mamba: Linear-Time Sequence Modeling with Selective State Spaces"
   - Mamba state-space model
   - https://arxiv.org/html/2312.00752v2

---

**Last Updated**: November 2025
**Version**: 1.0
**Status**: Experimental (Research Code)
