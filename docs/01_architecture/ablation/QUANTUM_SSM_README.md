# QuantumMambaSSM and QuantumHydraSSM

## Overview

This document describes the **QuantumMambaSSM** and **QuantumHydraSSM** models, which combine quantum computing principles with state-of-the-art state space models (SSMs). These models replace outdated LSTM-style gating with theoretically-aligned architectures based on the original Mamba and Hydra papers.

**Key Innovation**: True integration of quantum superposition and entanglement with selective state space mechanisms for sequence modeling.

---

## Table of Contents

1. [Model Architectures](#1-model-architectures)
2. [Unidirectional vs Bidirectional Processing](#2-unidirectional-vs-bidirectional-processing)
3. [Selective Mechanisms](#3-selective-mechanisms)
4. [Quantum Components](#4-quantum-components)
5. [Comparison with Classical Counterparts](#5-comparison-with-classical-counterparts)
6. [Usage Examples](#6-usage-examples)
7. [References](#7-references)

---

## 1. Model Architectures

### 1.1 QuantumMambaSSM

**Purpose**: Unidirectional sequence modeling with quantum-enhanced feature extraction and selective state space dynamics.

#### Architecture Diagram

```
Input: x ∈ ℝ^(B × T × D)
         │
         ▼
┌─────────────────────────────────────────────────────────────┐
│                    Feature Projection                        │
│              Linear(D_in → D_model) + LayerNorm + SiLU       │
└─────────────────────────────────────────────────────────────┘
         │
         ▼
┌─────────────────────────────────────────────────────────────┐
│              Quantum Superposition Branches                  │
│                                                              │
│   x_t ──┬──► [Quantum Circuit 1] ──► |ψ₁⟩ ──┐               │
│         │                                    │               │
│         ├──► [Quantum Circuit 2] ──► |ψ₂⟩ ──┼──► |ψ⟩ = α|ψ₁⟩ + β|ψ₂⟩ + γ|ψ₃⟩
│         │                                    │               │
│         └──► [Quantum Circuit 3] ──► |ψ₃⟩ ──┘               │
│                                                              │
│   Trainable complex coefficients: α, β, γ ∈ ℂ               │
└─────────────────────────────────────────────────────────────┘
         │
         ▼
┌─────────────────────────────────────────────────────────────┐
│                    Mamba Blocks (×N layers)                  │
│                                                              │
│   ┌─────────────────────────────────────────────────────┐   │
│   │  Input Projection: D_model → 2×D_inner              │   │
│   │         │                                           │   │
│   │    ┌────┴────┐                                      │   │
│   │    ▼         ▼                                      │   │
│   │  Conv1D    Gate                                     │   │
│   │    │      (SiLU)                                    │   │
│   │    ▼         │                                      │   │
│   │  SiLU        │                                      │   │
│   │    │         │                                      │   │
│   │    ▼         │                                      │   │
│   │  Selective   │                                      │   │
│   │    SSM       │                                      │   │
│   │    │         │                                      │   │
│   │    └────┬────┘                                      │   │
│   │         ▼                                           │   │
│   │      Multiply                                       │   │
│   │         │                                           │   │
│   │         ▼                                           │   │
│   │  Output Projection: D_inner → D_model               │   │
│   └─────────────────────────────────────────────────────┘   │
│                         + Residual                           │
└─────────────────────────────────────────────────────────────┘
         │
         ▼
┌─────────────────────────────────────────────────────────────┐
│                    Output Processing                         │
│         LayerNorm → Global Pooling → MLP → Prediction        │
└─────────────────────────────────────────────────────────────┘
         │
         ▼
Output: ŷ ∈ ℝ^(B × C)  (classification logits)
```

#### Processing Flow

1. **Input Processing**:
   - Input tensor `x` of shape `(batch, features, timesteps)` or `(batch, timesteps, features)`
   - Automatically handles format conversion to `(batch, timesteps, d_model)`
   - Feature projection with LayerNorm and SiLU activation

2. **Quantum Enhancement**:
   - For each timestep (up to 32 for efficiency), features pass through three parallel quantum circuits
   - Each circuit uses variational quantum gates (RX, RY, RZ) with CNOT entanglement
   - Outputs are combined via quantum superposition: `|ψ⟩ = α|ψ₁⟩ + β|ψ₂⟩ + γ|ψ₃⟩`
   - Quantum features are averaged with original features

3. **Selective SSM Processing**:
   - Multiple Mamba blocks process the sequence unidirectionally (left-to-right)
   - Each block uses input-dependent Δ, B, C for selective information propagation
   - Residual connections preserve gradient flow

4. **Output Prediction**:
   - LayerNorm followed by global mean pooling
   - MLP head produces classification logits or regression outputs

---

### 1.2 QuantumHydraSSM

**Purpose**: Bidirectional sequence modeling with quantum-enhanced feature extraction and Hydra-style state space dynamics.

#### Architecture Diagram

```
Input: x ∈ ℝ^(B × T × D)
         │
         ▼
┌─────────────────────────────────────────────────────────────┐
│                    Feature Projection                        │
│              Linear(D_in → D_model) + LayerNorm + SiLU       │
└─────────────────────────────────────────────────────────────┘
         │
         ▼
┌─────────────────────────────────────────────────────────────┐
│           Three-Branch Quantum Processing                    │
│                                                              │
│   x_first ──► [Quantum Forward Branch]  ──► q_forward       │
│                                                              │
│   x_last  ──► [Quantum Backward Branch] ──► q_backward      │
│                                                              │
│   x_mean  ──► [Quantum Diagonal Branch] ──► q_diagonal      │
│                                                              │
│   Combined: concat([q_forward, q_backward, q_diagonal])      │
│             → Linear projection → Add to sequence            │
└─────────────────────────────────────────────────────────────┘
         │
         ▼
┌─────────────────────────────────────────────────────────────┐
│                    Hydra Blocks (×N layers)                  │
│                                                              │
│   ┌─────────────────────────────────────────────────────┐   │
│   │  Input Projection: D_model → 2×D_inner              │   │
│   │         │                                           │   │
│   │    ┌────┴────┐                                      │   │
│   │    ▼         ▼                                      │   │
│   │  Bidirectional   Gate                               │   │
│   │     SSM        (SiLU)                               │   │
│   │    │             │                                  │   │
│   │    │  ┌──────────┴──────────┐                       │   │
│   │    │  │                     │                       │   │
│   │    ▼  ▼                     ▼                       │   │
│   │  ┌───────┐  ┌───────┐  ┌───────┐                   │   │
│   │  │Forward│  │Backward│ │Diagonal│                   │   │
│   │  │  SSM  │  │  SSM   │ │       │                   │   │
│   │  └───┬───┘  └───┬───┘  └───┬───┘                   │   │
│   │      │          │          │                        │   │
│   │      ▼          ▼          ▼                        │   │
│   │      α ×        β ×        γ ×                      │   │
│   │      └────┬─────┴──────────┘                        │   │
│   │           ▼                                         │   │
│   │    y = α·forward + β·backward + γ·diagonal         │   │
│   │           │                                         │   │
│   │           ▼                                         │   │
│   │        Multiply with Gate                           │   │
│   │           │                                         │   │
│   │           ▼                                         │   │
│   │  Output Projection: D_inner → D_model               │   │
│   └─────────────────────────────────────────────────────┘   │
│                         + Residual                           │
└─────────────────────────────────────────────────────────────┘
         │
         ▼
┌─────────────────────────────────────────────────────────────┐
│              Complex Coefficient Combination                 │
│                                                              │
│   x_forward  = mean(x[:, :T/2, :])                          │
│   x_backward = mean(x[:, T/2:, :])                          │
│   x_global   = mean(x[:, :, :])                             │
│                                                              │
│   output = |α·x_forward + β·x_backward + γ·x_global|        │
│            (complex coefficients α, β, γ ∈ ℂ)               │
└─────────────────────────────────────────────────────────────┘
         │
         ▼
Output: ŷ ∈ ℝ^(B × C)  (classification logits)
```

#### Processing Flow

1. **Input Processing**:
   - Same as QuantumMambaSSM for format handling
   - Feature projection to internal dimension

2. **Quantum Enhancement**:
   - Three specialized quantum branches:
     - **Forward Branch**: Processes first timestep `x[:, 0, :]`
     - **Backward Branch**: Processes last timestep `x[:, -1, :]`
     - **Diagonal Branch**: Processes global mean `x.mean(dim=1)`
   - Quantum features are concatenated and projected back
   - Added to the entire sequence as a learned bias

3. **Bidirectional SSM Processing**:
   - Multiple Hydra blocks process the sequence bidirectionally
   - Each block contains:
     - Forward SSM: Processes sequence left-to-right
     - Backward SSM: Processes sequence right-to-left
     - Diagonal: Direct transformation (skip connection)
   - Output: `y = α·forward + β·backward + γ·diagonal`

4. **Output Prediction**:
   - Split sequence into forward/backward halves plus global
   - Combine with complex coefficients `α, β, γ ∈ ℂ`
   - Take magnitude for real output: `|α·x_f + β·x_b + γ·x_g|`
   - Linear projection to final prediction

---

## 2. Unidirectional vs Bidirectional Processing

### 2.1 QuantumMambaSSM (Unidirectional)

```
Sequence:  [x₁] → [x₂] → [x₃] → [x₄] → [x₅]
               ↘     ↘     ↘     ↘
Hidden:        h₁ → h₂ → h₃ → h₄ → h₅
                              ↓
Output:                    Uses final hidden state
```

**Characteristics**:
- Processes sequence in a single direction (left-to-right)
- Each position can only attend to previous positions
- **Causal**: Suitable for autoregressive tasks (language modeling, time-series forecasting)
- Hidden state accumulates information from past only
- Recurrence: `h_t = Ā·h_{t-1} + B̄·x_t`

**Use Cases**:
- Real-time prediction (streaming data)
- Causal sequence modeling
- Language generation
- Online learning scenarios

### 2.2 QuantumHydraSSM (Bidirectional)

```
Forward:   [x₁] → [x₂] → [x₃] → [x₄] → [x₅]
               ↘     ↘     ↘     ↘
           h₁→ → h₂→ → h₃→ → h₄→ → h₅→

Backward:  [x₁] ← [x₂] ← [x₃] ← [x₄] ← [x₅]
               ↙     ↙     ↙     ↙
           h₁← ← h₂← ← h₃← ← h₄← ← h₅←

Diagonal:  [x₁]   [x₂]   [x₃]   [x₄]   [x₅]
             ↓      ↓      ↓      ↓      ↓
           d₁     d₂     d₃     d₄     d₅

Combined:  y_t = α·h_t→ + β·h_t← + γ·d_t
```

**Characteristics**:
- Processes sequence in both directions simultaneously
- Each position has access to full context (past and future)
- **Non-causal**: Suitable for classification, understanding tasks
- Combines forward, backward, and direct (diagonal) information
- Three learnable weights (α, β, γ) balance the contributions

**Use Cases**:
- Sequence classification (sentiment, intent)
- Named entity recognition
- EEG/biosignal classification
- Tasks where full sequence is available

### 2.3 Comparison Table

| Aspect | QuantumMambaSSM | QuantumHydraSSM |
|--------|-----------------|-----------------|
| Direction | Unidirectional (L→R) | Bidirectional (L↔R + Diagonal) |
| Causality | Causal | Non-causal |
| Context | Past only | Full sequence |
| SSM Components | 1 (forward) | 3 (forward + backward + diagonal) |
| Inference | Streaming possible | Requires full sequence |
| Parameters | Lower | Higher (~2-3× SSM params) |
| Complexity | O(T) | O(T) (still linear!) |

---

## 3. Selective Mechanisms

Both models use the **Mamba-style selective mechanism**, which is the key innovation distinguishing them from classical SSMs like S4 or LSSL.

### 3.1 Classical SSM (Non-Selective)

In classical SSMs, the state transition matrices A, B, C are **fixed** (input-independent):

```
Continuous: dx/dt = Ax + Bu,  y = Cx + Du
Discrete:   h_t = Āh_{t-1} + B̄x_t,  y_t = Ch_t + Dx_t

Where Ā, B̄, C, D are CONSTANT matrices (learned but fixed after training)
```

**Problem**: Cannot selectively filter or emphasize different parts of the input sequence.

### 3.2 Selective SSM (Mamba-Style)

In the selective SSM, the matrices **depend on the input**:

```python
# Input-dependent projections
x_proj = Linear(x)  # Project input

# Extract Δ (time step), B (input matrix), C (output matrix)
Δ_raw = x_proj[:, :dt_rank]
B = x_proj[:, dt_rank:dt_rank+d_state]     # Input-dependent!
C = x_proj[:, dt_rank+d_state:]             # Input-dependent!

# Compute time step
Δ = softplus(Linear(Δ_raw))                 # Input-dependent!

# Discretization
Ā = exp(Δ × A)      # Time step controls decay rate
B̄ = Δ × B          # Time step scales input influence

# Recurrence
h_t = Ā × h_{t-1} + B̄ × x_t
y_t = C × h_t + D × x_t
```

### 3.3 What Each Parameter Controls

| Parameter | Role | Effect of Large Value | Effect of Small Value |
|-----------|------|----------------------|----------------------|
| **Δ (Delta)** | Time step / Discretization | Faster forgetting, more weight on new input | Slower forgetting, retains history |
| **B** | Input projection | Stronger influence of current input on state | Weaker input → state influence |
| **C** | Output projection | Current state more visible in output | Current state less visible |
| **A** | State transition (fixed) | Decay rate of hidden state | (Always negative for stability) |
| **D** | Skip connection | More direct input → output | Less direct connection |

### 3.4 Selectivity in Action

```
Example: Processing sequence "The cat sat on the mat"

Position:  "The"  "cat"  "sat"  "on"  "the"  "mat"
Δ values:   0.01   0.08   0.03   0.01  0.01   0.09
            (low)  (HIGH) (med)  (low) (low)  (HIGH)

Interpretation:
- "cat" and "mat" have high Δ → model "pays attention" to these nouns
- "The", "on", "the" have low Δ → model retains context but doesn't emphasize
- This is learned automatically during training!
```

### 3.5 Selectivity in QuantumHydraSSM

The Hydra model applies selectivity in **three directions**:

```python
# Forward direction selectivity
forward_output = ForwardSSM(x)   # Each with its own Δ, B, C

# Backward direction selectivity
backward_output = BackwardSSM(flip(x))  # Different Δ, B, C

# Diagonal (direct)
diagonal_output = Linear(x)  # No state, direct transformation

# Combine with learned weights
y = α × forward + β × backward + γ × diagonal
```

This allows the model to:
- Use **different selective patterns** for forward vs backward processing
- **Learn which direction** is more important for different inputs
- Maintain a **direct path** (diagonal) for information that doesn't need sequential processing

---

## 4. Quantum Components

Both models leverage quantum computing principles through the **QuantumSuperpositionBranches** module.

### 4.1 Quantum Circuit Structure

Each quantum branch uses a variational quantum circuit (VQC):

```
┌─────────────────────────────────────────────────────────────┐
│                   Variational Quantum Circuit                │
│                                                              │
│  Input: x ∈ ℝ^D                                             │
│         │                                                    │
│         ▼                                                    │
│  ┌─────────────────┐                                        │
│  │  Input Encoding │  x → θ = tanh(Linear(x)) × π          │
│  │  (Angle Embed)  │                                        │
│  └────────┬────────┘                                        │
│           │                                                  │
│           ▼                                                  │
│  ┌─────────────────┐                                        │
│  │    RY(θᵢ)       │  Each qubit gets one angle             │
│  │    on each      │  |0⟩ → cos(θ/2)|0⟩ + sin(θ/2)|1⟩      │
│  │    qubit        │                                        │
│  └────────┬────────┘                                        │
│           │                                                  │
│  ┌────────┴────────┐  Repeated for n_layers                 │
│  │                 │                                        │
│  │  ┌───────────┐  │                                        │
│  │  │ RX(φ₁)   │  │  Parameterized rotations               │
│  │  │ RY(φ₂)   │  │  (learned during training)             │
│  │  │ RZ(φ₃)   │  │                                        │
│  │  └─────┬─────┘  │                                        │
│  │        │        │                                        │
│  │  ┌─────┴─────┐  │                                        │
│  │  │   CNOT    │  │  Entanglement layer                   │
│  │  │  ladder   │  │  (creates quantum correlations)       │
│  │  └───────────┘  │                                        │
│  │                 │                                        │
│  └────────┬────────┘                                        │
│           │                                                  │
│           ▼                                                  │
│  ┌─────────────────┐                                        │
│  │   Measurement   │  ⟨Z₀⟩, ⟨Z₁⟩, ..., ⟨Zₙ₋₁⟩              │
│  │  (Expectation)  │  Returns n_qubits values in [-1, 1]   │
│  └─────────────────┘                                        │
│                                                              │
│  Output: [⟨Z₀⟩, ⟨Z₁⟩, ..., ⟨Zₙ₋₁⟩] ∈ ℝ^n_qubits            │
└─────────────────────────────────────────────────────────────┘
```

### 4.2 Quantum Superposition

The three-branch quantum system creates a **superposition of quantum states**:

```
|ψ_total⟩ = α|ψ₁⟩ + β|ψ₂⟩ + γ|ψ₃⟩

Where:
- |ψ₁⟩ = Output of quantum circuit 1 (different parameters)
- |ψ₂⟩ = Output of quantum circuit 2 (different parameters)
- |ψ₃⟩ = Output of quantum circuit 3 (different parameters)
- α, β, γ ∈ ℂ are trainable complex coefficients

Normalization: |α|² + |β|² + |γ|² = 1 (enforced in forward pass)
```

**Why Three Branches?**

This mirrors the three-component structure in both Mamba and Hydra:
- **QuantumMambaSSM**: SSM path + Gate path + Skip connection
- **QuantumHydraSSM**: Forward + Backward + Diagonal

### 4.3 Entanglement

The CNOT ladder creates **quantum entanglement** between qubits:

```python
# CNOT ladder with wrap-around
for i in range(n_qubits - 1):
    qml.CNOT(wires=[i, i + 1])
qml.CNOT(wires=[n_qubits - 1, 0])  # Circular entanglement
```

```
Before CNOT:  |ψ₀⟩ ⊗ |ψ₁⟩ ⊗ |ψ₂⟩ ⊗ |ψ₃⟩  (product state)

After CNOT:   Entangled state where measuring one qubit
              affects the probabilities of others

Example (2 qubits):
|00⟩ → |00⟩
|01⟩ → |01⟩
|10⟩ → |11⟩  (CNOT flips target when control is 1)
|11⟩ → |10⟩
```

**Benefits of Entanglement**:
1. **Expressivity**: Entangled states can represent correlations that classical vectors cannot
2. **Feature Interaction**: Non-linear interactions between input features
3. **Compact Representation**: 2^n dimensional Hilbert space with only n qubits

### 4.4 Quantum Advantage in Feature Extraction

The quantum circuits provide advantages for feature extraction:

| Aspect | Classical | Quantum |
|--------|-----------|---------|
| Feature Space | Linear (d-dimensional) | Exponential (2^n dimensional Hilbert space) |
| Interactions | Require explicit layers | Automatic via entanglement |
| Non-linearity | Activation functions | Inherent in quantum gates |
| Expressivity | Polynomial in parameters | Exponential in qubits |

### 4.5 Complex Coefficients

The complex coefficients α, β, γ enable:

```python
# Each coefficient has real and imaginary parts
α = α_real + i·α_imag
β = β_real + i·β_imag
γ = γ_real + i·γ_imag

# Phase information
α = |α| × e^(iθ_α)  # Magnitude and phase

# Normalization ensures valid quantum state
norm = sqrt(|α|² + |β|² + |γ|²)
α, β, γ = α/norm, β/norm, γ/norm
```

**Why Complex?**
- Quantum states naturally have complex amplitudes
- Phases can encode additional information
- Interference effects between branches (constructive/destructive)

---

## 5. Comparison with Classical Counterparts

### 5.1 Classical Mamba

**Paper**: Gu & Dao (2024) "Mamba: Linear-Time Sequence Modeling with Selective State Spaces"

```
Classical Mamba Architecture:
┌─────────────────────────────────────────────────────────────┐
│  Input → Linear → Conv1D → SiLU → Selective SSM → Gate     │
│                                                     ↓       │
│                                              Linear → Output│
└─────────────────────────────────────────────────────────────┘

Key Components:
1. Input projection (expansion)
2. Depthwise convolution for local context
3. Selective SSM with input-dependent Δ, B, C
4. Multiplicative gating
5. Output projection
```

### 5.2 Classical Hydra

**Paper**: Hwang et al. (2024) "Hydra: Bidirectional State Space Models"

```
Classical Hydra Architecture:
┌─────────────────────────────────────────────────────────────┐
│                                                              │
│  Input ──┬──► Forward SSM  ───► α × ──┐                     │
│          │                            │                      │
│          ├──► Backward SSM ───► β × ──┼──► Sum → Output     │
│          │                            │                      │
│          └──► Diagonal     ───► γ × ──┘                     │
│                                                              │
└─────────────────────────────────────────────────────────────┘

Key Components:
1. Forward semi-separable matrix (captures past)
2. Backward semi-separable matrix (captures future)
3. Diagonal matrix (direct transformation)
4. Learnable combination weights
```

### 5.3 Comparison Table

| Aspect | Classical Mamba | QuantumMambaSSM | Classical Hydra | QuantumHydraSSM |
|--------|-----------------|-----------------|-----------------|-----------------|
| **Direction** | Unidirectional | Unidirectional | Bidirectional | Bidirectional |
| **Feature Extraction** | Linear/Conv | Quantum circuits | Linear/Conv | Quantum circuits |
| **State Space** | Real-valued | Real-valued | Real-valued | Complex-valued |
| **Superposition** | No | Yes (3 branches) | No | Yes (3 branches) |
| **Entanglement** | No | Yes (CNOT ladder) | No | Yes (CNOT ladder) |
| **Coefficients** | Real (α, β, γ) | Complex (α, β, γ ∈ ℂ) | Real (α, β, γ) | Complex (α, β, γ ∈ ℂ) |
| **Feature Dim** | d | 2^n (Hilbert space) | d | 2^n (Hilbert space) |
| **SSM Type** | Selective | Selective | Selective | Selective |
| **Parameters** | ~2M typical | ~4-8K (quantum) + ~1M | ~4M typical | ~8-10K (quantum) + ~2M |

### 5.4 What Quantum Adds

| Classical Component | Quantum Enhancement |
|---------------------|---------------------|
| Linear projection | Variational quantum circuit with exponential expressivity |
| Feature mixing | Automatic via entanglement |
| Non-linearity | Inherent in quantum gates |
| Parallel paths | Quantum superposition with interference |
| Weight combination | Complex coefficients with phase information |

### 5.5 Preserved from Classical

Both quantum models **preserve** the core innovations from their classical counterparts:

1. **Selective Mechanism**: Input-dependent Δ, B, C
2. **Linear Complexity**: O(T) for sequence length T
3. **Gated Architecture**: Multiplicative gating for information flow
4. **Residual Connections**: Gradient-friendly deep networks
5. **HiPPO Initialization**: Structured A matrix for stable dynamics

### 5.6 Unique to Quantum Models

1. **Quantum Superposition**: Three-branch parallel processing in superposition
2. **Entanglement-based Features**: Non-classical correlations in feature space
3. **Complex Amplitudes**: Phase information in coefficients
4. **Exponential Feature Space**: 2^n dimensions with n qubits
5. **Quantum Interference**: Constructive/destructive combination of branches

---

## 6. Usage Examples

### 6.1 Basic Usage

```python
from models.QuantumSSM import QuantumMambaSSM, QuantumHydraSSM
import torch

# Device setup
device = "cuda" if torch.cuda.is_available() else "cpu"

# QuantumMambaSSM for unidirectional tasks
model_mamba = QuantumMambaSSM(
    n_qubits=4,           # Number of qubits per branch
    n_timesteps=100,      # Expected sequence length
    qlcu_layers=2,        # Quantum circuit depth
    feature_dim=64,       # Input feature dimension
    d_model=128,          # Internal model dimension
    d_state=16,           # SSM state dimension
    n_layers=2,           # Number of Mamba blocks
    output_dim=10,        # Classification classes
    dropout=0.1,
    device=device
)

# QuantumHydraSSM for bidirectional tasks
model_hydra = QuantumHydraSSM(
    n_qubits=4,
    n_timesteps=100,
    qlcu_layers=2,
    feature_dim=64,
    d_model=128,
    d_state=16,
    n_layers=2,
    output_dim=10,
    dropout=0.1,
    device=device
)

# Forward pass
x = torch.randn(32, 64, 100, device=device)  # (batch, features, timesteps)
output_mamba = model_mamba(x)  # (32, 10)
output_hydra = model_hydra(x)  # (32, 10)
```

### 6.2 Training Example

```python
import torch.nn as nn
import torch.optim as optim

# Setup
model = QuantumMambaSSM(n_qubits=4, n_timesteps=100, feature_dim=64,
                        d_model=128, output_dim=10, device="cuda")
criterion = nn.CrossEntropyLoss()
optimizer = optim.AdamW(model.parameters(), lr=1e-3)

# Training loop
for epoch in range(100):
    model.train()
    for batch_x, batch_y in dataloader:
        batch_x = batch_x.to("cuda")
        batch_y = batch_y.to("cuda")

        optimizer.zero_grad()
        outputs = model(batch_x)
        loss = criterion(outputs, batch_y)
        loss.backward()

        # Gradient clipping (recommended for quantum models)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

        optimizer.step()

    print(f"Epoch {epoch+1}, Loss: {loss.item():.4f}")
```

### 6.3 Choosing Between Models

```python
# Use QuantumMambaSSM when:
# - Causal prediction is required (e.g., forecasting)
# - Streaming/online processing
# - Autoregressive generation
# - Lower parameter budget

# Use QuantumHydraSSM when:
# - Full sequence is available
# - Classification/understanding tasks
# - Bidirectional context is important
# - Higher accuracy is priority over speed
```

---

## 7. References

### Primary Papers

1. **Mamba**: Gu, A., & Dao, T. (2024). "Mamba: Linear-Time Sequence Modeling with Selective State Spaces." *arXiv:2312.00752*

2. **Hydra**: Hwang, I., et al. (2024). "Hydra: Bidirectional State Space Models Through Generalized Matrix Mixers." *arXiv*

3. **S4**: Gu, A., et al. (2022). "Efficiently Modeling Long Sequences with Structured State Spaces." *ICLR 2022*

### Quantum Computing

4. **VQC**: Benedetti, M., et al. (2019). "Parameterized quantum circuits as machine learning models." *Quantum Science and Technology*

5. **PennyLane**: Bergholm, V., et al. (2018). "PennyLane: Automatic differentiation of hybrid quantum-classical computations." *arXiv:1811.04968*

### Implementation

- **File Location**: `/pscratch/sd/j/junghoon/quantum_hydra_mamba/models/QuantumSSM.py`
- **Dependencies**: PyTorch, PennyLane, pennylane-lightning-gpu (for CUDA)
- **Tested On**: NVIDIA A100-PCIE-40GB

---

## Appendix: Mathematical Details

### A.1 SSM Discretization

Continuous SSM:
```
dx/dt = Ax + Bu
y = Cx + Du
```

Zero-Order Hold (ZOH) Discretization:
```
Ā = exp(ΔA)
B̄ = (ΔA)⁻¹(exp(ΔA) - I) · ΔB ≈ ΔB  (first-order approximation)
```

Discrete recurrence:
```
h_t = Ā·h_{t-1} + B̄·x_t
y_t = C·h_t + D·x_t
```

### A.2 Quantum State Normalization

For complex coefficients α, β, γ:
```
Normalization: |α|² + |β|² + |γ|² = 1

Implementation:
norm = sqrt(|α|² + |β|² + |γ|² + ε)
α_normalized = α / norm
β_normalized = β / norm
γ_normalized = γ / norm
```

### A.3 Measurement Statistics

For a Pauli-Z measurement on qubit i:
```
⟨Zᵢ⟩ = ⟨ψ|Zᵢ|ψ⟩ = P(|0⟩) - P(|1⟩) ∈ [-1, 1]
```

---

## 8. Comprehensive Six-Model Comparison

This section provides a detailed comparison of all six models in the quantum-classical Mamba/Hydra family.

### 8.1 Models Overview

| # | Model | Type | Gating Mechanism | Direction |
|---|-------|------|------------------|-----------|
| 1 | **QuantumMambaSSM** | Quantum | Mamba-style Selective SSM | Unidirectional |
| 2 | **QuantumHydraSSM** | Quantum | Hydra-style Bidirectional SSM | Bidirectional |
| 3 | **QuantumMambaGated** | Quantum | LSTM-style Gates | Unidirectional |
| 4 | **QuantumHydraGated** | Quantum | LSTM-style Gates | Bidirectional |
| 5 | **ClassicalMamba** | Classical | Mamba-style Selective SSM | Unidirectional |
| 6 | **ClassicalHydra** | Classical | Hydra-style Bidirectional SSM | Bidirectional |

### 8.2 Parameter Count Comparison

Test configuration:
- `n_qubits=4`, `qlcu_layers=2`
- `n_timesteps=100`, `feature_dim=64`
- `d_model=128`, `d_state=16`, `n_layers=2`
- `output_dim=10`

| Model | Parameters | Relative Size | Notes |
|-------|------------|---------------|-------|
| **QuantumMambaGated** | 49,296 | 1.0× (baseline) | Smallest due to chunked processing |
| **QuantumHydraGated** | 92,639 | 1.9× | 3 branches × quantum circuits |
| **ClassicalMamba** | 242,954 | 4.9× | Full Mamba block structure |
| **ClassicalHydra** | 240,842 | 4.9× | Forward + Backward + Diagonal SSMs |
| **QuantumMambaSSM** | 402,684 | 8.2× | Quantum + full Mamba blocks |
| **QuantumHydraSSM** | 834,377 | 16.9× | 3× quantum branches + Hydra blocks |

**Key Observations**:
- Gated models (LSTM-style) have fewer parameters due to simpler recurrence
- SSM models have more parameters due to input-dependent projections (Δ, B, C)
- Quantum models add ~50-100K parameters for quantum circuits
- Hydra variants have ~2× parameters of Mamba due to bidirectional processing

### 8.3 Computational Complexity

| Model | Time Complexity | Dominant Operation |
|-------|-----------------|-------------------|
| **QuantumMambaSSM** | O(T × n_qubits × 2^n_qubits) | Quantum circuit simulation |
| **QuantumHydraSSM** | O(T × n_qubits × 2^n_qubits) | Quantum circuit simulation |
| **QuantumMambaGated** | O(T/C × n_qubits × 2^n_qubits) | Quantum circuit (chunked) |
| **QuantumHydraGated** | O(T/C × n_qubits × 2^n_qubits) | Quantum circuit (chunked) |
| **ClassicalMamba** | O(T × d_model × d_state) | SSM recurrence |
| **ClassicalHydra** | O(T × d_model × d_state) | SSM recurrence (×2 directions) |

Where:
- T = sequence length
- C = chunk size (typically 16)
- n_qubits = number of qubits (typically 4-8)
- d_model = model dimension (typically 128)
- d_state = SSM state dimension (typically 16)

**Key Observations**:
- All models have **linear complexity** in sequence length T (no quadratic attention!)
- Quantum models are dominated by circuit simulation (exponential in qubits, but qubits are small)
- Chunked models (Gated) process T/C chunks instead of T timesteps
- Classical models are fastest per-step due to no quantum overhead

### 8.4 Memory Efficiency

| Model | Memory Complexity | Peak Memory Usage |
|-------|-------------------|-------------------|
| **QuantumMambaSSM** | O(B × T × d_model) | High (full sequence + quantum states) |
| **QuantumHydraSSM** | O(B × T × d_model) | Highest (bidirectional + 3× quantum) |
| **QuantumMambaGated** | O(B × C × d_model) | Low (only chunk in memory) |
| **QuantumHydraGated** | O(B × C × d_model) | Low (chunked processing) |
| **ClassicalMamba** | O(B × T × d_model) | Medium (full sequence) |
| **ClassicalHydra** | O(B × T × d_model) | Medium-High (stores both directions) |

Where B = batch size, T = sequence length, C = chunk size

**Key Observations**:
- **Gated models are most memory-efficient** due to chunked processing
- SSM models store full sequence for gradient computation
- Hydra variants require ~2× memory for bidirectional states
- Quantum circuit simulation adds minimal memory overhead (states are measured immediately)

### 8.5 Long Sequence Learning Capability

| Model | Long Sequence Capability | Mechanism | Theoretical Basis |
|-------|-------------------------|-----------|-------------------|
| **QuantumMambaSSM** | ⭐⭐⭐⭐⭐ Excellent | Selective SSM + Quantum | Input-dependent Δ controls memory horizon |
| **QuantumHydraSSM** | ⭐⭐⭐⭐⭐ Excellent | Bidirectional SSM + Quantum | Full context from both directions |
| **QuantumMambaGated** | ⭐⭐⭐ Good | LSTM gates + Chunking | Forget gate controls memory, but limited by LSTM dynamics |
| **QuantumHydraGated** | ⭐⭐⭐ Good | LSTM gates + Bidirectional | Same LSTM limitations |
| **ClassicalMamba** | ⭐⭐⭐⭐⭐ Excellent | Selective SSM | HiPPO theory ensures long-range dependencies |
| **ClassicalHydra** | ⭐⭐⭐⭐⭐ Excellent | Bidirectional SSM | Combines forward/backward memory |

**Why SSM > LSTM for Long Sequences**:

1. **HiPPO Initialization**: SSM models use structured A matrix that theoretically preserves history
   ```
   A_ii = -(i+1)  →  Polynomial basis for memory compression
   ```

2. **Selective Mechanism**: Input-dependent Δ allows adaptive memory:
   - Large Δ: "Remember this!" (important token)
   - Small Δ: "Keep context" (less important token)

3. **LSTM Limitations**:
   - Fixed forget gate dynamics
   - Vanishing gradients over very long sequences
   - No theoretical guarantees for memory preservation

### 8.6 Detailed Comparison Matrix

| Aspect | QuantumMambaSSM | QuantumHydraSSM | QuantumMambaGated | QuantumHydraGated | ClassicalMamba | ClassicalHydra |
|--------|-----------------|-----------------|-------------------|-------------------|----------------|----------------|
| **Direction** | Unidirectional | Bidirectional | Unidirectional | Bidirectional | Unidirectional | Bidirectional |
| **Gating** | Selective SSM | Selective SSM | LSTM-style | LSTM-style | Selective SSM | Selective SSM |
| **Quantum** | ✅ Yes | ✅ Yes | ✅ Yes | ✅ Yes | ❌ No | ❌ No |
| **Superposition** | ✅ 3-branch | ✅ 3-branch | ✅ 3-branch | ✅ 3-branch | ❌ No | ❌ No |
| **Entanglement** | ✅ CNOT | ✅ CNOT | ✅ CNOT | ✅ CNOT | ❌ No | ❌ No |
| **Complex Coeff** | ✅ α,β,γ∈ℂ | ✅ α,β,γ∈ℂ | ✅ α,β,γ∈ℂ | ✅ α,β,γ∈ℂ | ❌ Real | ❌ Real |
| **Parameters** | ~400K | ~830K | ~49K | ~93K | ~243K | ~241K |
| **Memory** | High | Highest | Low | Low | Medium | Medium |
| **Speed** | Slow | Slowest | Medium | Medium | Fast | Fast |
| **Long Seq** | Excellent | Excellent | Good | Good | Excellent | Excellent |
| **Streaming** | ✅ Yes | ❌ No | ✅ Yes | ❌ No | ✅ Yes | ❌ No |

### 8.7 Advantages and Disadvantages

#### QuantumMambaSSM
| Advantages | Disadvantages |
|------------|---------------|
| ✅ Best theoretical alignment with Mamba | ❌ Highest computational cost |
| ✅ Selective mechanism for adaptive memory | ❌ Requires quantum simulation |
| ✅ Quantum superposition for expressivity | ❌ Large parameter count |
| ✅ Supports streaming/causal inference | ❌ Slower training than classical |
| ✅ Entanglement for feature interaction | |

#### QuantumHydraSSM
| Advantages | Disadvantages |
|------------|---------------|
| ✅ Full bidirectional context | ❌ Largest model (most parameters) |
| ✅ Best for classification tasks | ❌ Cannot stream (needs full sequence) |
| ✅ Quantum + Hydra theoretical alignment | ❌ Slowest inference |
| ✅ Complex coefficients capture phase | ❌ Highest memory usage |
| ✅ Forward + Backward + Diagonal structure | |

#### QuantumMambaGated
| Advantages | Disadvantages |
|------------|---------------|
| ✅ Smallest model (fewest parameters) | ❌ LSTM gating is theoretically outdated |
| ✅ Memory-efficient chunked processing | ❌ May struggle with very long sequences |
| ✅ Fastest among quantum models | ❌ Not aligned with Mamba theory |
| ✅ Good for resource-constrained settings | ❌ Fixed gate dynamics (not selective) |
| ✅ Quantum superposition benefits | |

#### QuantumHydraGated
| Advantages | Disadvantages |
|------------|---------------|
| ✅ Bidirectional with low memory | ❌ LSTM gating limitations |
| ✅ Good parameter efficiency | ❌ Not aligned with Hydra theory |
| ✅ Chunked processing | ❌ May miss long-range dependencies |
| ✅ Quantum enhancement | ❌ Cannot stream |
| ✅ Three-branch quantum structure | |

#### ClassicalMamba
| Advantages | Disadvantages |
|------------|---------------|
| ✅ Fastest inference | ❌ No quantum enhancement |
| ✅ Selective SSM (theoretically sound) | ❌ Linear feature space only |
| ✅ Low memory usage | ❌ No entanglement benefits |
| ✅ Proven long-sequence capability | ❌ Cannot capture quantum correlations |
| ✅ Supports streaming | |

#### ClassicalHydra
| Advantages | Disadvantages |
|------------|---------------|
| ✅ Full bidirectional context | ❌ No quantum enhancement |
| ✅ Fast inference (no quantum) | ❌ Cannot stream |
| ✅ Selective mechanism in both directions | ❌ Linear feature space |
| ✅ Theoretically sound | ❌ More parameters than Mamba |
| ✅ Good for classification | |

### 8.8 When to Use Each Model

```
Decision Tree for Model Selection:

1. Is quantum hardware/simulation available?
   ├── NO → Use ClassicalMamba (streaming) or ClassicalHydra (classification)
   └── YES → Continue to 2

2. Is full sequence available at inference?
   ├── NO (streaming) → Use QuantumMambaSSM or QuantumMambaGated
   └── YES → Continue to 3

3. Is memory a constraint?
   ├── YES → Use QuantumMambaGated or QuantumHydraGated (chunked)
   └── NO → Continue to 4

4. Is theoretical alignment important?
   ├── YES → Use QuantumMambaSSM or QuantumHydraSSM
   └── NO → Use Gated variants for faster training

5. Unidirectional or Bidirectional?
   ├── Unidirectional → QuantumMambaSSM
   └── Bidirectional → QuantumHydraSSM
```

### 8.9 Recommended Use Cases

| Use Case | Recommended Model | Reason |
|----------|-------------------|--------|
| **Real-time EEG processing** | QuantumMambaGated | Streaming + memory efficient |
| **EEG classification** | QuantumHydraSSM | Full context + quantum enhancement |
| **Language modeling** | ClassicalMamba | Fast + streaming + proven |
| **Sentiment analysis** | QuantumHydraSSM | Bidirectional + quantum features |
| **Time-series forecasting** | QuantumMambaSSM | Causal + selective + quantum |
| **Resource-constrained** | QuantumMambaGated | Smallest model |
| **Maximum accuracy** | QuantumHydraSSM | Most expressive |
| **Production deployment** | ClassicalMamba/Hydra | Fastest inference |
| **Research (quantum advantage)** | QuantumMambaSSM/HydraSSM | Theoretically aligned |

### 8.10 Performance Summary

Based on theoretical analysis and architectural properties:

| Metric | Best Model | Runner-up |
|--------|------------|-----------|
| **Fewest Parameters** | QuantumMambaGated (49K) | QuantumHydraGated (93K) |
| **Fastest Inference** | ClassicalMamba | ClassicalHydra |
| **Lowest Memory** | QuantumMambaGated | QuantumHydraGated |
| **Best Long-Sequence** | QuantumMambaSSM / ClassicalMamba | QuantumHydraSSM / ClassicalHydra |
| **Best Expressivity** | QuantumHydraSSM | QuantumMambaSSM |
| **Best Streaming** | QuantumMambaSSM | ClassicalMamba |
| **Best Classification** | QuantumHydraSSM | ClassicalHydra |
| **Best Theory Alignment** | QuantumMambaSSM + QuantumHydraSSM | ClassicalMamba + ClassicalHydra |

---

**Author**: Junghoon Park
**Date**: December 2024
**Version**: 1.1
