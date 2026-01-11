# Ablation Study Plan V3: 2×2×3 Factorial Design

## Systematic Evaluation of Quantum vs Classical Components in Sequence Models

This document presents a refined ablation study design using a **2×2×3 factorial structure** to systematically evaluate where quantum computing provides advantages in sequence modeling.

---

## Table of Contents

1. [Overview](#1-overview)
2. [Factorial Design](#2-factorial-design)
3. [Complete Model Matrix](#3-complete-model-matrix)
4. [Model Architectures](#4-model-architectures)
5. [Research Questions](#5-research-questions)
6. [Implementation Status](#6-implementation-status)
7. [Experimental Protocol](#7-experimental-protocol)
8. [Expected Outcomes](#8-expected-outcomes)
9. [Paper Structure](#9-paper-structure-icmlneurips-format)

---

## 1. Overview

### 1.1 Design Philosophy

We systematically vary **three factors** to isolate the contribution of quantum computing:

| Factor | Levels | Description |
|--------|--------|-------------|
| **Feature Extraction** | Quantum, Classical | How input data is encoded |
| **Mixing Mechanism** | Quantum, Classical | How sequence information is combined |
| **Mixing Type** | Transformer, Mamba, Hydra | Architecture style for mixing |

This creates **2 × 2 × 3 = 12 base model combinations** plus **4 additional superposition variants** for a total of **16 models**.

### 1.2 Key Research Questions

By independently varying feature extraction and mixing, we answer **three core questions**:

| Question | Core Inquiry | Tested Across |
|----------|--------------|---------------|
| **Q1** | Where should quantum be applied? | Multiple datasets + sequence lengths |
| **Q2** | Which mixing mechanism benefits most from quantum? | Multiple datasets + sequence lengths |
| **Q3** | Does end-to-end quantum provide synergistic benefits? | Multiple datasets + sequence lengths |

**Sequence length** is not a separate question—it's a **moderating variable** that tests the robustness and boundaries of Q1, Q2, Q3.

### 1.3 Experimental Design for ICML/NeurIPS Rigor

To ensure findings generalize and satisfy top-tier venue reviewers:

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                    MULTI-DATASET + MULTI-CONDITION DESIGN                    │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│   DATASETS (Cross-Domain Validation):                                       │
│   ├── PhysioNet EEG (Biomedical signals)                                    │
│   ├── Genomic Benchmarks (DNA sequences)                                    │
│   ├── SST-2 GLUE (Natural language)                                         │
│   └── Forrelation (Theoretical quantum advantage)                           │
│                                                                             │
│   CONDITIONS (Sequence Length - PhysioNet only):                            │
│   ├── 40 Hz  → 124 timesteps (Short)                                        │
│   ├── 80 Hz  → 248 timesteps (Medium)                                       │
│   └── 160 Hz → 496 timesteps (Long)                                         │
│                                                                             │
│   MODELS: 12 (2×2×3 factorial)                                              │
│   SEEDS: 5 per configuration                                                │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

### 1.4 Architecture Diagram

```
┌─────────────────────────────────────────────────────────────────────────────────┐
│                              INPUT DATA                                          │
│                    (EEG, Genomic, NLP, Synthetic)                               │
└─────────────────────────────────────┬───────────────────────────────────────────┘
                                      │
                    ┌─────────────────┴─────────────────┐
                    │                                   │
                    ▼                                   ▼
        ┌───────────────────────┐           ┌───────────────────────┐
        │   QUANTUM FEATURE     │           │   CLASSICAL FEATURE   │
        │     EXTRACTION        │           │     EXTRACTION        │
        │                       │           │                       │
        │  • VQC encoding       │           │  • Conv2d + MaxPool   │
        │  • Angle embedding    │           │  • GLU gating         │
        │  • Quantum measurement│           │  • Sigmoid scaling    │
        │                       │           │                       │
        │  Output: features     │           │  Output: angles [0,π] │
        └───────────┬───────────┘           └───────────┬───────────┘
                    │                                   │
                    └─────────────────┬─────────────────┘
                                      │
                    ┌─────────────────┴─────────────────┐
                    │                                   │
                    ▼                                   ▼
        ┌───────────────────────┐           ┌───────────────────────┐
        │   CLASSICAL MIXING    │           │    QUANTUM MIXING     │
        │                       │           │                       │
        │  • Transformer:       │           │  • Transformer:       │
        │    Self-attention     │           │    QSVT + LCU Attn    │
        │                       │           │                       │
        │  • Mamba:             │           │  • Mamba:             │
        │    Selective SSM      │           │    QSVT + Selective Δ │
        │                       │           │                       │
        │  • Hydra:             │           │  • Hydra:             │
        │    Bidirectional SSM  │           │    QSVT + LCU Bidir   │
        └───────────┬───────────┘           └───────────┬───────────┘
                    │                                   │
                    └─────────────────┬─────────────────┘
                                      │
                                      ▼
                          ┌───────────────────────┐
                          │      OUTPUT HEAD      │
                          │   (Classification)    │
                          └───────────────────────┘
```

---

## 2. Factorial Design

### 2.1 Factor Definitions

#### Factor A: Feature Extraction
| Level | Name | Implementation |
|-------|------|----------------|
| **Quantum** | VQC Encoder | Variational quantum circuit with angle embedding |
| **Classical** | QTS Encoder | Conv2d → MaxPool → GLU → Sigmoid(·π) |

#### Factor B: Mixing Mechanism
| Level | Name | Implementation |
|-------|------|----------------|
| **Quantum** | QSVT-based | Quantum Singular Value Transformation |
| **Classical** | Standard | Classical neural network operations |

#### Factor C: Mixing Type (Architecture Style)
| Level | Name | Key Characteristic |
|-------|------|-------------------|
| **Transformer** | Self-Attention | Global mixing, all timesteps interact |
| **Mamba** | Selective SSM | Input-dependent Δ, selective forgetting |
| **Hydra** | Bidirectional SSM | Forward + backward passes, LCU combination |

### 2.2 Factorial Table

```
                              MIXING MECHANISM
                    ┌─────────────────┬─────────────────────────────────────┐
                    │    CLASSICAL    │              QUANTUM                │
                    │                 ├─────────────────┬───────────────────┤
                    │                 │    Standard     │ True Superposition│
┌───────────────────┼─────────────────┼─────────────────┼───────────────────┤
│                   │                 │                 │                   │
│  QUANTUM          │   Group (1)     │   Group (4)     │ Group (4) Super   │
│  FEATURE          │  Q-Feat→C-Mix   │  Q-Feat→Q-Mix   │ E2E + Superpos.   │
│  EXTRACTION       │                 │  (End-to-End)   │                   │
│                   │  1a: Transformer│  4a: Transformer│                   │
│                   │  1b: Mamba      │  4b: Mamba      │  4d: Mamba+Super  │
│                   │  1c: Hydra      │  4c: Hydra      │  4e: Hydra+Super  │
├───────────────────┼─────────────────┼─────────────────┼───────────────────┤
│                   │                 │                 │                   │
│  CLASSICAL        │   Group (3)     │   Group (2)     │ Group (2) Super   │
│  FEATURE          │  C-Feat→C-Mix   │  C-Feat→Q-Mix   │ C-Feat+Q-Super    │
│  EXTRACTION       │  (Baseline)     │                 │                   │
│                   │  3a: Transformer│  2a: Transformer│                   │
│                   │  3b: Mamba      │  2b: Mamba      │  2d: Mamba+Super  │
│                   │  3c: Hydra      │  2c: Hydra      │  2e: Hydra+Super  │
└───────────────────┴─────────────────┴─────────────────┴───────────────────┘
```

**Note**: The "True Superposition" variants (2d, 2e, 4d, 4e) combine three quantum branch states **before** measurement:
```
|ψ⟩ = α|ψ₁⟩ + β|ψ₂⟩ + γ|ψ₃⟩  (Forward, Backward, Diagonal branches)
```
This is fundamentally different from classical ensemble averaging or concatenation.

---

## 3. Complete Model Matrix

### 3.1 All 16 Models

| ID | Feature Extraction | Mixing Mechanism | Mixing Type | Superposition | Full Name |
|----|-------------------|------------------|-------------|---------------|-----------|
| **1a** | Quantum (VQC) | Classical | Transformer | No | QuantumFeat-ClassicalTransformer |
| **1b** | Quantum (VQC) | Classical | Mamba | No | QuantumFeat-ClassicalMamba |
| **1c** | Quantum (VQC) | Classical | Hydra | No | QuantumFeat-ClassicalHydra |
| **2a** | Classical (QTS) | Quantum | Transformer | No | QTSQuantumTransformer |
| **2b** | Classical (QTS) | Quantum | Mamba | No | QTSQuantumMambaSSM |
| **2c** | Classical (QTS) | Quantum | Hydra | No | QTSQuantumHydraSSM |
| **2d** | Classical (QTS) | Quantum | Mamba | **Yes** | QuantumMambaHydraSSM |
| **2e** | Classical (QTS) | Quantum | Hydra | **Yes** | QuantumHydraHydraSSM |
| **3a** | Classical | Classical | Transformer | No | ClassicalTransformer |
| **3b** | Classical | Classical | Mamba | No | ClassicalMamba (TrueClassicalMamba) |
| **3c** | Classical | Classical | Hydra | No | ClassicalHydra (TrueClassicalHydra) |
| **4a** | Quantum (VQC) | Quantum | Transformer | No | FullQuantumTransformer |
| **4b** | Quantum (VQC) | Quantum | Mamba | No | FullQuantumMamba |
| **4c** | Quantum (VQC) | Quantum | Hydra | No | FullQuantumHydra |
| **4d** | Quantum (VQC) | Quantum | Mamba | **Yes** | QuantumMambaE2E_Superposition |
| **4e** | Quantum (VQC) | Quantum | Hydra | **Yes** | QuantumHydraE2E_Superposition |

### 3.2 Model Groupings

#### Group 1: Quantum Features → Classical Mixing
```
Purpose: Test if quantum provides advantage for FEATURE EXTRACTION
Hypothesis: Quantum encoding may capture patterns classical methods miss

1a: VQC Encoder → Classical Self-Attention
1b: VQC Encoder → Classical Mamba SSM
1c: VQC Encoder → Classical Hydra SSM
```

#### Group 2: Classical Features → Quantum Mixing
```
Purpose: Test if quantum provides advantage for SEQUENCE DYNAMICS
Hypothesis: Quantum interference/entanglement improves temporal mixing

Standard Quantum Mixing:
2a: QTS Encoder → Quantum Self-Attention (QSVT + LCU)
2b: QTS Encoder → Quantum Mamba SSM (QSVT + Selective Δ)
2c: QTS Encoder → Quantum Hydra SSM (QSVT + LCU Bidirectional)

TRUE Quantum Superposition Variants (2d, 2e):
2d: QTS Encoder → QuantumMambaHydraSSM (unidirectional, true superposition)
2e: QTS Encoder → QuantumHydraHydraSSM (bidirectional, true superposition)

Key Innovation (2d, 2e):
- Three quantum branches (Forward SSM, Backward SSM, Diagonal)
- States combined in quantum state space BEFORE measurement:
  |ψ⟩ = α|ψ₁⟩ + β|ψ₂⟩ + γ|ψ₃⟩
- Delta-modulated selective forgetting (Mamba-style)
- Learnable complex superposition coefficients (α, β, γ)
```

#### Group 3: Fully Classical (Baseline)
```
Purpose: Establish classical baseline performance
Note: Essential for measuring quantum advantage

3a: Classical Encoder → Classical Self-Attention
3b: Classical Encoder → Classical Mamba SSM
3c: Classical Encoder → Classical Hydra SSM
```

#### Group 4: Fully Quantum (End-to-End)
```
Purpose: Test if end-to-end quantum provides synergistic benefits
Hypothesis: Quantum coherence maintained throughout may help

Standard E2E Quantum:
4a: VQC Encoder → Quantum Self-Attention (E2E, no intermediate measurement)
4b: VQC Encoder → Quantum Mamba SSM (E2E, no intermediate measurement)
4c: VQC Encoder → Quantum Hydra SSM (E2E, no intermediate measurement)

E2E + TRUE Quantum Superposition Variants (4d, 4e):
4d: QuantumMambaE2E_Superposition (unidirectional, E2E + true superposition)
4e: QuantumHydraE2E_Superposition (bidirectional, E2E + true superposition)

Key Innovation (4d, 4e) - COMBINES BOTH E2E AND TRUE SUPERPOSITION:
- End-to-End processing: NO intermediate measurements
- TRUE quantum superposition: |ψ⟩ = α|ψ₁⟩ + β|ψ₂⟩ + γ|ψ₃⟩
- Three branches:
  • Forward: Feature Extraction → Forward SSM Mixing → State Vector
  • Backward: Feature Extraction → Backward SSM Mixing → State Vector
  • Diagonal: Feature Extraction → Diagonal Mixing → State Vector
- States combined BEFORE single final measurement (PauliX, PauliY, PauliZ)
- Learnable complex superposition coefficients (α, β, γ)

Architecture Flow (4d, 4e):
Input → RY Embedding → Variational Feature Layers → SSM Mixing Layers → |ψᵢ⟩
                                                                        ↓
                                              |ψ⟩ = α|ψ₁⟩ + β|ψ₂⟩ + γ|ψ₃⟩
                                                                        ↓
                                              PauliXYZ Measurement (ONLY measurement)
```

---

## 4. Model Architectures

### 4.1 Feature Extraction Components

#### 4.1.1 Classical Feature Extraction (QTS Encoder)

**File**: `models/qts_encoder.py`

```python
class QTSFeatureEncoder(nn.Module):
    """
    Classical feature extraction from QTSTransformer.

    Architecture:
        Input → Conv2d → ReLU → MaxPool → Conv2d → ReLU → MaxPool → GLU → Sigmoid(·π)

    Output: angles ∈ [0, π]^{n_qubits} ready for quantum circuits
    """

    def __init__(self, feature_dim, n_timesteps, n_qubits, projection_type='Conv2d_GLU'):
        # Conv2d layers for spatial-temporal feature extraction
        # GLU gating for non-linear transformation
        # Sigmoid scaling to [0, π] for quantum angle encoding
```

**Projection Types:**
- `Linear`: Simple linear projection
- `Conv2d`: 2D CNN feature extraction
- `Conv2d_GLU`: 2D CNN + Gated Linear Unit (Recommended)
- `GLU`: Gated Linear Unit only

#### 4.1.2 Quantum Feature Extraction (VQC Encoder)

**File**: `models/vqc_encoder.py` (TO BUILD)

```python
class VQCFeatureEncoder(nn.Module):
    """
    Quantum feature extraction using Variational Quantum Circuit.

    Architecture:
        Input → Linear projection → Angle Embedding → VQC layers → Measurement

    Quantum Components:
        - RY angle embedding for input encoding
        - Variational layers (RX, RY, RZ rotations)
        - CNOT entanglement
        - PauliZ measurement
    """

    def __init__(self, feature_dim, n_timesteps, n_qubits, n_layers=2):
        # Builds parameterized quantum circuit for feature extraction
```

### 4.2 Mixing Components

#### 4.2.1 Classical Self-Attention (Transformer)

```python
class ClassicalSelfAttention(nn.Module):
    """
    Standard multi-head self-attention.

    Attention(Q, K, V) = softmax(QK^T / sqrt(d_k)) V
    """
```

#### 4.2.2 Classical Mamba SSM

```python
class ClassicalMambaSSM(nn.Module):
    """
    Selective State Space Model (Mamba-style).

    h[t] = exp(Δ[t]·A) · h[t-1] + Δ[t]·B[t]·x[t]
    y[t] = C[t]·h[t] + D·x[t]

    where Δ[t], B[t], C[t] = f(x[t]) are INPUT-DEPENDENT
    """
```

#### 4.2.3 Classical Hydra SSM

```python
class ClassicalHydraSSM(nn.Module):
    """
    Bidirectional State Space Model (Hydra-style).

    h_fwd[t] = A · h_fwd[t-1] + B · x[t]  (forward)
    h_bwd[t] = A · h_bwd[t+1] + B · x[t]  (backward)
    y[t] = combine(h_fwd[t], h_bwd[t])
    """
```

#### 4.2.4 Quantum Self-Attention (QSVT + LCU)

**File**: `models/quantum_attention_core.py`

```python
class QuantumAttentionCore(nn.Module):
    """
    Quantum attention using QSVT and LCU.

    Key mechanisms:
    1. Each timestep t → U(θ_t)|ψ⟩ via sim14 circuit
    2. LCU: |ψ_mixed⟩ = Σ_t α_t U(θ_t)|ψ⟩
    3. QSVT polynomial transformation
    4. Multi-observable measurement (X, Y, Z)

    This implements GLOBAL ATTENTION quantumly.
    """
```

#### 4.2.5 Quantum Mamba SSM (QSVT + Selective Δ)

**File**: `models/quantum_mamba_ssm_core_advanced.py`

```python
class QuantumMambaSSMCoreAdvanced(nn.Module):
    """
    Quantum Selective SSM with input-dependent Δ.

    Key mechanisms:
    1. Input-dependent Δ[t] = f(x[t]) modulates QSVT
    2. QSVT exp(Δ·A) for quantum state transition
    3. Selective input injection scaled by Δ
    4. sim14 ansatz + multi-observable measurement

    QUANTUM SELECTIVE FORGETTING:
    - High Δ: Strong transformation, remember input
    - Low Δ: Weak transformation, maintain state
    """
```

#### 4.2.6 Quantum Hydra SSM (QSVT + LCU Bidirectional)

**File**: `models/quantum_hydra_ssm_core_advanced.py`

```python
class QuantumHydraSSMCoreAdvanced(nn.Module):
    """
    Quantum Bidirectional SSM with QSVT and LCU.

    Key mechanisms:
    1. Forward quantum SSM pass
    2. Backward quantum SSM pass
    3. LCU combination of forward/backward states
    4. sim14 ansatz + multi-observable measurement

    Implements BIDIRECTIONAL QUANTUM STATE EVOLUTION.
    """
```

---

## 5. Research Questions

### 5.1 Three Core Questions (Tested Across All Conditions)

| Question | Comparison | What We Learn |
|----------|------------|---------------|
| **Q1: Where should quantum be applied?** | Groups (1) vs (2) vs (3) vs (4) | Feature extraction, mixing, both, or neither? |
| **Q2: Which mixing benefits from quantum?** | 2a vs 2b vs 2c | Transformer, Mamba, or Hydra? |
| **Q3: Does end-to-end quantum help?** | Group (3) vs (4) | Synergistic benefits from full quantum? |

### 5.2 Sequence Length as Moderating Variable

**Sequence length is NOT a separate question.** It tests the **robustness and boundaries** of Q1, Q2, Q3.

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                    INTEGRATED EXPERIMENTAL DESIGN                            │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│   Q1: Where to apply quantum?                                               │
│       → Tested at 40 Hz, 80 Hz, 160 Hz (PhysioNet)                         │
│       → Tested on Genomics, SST-2, Forrelation                             │
│       → If answer consistent: ROBUST finding                                │
│       → If answer varies: CONDITIONAL finding                               │
│                                                                             │
│   Q2: Which mixing benefits?                                                │
│       → Tested at 40 Hz, 80 Hz, 160 Hz (PhysioNet)                         │
│       → Tested on Genomics, SST-2, Forrelation                             │
│       → Hypothesis: SSMs benefit more at longer sequences                   │
│                                                                             │
│   Q3: Does E2E quantum help?                                                │
│       → Tested at 40 Hz, 80 Hz, 160 Hz (PhysioNet)                         │
│       → Tested on Genomics, SST-2, Forrelation                             │
│       → Hypothesis: E2E may only help at shorter sequences                  │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

### 5.3 Detailed Comparisons

#### Q1: Where to Apply Quantum

```
Compare across groups (fixing mixing type):

For Transformer:  1a vs 2a vs 3a vs 4a  (across all datasets & lengths)
For Mamba:        1b vs 2b vs 3b vs 4b  (across all datasets & lengths)
For Hydra:        1c vs 2c vs 3c vs 4c  (across all datasets & lengths)

Hypothesis: Group (2) > Group (1)
Rationale: Classical encoders excel at feature extraction (proven by deep learning),
           while quantum may excel at dynamics (interference, superposition).

Robustness Check:
- Does this hold at 40 Hz? 80 Hz? 160 Hz?
- Does this hold on Genomics? SST-2? Forrelation?
```

#### Q2: Which Quantum Mixing

```
Compare within Group (2):

2a (Q-Transformer) vs 2b (Q-Mamba) vs 2c (Q-Hydra)

Questions:
- Does quantum attention outperform quantum SSM?
- Does selective (Mamba) or bidirectional (Hydra) work better?

Sequence Length Insight:
- At short sequences (40 Hz): Transformer may win (global attention feasible)
- At long sequences (160 Hz): SSMs may win (O(n) vs O(n²) complexity)
```

#### Q3: End-to-End Quantum

```
Compare Group (3) vs Group (4):

3a vs 4a (Transformer)  - across all datasets & lengths
3b vs 4b (Mamba)        - across all datasets & lengths
3c vs 4c (Hydra)        - across all datasets & lengths

Hypothesis:
- If quantum coherence is important → Group (4) wins
- If mixing needs classical grounding → Group (4) may not help

Sequence Length Insight:
- Longer sequences = deeper quantum circuits = more decoherence
- E2E quantum may only help at shorter sequences
```

### 5.4 Interaction Effects

```
Full factorial analysis allows testing:

1. Feature × Mixing interaction:
   Does quantum mixing ONLY help with certain feature types?

2. Mixing Type × Quantum interaction:
   Does quantum help Transformer more than SSMs (or vice versa)?

3. Sequence Length × Architecture interaction:
   Do SSMs scale better than Transformers as sequence length increases?

4. Dataset × Quantum interaction:
   Does quantum advantage appear on some domains but not others?
```

### 5.5 What Constitutes a Robust Finding?

| Scenario | Interpretation | Confidence |
|----------|----------------|------------|
| Q1 answer same across all datasets AND all lengths | Universal finding | HIGH |
| Q1 answer same across datasets, varies by length | Length-dependent finding | MEDIUM |
| Q1 answer varies across datasets | Domain-specific finding | LOW (report as conditional) |
| Q1 answer inconsistent | No clear conclusion | Report null result |

---

## 6. Implementation Status

### 6.1 Current Status

**ALL 16 MODELS ARE IMPLEMENTED AND READY**

| ID | Model | Status | Class Name | File Location |
|----|-------|--------|------------|---------------|
| **1a** | Q-Feat + C-Transformer | ✅ READY | `QuantumTransformer` | `models/QuantumTransformer.py` |
| **1b** | Q-Feat + C-Mamba | ✅ READY | `QuantumMambaSSM` | `models/QuantumSSM.py` |
| **1c** | Q-Feat + C-Hydra | ✅ READY | `QuantumHydraSSM` | `models/QuantumSSM.py` |
| **2a** | C-Feat + Q-Transformer | ✅ READY | `ClassicalQuantumAttention` | `models/QuantumMixingSSM.py` |
| **2b** | C-Feat + Q-Mamba | ✅ READY | `ClassicalMambaQuantumSSM` | `models/QuantumMixingSSM.py` |
| **2c** | C-Feat + Q-Hydra | ✅ READY | `ClassicalHydraQuantumSSM` | `models/QuantumMixingSSM.py` |
| **2d** | C-Feat + Q-Mamba (Superposition) | ✅ READY | `QuantumMambaHydraSSM` | `models/QuantumHydraSSM.py` |
| **2e** | C-Feat + Q-Hydra (Superposition) | ✅ READY | `QuantumHydraHydraSSM` | `models/QuantumHydraSSM.py` |
| **3a** | C-Feat + C-Transformer | ✅ READY | `ClassicalTransformer` | `models/ClassicalTransformer.py` |
| **3b** | C-Feat + C-Mamba | ✅ READY | `TrueClassicalMamba` | `models/TrueClassicalMamba.py` |
| **3c** | C-Feat + C-Hydra | ✅ READY | `TrueClassicalHydra` | `models/TrueClassicalHydra.py` |
| **4a** | Q-Feat + Q-Transformer (E2E) | ✅ READY | `QuantumTransformerE2E` | `models/QuantumE2E.py` |
| **4b** | Q-Feat + Q-Mamba (E2E) | ✅ READY | `QuantumMambaE2E` | `models/QuantumE2E.py` |
| **4c** | Q-Feat + Q-Hydra (E2E) | ✅ READY | `QuantumHydraE2E` | `models/QuantumE2E.py` |
| **4d** | Q-Feat + Q-Mamba (E2E + Superposition) | ✅ READY | `QuantumMambaE2E_Superposition` | `models/QuantumE2E_Superposition.py` |
| **4e** | Q-Feat + Q-Hydra (E2E + Superposition) | ✅ READY | `QuantumHydraE2E_Superposition` | `models/QuantumE2E_Superposition.py` |

### 6.2 Implementation Summary

```
All 16 models are implemented:

Group 1 (Quantum Features → Classical Mixing):
├── [x] QuantumTransformer (1a) - models/QuantumTransformer.py
├── [x] QuantumMambaSSM (1b) - models/QuantumSSM.py
└── [x] QuantumHydraSSM (1c) - models/QuantumSSM.py

Group 2 (Classical Features → Quantum Mixing):
├── [x] ClassicalQuantumAttention (2a) - models/QuantumMixingSSM.py
├── [x] ClassicalMambaQuantumSSM (2b) - models/QuantumMixingSSM.py
├── [x] ClassicalHydraQuantumSSM (2c) - models/QuantumMixingSSM.py
│
│   TRUE Superposition Variants:
├── [x] QuantumMambaHydraSSM (2d) - models/QuantumHydraSSM.py
│       Classical features → Quantum mixing with TRUE superposition (unidirectional)
│       |ψ⟩ = α|ψ₁⟩ + β|ψ₂⟩ + γ|ψ₃⟩ + Delta-modulated selective forgetting
└── [x] QuantumHydraHydraSSM (2e) - models/QuantumHydraSSM.py
        Classical features → Quantum mixing with TRUE superposition (bidirectional)
        Same as 2d but with bidirectional sequence processing

Group 3 (Classical Features → Classical Mixing - Baseline):
├── [x] ClassicalTransformer (3a) - models/ClassicalTransformer.py
├── [x] TrueClassicalMamba (3b) - models/TrueClassicalMamba.py
└── [x] TrueClassicalHydra (3c) - models/TrueClassicalHydra.py

Group 4 (Quantum Features → Quantum Mixing - End-to-End):
├── [x] QuantumTransformerE2E (4a) - models/QuantumE2E.py
├── [x] QuantumMambaE2E (4b) - models/QuantumE2E.py
├── [x] QuantumHydraE2E (4c) - models/QuantumE2E.py
│
│   E2E + TRUE Superposition Variants:
├── [x] QuantumMambaE2E_Superposition (4d) - models/QuantumE2E_Superposition.py
│       E2E quantum + TRUE superposition (unidirectional)
│       NO intermediate measurements + |ψ⟩ = α|ψ₁⟩ + β|ψ₂⟩ + γ|ψ₃⟩
└── [x] QuantumHydraE2E_Superposition (4e) - models/QuantumE2E_Superposition.py
        E2E quantum + TRUE superposition (bidirectional)
        Same as 4d but with bidirectional sequence processing

Remaining Tasks (Scripts):
├── [x] run_ablation_eeg.py              # Main runner for 16 models
├── [x] generate_ablation_eeg_jobs.py    # SLURM job generator
├── [x] aggregate_ablation_results.py    # Results aggregation + statistics
└── [ ] plot_ablation_analysis.py        # Visualization (TO BUILD)
```

### 6.3 Shared Components

| Component | Used By | File |
|-----------|---------|------|
| ClassicalFeatureExtractor | 2a, 2b, 2c, 2d, 2e, 3a, 3b, 3c | `models/QuantumMixingSSM.py` |
| ChunkedQuantumProcessor | 1a, 1b, 1c | `models/QuantumSSM.py` |
| QuantumAttentionMixingCore | 2a | `models/QuantumMixingSSM.py` |
| QuantumSSMCore | 2b | `models/QuantumMixingSSM.py` |
| QuantumBidirectionalSSMCore | 2c | `models/QuantumMixingSSM.py` |
| QuantumHydraSSMCore | 2d, 2e | `models/QuantumHydraSSM.py` |
| MambaBlock (Classical) | 1b, 3b | `models/TrueClassicalMamba.py`, `models/QuantumSSM.py` |
| HydraBlock (Classical) | 1c, 3c | `models/TrueClassicalHydra.py`, `models/QuantumSSM.py` |
| TransformerBlock (Classical) | 1a, 3a | `models/ClassicalTransformer.py`, `models/QuantumTransformer.py` |
| E2E Quantum Circuits | 4a, 4b, 4c | `models/QuantumE2E.py` |
| QuantumE2ESuperpositionCore | 4d, 4e | `models/QuantumE2E_Superposition.py` |

---

## 7. Experimental Protocol

### 7.1 Dataset Selection for ICML/NeurIPS Rigor

#### 7.1.1 Primary Datasets (Required)

| Dataset | Domain | Seq Length | Classes | Purpose | Dataloader |
|---------|--------|------------|---------|---------|------------|
| **PhysioNet EEG** | Biomedical | 124/248/496 (40/80/160 Hz) | 2 | Main ablation + length scaling | `Load_PhysioNet_EEG_NoPrompt.py` |
| **Genomic Benchmarks** | Biology | ~200 bp | 2 | Cross-domain + compare HyenaDNA | `Load_Genomic_Benchmarks.py` |
| **SST-2 (GLUE)** | NLP | 128 tokens | 2 | Different modality (text) | `Load_GLUE.py` |
| **Forrelation** | Synthetic | 64 | 2 | Theoretical quantum advantage | `forrelation_dataloader.py` |

#### 7.1.2 Why These Datasets?

| Reviewer Concern | Your Response |
|------------------|---------------|
| "Only one dataset" | "We test on 4 datasets: EEG, Genomics, NLP, Synthetic" |
| "Only one domain" | "Biomedical signals, DNA sequences, text, and synthetic" |
| "No theoretical grounding" | "Forrelation has proven quantum advantage" |
| "No sequence length analysis" | "PhysioNet tested at 3 sampling rates (40/80/160 Hz)" |

#### 7.1.3 Optional Datasets (Appendix/Supplementary)

| Dataset | Domain | Seq Length | Classes | Purpose | Dataloader |
|---------|--------|------------|---------|---------|------------|
| **Sequential MNIST** | Image | 784 | 10 | Standard QML benchmark | `Load_Image_Datasets.py` |
| **Sequential CIFAR-10** | Image | 3072 | 10 | Harder image benchmark | `Load_Image_Datasets.py` |
| **Fashion-MNIST** | Image | 784 | 10 | Alternative to MNIST | `Load_Image_Datasets.py` |
| CoLA (GLUE) | NLP | ~32 | 2 | Linguistic acceptability | `Load_GLUE.py` |
| MRPC (GLUE) | NLP | ~128 | 2 | Sentence pair task | `Load_GLUE.py` |
| FACED EEG | Biomedical | Variable | 2 | Additional EEG dataset | `Load_FACED_EEG.py` |

**Why Image Datasets?**
- **Sequential MNIST/CIFAR**: Treat images as 1D sequences (pixel-by-pixel)
- Tests models on very long sequences (784-3072 timesteps)
- Standard benchmarks in QML literature for reproducibility
- Different domain from EEG/Genomics/NLP

### 7.2 Experiment Counts

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                         TOTAL EXPERIMENT COUNT                               │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│   PRIMARY EXPERIMENTS:                                                      │
│   ├── PhysioNet EEG (3 sampling rates × 12 models × 5 seeds) =  180 runs   │
│   ├── Genomic Benchmarks (1 config × 12 models × 5 seeds)    =   60 runs   │
│   ├── SST-2 GLUE (1 config × 12 models × 5 seeds)            =   60 runs   │
│   └── Forrelation (1 config × 12 models × 5 seeds)           =   60 runs   │
│                                                               ─────────────│
│                                                       TOTAL  =  360 runs   │
│                                                                             │
│   OPTIONAL (for appendix):                                                  │
│   ├── Sequential MNIST (12 × 5)                              =   60 runs   │
│   ├── Sequential CIFAR-10 (12 × 5)                           =   60 runs   │
│   ├── Fashion-MNIST (12 × 5)                                 =   60 runs   │
│   ├── CoLA (12 × 5)                                          =   60 runs   │
│   └── MRPC (12 × 5)                                          =   60 runs   │
│                                                               ─────────────│
│                                                       TOTAL  =  660 runs   │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

### 7.3 PhysioNet Sequence Length Conditions

| Sampling Frequency | Sequence Length | Purpose |
|-------------------|-----------------|---------|
| **40 Hz** | 124 timesteps | Short sequences - tests basic functionality |
| **80 Hz** | 248 timesteps | Medium sequences - balanced condition |
| **160 Hz** | 496 timesteps | Long sequences - tests scaling (SSM vs Attention) |

**Implementation**: Use `sampling_freq` parameter in `Load_PhysioNet_EEG_NoPrompt.py`:

```python
# Short sequences (40 Hz)
train_loader, val_loader, test_loader, input_dim = load_eeg_ts_revised(
    seed=2024, device=device, batch_size=32, sampling_freq=40, sample_size=109
)

# Medium sequences (80 Hz)
train_loader, val_loader, test_loader, input_dim = load_eeg_ts_revised(
    seed=2024, device=device, batch_size=32, sampling_freq=80, sample_size=109
)

# Long sequences (160 Hz)
train_loader, val_loader, test_loader, input_dim = load_eeg_ts_revised(
    seed=2024, device=device, batch_size=32, sampling_freq=160, sample_size=109
)
```

### 7.4 Hyperparameters (Fixed Across All Models)

```python
# Quantum parameters
N_QUBITS = 6
N_LAYERS = 2
QSVT_DEGREE = 3

# Training parameters
BATCH_SIZE = 32
LEARNING_RATE = 1e-3
WEIGHT_DECAY = 1e-4
N_EPOCHS = 50
OPTIMIZER = 'AdamW'
SCHEDULER = 'CosineAnnealingLR'
EARLY_STOPPING_PATIENCE = 10

# Statistical rigor
SEEDS = [2024, 2025, 2026, 2027, 2028]  # 5 runs per config
```

### 7.5 Evaluation Metrics

| Task Type | Primary Metric | Secondary Metrics |
|-----------|----------------|-------------------|
| Binary Classification | Accuracy | AUC, F1, Sensitivity, Specificity |
| Multi-class Classification | Accuracy | Macro-F1, Per-class Accuracy |
| Regression | RMSE | MAE, R² |

### 7.6 Statistical Analysis Requirements

#### 7.6.1 Per-Comparison Statistics

```python
from scipy import stats
import numpy as np

def compute_comparison_stats(results_a, results_b, alpha=0.05):
    """
    Compute all statistics needed for ICML/NeurIPS paper.

    Args:
        results_a: List of 5 accuracy values (one per seed) for model A
        results_b: List of 5 accuracy values (one per seed) for model B
        alpha: Significance level (default 0.05)

    Returns:
        Dictionary with all required statistics
    """
    # Paired t-test (same seeds for both models)
    t_stat, p_value = stats.ttest_rel(results_a, results_b)

    # Effect size (Cohen's d for paired samples)
    diff = np.array(results_a) - np.array(results_b)
    cohens_d = diff.mean() / diff.std()

    # 95% Confidence Interval
    ci = stats.t.interval(0.95, len(diff)-1, loc=diff.mean(), scale=stats.sem(diff))

    # Bonferroni correction (for multiple comparisons)
    # With 12 models, we have C(12,2) = 66 pairwise comparisons
    bonferroni_alpha = alpha / 66

    return {
        'mean_a': np.mean(results_a),
        'std_a': np.std(results_a),
        'mean_b': np.mean(results_b),
        'std_b': np.std(results_b),
        'mean_diff': diff.mean(),
        'p_value': p_value,
        'significant_uncorrected': p_value < alpha,
        'significant_bonferroni': p_value < bonferroni_alpha,
        'cohens_d': cohens_d,
        'effect_size': 'large' if abs(cohens_d) > 0.8 else 'medium' if abs(cohens_d) > 0.5 else 'small',
        'ci_95': ci
    }
```

#### 7.6.2 Cross-Dataset Consistency Check

```python
def check_finding_robustness(results_by_dataset):
    """
    Check if a finding is consistent across all datasets.

    Args:
        results_by_dataset: Dict mapping dataset name to comparison result

    Returns:
        Robustness assessment
    """
    significant_count = sum(1 for r in results_by_dataset.values() if r['significant_uncorrected'])
    total_datasets = len(results_by_dataset)

    if significant_count == total_datasets:
        return "ROBUST: Significant on all datasets"
    elif significant_count >= total_datasets * 0.75:
        return "MOSTLY ROBUST: Significant on most datasets"
    elif significant_count >= total_datasets * 0.5:
        return "CONDITIONAL: Significant on some datasets"
    else:
        return "NOT ROBUST: Inconsistent across datasets"
```

#### 7.6.3 Required Tables for Paper

**Table 1: Main Results (Per Dataset)**
```
| Model | PhysioNet-40Hz | PhysioNet-80Hz | PhysioNet-160Hz | Genomics | SST-2 | Forrelation |
|-------|----------------|----------------|-----------------|----------|-------|-------------|
| 1a    | 75.2 ± 1.3     | 76.1 ± 1.1     | 74.8 ± 1.5      | ...      | ...   | ...         |
| ...   | ...            | ...            | ...             | ...      | ...   | ...         |
```

**Table 2: Pairwise Comparisons for Q1 (Group 1 vs Group 2)**
```
| Comparison | PhysioNet (avg) | Genomics | SST-2 | Forrelation | Consistent? |
|------------|-----------------|----------|-------|-------------|-------------|
| 1a vs 2a   | Δ=+2.3*, d=0.6  | ...      | ...   | ...         | Yes/No      |
| 1b vs 2b   | Δ=+3.1*, d=0.8  | ...      | ...   | ...         | Yes/No      |
| 1c vs 2c   | Δ=+2.8*, d=0.7  | ...      | ...   | ...         | Yes/No      |
```

**Table 3: Sequence Length Scaling (PhysioNet Only)**
```
| Model | 40 Hz (124) | 80 Hz (248) | 160 Hz (496) | Scales Well? |
|-------|-------------|-------------|--------------|--------------|
| 2a    | 78.2 ± 1.1  | 77.5 ± 1.3  | 74.1 ± 1.8   | No (↓4.1%)   |
| 2b    | 76.5 ± 1.2  | 77.8 ± 1.0  | 78.2 ± 1.1   | Yes (↑1.7%)  |
| 2c    | 77.1 ± 1.0  | 78.5 ± 0.9  | 79.3 ± 1.0   | Yes (↑2.2%)  |
```

---

## 8. Expected Outcomes

### 8.1 Hypotheses

| Hypothesis | Comparison | Prediction |
|------------|------------|------------|
| **H1** | Group (2) > Group (1) | Classical features + Quantum mixing outperforms Quantum features + Classical mixing |
| **H2** | 2b, 2c > 2a | Quantum SSMs (Mamba/Hydra) outperform Quantum Attention for sequential data |
| **H3** | 2c > 2b | Bidirectional (Hydra) outperforms Selective (Mamba) for classification |
| **H4** | Group (4) ≈ Group (2) | End-to-end quantum doesn't significantly outperform hybrid |
| **H5** | Group (2) > Group (3) | Quantum mixing provides measurable advantage over classical |

### 8.2 Expected Results Table

```
                              MIXING MECHANISM
                    ┌─────────────────┬─────────────────┐
                    │    CLASSICAL    │     QUANTUM     │
┌───────────────────┼─────────────────┼─────────────────┤
│  QUANTUM FEATURE  │   ~75-80%       │   ~78-83%       │
│  EXTRACTION       │   (Group 1)     │   (Group 4)     │
├───────────────────┼─────────────────┼─────────────────┤
│  CLASSICAL        │   ~73-78%       │   ~80-85%       │
│  FEATURE          │   (Group 3)     │   (Group 2)     │
│  EXTRACTION       │   Baseline      │   BEST          │
└───────────────────┴─────────────────┴─────────────────┘

Expected ranking: Group (2) > Group (4) > Group (1) > Group (3)
```

### 8.3 Key Insights to Validate

1. **Quantum Mixing > Quantum Features**: Classical encoders are mature and effective; quantum adds value in dynamics.

2. **SSM Benefits from Quantum**: State evolution through quantum interference may capture temporal patterns better.

3. **No Need for End-to-End Quantum**: Hybrid (classical encoder + quantum mixing) may be optimal for NISQ era.

4. **Hydra vs Mamba**: Task-dependent; Hydra may win for classification, Mamba for generation.

---

## 9. Paper Structure (ICML/NeurIPS Format)

### 9.1 Suggested Paper Outline

```
1. Introduction (1 page)
   - Quantum ML motivation
   - Key question: Where does quantum help in sequence models?
   - Our contribution: Systematic 2×2×3 factorial ablation

2. Related Work (0.5-1 page)
   - Quantum ML for sequences
   - Classical SSMs (Mamba, Hydra)
   - Hybrid quantum-classical approaches

3. Method (2 pages)
   - 2×2×3 factorial design explanation
   - 12 model architectures
   - Quantum vs Classical components

4. Experiments (3-4 pages)
   4.1 Experimental Setup
       - Datasets (PhysioNet, Genomics, SST-2, Forrelation)
       - Sequence length conditions (40/80/160 Hz)
       - Hyperparameters and training details

   4.2 Q1: Where to Apply Quantum?
       - Table: Group (1) vs (2) across all datasets
       - Finding: [Robust/Conditional]

   4.3 Q2: Which Mixing Benefits from Quantum?
       - Table: 2a vs 2b vs 2c across all datasets
       - Finding: [Transformer/Mamba/Hydra]

   4.4 Q3: Does End-to-End Quantum Help?
       - Table: Group (3) vs (4) across all datasets
       - Finding: [Yes/No/Conditional]

   4.5 Sequence Length Analysis
       - Table: Performance vs sequence length
       - Finding: SSM scaling vs Attention scaling

5. Discussion (0.5 page)
   - Implications for quantum ML practitioners
   - When to use quantum features vs quantum mixing
   - Limitations

6. Conclusion (0.25 page)

Appendix (Supplementary):
   - Full hyperparameters
   - All 360 individual results
   - Additional datasets (CoLA, MRPC, MNIST)
   - Ablation on quantum circuit depth
```

### 9.2 Key Figures

| Figure | Content | Purpose |
|--------|---------|---------|
| Fig 1 | Architecture diagram (2×2×3 grid) | Method overview |
| Fig 2 | Main results heatmap (12 models × 6 conditions) | Q1, Q2, Q3 answers at a glance |
| Fig 3 | Sequence length scaling curves | SSM vs Attention complexity |
| Fig 4 | Cross-dataset consistency plot | Robustness visualization |

### 9.3 Addressing Reviewer Concerns

| Potential Concern | Pre-emptive Response in Paper |
|-------------------|-------------------------------|
| "Only simulation, no real quantum" | "We focus on algorithmic advantage; hardware experiments are future work" |
| "Classical baselines too weak" | "Group 3 uses state-of-the-art Mamba/Hydra architectures" |
| "Dataset-specific results" | "We test on 4 diverse datasets; findings are consistent/conditional" |
| "Not enough statistical rigor" | "5 seeds, paired t-tests, effect sizes, Bonferroni correction" |
| "No theoretical analysis" | "Forrelation provides theoretical grounding; empirical focus is intentional" |

---

## Appendix A: File Structure

```
quantum_hydra_mamba/
├── models/
│   ├── __init__.py                         # Model exports (all 16 models)
│   │
│   │   # ========== Group 1: Quantum Features → Classical Mixing ==========
│   ├── QuantumTransformer.py               # Model 1a: QuantumTransformer
│   ├── QuantumSSM.py                       # Models 1b, 1c: QuantumMambaSSM, QuantumHydraSSM
│   │
│   │   # ========== Group 2: Classical Features → Quantum Mixing ==========
│   ├── QuantumMixingSSM.py                 # Models 2a, 2b, 2c:
│   │                                       #   ClassicalQuantumAttention
│   │                                       #   ClassicalMambaQuantumSSM
│   │                                       #   ClassicalHydraQuantumSSM
│   │
│   ├── QuantumHydraSSM.py                  # Models 2d, 2e (TRUE Superposition):
│   │                                       #   QuantumMambaHydraSSM (2d)
│   │                                       #   QuantumHydraHydraSSM (2e)
│   │                                       #   Key: |ψ⟩ = α|ψ₁⟩ + β|ψ₂⟩ + γ|ψ₃⟩
│   │                                       #   + Delta-modulated selective forgetting
│   │
│   │   # ========== Group 3: Classical Features → Classical Mixing ==========
│   ├── ClassicalTransformer.py             # Model 3a: ClassicalTransformer
│   ├── TrueClassicalMamba.py               # Model 3b: TrueClassicalMamba
│   ├── TrueClassicalHydra.py               # Model 3c: TrueClassicalHydra
│   │
│   │   # ========== Group 4: Quantum Features → Quantum Mixing (E2E) ==========
│   ├── QuantumE2E.py                       # Models 4a, 4b, 4c:
│   │                                       #   QuantumTransformerE2E
│   │                                       #   QuantumMambaE2E
│   │                                       #   QuantumHydraE2E
│   │
│   ├── QuantumE2E_Superposition.py         # Models 4d, 4e (E2E + TRUE Superposition):
│   │                                       #   QuantumMambaE2E_Superposition (4d)
│   │                                       #   QuantumHydraE2E_Superposition (4e)
│   │                                       #   Key: E2E (no intermediate measurement)
│   │                                       #   + |ψ⟩ = α|ψ₁⟩ + β|ψ₂⟩ + γ|ψ₃⟩
│   │
│   │   # ========== Shared Components ==========
│   ├── qts_encoder.py                      # Classical feature encoder
│   ├── quantum_attention_core.py           # Quantum attention core
│   ├── quantum_mamba_ssm_core_advanced.py  # Quantum Mamba SSM core
│   └── quantum_hydra_ssm_core_advanced.py  # Quantum Hydra SSM core
│
├── scripts/
│   ├── run_single_model_eeg.py             # Single model training script
│   ├── run_ablation_eeg.py                 # Ablation study runner (16 models)
│   ├── generate_ablation_eeg_jobs.py       # SLURM job generator (180 jobs)
│   ├── aggregate_ablation_results.py       # Results aggregation + statistics
│   └── plot_ablation_analysis.py           # Visualization (TO BUILD)
│
├── data_loaders/
│   │   # ========== EEG Datasets ==========
│   ├── Load_PhysioNet_EEG_NoPrompt.py      # PhysioNet EEG (PRIMARY)
│   ├── Load_FACED_EEG.py                   # FACED EEG (optional)
│   ├── Load_SEED_EEG.py                    # SEED EEG (optional)
│   ├── eeg_datasets.py                     # Unified EEG loader
│   │
│   │   # ========== Genomic Datasets ==========
│   ├── Load_Genomic_Benchmarks.py          # Genomic Benchmarks (PRIMARY)
│   │
│   │   # ========== NLP Datasets ==========
│   ├── Load_GLUE.py                        # GLUE Benchmarks - SST-2 (PRIMARY)
│   │
│   │   # ========== Image Datasets ==========
│   ├── Load_Image_Datasets.py              # MNIST, Fashion-MNIST, CIFAR-10 (optional)
│   │                                       #   load_mnist(), load_fashion(), load_cifar()
│   │
│   │   # ========== Synthetic Datasets ==========
│   └── forrelation_dataloader.py           # Forrelation (PRIMARY - quantum advantage)
│
└── docs/
    ├── ABLATION_STUDY_PLAN_V3.md           # This document
    ├── ABLATION_STUDY_IMPLEMENTATION_PLAN_V2.md  # Previous version
    └── EEG_DATASETS_SETUP_GUIDE.md
```

---

## Appendix B: Quick Reference Card

### Model ID Lookup

| ID | Feature | Mixing | Type | Superposition | Short Name |
|----|---------|--------|------|---------------|------------|
| 1a | Q | C | Trans | No | QF-CT |
| 1b | Q | C | Mamba | No | QF-CM |
| 1c | Q | C | Hydra | No | QF-CH |
| 2a | C | Q | Trans | No | CF-QT |
| 2b | C | Q | Mamba | No | CF-QM |
| 2c | C | Q | Hydra | No | CF-QH |
| 2d | C | Q | Mamba | **Yes** | CF-QM-S |
| 2e | C | Q | Hydra | **Yes** | CF-QH-S |
| 3a | C | C | Trans | No | CF-CT |
| 3b | C | C | Mamba | No | CF-CM |
| 3c | C | C | Hydra | No | CF-CH |
| 4a | Q | Q | Trans | No | QF-QT |
| 4b | Q | Q | Mamba | No | QF-QM |
| 4c | Q | Q | Hydra | No | QF-QH |
| 4d | Q | Q | Mamba | **Yes** | QF-QM-S |
| 4e | Q | Q | Hydra | **Yes** | QF-QH-S |

### Model Abbreviations

```
Q  = Quantum          C  = Classical
S  = Superposition (True quantum state superposition: |ψ⟩ = α|ψ₁⟩ + β|ψ₂⟩ + γ|ψ₃⟩)
E2E = End-to-End (no intermediate measurements)

-S suffix indicates TRUE quantum superposition variant
```

### Comparison Shortcuts

```
Ablation 1 (Where Quantum?):     Compare rows (1 vs 2 vs 3 vs 4)
Ablation 2 (Which Mixing?):      Compare columns (a vs b vs c)
Ablation 3 (End-to-End?):        Compare 3 vs 4
Ablation 4 (Superposition?):     Compare 2b/2c vs 2d/2e, 4b/4c vs 4d/4e
Baseline:                        Group 3 (all classical)
Best Expected:                   Group 2 (classical feat + quantum mix)

Superposition Research Questions:
- Q4a: Does TRUE superposition improve quantum mixing? (2b vs 2d, 2c vs 2e)
- Q4b: Does TRUE superposition improve E2E quantum? (4b vs 4d, 4c vs 4e)
- Q4c: Do learnable coefficients (α, β, γ) provide meaningful structure?
```

---

## Document Information

- **Version**: 3.4 (Complete 16-Model Design with TRUE Superposition Variants)
- **Author**: Junghoon Park
- **Created**: December 2024
- **Updated**: January 2026
- **Key Changes**:
  - v3.0: Simplified to clean factorial design with 12 models
  - v3.1: All 12 models implemented and verified ready
  - v3.2: Multi-dataset experimental design for ICML/NeurIPS rigor
    - Added 4 primary datasets (PhysioNet, Genomics, SST-2, Forrelation)
    - Integrated sequence length as moderating variable (not separate question)
    - Added statistical analysis requirements (paired t-test, Cohen's d, Bonferroni)
    - Added Paper Structure section (Section 9)
    - 360 total primary experiments (12 models × 6 conditions × 5 seeds)
  - v3.3: Added image datasets as optional benchmarks
    - Sequential MNIST (784 timesteps), CIFAR-10 (3072 timesteps), Fashion-MNIST
    - Standard QML benchmarks for reproducibility and comparison with literature
    - 660 total runs including optional experiments
  - **v3.4**: Expanded to 16 models with TRUE quantum superposition variants
    - Added 4 new models with TRUE quantum superposition (2d, 2e, 4d, 4e)
    - **2d**: QuantumMambaHydraSSM - Classical features → Quantum mixing with true superposition (unidirectional)
    - **2e**: QuantumHydraHydraSSM - Classical features → Quantum mixing with true superposition (bidirectional)
    - **4d**: QuantumMambaE2E_Superposition - E2E quantum + true superposition (unidirectional)
    - **4e**: QuantumHydraE2E_Superposition - E2E quantum + true superposition (bidirectional)
    - TRUE superposition: |ψ⟩ = α|ψ₁⟩ + β|ψ₂⟩ + γ|ψ₃⟩ (states combined BEFORE measurement)
    - Added new research question (Q4): Does TRUE superposition improve performance?
    - Updated all sections to reflect 16 models
- **Previous Version**: `ABLATION_STUDY_IMPLEMENTATION_PLAN_V2.md`
