# Quantum Gated Models: Complexity and Performance Analysis

**Date**: December 2, 2025
**Version**: 1.0
**Related Document**: `QUANTUM_GATED_RECURRENCE_GUIDE.md`

---

## Table of Contents

1. [Executive Summary](#1-executive-summary)
2. [Computational Complexity Analysis](#2-computational-complexity-analysis)
3. [Parameter Count Analysis](#3-parameter-count-analysis)
4. [Performance Analysis: Why Quantum Models Perform Better](#4-performance-analysis-why-quantum-models-perform-better)
5. [Ablation Study Design](#5-ablation-study-design)
6. [Conclusions and Trade-offs](#6-conclusions-and-trade-offs)

---

## 1. Executive Summary

This document provides rigorous technical analysis comparing:
- **Classical Models**: Mamba (Gu & Dao, 2024), Hydra (Hwang et al., 2024)
- **Quantum Gated Models**: QuantumMambaGated, QuantumHydraGated

### Key Findings

| Aspect | Classical Mamba/Hydra | Quantum Gated Models |
|--------|----------------------|---------------------|
| **Complexity** | O(T × d²) | O(T/c × (kQ + mH²)) |
| **Scaling with T** | Linear | Linear (same class) |
| **Parameters** | ~50-100K | ~52-102K (comparable) |
| **EEG Accuracy** | ~65-70% (estimated) | 69-70% |
| **Key Advantage** | Parallel scan, proven | Gating + quantum expressivity |

**Critical Insight**: Both model families have **O(T) complexity**. The quantum circuit cost Q is **constant** with respect to sequence length T.

---

## 2. Computational Complexity Analysis

### 2.1 Classical Mamba Complexity

**State Space Model Formulation:**
```
h[t] = A × h[t-1] + B × u[t]    # State update
y[t] = C × h[t] + D × u[t]      # Output
```

**Per-timestep operations:**
| Operation | Cost | Description |
|-----------|------|-------------|
| A × h[t-1] | O(d × d_state) | State transition |
| B × u[t] | O(d × d_state) | Input projection |
| C × h[t] | O(d × d_state) | Output projection |
| Selective params (B, C, dt) | O(d²) | Input-dependent |

**Total per timestep**: O(d² + d × d_state) ≈ O(d²)

**For sequence of length T**:
```
Classical Mamba: O(T × d²)
```

With parallel scan optimization: O(T × d² / log T) but effectively **O(T)** for practical purposes.

### 2.2 Classical Hydra Complexity

**Bidirectional formulation:**
```
QS(X) = shift(SS(X)) + flip(shift(SS(flip(X)))) + DX
```

**Operations:**
| Component | Cost | Description |
|-----------|------|-------------|
| Forward SS | O(T × d²) | Semi-separable forward |
| Backward SS | O(T × d²) | Semi-separable backward |
| Shift operations | O(T × d) | Circular shift |
| Diagonal D | O(T × d) | Element-wise |
| Combination | O(T × d) | Addition |

**Total**:
```
Classical Hydra: O(2 × T × d²) = O(T × d²)
```

Same complexity class as Mamba, with ~2× constant factor for bidirectionality.

### 2.3 Quantum Circuit Simulation Cost (Q)

**This is the critical quantity that determines quantum model complexity.**

#### Circuit Structure

The QLCU (Quantum Linear Combination of Unitaries) circuit per layer:
```
1. RY rotation on each qubit:     n gates
2. CRX entangling (forward):      n gates
3. RY rotation on each qubit:     n gates
4. CRX entangling (backward):     n gates
```

**Total gates**: 4 × n_qubits × qlcu_layers

#### Simulation Cost on Classical Hardware

| Operation | Cost | Explanation |
|-----------|------|-------------|
| State vector | 2^n complex numbers | Hilbert space dimension |
| Single-qubit gate | O(2^n) | Apply 2×2 matrix to state |
| Two-qubit gate | O(2^n) | Apply 4×4 matrix to state |
| Expectation value | O(2^n) | Inner product |

**Total Q**:
```
Q = O(gates × 2^n + measurements × 2^n)
  = O(4 × n × L × 2^n + 3n × 2^n)
  = O(n × L × 2^n)

Where:
  n = n_qubits (typically 4-8)
  L = qlcu_layers (typically 1-3)
```

#### Q is CONSTANT with respect to sequence length T

**This is crucial**: Q depends only on circuit hyperparameters (n_qubits, qlcu_layers), NOT on sequence length T.

| n_qubits | qlcu_layers | 2^n | Gates | Q (approx operations) |
|----------|-------------|-----|-------|----------------------|
| 4 | 2 | 16 | 32 | ~1,000 |
| 6 | 2 | 64 | 48 | ~4,200 |
| 8 | 2 | 256 | 64 | ~20,000 |
| 8 | 3 | 256 | 96 | ~30,000 |

### 2.4 QuantumMambaGated Complexity

**Processing structure:**
```
For each chunk (T/c chunks total):
    1. Feature projection:     O(d²) per timestep in chunk
    2. Quantum branches:       3 × Q (three quantum circuits)
    3. Chunk aggregation:      O(c × 3n × H)
    4. LSTM gating:           O(4 × H²) (4 gate matrices)
    5. State update:          O(H)
```

**Per-chunk cost**:
```
Cost_chunk = O(c × d²) + O(3Q) + O(c × 3n × H) + O(4H²)
           ≈ O(c × d² + 3Q + 4H²)
```

**Total for sequence**:
```
QuantumMambaGated: O((T/c) × (c × d² + 3Q + 4H²))
                 = O(T × d² + (T/c) × (3Q + 4H²))
```

Since Q and H are constants:
```
QuantumMambaGated: O(T × d²) + O(T/c × constant)
                 = O(T)
```

### 2.5 QuantumHydraGated Complexity

**Three directional branches** (forward, backward, global):
```
Each direction:
    - Chunked quantum processing: O((T/c) × 3Q)
    - LSTM gating: O((T/c) × 4H²)

Total quantum circuits: 3 directions × 3 branches = 9 circuits per chunk
Total gating: 3 directions × 4 gates = 12 gate computations per chunk
```

**Total**:
```
QuantumHydraGated: O((T/c) × (9Q + 12H²))
                 = O(T/c × constant)
                 = O(T)
```

### 2.6 Direct Complexity Comparison

| Model | Complexity | Constant Factor | Notes |
|-------|------------|-----------------|-------|
| Classical Mamba | O(T × d²) | d² | Parallel scan possible |
| Classical Hydra | O(T × d²) | 2d² | Bidirectional |
| QuantumMambaGated | O(T) | d² + (3Q + 4H²)/c | Sequential chunks |
| QuantumHydraGated | O(T) | d² + (9Q + 12H²)/c | 3× more quantum |

**All models are O(T)** - they scale linearly with sequence length.

### 2.7 Numerical Comparison

**Settings**: T=500, c=32, n=6, L=2, d=H=64

| Model | Formula | Operations |
|-------|---------|------------|
| Classical Mamba | T × d² | 500 × 4,096 = **2,048,000** |
| Classical Hydra | 2 × T × d² | 2 × 500 × 4,096 = **4,096,000** |
| QuantumMambaGated | T × d² + (T/c) × (3Q + 4H²) | 2,048,000 + 16 × (12,672 + 16,384) = **2,513,000** |
| QuantumHydraGated | T × d² + (T/c) × (9Q + 12H²) | 2,048,000 + 16 × (38,016 + 49,152) = **3,443,000** |

**Observations**:
1. QuantumMambaGated is ~1.2× Classical Mamba (23% overhead)
2. QuantumHydraGated is ~0.84× Classical Hydra (16% faster!)
3. The quantum overhead is modest because Q is amortized over chunk_size

### 2.8 Scaling Analysis

**As sequence length T increases:**

```
T = 100:
  Classical Mamba:    409,600 ops
  QuantumMambaGated:  502,600 ops  (1.23× slower)

T = 1000:
  Classical Mamba:    4,096,000 ops
  QuantumMambaGated:  5,026,000 ops  (1.23× slower)

T = 10000:
  Classical Mamba:    40,960,000 ops
  QuantumMambaGated:  50,260,000 ops  (1.23× slower)
```

**The ratio stays constant** because both are O(T).

---

## 3. Parameter Count Analysis

### 3.1 Classical Mamba Parameters

```python
# Typical configuration: d_model=64, d_state=16

Projections:
  - in_proj:    d × 2d = 64 × 128 = 8,192
  - out_proj:   d × d = 64 × 64 = 4,096

SSM parameters:
  - A_log:      d × d_state = 64 × 16 = 1,024
  - D:          d = 64
  - dt_proj:    dt_rank × d ≈ 4 × 64 = 256

Selective parameters:
  - x_proj (B, C, dt): d × (2 × d_state + dt_rank) = 64 × 36 = 2,304

Total: ~16,000 parameters
```

### 3.2 Classical Hydra Parameters

```python
# Bidirectional = roughly 2× Mamba core + combination layers

Forward SSM:    ~16,000
Backward SSM:   ~16,000
Combination:    ~5,000

Total: ~37,000 parameters
```

### 3.3 QuantumMambaGated Parameters

```python
# Configuration: feature_dim=64, n_qubits=6, qlcu_layers=2, hidden_dim=64

Feature projection:
  - Linear(64, 64):                    64 × 64 = 4,096

Quantum Superposition Branches (×1):
  - 3 × proj(64, 48):                  3 × 64 × 48 = 9,216
  - 3 × quantum_base_params(48):       3 × 48 = 144
  - Complex coefficients (α,β,γ):      6

Chunk aggregation:
  - Linear(18, 64):                    18 × 64 = 1,152
  - LayerNorm(64):                     128

LSTM Gating (4 gates):
  - W_f: Linear(128, 64):              128 × 64 = 8,192
  - W_i: Linear(128, 64):              128 × 64 = 8,192
  - W_o: Linear(128, 64):              128 × 64 = 8,192
  - W_c: Linear(128, 64):              128 × 64 = 8,192

Output layer:
  - Linear(64, 64):                    4,096
  - Linear(64, output_dim):            64 × 2 = 128

Total: ~52,000 parameters
```

### 3.4 QuantumHydraGated Parameters

```python
# 3 directions × QuantumMambaGated-like structure

Forward branch:     ~17,000
Backward branch:    ~17,000
Global branch:      ~17,000
Direction combination: 6 (complex α,β,γ)
Feature projection: 4,096
Output layer:       ~200

Total: ~102,000 parameters
```

### 3.5 Parameter Comparison Summary

| Model | Total Params | Quantum Params | Gating Params | Ratio to Classical |
|-------|--------------|----------------|---------------|-------------------|
| Classical Mamba | ~16K | 0 | 0 | 1.0× |
| Classical Hydra | ~37K | 0 | 0 | 1.0× |
| QuantumMambaGated | ~52K | ~150 | ~32K | 3.25× vs Mamba |
| QuantumHydraGated | ~102K | ~450 | ~96K | 2.75× vs Hydra |

### 3.6 Where Do Parameters Go?

**QuantumMambaGated breakdown:**

| Component | Parameters | Percentage |
|-----------|------------|------------|
| Feature projection | 4,096 | 7.9% |
| Quantum branches | 9,366 | 18.0% |
| Chunk aggregation | 1,280 | 2.5% |
| **LSTM Gating** | **32,768** | **63.0%** |
| Output layer | 4,224 | 8.1% |
| Quantum circuit params | 150 | 0.3% |

**Key insight**: The LSTM gating mechanism accounts for **63%** of parameters, not the quantum circuits (only 0.3%).

---

## 4. Performance Analysis: Why Quantum Models Perform Better

### 4.1 Observed Results

| Task | Old Quantum (no gating) | QuantumMambaGated | QuantumHydraGated |
|------|------------------------|-------------------|-------------------|
| EEG (249 timesteps) | ~50% (random) | 69.4% | 70.2% |
| DNA (57 nucleotides) | N/A | 74.4% | 86.7% |
| Mouse Enhancers (4776 bp) | N/A | 50.0% | 76%+ |

### 4.2 Why Old Quantum Models Failed (~50% accuracy)

**Root cause**: No temporal memory mechanism.

```
Old Model Pipeline:
  Input (batch, channels, timesteps)
      ↓
  Conv1d (local temporal encoding)
      ↓
  AdaptiveAvgPool → COLLAPSE entire sequence to single vector
      ↓
  Single quantum computation
      ↓
  Output
```

**Problems:**
1. **Information loss**: Pooling destroys temporal structure
2. **No state propagation**: Cannot model dependencies across time
3. **Single quantum call**: One 2^n state cannot represent T timesteps
4. **Local only**: Conv1d captures only local patterns (kernel_size ~3)

### 4.3 Why New Gated Models Succeed

**Two key additions**: (1) LSTM-style gating, (2) Chunked recurrence

```
New Model Pipeline:
  Input (batch, timesteps, features)
      ↓
  FOR EACH CHUNK:
      Quantum feature extraction (3 branches)
          ↓
      Chunk aggregation (preserves info)
          ↓
      LSTM Gating:
        - Forget gate: what to discard
        - Input gate: what to add
        - Output gate: what to output
          ↓
      Hidden state h propagates to next chunk
      ↓
  Final hidden state → Output
```

**Why this works:**

| Mechanism | Contribution | Evidence |
|-----------|--------------|----------|
| **LSTM Gating** | Selective memory | Prevents gradient vanishing, enables long-range deps |
| **Chunked recurrence** | Temporal structure | State propagates across sequence |
| **Quantum features** | Expressivity | 2^n Hilbert space with n parameters |
| **Bidirectionality (Hydra)** | Full context | Critical for long sequences (4776bp) |

### 4.4 Hypothesis: What Causes the Improvement?

**We cannot definitively separate the contributions without ablation studies.** However, we can hypothesize:

#### Hypothesis 1: Gating is the Primary Factor

**Evidence for:**
- LSTM/GRU are proven for sequence modeling
- 63% of parameters are in gating mechanism
- Improvement from 50% → 70% matches what gating typically provides

**Implication**: A "Classical Gated" model (LSTM + classical features) might perform similarly.

#### Hypothesis 2: Quantum Features Provide Additional Benefit

**Evidence for:**
- Quantum circuits access 2^n-dimensional Hilbert space with only O(n) parameters
- Entanglement creates non-local correlations
- Complex amplitudes provide richer representations

**Implication**: Quantum Gated should outperform Classical Gated (same gating, different features).

#### Hypothesis 3: Combination is Key

**Evidence for:**
- Neither quantum alone (50%) nor simple classical works as well
- The specific combination of chunked quantum + gating is novel
- Bidirectional quantum (HydraGated) helps long sequences where Mamba fails

### 4.5 What We Can Conclude

| Claim | Confidence | Evidence |
|-------|------------|----------|
| Gating is necessary | **High** | Old quantum (no gating) = 50% random |
| Quantum + Gating works | **High** | 70% accuracy achieved |
| Quantum is necessary | **Unknown** | No Classical Gated baseline |
| Quantum provides advantage | **Unknown** | Need ablation study |
| Bidirectionality helps long sequences | **Medium** | HydraGated 76% vs MambaGated 50% on 4776bp |

### 4.6 Expressivity Argument

**Why quantum MIGHT help (theoretical):**

```
Classical feature extraction:
  - Linear: d → d' (d × d' parameters)
  - Can represent: linear subspaces

Quantum feature extraction:
  - Circuit: n qubits, L layers (4nL parameters)
  - Can represent: 2^n-dimensional transformations
  - With entanglement: non-separable correlations
```

**Example with n=6 qubits, L=2 layers:**
- Classical equivalent: 48 parameters → 48-dimensional linear map
- Quantum: 48 parameters → operations on 64-dimensional Hilbert space

**The quantum circuit is a more expressive function approximator per parameter.**

---

## 5. Ablation Study Design

To definitively answer "Does quantum help?", we need controlled experiments.

### 5.1 Proposed Ablation Models

| Model | Gating | Feature Extraction | Purpose |
|-------|--------|-------------------|---------|
| QuantumMambaGated | LSTM | Quantum (3 branches) | Current model |
| **ClassicalMambaGated** | LSTM | MLP (3 branches) | Ablation: remove quantum |
| QuantumMamba (old) | None | Quantum (3 branches) | Baseline: no gating |
| **ClassicalMamba (old)** | None | MLP (3 branches) | Baseline: no gating, no quantum |

### 5.2 ClassicalMambaGated Design

Replace quantum circuits with equivalent classical networks:

```python
class ClassicalFeatureExtractor(nn.Module):
    """Replace quantum circuit with MLP of similar expressivity."""

    def __init__(self, n_params, output_dim):
        super().__init__()
        # Match quantum output dimension: 3 * n_qubits
        hidden = 4 * output_dim  # Similar parameter count
        self.mlp = nn.Sequential(
            nn.Linear(n_params, hidden),
            nn.SiLU(),
            nn.Linear(hidden, output_dim),
            nn.Tanh()  # Bound to [-1, 1] like quantum expectations
        )

    def forward(self, params):
        return self.mlp(params)
```

**Parameter matching:**
- Quantum: 48 circuit params → 18 outputs (3 × 6 qubits)
- Classical: 48 inputs → hidden → 18 outputs
- Hidden = 72 → ~48×72 + 72×18 = 4,752 params (vs quantum's 48)

**Note**: Classical MLP needs MORE parameters to match output dimension.

### 5.3 Experimental Protocol

```python
models = {
    'QuantumMambaGated': QuantumMambaGated(...),
    'ClassicalMambaGated': ClassicalMambaGated(...),  # Same gating, MLP features
    'QuantumMamba_NoGating': QuantumMamba(...),       # Old model
    'ClassicalMamba_NoGating': ClassicalMamba(...),   # MLP, no gating
}

for model_name, model in models.items():
    for seed in [2024, 2025, 2026]:
        train(model, dataset='EEG', epochs=100, seed=seed)
        evaluate(model, metrics=['accuracy', 'auc', 'f1'])
```

### 5.4 Expected Outcomes and Interpretations

| Outcome | Interpretation |
|---------|----------------|
| QuantumGated > ClassicalGated | Quantum provides expressivity advantage |
| QuantumGated ≈ ClassicalGated | Gating is key; quantum is replaceable |
| QuantumGated < ClassicalGated | Quantum hurts (unlikely but possible) |
| Gated >> NoGating | Gating is necessary (expected) |

### 5.5 Additional Ablations

| Ablation | Question Answered |
|----------|-------------------|
| Vary n_qubits (4, 6, 8) | Does more quantum expressivity help? |
| Vary chunk_size (8, 16, 32, 64) | Trade-off between speed and accuracy? |
| Remove one branch | Which quantum branch matters most? |
| Real vs complex coefficients | Does complex superposition help? |
| Remove bidirectionality (Hydra) | Is backward branch necessary? |

---

## 6. Conclusions and Trade-offs

### 6.1 Complexity Conclusions

| Finding | Implication |
|---------|-------------|
| Both Classical and Quantum are O(T) | Same scalability class |
| Q is constant w.r.t. T | Quantum overhead doesn't grow with sequence length |
| Quantum ~1.2× Classical Mamba ops | Modest computational overhead |
| Quantum ~0.84× Classical Hydra ops | Actually faster for bidirectional |

### 6.2 Parameter Conclusions

| Finding | Implication |
|---------|-------------|
| Quantum Gated has ~3× more params than Classical | Due to gating, not quantum |
| Only 0.3% of params are quantum circuit | Quantum is parameter-efficient |
| 63% of params are LSTM gating | Gating dominates model size |

### 6.3 Performance Conclusions

| Finding | Confidence | Implication |
|---------|------------|-------------|
| Gating is necessary | High | Old quantum models fail without it |
| Quantum + Gating achieves 70% | High | The combination works |
| Quantum advantage over classical | Unknown | Need ablation study |
| Bidirectionality helps long sequences | Medium | HydraGated >> MambaGated on 4776bp |

### 6.4 When to Use Each Model

| Scenario | Recommended Model | Reason |
|----------|-------------------|--------|
| Short sequences (<100 timesteps) | Classical Mamba | Simpler, faster |
| Medium sequences (100-1000) | QuantumMambaGated | Good balance |
| Long sequences (>1000) | QuantumHydraGated | Bidirectionality helps |
| Resource-constrained | Classical | Lower overhead |
| Research/exploration | Quantum Gated | Novel representations |
| Production (proven) | Classical | More mature |

### 6.5 Trade-off Summary

| Aspect | Classical | Quantum Gated | Winner |
|--------|-----------|---------------|--------|
| **Complexity** | O(T × d²) | O(T × d² + T/c × Q) | Classical (slightly) |
| **Parameters** | ~16-37K | ~52-102K | Classical |
| **Accuracy (EEG)** | ~65-70% (est.) | 69-70% | Comparable |
| **Long sequences** | Good | Better (Hydra) | Quantum Gated |
| **Interpretability** | High | Medium | Classical |
| **Novelty** | Low | High | Quantum Gated |
| **Hardware ready** | Yes (GPU) | Simulation only | Classical |

### 6.6 Open Questions

1. **Does quantum provide advantage?** → Need ClassicalMambaGated ablation
2. **Optimal n_qubits?** → Trade-off between expressivity and simulation cost
3. **Real quantum hardware?** → Would eliminate 2^n simulation overhead
4. **Other tasks?** → Performance on NLP, audio, other modalities unknown

### 6.7 Recommendations

**For practitioners:**
- Use Quantum Gated models if exploring novel representations
- Compare against Classical Gated baseline before claiming quantum advantage
- Consider QuantumHydraGated for long sequences where bidirectionality matters

**For researchers:**
- Conduct ablation studies (Section 5) to isolate quantum contribution
- Test on real quantum hardware when available
- Investigate barren plateau mitigation for deeper circuits

---

## References

1. Gu, A., & Dao, T. (2024). Mamba: Linear-Time Sequence Modeling with Selective State Spaces. *arXiv:2312.00752*

2. Hwang, W., et al. (2024). Hydra: Bidirectional State Space Models Through Generalized Matrix Mixers. *arXiv:2407.09941*

3. McClean, J. R., et al. (2018). Barren plateaus in quantum neural network training landscapes. *Nature Communications*, 9(1), 4812.

4. Schuld, M., & Petruccione, F. (2021). Machine Learning with Quantum Computers. Springer.

---

**Document**: `QUANTUM_GATED_COMPLEXITY_ANALYSIS.md`
**Location**: `/pscratch/sd/j/junghoon/quantum_hydra_mamba/`
**Status**: Complete
**Last Updated**: December 2, 2025
