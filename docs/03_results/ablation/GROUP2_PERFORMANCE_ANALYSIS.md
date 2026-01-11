# Group 2 Performance Analysis: Why ClassicalQuantumAttention (2a) Outperforms Other Quantum Mixing Models

**Date:** December 27, 2025
**Author:** Research Team
**Related Files:** `models/QuantumMixingSSM.py`, `models/QuantumHydraSSM.py`

---

## Executive Summary

This document analyzes why **ClassicalQuantumAttention (2a)** consistently outperforms other Group 2 models that use classical feature extraction with quantum mixing:

| Model | 40Hz | 80Hz | 160Hz | Pattern |
|-------|------|------|-------|---------|
| **2a** ClassicalQuantumAttention | **72.86%** | **71.56%** | **72.06%** | Stable, best |
| 2b ClassicalMambaQuantumSSM | 59.41% | 50.77% | 49.89% | Catastrophic failure |
| 2c ClassicalHydraQuantumSSM | 68.68% | 51.74% | 52.21% | Catastrophic failure |
| 2d QuantumMambaHydraSSM | 69.95% | 69.91% | 70.18% | Stable, ~2-3% below 2a |
| 2e QuantumHydraHydraSSM | 70.27% | 71.25% | 69.13% | Stable, ~1-2% below 2a |

The analysis reveals **two distinct failure modes**:
1. **2b/2c**: Sequential quantum SSM causes vanishing gradients at longer sequences
2. **2d/2e**: Local quantum processing loses cross-chunk correlations

---

## Part 1: Shared Components (Feature Extraction)

All Group 2 models share similar classical feature extraction, but with minor differences:

### 2a, 2b, 2c: ClassicalFeatureExtractor

```python
# models/QuantumMixingSSM.py, lines 46-102
class ClassicalFeatureExtractor(nn.Module):
    def __init__(self, n_channels, d_model, dropout=0.1):
        self.embedding = nn.Linear(n_channels, d_model)  # Simple linear
        self.dropout = nn.Dropout(dropout)
```

### 2d, 2e: Input Projection

```python
# models/QuantumHydraSSM.py, lines 434-439
self.input_proj = nn.Sequential(
    nn.Linear(actual_channels, d_model),
    nn.LayerNorm(d_model),  # Additional normalization
    nn.SiLU(),               # Additional activation
    nn.Dropout(dropout)
)
```

**Note:** 2d/2e have slightly more complex feature extraction, but this is not the primary performance differentiator.

---

## Part 2: Why 2a Outperforms 2b/2c (Catastrophic Failure)

### The Problem: Sequential Quantum SSM

Models 2b and 2c use `QuantumSSMCore` which implements **sequential state evolution**:

```python
# models/QuantumMixingSSM.py, lines 304-333
def forward(self, x, h=None):
    # Sequential loop over timesteps
    for t in range(seq_len):
        x_t = x_quantum[:, t, :]  # Current input
        dt_t = delta_scale[:, t, 0]  # Current Delta

        # Quantum circuit called at EACH timestep
        measurements = self.qnode(h, x_t, dt_t, self.circuit_params)

        # State update: h[t] depends on h[t-1]
        h = torch.tanh(q_features[:, -self.n_qubits:]) * np.pi

        outputs.append(y_t)
```

This implements the SSM recurrence: **h[t] = f(h[t-1], x[t])**

### Why This Causes Failure

#### 1. Vanishing Gradients Through Sequential Quantum Circuits

```
Gradient Flow (Backpropagation):

Loss → y[T] → QNode[T] → h[T-1] → QNode[T-1] → h[T-2] → ... → QNode[1] → h[0]
              ↓              ↓                   ↓              ↓
           ∂L/∂θ          ∂L/∂θ               ∂L/∂θ          ∂L/∂θ

At each QNode, gradients pass through:
- Quantum measurements (expectation values)
- State normalization (tanh)
- Parameter projections

Result: Gradients diminish exponentially with sequence length
```

#### 2. Frequency-Dependent Sequence Length

| Frequency | Sequence Length | Quantum Circuit Calls | Result |
|-----------|-----------------|----------------------|--------|
| 40Hz | Short | Manageable | Modest performance (59-69%) |
| 80Hz | Medium | Many | **~50% (random chance)** |
| 160Hz | Long | Very many | **~50% (random chance)** |

At 80Hz and 160Hz, the gradient chain becomes too long, causing complete training collapse.

#### 3. Model 2c is Worse Than 2b

2c uses `QuantumBidirectionalSSMCore` which has **two** sequential SSM passes:

```python
# models/QuantumMixingSSM.py, lines 398-427
def forward(self, x):
    # Forward pass: seq_len quantum calls
    forward_out, _ = self.forward_ssm(x)

    # Backward pass: seq_len more quantum calls
    x_reversed = torch.flip(x, dims=[1])
    backward_out, _ = self.backward_ssm(x_reversed)
    backward_out = torch.flip(backward_out, dims=[1])

    # Combine
    y = alpha * forward_out + beta * backward_out + gamma * diagonal_out
```

**2× more quantum circuit calls = 2× worse gradient flow**

### Evidence: Training Time

| Model | 40Hz | 80Hz | 160Hz |
|-------|------|------|-------|
| 2a | 1.20h | 1.14h | 1.47h |
| 2b | 1.90h | 1.74h | **2.98h** |
| 2c | 3.83h | 3.34h | **13.30h** |

The dramatically longer training times for 2b/2c (especially 2c at 160Hz = 13.30h) indicate excessive quantum computation through sequential processing.

### Comparison with 2a

**2a (ClassicalQuantumAttention)** uses `QuantumAttentionMixingCore` with **global LCU mixing**:

```python
# models/QuantumMixingSSM.py, lines 855-893
def forward(self, x):
    # Project ALL chunks to circuit parameters at once
    chunk_params = torch.sigmoid(self.param_proj(x))  # (batch, n_chunks, n_params)

    # Vectorized state evolution for ALL chunks simultaneously
    evolved_states = self.timestep_state_qnode(repeated_base, flat_params)
    evolved_states = evolved_states.reshape(batch_size, n_chunks, state_dim)

    # Global LCU mixing: |ψ_mixed⟩ = Σ_t α_t U(θ_t)|ψ⟩
    mixed_states = torch.einsum('bti,t->bi', evolved_states, coeffs_norm)
```

**Key difference:** No sequential loop over timesteps. All chunks are processed in parallel, then mixed via quantum superposition.

### Summary: 2a > 2b/2c

| Aspect | 2a (Attention) | 2b/2c (SSM) |
|--------|----------------|-------------|
| Processing | Parallel (all chunks at once) | Sequential (timestep by timestep) |
| Gradient flow | Direct to all timesteps | Through long recurrent chain |
| Circuit calls | O(1) per forward pass | O(seq_len) per forward pass |
| Failure mode | None | Vanishing gradients at long sequences |

---

## Part 3: Why 2a Outperforms 2d/2e (Stable but Lower)

Unlike 2b/2c, models 2d and 2e are **stable** across all frequencies but consistently ~2-3% below 2a.

### Clarification: 2d/2e Are NOT Sequential SSM

Despite the "SSM" in their names, 2d/2e use a **different architecture**:

```python
# models/QuantumHydraSSM.py, lines 339-351
def forward(self, input_angles, forward_params, backward_params, diagonal_params, dt_scale):
    # Three parallel quantum circuits (NOT sequential)
    psi1 = self.forward_circuit(input_angles, forward_params, dt_scale)
    psi2 = self.backward_circuit(input_angles, backward_params, dt_scale)
    psi3 = self.diagonal_circuit(input_angles, diagonal_params, dt_scale)

    # True quantum superposition: |ψ⟩ = α|ψ₁⟩ + β|ψ₂⟩ + γ|ψ₃⟩
    psi_combined = alpha * psi1 + beta * psi2 + gamma * psi3
```

This is **three-branch superposition**, not sequential state evolution. The "Delta-modulation" affects gate scaling, not recurrence.

### Architecture Comparison

| Model | Architecture Type | Processing Scope |
|-------|------------------|------------------|
| 2a | Global LCU Attention | All chunks mixed quantum-mechanically |
| 2b/2c | Sequential Quantum SSM | Per-timestep with state recurrence |
| 2d/2e | Three-Branch Superposition | Per-chunk, then classical aggregation |

### Reason 1: Global vs Local Quantum Mixing

**2a: Global Quantum Mixing**

```python
# models/QuantumMixingSSM.py, lines 882-893
# State evolution for ALL chunks
evolved_states = self.timestep_state_qnode(repeated_base, flat_params)
evolved_states = evolved_states.reshape(batch_size, n_chunks, state_dim)

# LCU mixes ALL chunks in quantum superposition
mixed_states = torch.einsum('bti,t->bi', evolved_states, coeffs_norm)
# Output: (batch, 2^n_qubits) - single mixed state capturing all temporal info
```

All n_chunks contribute to ONE quantum superposed state. Cross-chunk correlations are captured quantum-mechanically.

**2d/2e: Local Quantum + Classical Aggregation**

```python
# models/QuantumHydraSSM.py, lines 543-554
# Each chunk processed INDEPENDENTLY
quantum_features = self.quantum_core(
    input_angles, forward_params, backward_params, diagonal_params, dt_scale
)  # (batch*n_chunks, 3*n_qubits)

# Reshape to separate chunks
chunk_features = self.output_proj(quantum_features)
chunk_features = chunk_features.reshape(batch_size, n_chunks, self.d_model)

# CLASSICAL aggregation across chunks
seq_attn = self.seq_attention(chunk_features)
seq_weights = F.softmax(seq_attn, dim=1)
sequence_repr = (seq_weights * chunk_features).sum(dim=1)  # Weighted sum
```

Each chunk is processed through quantum circuits independently. Cross-chunk information is combined via **classical attention**, losing potential quantum correlations.

```
2a: Chunk1 ──┐
     Chunk2 ──┼──► Quantum LCU ──► Single Mixed State ──► Classification
     Chunk3 ──┤    Superposition
     ...    ──┘

2d/2e: Chunk1 ──► Quantum 3-Branch ──┐
       Chunk2 ──► Quantum 3-Branch ──┼──► Classical Attention ──► Classification
       Chunk3 ──► Quantum 3-Branch ──┤
       ...                          ──┘
```

### Reason 2: Mixing Coefficient Flexibility

| Model | Mixing Coefficients | Flexibility |
|-------|---------------------|-------------|
| **2a** | n_chunks complex numbers | Each chunk has its own learnable weight α_t |
| **2d/2e** | 3 real numbers (α, β, γ) | Same 3-way split for ALL chunks |

**2a** can learn that specific chunks (timesteps) are more important:
```python
# models/QuantumMixingSSM.py, lines 756-758
self.mix_coeffs = nn.Parameter(
    torch.rand(n_chunks, dtype=torch.complex64)
)
```

**2d/2e** use the same three coefficients for all chunks:
```python
# models/QuantumHydraSSM.py, lines 230-235
self.alpha_real = nn.Parameter(torch.tensor(init_val))
self.beta_real = nn.Parameter(torch.tensor(init_val))
self.gamma_real = nn.Parameter(torch.tensor(init_val))
```

### Reason 3: Architecture Complexity

**2a**: Simpler architecture
- 1 state evolution circuit per chunk
- 1 QFF circuit for final measurement

**2d/2e**: More complex architecture
- 3 state circuits per chunk (forward, backward, diagonal)
- 3× more quantum parameters to optimize

```python
# models/QuantumHydraSSM.py, lines 449-451
self.forward_param_proj = nn.Linear(d_model, self.params_per_branch)
self.backward_param_proj = nn.Linear(d_model, self.params_per_branch)
self.diagonal_param_proj = nn.Linear(d_model, self.params_per_branch)
```

More parameters with limited training data leads to:
- Harder optimization landscape
- Potential overfitting
- Less efficient use of quantum resources

### Note: Per-Sample Measurement Loop

Both 2a and 2d/2e have per-sample measurement loops due to PennyLane's StatePrep limitation:

**2a** (lines 901-905):
```python
for b in range(batch_size):
    expvals = self.qff_qnode(mixed_states[b], self.qff_params)
```

**2d/2e** (lines 370-376):
```python
for b in range(batch_size):
    m = self.measurement_circuit(psi_normalized[b])
```

This is **NOT** a differentiating factor between 2a and 2d/2e.

### Summary: 2a > 2d/2e

| Aspect | 2a (Attention) | 2d/2e (Three-Branch) |
|--------|----------------|---------------------|
| Quantum mixing scope | **Global** (all chunks) | Local (per chunk) |
| Cross-chunk correlation | Quantum superposition | Classical attention |
| Mixing coefficients | n_chunks (adaptive) | 3 (fixed structure) |
| Quantum circuits/chunk | 1 | 3 |
| Architecture | Simpler | More complex |

---

## Part 4: Design Implications

### For Quantum Mixing Architectures

1. **Prefer global quantum mixing** over local processing with classical aggregation
2. **Avoid sequential quantum circuits** that create long gradient chains
3. **Use adaptive mixing coefficients** (per-timestep) rather than fixed structures
4. **Simpler quantum architectures** often outperform complex ones with limited data

### Architecture Recommendations

| Design Goal | Recommended Approach | Avoid |
|-------------|---------------------|-------|
| Temporal mixing | Global LCU superposition | Sequential SSM recurrence |
| Multi-branch | Single global superposition | Per-chunk independent processing |
| Coefficients | Learnable per-timestep | Fixed small set |
| Complexity | Minimal circuits | Overparameterized designs |

---

## Appendix: Model Architecture Summary

### 2a: ClassicalQuantumAttention
```
Input → ClassicalFeatureExtractor → Chunk Aggregation
      → QuantumAttentionMixingCore (Global LCU)
      → Classification Head
```

### 2b: ClassicalMambaQuantumSSM
```
Input → ClassicalFeatureExtractor → Chunk Aggregation
      → QuantumSSMCore (Sequential, unidirectional)
      → Classification Head
```

### 2c: ClassicalHydraQuantumSSM
```
Input → ClassicalFeatureExtractor → Chunk Aggregation
      → QuantumBidirectionalSSMCore (Sequential, forward + backward)
      → Classification Head
```

### 2d: QuantumMambaHydraSSM
```
Input → Input Projection (Linear + LayerNorm + SiLU)
      → Chunk Aggregation
      → QuantumHydraSSMCore (Three-branch superposition, per-chunk)
      → Classical Sequence Aggregation
      → Classification Head
```

### 2e: QuantumHydraHydraSSM
```
Input → Input Projection (Linear + LayerNorm + SiLU)
      → Chunk Aggregation
      → 2× QuantumHydraSSMCore (Bidirectional three-branch)
      → Classical Sequence Aggregation
      → Classification Head
```

---

## References

- `models/QuantumMixingSSM.py`: Contains 2a, 2b, 2c implementations
- `models/QuantumHydraSSM.py`: Contains 2d, 2e implementations
- `docs/ABLATION_EEG_RESULTS_SUMMARY.md`: Full experimental results

---

*Analysis based on code review and experimental results from the EEG classification ablation study.*
