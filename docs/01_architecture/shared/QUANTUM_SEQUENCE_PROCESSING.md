# Quantum Sequence Processing: Analysis and Solutions

**Date**: November 22, 2025

---

## 1. Original QuantumMamba and QuantumHydra: Sequence Processing

### QuantumMambaTS

```python
def forward(self, x):
    x_conv = self.temporal_conv(x)       # (batch, channels, timesteps)
    x_pooled = self.pool(x_conv).squeeze(-1)  # Pool ALL timesteps
    output = self.quantum_mamba(x_pooled)     # Single quantum call
```

**Behavior**:
- Destroys temporal structure via average pooling
- All timesteps collapsed before quantum processing
- **No sequence learning capability**

### QuantumHydraTS

```python
def forward(self, x):
    x_flat = x.reshape(batch * n_timesteps, -1)  # Flatten
    out_flat = self.quantum_hydra(x_flat)        # Independent processing
    aggregated = weighted_sum(out_flat)          # Aggregate
```

**Behavior**:
- Processes each timestep independently
- No cross-timestep information flow
- **No sequential dependencies learned**

### Summary

| Model | Processing | Cross-timestep Info | Sequence Learning |
|-------|-----------|---------------------|-------------------|
| QuantumMambaTS | Pool | None | None |
| QuantumHydraTS | Independent | None | None |

---

## 2. QuantumParallelScan: Improvements and Limitations

### How It Works

Computes cumulative unitary products using parallel scan:

```python
U_cum[t] = U_t @ U_{t-1} @ ... @ U_1
psi[t] = U_cum[t] @ psi_0
```

### Improvements

1. **True Recurrence**: Each timestep depends on all previous timesteps
2. **Long-Range Dependencies**: Information flows through cumulative composition
3. **Efficient Computation**: O(log T) sequential steps via parallel scan

### Parallel Scan Algorithm

```
Input: [U_1, U_2, U_3, U_4, U_5, U_6, U_7, U_8]

Step 1 (stride=1): Pairwise products
Step 2 (stride=2): Quad products
Step 3 (stride=4): Full cumulative products

Result: All cumulative products in log₂(T) steps
```

### Limitations

#### 1. No Forgetting Mechanism

**Classical Mamba**:
```python
h_t = A_t * h_{t-1} + B_t * x_t  # A_t ∈ [0,1] enables forgetting
```

**QuantumParallelScan**:
```python
U_cum[t] = U_t @ U_cum[t-1]  # Pure composition, no decay
```

**Problem**: Information only accumulates, causing state saturation for long sequences.

#### 2. Memory Overhead

```python
unitaries: (batch, seq_len, 2^n_qubits, 2^n_qubits)
# For n_qubits=6, seq_len=1000: ~32 MB per sample
```

#### 3. Unitary Constraints

- Norm preservation: ||U @ x|| = ||x||
- No amplitude decay or selective emphasis
- Information rotated but not filtered

#### 4. Gradient Flow

Long chains of matrix multiplications may cause vanishing/exploding gradients.

### Summary

| Aspect | Improvement | Limitation |
|--------|-------------|------------|
| Recurrence | Cumulative unitaries | No forgetting |
| Complexity | O(log T) | O(T × 4^n) memory |
| Dependencies | Long-range | No selective filtering |

---

## 3. Gating Mechanism for Long Sequence Learning

### Why Gating Works: Classical Mamba's Insight

```python
dt = softplus(dt_proj(x))      # Input-dependent time step
A = exp(-dt * A_base)          # Decay factor [0, 1]
B = dt * B_proj(x)             # Input-dependent input matrix

h_t = A * h_{t-1} + B * x_t    # Selective update
```

- `A ≈ 0`: Forget previous state
- `A ≈ 1`: Remember previous state
- All parameters are **input-dependent** (selective)

### Gating Approaches for Quantum Circuits

#### Approach 1: State Reset Gate

```python
gate = sigmoid(gate_proj(x[:, t]))  # [0, 1]
h_new = U_t @ h
h = gate * h_new + (1 - gate) * initial_state
```

- `gate ≈ 0`: Reset to initial state
- `gate ≈ 1`: Keep evolved state

#### Approach 2: Selective State Mixing

```python
forget_gate = sigmoid(forget_proj(x))
input_gate = sigmoid(input_proj(x))

q_features = quantum_circuit(x[:, t])
h = forget_gate * h + input_gate * q_features
```

- Forget gate: Memory retention control
- Input gate: New information integration

#### Approach 3: Hybrid Classical-Quantum SSM

```python
# Quantum for features, classical SSM for temporal
q_features = quantum_encoder(x)
h = A * h + B * q_features  # Classical selective SSM
```

### Why Gating Helps

| Challenge | Without Gating | With Gating |
|-----------|----------------|-------------|
| Irrelevant early info | Accumulates | Can be forgotten |
| Important info | May dilute | Can be preserved |
| State saturation | Guaranteed | Controlled |
| Gradient flow | Through all | Shortcuts via gates |

### Concrete Example

**Task**: Sentiment classification of 1000-word review

**Without gating**:
- Early positive descriptions accumulate
- Crucial negative ending may be overwhelmed

**With gating**:
- `gate ≈ 0.1` at timestep 500: Forget old descriptions
- `gate ≈ 0.9` at timestep 900-1000: Remember conclusion

---

## 4. Implementation Comparison

| Model | Memory | Speed | Long Seq | Forgetting |
|-------|--------|-------|----------|------------|
| Original TS | Low | Fast | Poor | N/A |
| ParallelScan | High | Medium | Medium | No |
| Gated | Medium | Medium | **Good** | **Yes** |

---

## 5. Recommendations

### For Best Long Sequence Learning

1. **Chunked Selective Processing**
   - Process chunks in parallel
   - Selective gating between chunks
   - Balances speed and recurrence

2. **Avoid Explicit Unitary Matrices**
   - Use measurement-based approach
   - Reduces memory from O(4^n) to O(2^n)

3. **Input-Dependent Gating**
   - Essential for selective forgetting
   - Enables adaptive memory management

### Implementation Priority

```
Chunked Selective State Mixing
├── Memory efficient (no explicit unitaries)
├── Parallelizable (within chunks)
├── Selective forgetting (between chunks)
└── Proven effective (LSTM/Mamba-style gating)
```

---

## Files Reference

- Original models: `QuantumMamba.py`, `QuantumHydra.py`
- Parallel scan: `QuantumParallelScan.py`
- Gated version: `QuantumGatedRecurrence.py` (to be implemented)
