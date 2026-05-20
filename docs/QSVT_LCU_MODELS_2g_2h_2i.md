# QSVT+LCU Models: 2g, 2h, and 2i

## 1. Motivation

Models 2d and 2e (`QuantumMambaHydraSSM` / `QuantumHydraHydraSSM`) were designed as "Classical Feature Extraction + Quantum Mixing" models. However, they fail this goal in practice. Both aggregate timesteps via a **classical attention** layer *before* the quantum circuit ever sees the data:

```
2d/2e data flow (PROBLEMATIC):
  chunks → classical chunk_attention → single summary vector → quantum circuit
```

The quantum core only processes a single pre-aggregated vector per chunk. Cross-timestep mixing is entirely classical; the quantum circuit never sees individual timesteps.

Models 2g, 2h, and 2i fix this by using the **QSVT+LCU** circuit architecture, where each timestep in a chunk gets its own sim14 unitary and `qml.Select` superposes them coherently inside the quantum circuit. This achieves **true quantum cross-timestep mixing**.

```
2g/2h/2i data flow (CORRECT):
  chunks → per-timestep rotation angles → QSVT+LCU quantum circuit
                                           (all timesteps mixed quantumly)
```

---

## 2. The QSVT+LCU Quantum Circuit

All three models share the same quantum circuit. The circuit is built from three key quantum algorithmic primitives: the **sim14 variational ansatz**, **Linear Combination of Unitaries (LCU)**, and **Quantum Singular Value Transformation (QSVT)**.

### 2.1. sim14 Ansatz (Sim et al., 2019)

Each timestep is encoded as a parameterized unitary via the sim14 circuit structure:

```
Per layer (4 * n_qubits parameters):
  RY on each qubit                      (n_qubits params)
  CRX ring (reverse order)              (n_qubits params)
  RY on each qubit                      (n_qubits params)
  CRX counter-ring                      (n_qubits params)
```

With `n_ansatz_layers` layers, each timestep uses `4 * n_qubits * n_ansatz_layers` rotation parameters. These are **data-dependent** -- derived from the input features via a learned linear projection followed by `Sigmoid * 2pi` to map into `[0, 2pi]`.

### 2.2. Qubit Registers

The circuit operates on two registers:

| Register | Wires | Count | Purpose |
|----------|-------|-------|---------|
| **Main** | `0` to `n_qubits-1` | `n_qubits` | Carries the computational state; measured at the end |
| **Ancilla** | `n_qubits` to `n_qubits + n_ancilla - 1` | `ceil(log2(chunk_size))` | Controls `qml.Select`; addresses which timestep unitary to apply |

Example: with `n_qubits=6` and `chunk_size=32`, `n_ancilla = ceil(log2(32)) = 5`, giving 11 total wires. The ancilla register can address `2^5 = 32` unitaries (one per timestep). If `chunk_size` is not a power of 2, remaining slots are padded with Identity operators.

### 2.3. Circuit Structure

The full quantum circuit for one chunk of `chunk_size` timesteps:

```
PCPhase(phi_0)                                    <- initial signal processing angle

for k in range(degree):
    PREPARE(ancilla)                              <- learnable V on ancilla register
    if k is even:
        SELECT([U(x_0), U(x_1), ..., U(x_T)])    <- data-dependent unitaries
    else:
        SELECT_dagger(...)                        <- adjoint for alternating pattern
    PREPARE_dagger(ancilla)
    PCPhase(phi_{k+1})                            <- signal processing angle

QFF_sim14(main)                                   <- quantum feed-forward on main register

Measure PauliX, PauliY, PauliZ on each main wire <- 3 * n_qubits outputs
```

#### PREPARE

A learnable unitary `V` on the ancilla register that encodes the LCU coefficients. It consists of `n_prep_layers = n_ancilla` layers:

```
Per layer:
  RY(theta), RZ(phi) on each ancilla qubit       (2 * n_ancilla params per layer)
  CNOT chain along ancilla qubits                 (entanglement)
```

Total trainable parameters: `n_ancilla * n_ancilla * 2`.

#### SELECT

`qml.Select` is a multiplexed operation controlled by the ancilla register. When the ancilla is in state `|t>`, the corresponding sim14 unitary `U(x_t)` is applied to the main register. In effect:

```
SELECT |t>|psi> = |t> U(x_t)|psi>
```

This is the mechanism that achieves **quantum cross-timestep mixing** -- PREPARE puts the ancilla into a superposition of all `|t>` states, so SELECT applies a superposition of all timestep unitaries simultaneously:

```
V|0> = sum_t alpha_t |t>

SELECT (V|0>) |psi> = sum_t alpha_t |t> U(x_t) |psi>
```

After PREPARE-dagger projects back, the main register holds a coherent mixture of all timestep-transformed states. The mixing coefficients `alpha_t` are learned through PREPARE's parameters.

#### PCPhase (Projector-Controlled Phase)

`qml.PCPhase(phi, dim, wires)` applies a phase `e^{i*phi}` to the first `dim` basis states and `e^{-i*phi}` to the rest. This is the signal processing primitive of QSVT, enabling the circuit to implement polynomial transformations of the block-encoded operator.

With `degree` iterations, the circuit can implement a degree-`degree` polynomial of the LCU block encoding. Higher degree = more expressive signal processing, but also more circuit depth.

#### QFF (Quantum Feed-Forward)

A single-layer sim14 circuit applied to the main register after the QSVT iterations. This acts as a quantum analogue of a feed-forward layer, providing additional expressivity with `4 * n_qubits` trainable parameters.

#### Measurement

Three Pauli observables are measured on each main-register qubit:

```
[PauliX(0), ..., PauliX(n-1), PauliY(0), ..., PauliY(n-1), PauliZ(0), ..., PauliZ(n-1)]
```

This yields `3 * n_qubits` real-valued expectation values per chunk.

### 2.4. Trainable Quantum Parameters

| Parameter | Shape | Count | Role |
|-----------|-------|-------|------|
| `prepare_params` | `(n_ancilla, n_ancilla, 2)` | `2 * n_ancilla^2` | PREPARE ansatz angles |
| `signal_angles` | `(degree + 1,)` | `degree + 1` | PCPhase signal processing angles |
| `qff_params` | `(4 * n_qubits,)` | `4 * n_qubits` | QFF sim14 layer params |

Note: the per-timestep sim14 unitaries are **not** trainable quantum parameters -- they are derived from the input data through classical layers.

---

## 3. Model 2g: QuantumTSTransformer

**File:** `models/QTSTransformer_v2.py`
**Class:** `QuantumTSTransformer`

The original standalone model that introduced the QSVT+LCU circuit to this codebase. It is a self-contained transformer-style model.

### Architecture

```
Input: (B, feature_dim, T)
  |
  v
Permute -> (B, T, feature_dim)
  |
  v
+ Sinusoidal PE (feature_dim)          <- PE applied in feature_dim space
  |
  v
Linear(feature_dim -> n_rots) + Dropout
  |
  v
Sigmoid * 2pi -> (B, T, n_rots)        <- rotation angles in [0, 2pi]
  |
  v
Chunk into windows of chunk_size
  |
  v
Per-chunk QSVT+LCU -> (B, 3*n_qubits)
  |
  v
Mean pool across chunks -> (B, 3*n_qubits)
  |
  v
Linear(3*n_qubits -> output_dim)        <- single output layer
```

### Key Characteristics

- **PE dimension:** `feature_dim` (applied before the linear projection)
- **Chunk aggregation:** Simple mean pooling across chunk outputs
- **Output head:** Single linear layer, no hidden layers
- **No SSM-style recurrence or attention-based aggregation**

### Constructor Signature

```python
QuantumTSTransformer(
    n_qubits=6, n_timesteps=200, degree=2, n_ansatz_layers=2,
    feature_dim=64, output_dim=2, dropout=0.1, device="cpu",
    chunk_size=32
)
```

---

## 4. Model 2h: QuantumQSVTMambaSSM (Unidirectional)

**File:** `models/QuantumQSVTSSM.py`
**Class:** `QuantumQSVTMambaSSM`

A unidirectional SSM-style wrapper around the QSVT+LCU quantum core. Replaces model 2d (`QuantumMambaHydraSSM`).

### Architecture

```
Input: (B, n_channels, T)
  |
  v
Permute -> (B, T, n_channels)
  |
  v
input_proj: Linear(n_channels -> d_model) + LayerNorm + SiLU + Dropout
  |
  v
+ Sinusoidal PE (d_model)              <- PE applied in d_model space
  |
  v
feature_projection: Linear(d_model -> n_rots) + Dropout
  |
  v
Sigmoid * 2pi -> (B, T, n_rots)
  |
  v
Chunk into windows of chunk_size
  |
  v
Per-chunk QSVTLCUCore -> (B, 3*n_qubits)
  |
  v
output_proj: Linear(3*n_qubits -> d_model) + LayerNorm + SiLU + Dropout
  |
  v
seq_attention: attention-weighted aggregation across chunks -> (B, d_model)
  |
  v
classifier: Linear(d_model -> d_model//2) + SiLU + Dropout + Linear(-> output_dim)
```

### Key Characteristics

- **Two-stage projection:** `n_channels -> d_model -> n_rots` (2g projects directly from `feature_dim -> n_rots`)
- **PE dimension:** `d_model` (richer positional information)
- **Chunk aggregation:** Learned attention weights (not mean pooling)
- **Output head:** Two-layer MLP classifier with SiLU activation
- **Output projection:** Maps quantum outputs back to `d_model` before aggregation, allowing richer per-chunk representations

### Constructor Signature

```python
QuantumQSVTMambaSSM(
    n_qubits=6, qlcu_layers=2, d_model=128, d_state=16,
    feature_dim=64, n_timesteps=125, output_dim=2, dropout=0.1,
    chunk_size=32, device="cpu", n_layers=None, n_channels=None,
    degree=2
)
```

---

## 5. Model 2i: QuantumQSVTHydraSSM (Bidirectional)

**File:** `models/QuantumQSVTSSM.py`
**Class:** `QuantumQSVTHydraSSM`

A bidirectional SSM-style wrapper with two independent QSVT+LCU cores. Replaces model 2e (`QuantumHydraHydraSSM`).

### Architecture

```
Input: (B, n_channels, T)
  |
  v
Permute -> (B, T, n_channels)
  |
  v
input_proj: Linear(n_channels -> d_model) + LayerNorm + SiLU + Dropout
  |
  v
+ Sinusoidal PE (d_model)
  |
  v
feature_projection: Linear(d_model -> n_rots) + Dropout
  |
  v
Sigmoid * 2pi -> (B, T, n_rots)
  |
  v
+------ Forward Pass: chunks [0,1,...,N-1] in natural order ------+
|   Per-chunk QSVTLCUCore_fwd -> (B, 3*n_qubits)                 |
+-----------------------------------------------------------------+
  |
  v
+------ Backward Pass: chunks [N-1,...,1,0] reversed, timesteps flipped ------+
|   Per-chunk QSVTLCUCore_bwd -> (B, 3*n_qubits)                              |
|   (reversed back to align with forward order)                                |
+------------------------------------------------------------------------------+
  |
  v
Concatenate [fwd, bwd] per chunk -> (B, n_chunks, 2 * 3*n_qubits)
  |
  v
output_proj: Linear(2 * 3*n_qubits -> d_model) + LayerNorm + SiLU + Dropout
  |
  v
seq_attention: attention-weighted aggregation -> (B, d_model)
  |
  v
classifier: Linear(d_model -> d_model//2) + SiLU + Dropout + Linear(-> output_dim)
```

### Bidirectional Processing Details

The forward and backward passes see the data differently:

| Pass | Chunk Order | Timestep Order Within Chunk |
|------|-------------|----------------------------|
| Forward | `[chunk_0, chunk_1, ..., chunk_{N-1}]` | Natural: `[t_0, t_1, ..., t_{C-1}]` |
| Backward | `[chunk_{N-1}, ..., chunk_1, chunk_0]` | Flipped: `[t_{C-1}, ..., t_1, t_0]` |

Each direction has its **own** `QSVTLCUCore` instance with independent trainable parameters (`prepare_params`, `signal_angles`, `qff_params`). The backward results are re-ordered to align with the forward chunk indices before concatenation.

### Constructor Signature

```python
QuantumQSVTHydraSSM(
    n_qubits=6, qlcu_layers=2, d_model=128, d_state=16,
    feature_dim=64, n_timesteps=125, output_dim=2, dropout=0.1,
    chunk_size=32, device="cpu", n_layers=None, n_channels=None,
    degree=2
)
```

---

## 6. Comparison

### 6.1. Model 2g vs 2h vs 2i

| Property | 2g (Transformer) | 2h (Mamba SSM) | 2i (Hydra SSM) |
|----------|-------------------|----------------|-----------------|
| **Class** | `QuantumTSTransformer` | `QuantumQSVTMambaSSM` | `QuantumQSVTHydraSSM` |
| **File** | `QTSTransformer_v2.py` | `QuantumQSVTSSM.py` | `QuantumQSVTSSM.py` |
| **Quantum core** | Inline QSVT+LCU | `QSVTLCUCore` (shared) | 2x `QSVTLCUCore` |
| **Direction** | Unidirectional | Unidirectional | Bidirectional |
| **Input projection** | `Linear(feat -> n_rots)` | `Linear(feat -> d_model) -> Linear(d_model -> n_rots)` | Same as 2h |
| **PE dimension** | `feature_dim` | `d_model` | `d_model` |
| **Chunk aggregation** | Mean pooling | Learned attention | Learned attention |
| **Output head** | Single linear layer | 2-layer MLP | 2-layer MLP |
| **Output proj** | None (direct to classifier) | `Linear(3q -> d_model) + LN + SiLU` | `Linear(6q -> d_model) + LN + SiLU` |
| **Quantum output** | `3 * n_qubits` per chunk | `3 * n_qubits` per chunk | `2 * 3 * n_qubits` per chunk (fwd + bwd) |
| **QSVTLCUCore instances** | 1 (inline) | 1 | 2 (fwd + bwd) |

### 6.2. Comparison with 2d/2e (StatePrep-based)

| Property | 2d/2e (StatePrep) | 2h/2i (QSVT+LCU) |
|----------|-------------------|-------------------|
| **Quantum mixing** | Fake -- classical attention aggregates before quantum | True -- per-timestep unitaries mixed via `qml.Select` |
| **Circuit type** | StatePrep + RX/RY/RZ + CRX | sim14 + QSVT + LCU + QFF |
| **Cross-timestep mixing** | Classical `chunk_attention` MLP | Quantum `qml.Select` (coherent superposition) |
| **Ancilla qubits** | None | `ceil(log2(chunk_size))` |
| **Per-timestep encoding** | No -- single summary vector per chunk | Yes -- each timestep gets its own sim14 unitary |
| **Trainable quantum params** | Superposition coefficients (alpha, beta, gamma) | PREPARE angles, signal angles, QFF params |
| **QSVT degree** | N/A | Configurable (default 2) |

---

## 7. Hyperparameters

### Quantum-Specific

| Parameter | Default | Description |
|-----------|---------|-------------|
| `n_qubits` | 6 | Number of main-register qubits |
| `degree` | 2 | QSVT polynomial degree (higher = more expressive but deeper circuit) |
| `n_ansatz_layers` / `qlcu_layers` | 2 | sim14 layers per timestep unitary |
| `chunk_size` | 32 | Timesteps per chunk (determines `n_ancilla = ceil(log2(chunk_size))`) |

### Classical Wrapper (2h/2i only)

| Parameter | Default | Description |
|-----------|---------|-------------|
| `d_model` | 128 | Hidden dimension for input/output projections |
| `d_state` | 16 | (Accepted for API compatibility; not used internally) |
| `dropout` | 0.1 | Dropout rate throughout |

### Derived Quantities

| Quantity | Formula | Example (n_qubits=6, chunk_size=32, degree=2, layers=2) |
|----------|---------|----------------------------------------------------------|
| `n_rots` | `4 * n_qubits * n_ansatz_layers` | `4 * 6 * 2 = 48` |
| `n_ancilla` | `ceil(log2(chunk_size))` | `ceil(log2(32)) = 5` |
| `total_wires` | `n_qubits + n_ancilla` | `6 + 5 = 11` |
| `n_select_ops` | `2^n_ancilla` | `2^5 = 32` |
| `q_dim` (output) | `3 * n_qubits` | `3 * 6 = 18` |
| `prepare_params` | `n_ancilla * n_ancilla * 2` | `5 * 5 * 2 = 50` |
| `signal_angles` | `degree + 1` | `3` |
| `qff_params` | `4 * n_qubits` | `24` |

---

## 8. Usage

### Running Ablation Experiments

```bash
# Model 2h (unidirectional QSVT+LCU SSM) on EEG
python scripts/run_ablation_eeg.py \
    --model-id 2h \
    --sampling-freq 80 \
    --seed 2024 \
    --n-epochs 50

# Model 2i (bidirectional QSVT+LCU SSM) on EEG
python scripts/run_ablation_eeg.py \
    --model-id 2i \
    --sampling-freq 80 \
    --seed 2024 \
    --n-epochs 50

# Synthetic benchmark
python scripts/run_synthetic_benchmark.py \
    --model-id 2h \
    --task forrelation \
    --seq-len 200 \
    --seed 2024
```

### Direct Instantiation

```python
from models.QuantumQSVTSSM import QSVTLCUCore, QuantumQSVTMambaSSM, QuantumQSVTHydraSSM

# Shared quantum core alone
core = QSVTLCUCore(n_qubits=6, chunk_size=32, degree=2, n_ansatz_layers=2)

# Model 2h
model_2h = QuantumQSVTMambaSSM(
    n_qubits=6, qlcu_layers=2, d_model=128, feature_dim=64,
    n_timesteps=125, output_dim=2, chunk_size=32, degree=2
)

# Model 2i
model_2i = QuantumQSVTHydraSSM(
    n_qubits=6, qlcu_layers=2, d_model=128, feature_dim=64,
    n_timesteps=125, output_dim=2, chunk_size=32, degree=2
)
```

### Running Tests

```bash
cd /pscratch/sd/j/junghoon/quantum_hydra_mamba
python -c "from models.QuantumQSVTSSM import test_quantum_qsvt_ssm; test_quantum_qsvt_ssm()"
```

---

## 9. File Map

| File | Contents |
|------|----------|
| `models/QTSTransformer_v2.py` | `sim14_circuit()`, `QuantumTSTransformer` (model 2g) |
| `models/QuantumQSVTSSM.py` | `QSVTLCUCore`, `QuantumQSVTMambaSSM` (2h), `QuantumQSVTHydraSSM` (2i) |
| `models/QuantumHydraSSM.py` | `QuantumMambaHydraSSM` (2d), `QuantumHydraHydraSSM` (2e) -- predecessors |
| `scripts/run_ablation_eeg.py` | EEG ablation runner (supports `--model-id 2h` and `2i`) |
| `scripts/run_synthetic_benchmark.py` | Synthetic benchmark runner (supports `--model-id 2h` and `2i`) |
