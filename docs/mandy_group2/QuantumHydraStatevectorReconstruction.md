## QuantumHydra Statevector Reconstruction

This document explains the `QuantumHydraStatevectorReconstruction` model in
`models/mandy_group2/QuantumHydraStatevectorReconstruction.py`.

### High-Level Idea

The model reproduces the QuantumHydra superposition pipeline while avoiding a
second quantum circuit for measurement. It does so by:

1. Running three quantum branches (forward, backward, diagonal) to obtain
   statevectors.
2. Combining those statevectors with complex coefficients (alpha, beta, gamma).
3. Computing Pauli X/Y/Z expectation values directly from the combined
   statevector using PennyLane’s matrix definitions.

This yields the same measurements as the original superposition-then-measure
design, but the final measurement is computed classically from the statevector.

### Branches and Superposition

The core (`ClassicalReconstructionHydraSSMCore`) builds three QNodes:

- Forward branch: delta-modulated ansatz layers (selective forgetting).
- Backward branch: same layers, but applied in reversed qubit order.
- Diagonal branch: unified ansatz without delta scaling.

Each branch returns `qml.state()`. The combined state is:

```
|psi> = alpha * |psi1> + beta * |psi2> + gamma * |psi3>
```

The complex coefficients are trainable and normalized so that
`|alpha|^2 + |beta|^2 + |gamma|^2 = 1`.

### Measurement Without a Circuit

Instead of preparing the combined state into another QNode, the model computes
expectation values directly:

- It precomputes Pauli X/Y/Z matrices with `qml.matrix`.
- For each operator `O`, it evaluates `<psi|O|psi>` on the combined state.

This uses the same operator definitions and wire order as PennyLane, but avoids
the extra StatePrep and measurement circuit.

### End-to-End Flow

The full model wrappers (`ClassicalReconstructionHydraSSM` and the bidirectional
variant) follow the same high-level flow as other ablation models:

1. Project inputs to `d_model`.
2. Chunk the sequence and summarize each chunk with attention.
3. Produce branch parameters, input angles, and delta scales per chunk.
4. Run the reconstruction core to obtain quantum features.
5. Project features back to `d_model`, aggregate over time, and classify.

### When to Use

Use this model to:

- Validate that statevector reconstruction matches quantum-superposition
  measurements.
- Separate the effect of superposition from the effect of a final measurement
  circuit.
- Speed up experimentation by removing one QNode from the pipeline.

### How to Instantiate

You can import the classes directly from `models.mandy_group2`:

```python
from models.mandy_group2 import (
    ClassicalReconstructionHydraSSM,
    ClassicalReconstructionHydraSSMBidirectional,
)

# Unidirectional
model = ClassicalReconstructionHydraSSM(
    n_qubits=6,
    qlcu_layers=2,
    d_model=128,
    d_state=16,
    feature_dim=64,
    n_timesteps=125,
    output_dim=2,
    chunk_size=32,
    device="cpu",
)

# Bidirectional
model_bidir = ClassicalReconstructionHydraSSMBidirectional(
    n_qubits=6,
    qlcu_layers=2,
    d_model=128,
    d_state=16,
    feature_dim=64,
    n_timesteps=125,
    output_dim=2,
    chunk_size=32,
    device="cpu",
)
```

### Related Files

- `models/mandy_group2/QuantumHydraStatevectorReconstruction.py`
