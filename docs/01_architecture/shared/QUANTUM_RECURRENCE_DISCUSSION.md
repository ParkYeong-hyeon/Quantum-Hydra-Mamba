# Quantum Recurrence and Parallel Scan Discussion

**Date**: November 21, 2025
**Context**: Follow-up discussion on DNA classification experiments

---

## Key Question: Do Quantum Models Have Selective State Space?

### Answer: Yes, but with Important Caveats

Quantum Mamba/Hydra models **DO** have input-dependent selective mechanisms:

1. **Input-Dependent Parameters**:
   - QLCU parameters are derived from input: `qlcu_params = self.param_proj(x_projected)`
   - This is analogous to classical Mamba's selective mechanism (B, C from input)

2. **Current Issue**: TS versions lack recurrent state propagation
   - Original loop (`for t in range(seq_len):`) was removed due to OOM/timeout
   - This loop was NOT recurrent - it processed timesteps independently
   - True recurrence would propagate hidden state: `h_t = f(h_{t-1}, x_t)`

---

## Why Quantum Recurrence is Challenging

### Memory and Speed Concerns

1. **Sequential Bottleneck**: O(T) forward passes through quantum circuit
2. **Memory**: Each quantum operation creates computation graph nodes
3. **No Parallelization**: Can't batch across time dimension

### Classical Mamba's Solution: Parallel Scan

Classical Mamba uses parallel scan algorithm:
- Reduces O(T) sequential steps to O(log T)
- Works because linear recurrence is associative: `h_t = A_t h_{t-1} + B_t x_t`

---

## Implementing Parallel Scan for Quantum Circuits

### Challenge

Quantum operations are **unitary**, not linear recurrences. Standard parallel scan doesn't directly apply.

### Solution: Parallel Unitary Composition

```python
class QuantumParallelScan(nn.Module):
    """
    Parallel scan for quantum circuits using tree reduction.
    Instead of h_t = A*h_{t-1} + B*x_t (classical)
    We compute: U_cumulative[t] = U_t @ U_{t-1} @ ... @ U_1
    """

    def forward(self, x):
        batch_size, seq_len, d_model = x.shape
        device = x.device

        # Generate per-timestep unitary parameters
        params = self.param_generator(x)  # [B, T, n_params]

        # Compute individual unitary matrices
        unitaries = []
        for t in range(seq_len):
            U_t = self.compute_unitary_matrix(params[:, t, :])  # [B, 2^n, 2^n]
            unitaries.append(U_t)

        unitaries = torch.stack(unitaries, dim=1)  # [B, T, 2^n, 2^n]

        # Parallel scan: tree reduction for cumulative products
        cumulative = self._parallel_scan_unitaries(unitaries)

        # Apply cumulative unitaries to initial state
        psi_0 = self.get_initial_state(batch_size, device)  # [B, 2^n]

        outputs = []
        for t in range(seq_len):
            psi_t = torch.bmm(cumulative[:, t], psi_0.unsqueeze(-1)).squeeze(-1)
            measurement = self.measure(psi_t)
            outputs.append(measurement)

        return torch.stack(outputs, dim=1)

    def _parallel_scan_unitaries(self, unitaries):
        """
        Compute cumulative unitary products using parallel scan.

        For sequence [U_1, U_2, U_3, U_4]:
        Step 1: [U_1, U_2@U_1, U_3, U_4@U_3]
        Step 2: [U_1, U_2@U_1, U_3@U_2@U_1, U_4@U_3@U_2@U_1]

        Complexity: O(log T) sequential steps
        """
        B, T, dim, _ = unitaries.shape
        result = unitaries.clone()

        stride = 1
        while stride < T:
            # Parallel step: multiply pairs
            new_result = result.clone()
            for i in range(stride, T):
                # result[i] = result[i] @ result[i - stride]
                new_result[:, i] = torch.bmm(
                    result[:, i],
                    result[:, i - stride]
                )
            result = new_result
            stride *= 2

        return result
```

---

## Practical Alternative: Chunked Processing

For immediate implementation, chunked processing offers a simpler approach:

```python
class QuantumChunkedRecurrence(nn.Module):
    """
    Process sequence in chunks with recurrence between chunks.

    Trade-off:
    - Chunk size C: parallel within chunk, sequential across chunks
    - T/C sequential steps instead of T
    - Each chunk processes C timesteps in parallel
    """

    def __init__(self, n_qubits, d_model, chunk_size=8):
        super().__init__()
        self.chunk_size = chunk_size
        self.n_qubits = n_qubits

        # Quantum circuit for processing chunk
        self.quantum_chunk = QuantumChunkProcessor(n_qubits, d_model)

        # State transition between chunks
        self.state_transition = nn.Linear(d_model, d_model)

    def forward(self, x):
        batch_size, seq_len, d_model = x.shape
        device = x.device

        # Initialize hidden state
        h = torch.zeros(batch_size, d_model, device=device)

        outputs = []

        # Process in chunks
        for chunk_start in range(0, seq_len, self.chunk_size):
            chunk_end = min(chunk_start + self.chunk_size, seq_len)
            x_chunk = x[:, chunk_start:chunk_end, :]  # [B, C, D]

            # Combine chunk with previous hidden state
            x_chunk_with_state = x_chunk + h.unsqueeze(1)

            # Process chunk through quantum circuit (parallel over C)
            chunk_output = self.quantum_chunk(x_chunk_with_state)
            outputs.append(chunk_output)

            # Update hidden state for next chunk
            h = self.state_transition(chunk_output[:, -1, :])

        return torch.cat(outputs, dim=1)
```

---

## Recommendations

### For Immediate Use
1. **Chunked Processing** (chunk_size=8-16)
   - Easier to implement
   - Reduces sequential steps by factor of chunk_size
   - Good balance of speed and recurrence

### For Future Research
2. **Full Parallel Scan**
   - Requires explicit unitary matrix computation
   - O(log T) complexity
   - More complex implementation

3. **Quantum Reservoir Computing**
   - Use quantum circuit as fixed reservoir
   - Train only classical readout layer
   - Avoids gradient issues entirely

---

## Implications for Current DNA Results

The current DNA experiments (57 timesteps) work well because:
1. Short sequences don't require long-range recurrence
2. Independent timestep processing sufficient for local patterns
3. Quantum circuits capture feature interactions at each position

For longer sequences (1000+ bp genomic data), implementing chunked recurrence would likely improve performance by enabling:
- Information flow across distant positions
- Progressive feature refinement
- Context-aware processing

---

## Files Reference

- Model implementations: `/pscratch/sd/j/junghoon/QuantumMamba.py`, `/pscratch/sd/j/junghoon/QuantumHydra.py`
- DNA results: `/pscratch/sd/j/junghoon/results/dna_results/`
- Main results document: `/pscratch/sd/j/junghoon/results/dna_results/DNA_CLASSIFICATION_RESULTS.md`
