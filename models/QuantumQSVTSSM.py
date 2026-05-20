"""
Quantum QSVT+LCU SSM Models: True Quantum Mixing via QSVT and LCU

These models replace 2d/2e (QuantumHydraSSM) with the QSVT+LCU quantum core
from QTSTransformer_v2.py. Unlike 2d/2e which aggregate timesteps via classical
attention BEFORE the quantum circuit, these models send raw per-timestep rotation
angles directly into the QSVT+LCU circuit, achieving TRUE quantum cross-timestep
mixing.

Model IDs for ablation study:
- 2h: QuantumQSVTMambaSSM (unidirectional QSVT+LCU SSM)
- 2i: QuantumQSVTHydraSSM (bidirectional QSVT+LCU SSM)

Architecture flow (2h):
    Input (B, feat, T) -> permute -> (B, T, feat)
      -> input_proj: Linear(feat->d_model) + LN + SiLU + Dropout
      -> + sinusoidal PE (d_model dim)
      -> feature_projection: Linear(d_model->n_rots) -> Sigmoid * 2pi
      -> Chunk into windows of chunk_size
      -> Per-chunk QSVTLCUCore -> (B, 3*n_qubits)
      -> output_proj: Linear(3*n_qubits->d_model) + LN + SiLU + Dropout
      -> seq_attention aggregation across chunks -> (B, d_model)
      -> classifier: Linear(d_model->d_model//2) + SiLU + Drop + Linear(->output_dim)

Architecture flow (2i):
    Same as 2h but with two QSVTLCUCore instances (forward + backward):
    - Forward pass: chunks in natural order
    - Backward pass: chunks reversed, timesteps flipped within chunk
    - Concatenate -> Linear(2*3*n_qubits -> d_model)

Author: Junghoon Park
Date: February 2026
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F
import pennylane as qml
from math import ceil, log2
from typing import Optional

from models.QTSTransformer_v2 import sim14_circuit


# ================================================================================
# QSVTLCUCore: Shared Quantum Core (extracted from QTSTransformer_v2)
# ================================================================================

class QSVTLCUCore(nn.Module):
    """
    QSVT+LCU quantum core for processing a chunk of timesteps.

    Takes (B, chunk_size, n_rots) rotation parameters and returns
    (B, 3*n_qubits) expectation values via the QSVT+LCU circuit.

    This is the same circuit as QTSTransformer_v2.QuantumTSTransformer
    but extracted as a reusable module without classical pre/post processing.
    """

    def __init__(
        self,
        n_qubits: int = 6,
        chunk_size: int = 32,
        degree: int = 2,
        n_ansatz_layers: int = 2,
    ):
        super().__init__()

        self.n_qubits = n_qubits
        self.chunk_size = chunk_size
        self.degree = degree
        self.n_ansatz_layers = n_ansatz_layers

        # Qubit registers
        self.n_ancilla = ceil(log2(max(chunk_size, 2)))
        self.main_wires = list(range(n_qubits))
        self.anc_wires = list(range(n_qubits, n_qubits + self.n_ancilla))
        self.total_wires = n_qubits + self.n_ancilla
        self.n_select_ops = 2 ** self.n_ancilla

        # Parameter counts
        self.n_rots = 4 * n_qubits * n_ansatz_layers
        self.qff_n_rots = 4 * n_qubits * 1

        # Quantum output dimension
        self.q_dim = 3 * n_qubits

        # Trainable quantum parameters
        self.n_prep_layers = self.n_ancilla
        self.prepare_params = nn.Parameter(
            0.1 * torch.randn(self.n_prep_layers, self.n_ancilla, 2))

        self.signal_angles = nn.Parameter(
            0.1 * torch.randn(degree + 1))

        self.qff_params = nn.Parameter(torch.rand(self.qff_n_rots))

        # PennyLane device and QNode
        self.dev = qml.device("default.qubit", wires=self.total_wires)

        # Capture instance attributes as locals for the QNode closure
        _n_qubits = n_qubits
        _chunk_size = chunk_size
        _n_ancilla = self.n_ancilla
        _n_ansatz_layers = n_ansatz_layers
        _n_prep_layers = self.n_prep_layers
        _main_wires = self.main_wires
        _anc_wires = self.anc_wires
        _degree = degree
        _n_select_ops = self.n_select_ops
        _pcphase_wires = self.anc_wires + self.main_wires
        _pcphase_dim = 2 ** n_qubits

        @qml.qnode(self.dev, interface="torch", diff_method="backprop")
        def _circuit(ts_params, prep_p, sig_ang, qff_p):
            """
            QSVT+LCU circuit for one chunk.

            Args:
                ts_params: (B, chunk_size, n_rots) rotation params
                prep_p:    (n_prep_layers, n_ancilla, 2) PREPARE params
                sig_ang:   (degree+1,) signal processing angles
                qff_p:     (qff_n_rots,) QFF circuit params
            """

            def prepare():
                for ly in range(_n_prep_layers):
                    for qi, q in enumerate(_anc_wires):
                        qml.RY(prep_p[ly, qi, 0], wires=q)
                        qml.RZ(prep_p[ly, qi, 1], wires=q)
                    for i in range(_n_ancilla - 1):
                        qml.CNOT(wires=[_anc_wires[i], _anc_wires[i + 1]])

            def build_select_ops():
                select_ops = []
                for t in range(_chunk_size):
                    gates = []
                    param_idx = 0
                    for _ in range(_n_ansatz_layers):
                        for i in range(_n_qubits):
                            gates.append(qml.RY(
                                ts_params[..., t, param_idx],
                                wires=_main_wires[i]))
                            param_idx += 1
                        for i in range(_n_qubits - 1, -1, -1):
                            gates.append(qml.CRX(
                                ts_params[..., t, param_idx],
                                wires=[_main_wires[i],
                                       _main_wires[(i + 1) % _n_qubits]]))
                            param_idx += 1
                        for i in range(_n_qubits):
                            gates.append(qml.RY(
                                ts_params[..., t, param_idx],
                                wires=_main_wires[i]))
                            param_idx += 1
                        wire_order = [_n_qubits - 1] + list(range(_n_qubits - 1))
                        for i in wire_order:
                            gates.append(qml.CRX(
                                ts_params[..., t, param_idx],
                                wires=[_main_wires[i],
                                       _main_wires[(i - 1) % _n_qubits]]))
                            param_idx += 1
                    select_ops.append(qml.prod(*reversed(gates)))
                while len(select_ops) < _n_select_ops:
                    select_ops.append(qml.Identity(wires=_main_wires[0]))
                return select_ops

            select_ops = build_select_ops()

            qml.PCPhase(sig_ang[0], dim=_pcphase_dim, wires=_pcphase_wires)

            for k in range(_degree):
                prepare()
                if k % 2 == 0:
                    qml.Select(select_ops, control=_anc_wires)
                else:
                    qml.adjoint(qml.Select)(select_ops, control=_anc_wires)
                qml.adjoint(prepare)()
                qml.PCPhase(sig_ang[k + 1], dim=_pcphase_dim,
                            wires=_pcphase_wires)

            sim14_circuit(qff_p, wires=_n_qubits, layers=1)

            observables = (
                [qml.PauliX(i) for i in _main_wires] +
                [qml.PauliY(i) for i in _main_wires] +
                [qml.PauliZ(i) for i in _main_wires])
            return [qml.expval(op) for op in observables]

        self._circuit = _circuit

    def forward(self, ts_params: torch.Tensor) -> torch.Tensor:
        """
        Process one chunk through the QSVT+LCU circuit.

        Args:
            ts_params: (B, chunk_size, n_rots) rotation parameters in [0, 2pi]

        Returns:
            (B, 3*n_qubits) expectation values
        """
        exps = self._circuit(
            ts_params, self.prepare_params,
            self.signal_angles, self.qff_params)
        return torch.stack(exps, dim=1).float()


# ================================================================================
# Model 2h: Unidirectional QSVT+LCU SSM
# ================================================================================

class QuantumQSVTMambaSSM(nn.Module):
    """
    Unidirectional SSM with QSVT+LCU quantum core.

    Replaces 2d (QuantumMambaHydraSSM) with true quantum cross-timestep mixing.
    Instead of classical chunk attention before the quantum circuit, this model
    sends raw per-timestep rotation angles directly into the QSVT+LCU circuit.

    Model ID: 2h
    """

    def __init__(
        self,
        n_qubits: int = 6,
        qlcu_layers: int = 2,
        d_model: int = 128,
        d_state: int = 16,
        feature_dim: int = 64,
        n_timesteps: int = 125,
        output_dim: int = 2,
        dropout: float = 0.1,
        chunk_size: int = 32,
        device: str = "cpu",
        n_layers: Optional[int] = None,
        n_channels: Optional[int] = None,
        degree: int = 2,
    ):
        super().__init__()

        actual_layers = n_layers if n_layers is not None else qlcu_layers
        actual_channels = n_channels if n_channels is not None else feature_dim

        self.n_qubits = n_qubits
        self.n_layers = actual_layers
        self.d_model = d_model
        self.n_timesteps = n_timesteps
        self.chunk_size = chunk_size
        self.torch_device = torch.device(device) if isinstance(device, str) else device

        # Quantum core
        self.quantum_core = QSVTLCUCore(
            n_qubits=n_qubits,
            chunk_size=chunk_size,
            degree=degree,
            n_ansatz_layers=actual_layers,
        )

        self.q_dim = self.quantum_core.q_dim
        n_rots = self.quantum_core.n_rots

        # Input projection: (B, T, n_channels) -> (B, T, d_model)
        self.input_proj = nn.Sequential(
            nn.Linear(actual_channels, d_model),
            nn.LayerNorm(d_model),
            nn.SiLU(),
            nn.Dropout(dropout)
        )

        # Sinusoidal positional encoding (on d_model dimension)
        pe = torch.zeros(n_timesteps, d_model)
        pos = torch.arange(n_timesteps).unsqueeze(1).float()
        div = torch.exp(
            torch.arange(0, d_model, 2).float()
            * -(math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div[:d_model // 2])
        self.register_buffer('pe', pe.unsqueeze(0))  # (1, T, d_model)

        # Feature projection: d_model -> n_rots rotation angles
        self.feature_projection = nn.Linear(d_model, n_rots)
        self.dropout_layer = nn.Dropout(dropout)
        self.rot_sigm = nn.Sigmoid()

        # Output projection: 3*n_qubits -> d_model
        self.output_proj = nn.Sequential(
            nn.Linear(self.q_dim, d_model),
            nn.LayerNorm(d_model),
            nn.SiLU(),
            nn.Dropout(dropout)
        )

        # Sequence aggregation across chunks
        self.seq_attention = nn.Sequential(
            nn.Linear(d_model, d_model // 4),
            nn.Tanh(),
            nn.Linear(d_model // 4, 1)
        )

        # Classification head
        self.classifier = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, output_dim)
        )

        self.to(self.torch_device)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Args:
            x: (batch, n_channels, n_timesteps) or (batch, n_timesteps, n_channels)

        Returns:
            logits: (batch, output_dim)
        """
        if x.dim() == 2:
            x = x.unsqueeze(1)

        if x.shape[1] != self.n_timesteps and x.shape[2] == self.n_timesteps:
            x = x.permute(0, 2, 1)  # -> (B, T, n_channels)

        B = x.size(0)

        # Input projection + positional encoding
        x = self.input_proj(x)                            # (B, T, d_model)
        x = x + self.pe[:, :x.size(1)]                    # + sinusoidal PE

        # Feature projection -> rotation angles
        x = self.feature_projection(self.dropout_layer(x))
        ts_params = self.rot_sigm(x) * (2 * math.pi)     # (B, T, n_rots)

        # Process chunks
        chunk_results = []
        for start in range(0, self.n_timesteps, self.chunk_size):
            chunk = ts_params[:, start:start + self.chunk_size]
            pad_len = self.chunk_size - chunk.size(1)
            if pad_len > 0:
                chunk = F.pad(chunk, (0, 0, 0, pad_len))
            chunk_out = self.quantum_core(chunk)           # (B, 3*n_qubits)
            chunk_results.append(chunk_out)

        # Output projection per chunk
        chunk_features = torch.stack(chunk_results, dim=1)  # (B, n_chunks, 3*n_qubits)
        chunk_features = self.output_proj(chunk_features)   # (B, n_chunks, d_model)

        # Sequence aggregation via attention
        seq_attn = self.seq_attention(chunk_features)       # (B, n_chunks, 1)
        seq_weights = F.softmax(seq_attn, dim=1)
        sequence_repr = (seq_weights * chunk_features).sum(dim=1)  # (B, d_model)

        # Classification
        logits = self.classifier(sequence_repr)
        return logits


# ================================================================================
# Model 2i: Bidirectional QSVT+LCU SSM
# ================================================================================

class QuantumQSVTHydraSSM(nn.Module):
    """
    Bidirectional SSM with QSVT+LCU quantum core.

    Replaces 2e (QuantumHydraHydraSSM) with true quantum cross-timestep mixing.
    Uses two QSVTLCUCore instances:
    - Forward: chunks in natural order
    - Backward: chunks reversed, timesteps flipped within each chunk

    Model ID: 2i
    """

    def __init__(
        self,
        n_qubits: int = 6,
        qlcu_layers: int = 2,
        d_model: int = 128,
        d_state: int = 16,
        feature_dim: int = 64,
        n_timesteps: int = 125,
        output_dim: int = 2,
        dropout: float = 0.1,
        chunk_size: int = 32,
        device: str = "cpu",
        n_layers: Optional[int] = None,
        n_channels: Optional[int] = None,
        degree: int = 2,
    ):
        super().__init__()

        actual_layers = n_layers if n_layers is not None else qlcu_layers
        actual_channels = n_channels if n_channels is not None else feature_dim

        self.n_qubits = n_qubits
        self.n_layers = actual_layers
        self.d_model = d_model
        self.n_timesteps = n_timesteps
        self.chunk_size = chunk_size
        self.torch_device = torch.device(device) if isinstance(device, str) else device

        # Two quantum cores (forward + backward)
        self.quantum_core_fwd = QSVTLCUCore(
            n_qubits=n_qubits,
            chunk_size=chunk_size,
            degree=degree,
            n_ansatz_layers=actual_layers,
        )
        self.quantum_core_bwd = QSVTLCUCore(
            n_qubits=n_qubits,
            chunk_size=chunk_size,
            degree=degree,
            n_ansatz_layers=actual_layers,
        )

        self.q_dim = self.quantum_core_fwd.q_dim
        n_rots = self.quantum_core_fwd.n_rots

        # Input projection
        self.input_proj = nn.Sequential(
            nn.Linear(actual_channels, d_model),
            nn.LayerNorm(d_model),
            nn.SiLU(),
            nn.Dropout(dropout)
        )

        # Sinusoidal positional encoding
        pe = torch.zeros(n_timesteps, d_model)
        pos = torch.arange(n_timesteps).unsqueeze(1).float()
        div = torch.exp(
            torch.arange(0, d_model, 2).float()
            * -(math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div[:d_model // 2])
        self.register_buffer('pe', pe.unsqueeze(0))

        # Feature projection
        self.feature_projection = nn.Linear(d_model, n_rots)
        self.dropout_layer = nn.Dropout(dropout)
        self.rot_sigm = nn.Sigmoid()

        # Output projection (combines both directions: 2 * 3*n_qubits -> d_model)
        self.output_proj = nn.Sequential(
            nn.Linear(2 * self.q_dim, d_model),
            nn.LayerNorm(d_model),
            nn.SiLU(),
            nn.Dropout(dropout)
        )

        # Sequence aggregation
        self.seq_attention = nn.Sequential(
            nn.Linear(d_model, d_model // 4),
            nn.Tanh(),
            nn.Linear(d_model // 4, 1)
        )

        # Classification head
        self.classifier = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, output_dim)
        )

        self.to(self.torch_device)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass with bidirectional QSVT+LCU processing.

        Args:
            x: (batch, n_channels, n_timesteps) or (batch, n_timesteps, n_channels)

        Returns:
            logits: (batch, output_dim)
        """
        if x.dim() == 2:
            x = x.unsqueeze(1)

        if x.shape[1] != self.n_timesteps and x.shape[2] == self.n_timesteps:
            x = x.permute(0, 2, 1)

        B = x.size(0)

        # Input projection + PE
        x = self.input_proj(x)
        x = x + self.pe[:, :x.size(1)]

        # Feature projection -> rotation angles
        x = self.feature_projection(self.dropout_layer(x))
        ts_params = self.rot_sigm(x) * (2 * math.pi)     # (B, T, n_rots)

        # Build chunk start indices
        chunk_starts = list(range(0, self.n_timesteps, self.chunk_size))
        n_chunks = len(chunk_starts)

        # Forward pass: chunks in natural order
        fwd_results = []
        for start in chunk_starts:
            chunk = ts_params[:, start:start + self.chunk_size]
            pad_len = self.chunk_size - chunk.size(1)
            if pad_len > 0:
                chunk = F.pad(chunk, (0, 0, 0, pad_len))
            fwd_results.append(self.quantum_core_fwd(chunk))

        # Backward pass: chunks in reversed order, timesteps flipped within chunk
        bwd_results = []
        for start in reversed(chunk_starts):
            chunk = ts_params[:, start:start + self.chunk_size]
            pad_len = self.chunk_size - chunk.size(1)
            if pad_len > 0:
                chunk = F.pad(chunk, (0, 0, 0, pad_len))
            # Flip timesteps within this chunk
            chunk = torch.flip(chunk, dims=[1])
            bwd_results.append(self.quantum_core_bwd(chunk))

        # Reverse backward results to align with forward chunk order
        bwd_results = list(reversed(bwd_results))

        # Concatenate forward + backward per chunk
        chunk_features = []
        for fwd, bwd in zip(fwd_results, bwd_results):
            combined = torch.cat([fwd, bwd], dim=-1)  # (B, 2*3*n_qubits)
            chunk_features.append(combined)

        chunk_features = torch.stack(chunk_features, dim=1)  # (B, n_chunks, 2*q_dim)
        chunk_features = self.output_proj(chunk_features)     # (B, n_chunks, d_model)

        # Sequence aggregation
        seq_attn = self.seq_attention(chunk_features)
        seq_weights = F.softmax(seq_attn, dim=1)
        sequence_repr = (seq_weights * chunk_features).sum(dim=1)

        # Classification
        logits = self.classifier(sequence_repr)
        return logits


# ================================================================================
# Testing
# ================================================================================

def test_quantum_qsvt_ssm():
    """Test QSVTLCUCore, QuantumQSVTMambaSSM (2h), and QuantumQSVTHydraSSM (2i)."""
    print("=" * 80)
    print("Testing QSVT+LCU SSM Models (2h/2i)")
    print("True quantum cross-timestep mixing via QSVT and LCU")
    print("=" * 80)

    device = "cpu"
    batch_size = 2
    n_channels = 16
    n_timesteps = 32
    n_qubits = 4
    n_layers = 1
    chunk_size = 8
    degree = 2
    output_dim = 2

    # ── Test QSVTLCUCore ──
    print(f"\n[1] Testing QSVTLCUCore...")
    core = QSVTLCUCore(
        n_qubits=n_qubits,
        chunk_size=chunk_size,
        degree=degree,
        n_ansatz_layers=n_layers,
    )
    n_rots = core.n_rots
    ts_params = torch.rand(batch_size, chunk_size, n_rots) * 2 * math.pi
    out = core(ts_params)
    print(f"  Input:  ts_params {ts_params.shape}")
    print(f"  Output: {out.shape} (expected ({batch_size}, {3 * n_qubits}))")
    assert out.shape == (batch_size, 3 * n_qubits), f"Shape mismatch: {out.shape}"
    print(f"  n_ancilla={core.n_ancilla}, total_wires={core.total_wires}")
    print(f"  Trainable quantum params: prepare={core.prepare_params.numel()}, "
          f"signal={core.signal_angles.numel()}, qff={core.qff_params.numel()}")

    # ── Test Model 2h ──
    print(f"\n[2] Testing QuantumQSVTMambaSSM (Model 2h, unidirectional)...")
    model_2h = QuantumQSVTMambaSSM(
        n_qubits=n_qubits,
        qlcu_layers=n_layers,
        d_model=64,
        d_state=8,
        feature_dim=n_channels,
        n_timesteps=n_timesteps,
        output_dim=output_dim,
        dropout=0.1,
        chunk_size=chunk_size,
        device=device,
        degree=degree,
    )

    x = torch.randn(batch_size, n_channels, n_timesteps)
    out_2h = model_2h(x)
    print(f"  Input:  {x.shape}")
    print(f"  Output: {out_2h.shape} (expected ({batch_size}, {output_dim}))")
    assert out_2h.shape == (batch_size, output_dim), f"Shape mismatch: {out_2h.shape}"

    total_params_2h = sum(p.numel() for p in model_2h.parameters() if p.requires_grad)
    print(f"  Trainable parameters: {total_params_2h:,}")

    # ── Test Model 2i ──
    print(f"\n[3] Testing QuantumQSVTHydraSSM (Model 2i, bidirectional)...")
    model_2i = QuantumQSVTHydraSSM(
        n_qubits=n_qubits,
        qlcu_layers=n_layers,
        d_model=64,
        d_state=8,
        feature_dim=n_channels,
        n_timesteps=n_timesteps,
        output_dim=output_dim,
        dropout=0.1,
        chunk_size=chunk_size,
        device=device,
        degree=degree,
    )

    out_2i = model_2i(x)
    print(f"  Input:  {x.shape}")
    print(f"  Output: {out_2i.shape} (expected ({batch_size}, {output_dim}))")
    assert out_2i.shape == (batch_size, output_dim), f"Shape mismatch: {out_2i.shape}"

    total_params_2i = sum(p.numel() for p in model_2i.parameters() if p.requires_grad)
    print(f"  Trainable parameters: {total_params_2i:,}")

    # ── Test gradient flow ──
    print(f"\n[4] Testing gradient flow (2h)...")
    model_2h.train()
    optimizer = torch.optim.Adam(model_2h.parameters(), lr=1e-3)
    criterion = nn.CrossEntropyLoss()

    out = model_2h(x)
    labels = torch.randint(0, output_dim, (batch_size,))
    loss = criterion(out, labels)
    loss.backward()
    optimizer.step()
    print(f"  Loss: {loss.item():.4f}")
    print(f"  Gradient flow: OK")

    print(f"\n[5] Testing gradient flow (2i)...")
    model_2i.train()
    optimizer_2i = torch.optim.Adam(model_2i.parameters(), lr=1e-3)

    out = model_2i(x)
    loss = criterion(out, labels)
    loss.backward()
    optimizer_2i.step()
    print(f"  Loss: {loss.item():.4f}")
    print(f"  Gradient flow: OK")

    print("\n" + "=" * 80)
    print("All tests passed!")
    print("=" * 80)
    print("\nKey differences from 2d/2e:")
    print("  - NO classical chunk attention before quantum circuit")
    print("  - Raw per-timestep rotation angles go directly to QSVT+LCU")
    print("  - TRUE quantum cross-timestep mixing via qml.Select")
    print("  - sim14 ansatz (from QTSTransformer_v2) instead of StatePrep")
    print("=" * 80)

    return model_2h, model_2i


if __name__ == "__main__":
    test_quantum_qsvt_ssm()
