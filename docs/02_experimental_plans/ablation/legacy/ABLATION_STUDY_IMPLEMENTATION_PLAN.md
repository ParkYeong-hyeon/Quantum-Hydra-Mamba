# Ablation Study Implementation Plan (Extended)

## Classical Features + Quantum SSM: A Rigorous Evaluation

This document provides a detailed implementation plan for conducting ablation studies to validate the "Classical Features + Quantum SSM" architecture, including both **Hydra-style (bidirectional)** and **Mamba-style (selective)** variants.

---

## Table of Contents

1. [Overview](#1-overview)
2. [Model Specifications](#2-model-specifications)
3. [Shared Components](#3-shared-components)
4. [Ablation 1: Where to Apply Quantum](#4-ablation-1-where-to-apply-quantum)
5. [Ablation 2: Which Quantum Sequence Mixing](#5-ablation-2-which-quantum-sequence-mixing)
6. [Ablation 3: SSM vs Attention Comparison](#6-ablation-3-ssm-vs-attention-comparison)
7. [Quantum SSM Variants: Hydra vs Mamba](#7-quantum-ssm-variants-hydra-vs-mamba)
8. [Experimental Protocol](#8-experimental-protocol)
9. [Implementation Checklist](#9-implementation-checklist)

---

## 1. Overview

### Research Questions

| Question | Ablation | Key Comparison |
|----------|----------|----------------|
| Where should quantum be applied? | Ablation 1 | Quantum features vs Quantum dynamics |
| Which quantum mixing is best? | Ablation 2 | Quantum SSM vs Attention vs Gated |
| Bidirectional vs Selective? | Ablation 2 | Hydra-style vs Mamba-style |
| How do complete architectures compare? | Ablation 3 | New architecture vs QTSTransformer |

### Complete Model List

| ID | Model Name | Feature Extraction | Sequence Mixing | Style | Status |
|----|------------|-------------------|-----------------|-------|--------|
| A1 | Classical Hydra | Classical | Classical SSM | Bidir | EXISTS |
| A2a | Current QuantumHydraSSM | Quantum VQC | Classical Hydra | Bidir | EXISTS |
| A2b | Current QuantumMambaSSM | Quantum VQC | Classical Mamba | Select | EXISTS |
| **A3a** | Classical + Quantum Hydra SSM | Classical | Quantum SSM (QSVT) | Bidir | **TO BUILD** |
| **A3b** | Classical + Quantum Mamba SSM | Classical | Quantum SSM (QSVT) | Select | **TO BUILD** |
| A4a | End-to-End Quantum Hydra | Quantum VQC | Quantum Hydra SSM | Bidir | **TO BUILD** |
| A4b | End-to-End Quantum Mamba | Quantum VQC | Quantum Mamba SSM | Select | **TO BUILD** |
| B1a | Classical + Quantum Hydra SSM | Shared | Quantum SSM (QSVT) | Bidir | = A3a |
| B1b | Classical + Quantum Mamba SSM | Shared | Quantum SSM (QSVT) | Select | = A3b |
| B2a | Classical + Quantum Hydra Attention | Shared | Quantum Attention | Bidir | **TO BUILD** |
| B2b | Classical + Quantum Mamba Attention | Shared | Quantum Attention | Select | **TO BUILD** |
| B3a | Classical + Quantum Hydra Gated | Shared | Quantum Gated | Bidir | **TO BUILD** |
| B3b | Classical + Quantum Mamba Gated | Shared | Quantum Gated | Select | **TO BUILD** |
| B4 | Classical + Classical SSM | Shared | Classical SSM | Baseline | **TO BUILD** |
| C1a | Classical + Quantum Hydra SSM | Classical | Quantum Hydra SSM | Bidir | = A3a |
| C1b | Classical + Quantum Mamba SSM | Classical | Quantum Mamba SSM | Select | = A3b |
| C2 | QTSTransformer | Built-in | Quantum Attention | N/A | EXISTS |
| C3a | Classical Hydra | Classical | Classical Hydra | Bidir | = A1 |
| C3b | Classical Mamba | Classical | Classical Mamba | Select | **TO BUILD** |
| C4 | Classical Transformer | Classical | Classical Attention | N/A | **TO BUILD** |

---

## 2. Model Specifications

### 2.1 Standardized Hyperparameters

All models must use these parameters for fair comparison:

```python
# Quantum parameters
N_QUBITS = 4                    # Number of qubits
N_QUANTUM_LAYERS = 2            # Variational circuit depth
QSVT_DEGREE = 3                 # Polynomial degree for QSVT

# Classical parameters
D_MODEL = 64                    # Model dimension
D_STATE = 16                    # SSM state dimension
D_HIDDEN = 128                  # Classical encoder hidden dim
CHUNK_SIZE = 32                 # For chunked processing

# Training parameters
BATCH_SIZE = 32
LEARNING_RATE = 1e-3
N_EPOCHS = 30
SEEDS = [2024, 2025, 2026, 2027, 2028]
```

### 2.2 Hydra vs Mamba Style Comparison

| Aspect | Hydra (Bidirectional) | Mamba (Selective) |
|--------|----------------------|-------------------|
| **Direction** | Forward + Backward | Forward only |
| **Parameters** | Fixed A, B, C | Input-dependent Δ, B, C |
| **SSM Equation** | h_fwd + h_bwd | h = exp(ΔA)h + ΔBx |
| **QSVT Application** | exp(A) same for all t | exp(Δ[t]A) varies per timestep |
| **Best For** | Global context tasks | Selective memory tasks |
| **Complexity** | 2× quantum calls | More complex per call |

### 2.3 Parameter Count Targets

| Component | Target Parameters |
|-----------|------------------|
| Classical Encoder | ~16K-20K |
| Quantum Circuit | ~50-100 (VQC params) |
| Sequence Mixing | ~10K-15K |
| Output Head | ~2K-5K |
| **Total** | **~30K-40K** |

---

## 3. Shared Components

### 3.1 Shared Classical Encoder

**File**: `models/shared_components.py`

All models in Ablation 2 must use this IDENTICAL encoder:

```python
import torch
import torch.nn as nn

class SharedClassicalEncoder(nn.Module):
    """
    Shared classical feature encoder for controlled ablation studies.

    All models in Ablation 2 MUST use this exact encoder to ensure
    that performance differences come only from the mixing mechanism.

    Architecture:
        Input (seq_len, input_dim)
        → Linear (input_dim → d_hidden)
        → LayerNorm
        → GELU
        → Linear (d_hidden → d_model)
        → LayerNorm
        Output (seq_len, d_model)

    Parameters: ~16K for d_hidden=128, d_model=64
    """

    def __init__(
        self,
        input_dim: int,
        d_model: int = 64,
        d_hidden: int = 128,
        dropout: float = 0.1
    ):
        super().__init__()

        self.encoder = nn.Sequential(
            nn.Linear(input_dim, d_hidden),
            nn.LayerNorm(d_hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_hidden, d_model),
            nn.LayerNorm(d_model)
        )

        self.input_dim = input_dim
        self.d_model = d_model
        self.d_hidden = d_hidden

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch, seq_len, input_dim)
        Returns:
            (batch, seq_len, d_model)
        """
        return self.encoder(x)

    def get_param_count(self) -> int:
        return sum(p.numel() for p in self.parameters())
```

### 3.2 Shared Output Head

```python
class SharedOutputHead(nn.Module):
    """
    Shared classification head for all models.
    """

    def __init__(
        self,
        d_model: int = 64,
        num_classes: int = 2,
        dropout: float = 0.1
    ):
        super().__init__()

        self.head = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, num_classes)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(x)
```

### 3.3 Shared Pooling Strategy

```python
class SequencePooling(nn.Module):
    """
    Shared pooling strategy: attention-weighted mean pooling.
    """

    def __init__(self, d_model: int = 64):
        super().__init__()

        self.attention = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.Tanh(),
            nn.Linear(d_model // 2, 1)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        attn_scores = self.attention(x)
        attn_weights = torch.softmax(attn_scores, dim=1)
        pooled = (attn_weights * x).sum(dim=1)
        return pooled
```

---

## 4. Ablation 1: Where to Apply Quantum (Extended)

### Purpose
Determine whether quantum computation is more effective for feature extraction or sequence dynamics, comparing both Hydra and Mamba styles.

### 4.1 Model List

| Model | Features | Mixing | Style | Description |
|-------|----------|--------|-------|-------------|
| A1 | Classical | Classical Hydra | Bidir | Classical baseline (bidirectional) |
| A2a | Quantum VQC | Classical Hydra | Bidir | Current QuantumHydraSSM |
| A2b | Quantum VQC | Classical Mamba | Select | Current QuantumMambaSSM |
| **A3a** | Classical | **Quantum Hydra SSM** | Bidir | **PRIMARY MODEL (Hydra)** |
| **A3b** | Classical | **Quantum Mamba SSM** | Select | **PRIMARY MODEL (Mamba)** |
| A4a | Quantum VQC | Quantum Hydra SSM | Bidir | End-to-End Quantum (Hydra) |
| A4b | Quantum VQC | Quantum Mamba SSM | Select | End-to-End Quantum (Mamba) |

### 4.2 Model A3a: Classical + Quantum Hydra SSM (Bidirectional)

**File**: `models/ClassicalFeaturesQuantumHydraSSM.py`

**Architecture**:
```
Input → [Classical Encoder] → [Quantum Hydra SSM] → [Output Head]
                                 ├─ Forward QSVT
                                 ├─ Backward QSVT
                                 └─ Quantum Combine
```

**Implementation**:

```python
"""
Classical Features + Quantum Hydra SSM (Model A3a)

Bidirectional Quantum SSM using QSVT for both forward and backward passes.

Key Design:
1. Classical encoder extracts features
2. Forward quantum SSM processes left-to-right
3. Backward quantum SSM processes right-to-left
4. Quantum combination of both directions
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import pennylane as qml
import numpy as np
from typing import Tuple

from models.shared_components import (
    SharedClassicalEncoder,
    SharedOutputHead,
    SequencePooling
)


class QuantumHydraSSMCore(nn.Module):
    """
    Bidirectional Quantum SSM using QSVT.

    Implements:
        Forward:  |h_fwd[t]⟩ = QSVT_exp(A)|h_fwd[t-1]⟩ + B|x[t]⟩
        Backward: |h_bwd[t]⟩ = QSVT_exp(A)|h_bwd[t+1]⟩ + B|x[t]⟩
        Output:   Combine forward and backward via LCU
    """

    def __init__(
        self,
        n_qubits: int = 4,
        n_layers: int = 2,
        qsvt_degree: int = 3,
        d_model: int = 64,
        device: str = "cpu"
    ):
        super().__init__()

        self.n_qubits = n_qubits
        self.n_layers = n_layers
        self.qsvt_degree = qsvt_degree
        self.d_model = d_model

        # Dimension reduction for angle encoding
        self.input_proj = nn.Linear(d_model, n_qubits)

        # QSVT polynomial coefficients (shared for forward/backward)
        self.poly_coeffs = nn.Parameter(torch.randn(qsvt_degree + 1) * 0.1)

        # LCU coefficients for combining directions
        self.fwd_coeff = nn.Parameter(torch.tensor([0.5]))
        self.bwd_coeff = nn.Parameter(torch.tensor([0.5]))

        # Variational circuit parameters
        self.n_circuit_params = n_qubits * n_layers * 3
        self.fwd_params = nn.Parameter(torch.randn(self.n_circuit_params) * 0.1)
        self.bwd_params = nn.Parameter(torch.randn(self.n_circuit_params) * 0.1)

        # Setup quantum device
        if "cuda" in str(device):
            try:
                self.dev = qml.device("lightning.kokkos", wires=n_qubits, batch_obs=True)
            except:
                self.dev = qml.device("lightning.qubit", wires=n_qubits)
        else:
            self.dev = qml.device("lightning.qubit", wires=n_qubits)

        self._build_qnode()

    def _build_qnode(self):
        """Build QSVT-based SSM circuit."""

        @qml.qnode(self.dev, interface="torch", diff_method="best")
        def qsvt_ssm_circuit(state_angles, input_angles, circuit_params, poly_coeffs):
            n_qubits = self.n_qubits

            # Encode previous state
            for i in range(n_qubits):
                qml.RY(state_angles[i], wires=i)

            # QSVT transformation (approximates exp(A))
            for deg in range(len(poly_coeffs)):
                coeff = poly_coeffs[deg]
                for i in range(n_qubits):
                    qml.RZ(coeff * np.pi, wires=i)
                for i in range(n_qubits - 1):
                    qml.CNOT(wires=[i, i + 1])
                if n_qubits > 1:
                    qml.CNOT(wires=[n_qubits - 1, 0])

            # Variational layer
            param_idx = 0
            for layer in range(self.n_layers):
                for i in range(n_qubits):
                    qml.RX(circuit_params[param_idx], wires=i)
                    qml.RY(circuit_params[param_idx + 1], wires=i)
                    qml.RZ(circuit_params[param_idx + 2], wires=i)
                    param_idx += 3
                for i in range(n_qubits - 1):
                    qml.CNOT(wires=[i, i + 1])

            # Input injection
            for i in range(n_qubits):
                qml.RY(input_angles[i], wires=i)

            return [qml.expval(qml.PauliZ(i)) for i in range(n_qubits)]

        self.qnode = qsvt_ssm_circuit

    def _ssm_pass(self, x_angles: torch.Tensor, circuit_params: torch.Tensor, reverse: bool = False) -> torch.Tensor:
        """
        Single direction SSM pass.

        Args:
            x_angles: (batch, seq_len, n_qubits) angle-encoded inputs
            circuit_params: variational parameters
            reverse: if True, process sequence in reverse

        Returns:
            (batch, seq_len, n_qubits) SSM outputs
        """
        batch_size, seq_len, _ = x_angles.shape
        device = x_angles.device

        if reverse:
            x_angles = x_angles.flip(dims=[1])

        h = torch.zeros(batch_size, self.n_qubits, device=device)
        outputs = []

        for t in range(seq_len):
            x_t = x_angles[:, t, :]
            batch_outputs = []

            for b in range(batch_size):
                measurements = self.qnode(
                    h[b], x_t[b], circuit_params, self.poly_coeffs
                )
                batch_outputs.append(torch.stack(measurements))

            h = torch.stack(batch_outputs).to(device)
            outputs.append(h)

        output = torch.stack(outputs, dim=1)

        if reverse:
            output = output.flip(dims=[1])

        return output

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Bidirectional quantum SSM processing.

        Args:
            x: (batch, seq_len, d_model)
        Returns:
            (batch, seq_len, n_qubits)
        """
        # Project and encode
        x_proj = self.input_proj(x)
        x_angles = torch.tanh(x_proj) * np.pi

        # Forward pass
        h_fwd = self._ssm_pass(x_angles, self.fwd_params, reverse=False)

        # Backward pass
        h_bwd = self._ssm_pass(x_angles, self.bwd_params, reverse=True)

        # Combine via learned coefficients (Hydra-style)
        fwd_weight = torch.sigmoid(self.fwd_coeff)
        bwd_weight = torch.sigmoid(self.bwd_coeff)
        combined = fwd_weight * h_fwd + bwd_weight * h_bwd

        return combined


class ClassicalFeaturesQuantumHydraSSM(nn.Module):
    """
    Classical Features + Quantum Hydra SSM (Model A3a)

    Architecture:
        Input → SharedClassicalEncoder → QuantumHydraSSMCore → Pooling → OutputHead
                     Classical               QUANTUM (Bidir)    Classical
    """

    def __init__(
        self,
        input_dim: int,
        num_classes: int,
        d_model: int = 64,
        d_hidden: int = 128,
        n_qubits: int = 4,
        n_layers: int = 2,
        qsvt_degree: int = 3,
        dropout: float = 0.1,
        device: str = "cpu"
    ):
        super().__init__()

        self.encoder = SharedClassicalEncoder(
            input_dim=input_dim,
            d_model=d_model,
            d_hidden=d_hidden,
            dropout=dropout
        )

        self.quantum_ssm = QuantumHydraSSMCore(
            n_qubits=n_qubits,
            n_layers=n_layers,
            qsvt_degree=qsvt_degree,
            d_model=d_model,
            device=device
        )

        self.quantum_proj = nn.Linear(n_qubits, d_model)
        self.pooling = SequencePooling(d_model=d_model)
        self.output_head = SharedOutputHead(
            d_model=d_model,
            num_classes=num_classes,
            dropout=dropout
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.encoder(x)
        quantum_out = self.quantum_ssm(features)
        projected = self.quantum_proj(quantum_out)
        combined = features + projected
        pooled = self.pooling(combined)
        return self.output_head(pooled)
```

### 4.3 Model A3b: Classical + Quantum Mamba SSM (Selective)

**File**: `models/ClassicalFeaturesQuantumMambaSSM.py`

**Architecture**:
```
Input → [Classical Encoder] → [Quantum Mamba SSM] → [Output Head]
                                 ├─ Input-dependent Δ, B, C
                                 └─ QSVT for exp(Δ·A)
```

**Implementation**:

```python
"""
Classical Features + Quantum Mamba SSM (Model A3b)

Selective Quantum SSM with input-dependent parameters.

Key Design:
1. Classical encoder extracts features
2. Compute input-dependent Δ, B, C (selectivity)
3. QSVT applies exp(Δ·A) transformation
4. Unidirectional processing with selective memory
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import pennylane as qml
import numpy as np

from models.shared_components import (
    SharedClassicalEncoder,
    SharedOutputHead,
    SequencePooling
)


class QuantumMambaSSMCore(nn.Module):
    """
    Selective Quantum SSM using QSVT (Mamba-style).

    Implements:
        Δ[t], B[t], C[t] = f(x[t])           # Input-dependent (selective)
        |h[t]⟩ = QSVT_exp(Δ[t]·A)|h[t-1]⟩ + Δ[t]·B[t]|x[t]⟩

    The key difference from Hydra: parameters vary with input (selectivity).
    """

    def __init__(
        self,
        n_qubits: int = 4,
        n_layers: int = 2,
        qsvt_degree: int = 3,
        d_model: int = 64,
        dt_rank: str = "auto",
        dt_min: float = 0.001,
        dt_max: float = 0.1,
        device: str = "cpu"
    ):
        super().__init__()

        self.n_qubits = n_qubits
        self.n_layers = n_layers
        self.qsvt_degree = qsvt_degree
        self.d_model = d_model

        # dt_rank for selective parameter projection
        if dt_rank == "auto":
            self.dt_rank = max(d_model // 16, 1)
        else:
            self.dt_rank = dt_rank

        # Input projection
        self.input_proj = nn.Linear(d_model, n_qubits)

        # Selective parameter projections (Mamba-style)
        # Project input to get Δ, B, C
        self.x_proj = nn.Linear(d_model, self.dt_rank + n_qubits * 2, bias=False)

        # Δ projection with proper initialization
        self.dt_proj = nn.Linear(self.dt_rank, n_qubits, bias=True)

        # Initialize dt bias for desired range
        dt = torch.exp(
            torch.rand(n_qubits) * (np.log(dt_max) - np.log(dt_min)) + np.log(dt_min)
        )
        inv_dt = dt + torch.log(-torch.expm1(-dt))
        with torch.no_grad():
            self.dt_proj.bias.copy_(inv_dt)

        # Base QSVT polynomial (will be modulated by Δ)
        self.poly_coeffs_base = nn.Parameter(torch.randn(qsvt_degree + 1) * 0.1)

        # Variational circuit parameters
        self.n_circuit_params = n_qubits * n_layers * 3
        self.circuit_params = nn.Parameter(torch.randn(self.n_circuit_params) * 0.1)

        # D parameter (skip connection)
        self.D = nn.Parameter(torch.ones(n_qubits))

        # Setup quantum device
        if "cuda" in str(device):
            try:
                self.dev = qml.device("lightning.kokkos", wires=n_qubits, batch_obs=True)
            except:
                self.dev = qml.device("lightning.qubit", wires=n_qubits)
        else:
            self.dev = qml.device("lightning.qubit", wires=n_qubits)

        self._build_qnode()

    def _build_qnode(self):
        """Build selective QSVT circuit."""

        @qml.qnode(self.dev, interface="torch", diff_method="best")
        def selective_qsvt_circuit(state_angles, input_angles, dt_angles, circuit_params, poly_coeffs):
            """
            Selective QSVT circuit with input-dependent Δ modulation.

            Args:
                state_angles: Previous state
                input_angles: Current input
                dt_angles: Input-dependent time step (Δ)
                circuit_params: Variational parameters
                poly_coeffs: QSVT polynomial coefficients
            """
            n_qubits = self.n_qubits

            # Encode previous state
            for i in range(n_qubits):
                qml.RY(state_angles[i], wires=i)

            # QSVT with Δ-modulated transformation
            # Key: polynomial coefficients are scaled by dt_angles
            for deg in range(len(poly_coeffs)):
                coeff = poly_coeffs[deg]
                for i in range(n_qubits):
                    # Δ modulates the transformation strength
                    qml.RZ(coeff * dt_angles[i] * np.pi, wires=i)
                for i in range(n_qubits - 1):
                    qml.CNOT(wires=[i, i + 1])
                if n_qubits > 1:
                    qml.CNOT(wires=[n_qubits - 1, 0])

            # Variational layer
            param_idx = 0
            for layer in range(self.n_layers):
                for i in range(n_qubits):
                    qml.RX(circuit_params[param_idx], wires=i)
                    qml.RY(circuit_params[param_idx + 1], wires=i)
                    qml.RZ(circuit_params[param_idx + 2], wires=i)
                    param_idx += 3
                for i in range(n_qubits - 1):
                    qml.CNOT(wires=[i, i + 1])

            # Input injection (scaled by Δ)
            for i in range(n_qubits):
                qml.RY(input_angles[i] * dt_angles[i], wires=i)

            return [qml.expval(qml.PauliZ(i)) for i in range(n_qubits)]

        self.qnode = selective_qsvt_circuit

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Selective quantum SSM processing.

        Args:
            x: (batch, seq_len, d_model)
        Returns:
            (batch, seq_len, n_qubits)
        """
        batch_size, seq_len, _ = x.shape
        device = x.device

        # Project input features
        x_proj = self.input_proj(x)
        x_angles = torch.tanh(x_proj) * np.pi

        # Compute selective parameters (Δ, B, C)
        x_dbc = self.x_proj(x)  # (batch, seq_len, dt_rank + 2*n_qubits)

        # Split into components
        dt_raw = x_dbc[..., :self.dt_rank]
        B = x_dbc[..., self.dt_rank:self.dt_rank + self.n_qubits]
        C = x_dbc[..., self.dt_rank + self.n_qubits:]

        # Compute Δ with softplus activation
        dt = F.softplus(self.dt_proj(dt_raw))  # (batch, seq_len, n_qubits)

        # Normalize dt for quantum circuit
        dt_angles = torch.tanh(dt) * np.pi

        # Initialize hidden state
        h = torch.zeros(batch_size, self.n_qubits, device=device)

        outputs = []

        # Sequential selective processing
        for t in range(seq_len):
            x_t = x_angles[:, t, :]
            dt_t = dt_angles[:, t, :]
            B_t = B[:, t, :]
            C_t = C[:, t, :]

            batch_outputs = []
            for b in range(batch_size):
                measurements = self.qnode(
                    h[b],
                    x_t[b],
                    dt_t[b],
                    self.circuit_params,
                    self.poly_coeffs_base
                )
                new_state = torch.stack(measurements)
                batch_outputs.append(new_state)

            h = torch.stack(batch_outputs).to(device)

            # Output with selective C and skip connection D
            y_t = (C_t * h).sum(dim=-1, keepdim=True).expand_as(h) + self.D * x_proj[:, t, :]
            outputs.append(h)

        return torch.stack(outputs, dim=1)


class ClassicalFeaturesQuantumMambaSSM(nn.Module):
    """
    Classical Features + Quantum Mamba SSM (Model A3b)

    Architecture:
        Input → SharedClassicalEncoder → QuantumMambaSSMCore → Pooling → OutputHead
                     Classical               QUANTUM (Select)    Classical
    """

    def __init__(
        self,
        input_dim: int,
        num_classes: int,
        d_model: int = 64,
        d_hidden: int = 128,
        n_qubits: int = 4,
        n_layers: int = 2,
        qsvt_degree: int = 3,
        dropout: float = 0.1,
        device: str = "cpu"
    ):
        super().__init__()

        self.encoder = SharedClassicalEncoder(
            input_dim=input_dim,
            d_model=d_model,
            d_hidden=d_hidden,
            dropout=dropout
        )

        self.quantum_ssm = QuantumMambaSSMCore(
            n_qubits=n_qubits,
            n_layers=n_layers,
            qsvt_degree=qsvt_degree,
            d_model=d_model,
            device=device
        )

        self.quantum_proj = nn.Linear(n_qubits, d_model)
        self.pooling = SequencePooling(d_model=d_model)
        self.output_head = SharedOutputHead(
            d_model=d_model,
            num_classes=num_classes,
            dropout=dropout
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.encoder(x)
        quantum_out = self.quantum_ssm(features)
        projected = self.quantum_proj(quantum_out)
        combined = features + projected
        pooled = self.pooling(combined)
        return self.output_head(pooled)
```

---

## 5. Ablation 2: Which Quantum Sequence Mixing (Extended)

### Purpose
Given IDENTICAL classical feature extraction, compare quantum mixing mechanisms in both Hydra and Mamba styles.

### Critical Requirement
**ALL models in Ablation 2 MUST use the SAME `SharedClassicalEncoder` instance or identical architecture.**

### 5.1 Model List

| Model | Encoder | Mixing Type | Style | Description |
|-------|---------|-------------|-------|-------------|
| **B1a** | Shared | Quantum SSM | Hydra (Bidir) | Quantum Hydra SSM |
| **B1b** | Shared | Quantum SSM | Mamba (Select) | Quantum Mamba SSM |
| **B2a** | Shared | Quantum Attention | Hydra-style | Bidirectional attention |
| **B2b** | Shared | Quantum Attention | Mamba-style | Selective attention |
| **B3a** | Shared | Quantum Gated | Hydra (Bidir) | Bidirectional gating |
| **B3b** | Shared | Quantum Gated | Mamba (Select) | Selective gating |
| B4 | Shared | Classical SSM | Baseline | Classical control |

### 5.2 Comparison Diagram

```
                    ┌─────────────────────────────────────────────┐
                    │         SharedClassicalEncoder              │
                    │              (IDENTICAL)                    │
                    └─────────────────┬───────────────────────────┘
                                      │
        ┌─────────────────────────────┼─────────────────────────────┐
        │                             │                             │
        ▼                             ▼                             ▼
 ┌─────────────┐              ┌─────────────┐              ┌─────────────┐
 │  Quantum    │              │  Quantum    │              │  Quantum    │
 │    SSM      │              │  Attention  │              │   Gated     │
 └──────┬──────┘              └──────┬──────┘              └──────┬──────┘
        │                             │                             │
   ┌────┴────┐                   ┌────┴────┐                   ┌────┴────┐
   │         │                   │         │                   │         │
   ▼         ▼                   ▼         ▼                   ▼         ▼
  B1a       B1b                 B2a       B2b                 B3a       B3b
 Hydra     Mamba               Hydra     Mamba               Hydra     Mamba
(Bidir)   (Select)            (Bidir)   (Select)            (Bidir)   (Select)
```

### 5.3 Model B2: Classical + Quantum Attention (Hydra and Mamba Styles)

**File**: `models/ClassicalFeaturesQuantumAttention.py`

```python
"""
Classical Features + Quantum Attention (Models B2a and B2b)

B2a: Hydra-style bidirectional quantum attention
B2b: Mamba-style selective quantum attention
"""

import torch
import torch.nn as nn
import pennylane as qml
import numpy as np

from models.shared_components import (
    SharedClassicalEncoder,
    SharedOutputHead,
    SequencePooling
)


class QuantumHydraAttentionCore(nn.Module):
    """
    Bidirectional Quantum Attention (Hydra-style).

    Processes attention in both directions and combines results.
    """

    def __init__(
        self,
        n_qubits: int = 4,
        n_layers: int = 2,
        qsvt_degree: int = 3,
        d_model: int = 64,
        device: str = "cpu"
    ):
        super().__init__()

        self.n_qubits = n_qubits
        self.n_layers = n_layers
        self.d_model = d_model

        # Query, Key, Value projections
        self.q_proj = nn.Linear(d_model, n_qubits)
        self.k_proj = nn.Linear(d_model, n_qubits)
        self.v_proj = nn.Linear(d_model, n_qubits)

        # QSVT polynomial for attention weights
        self.poly_coeffs = nn.Parameter(torch.randn(qsvt_degree + 1) * 0.1)

        # Direction combination
        self.fwd_coeff = nn.Parameter(torch.tensor([0.5]))
        self.bwd_coeff = nn.Parameter(torch.tensor([0.5]))

        # Variational parameters
        self.n_circuit_params = n_qubits * n_layers * 3
        self.circuit_params = nn.Parameter(torch.randn(self.n_circuit_params) * 0.1)

        # Quantum device
        self.dev = qml.device("lightning.qubit", wires=n_qubits)
        self._build_qnode()

    def _build_qnode(self):
        @qml.qnode(self.dev, interface="torch", diff_method="best")
        def attention_circuit(q_angles, k_angles, v_angles, circuit_params, poly_coeffs):
            n_qubits = self.n_qubits

            # Encode Q, K, V
            for i in range(n_qubits):
                qml.RY(q_angles[i], wires=i)
                qml.RX(k_angles[i], wires=i)

            # QSVT for attention scores
            for deg, coeff in enumerate(poly_coeffs):
                for i in range(n_qubits):
                    qml.RZ(coeff * np.pi, wires=i)
                for i in range(n_qubits - 1):
                    qml.CNOT(wires=[i, i + 1])

            # Value integration
            for i in range(n_qubits):
                qml.RY(v_angles[i], wires=i)

            # Variational layer
            param_idx = 0
            for layer in range(self.n_layers):
                for i in range(n_qubits):
                    qml.RX(circuit_params[param_idx], wires=i)
                    qml.RY(circuit_params[param_idx + 1], wires=i)
                    qml.RZ(circuit_params[param_idx + 2], wires=i)
                    param_idx += 3

            return [qml.expval(qml.PauliZ(i)) for i in range(n_qubits)]

        self.qnode = attention_circuit

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, seq_len, _ = x.shape
        device = x.device

        Q = torch.tanh(self.q_proj(x)) * np.pi
        K = torch.tanh(self.k_proj(x)) * np.pi
        V = torch.tanh(self.v_proj(x)) * np.pi

        # Forward attention
        outputs_fwd = []
        for t in range(seq_len):
            batch_out = []
            for b in range(batch_size):
                # Attend to positions 0..t
                context = torch.zeros(self.n_qubits, device=device)
                for s in range(t + 1):
                    attn = self.qnode(Q[b, t], K[b, s], V[b, s],
                                     self.circuit_params, self.poly_coeffs)
                    context = context + torch.stack(attn).to(device) / (t + 1)
                batch_out.append(context)
            outputs_fwd.append(torch.stack(batch_out))

        # Backward attention
        outputs_bwd = []
        for t in range(seq_len):
            batch_out = []
            for b in range(batch_size):
                # Attend to positions t..seq_len-1
                context = torch.zeros(self.n_qubits, device=device)
                for s in range(t, seq_len):
                    attn = self.qnode(Q[b, t], K[b, s], V[b, s],
                                     self.circuit_params, self.poly_coeffs)
                    context = context + torch.stack(attn).to(device) / (seq_len - t)
                batch_out.append(context)
            outputs_bwd.append(torch.stack(batch_out))

        h_fwd = torch.stack(outputs_fwd, dim=1)
        h_bwd = torch.stack(outputs_bwd, dim=1)

        # Combine directions
        fwd_weight = torch.sigmoid(self.fwd_coeff)
        bwd_weight = torch.sigmoid(self.bwd_coeff)
        return fwd_weight * h_fwd + bwd_weight * h_bwd


class QuantumMambaAttentionCore(nn.Module):
    """
    Selective Quantum Attention (Mamba-style).

    Uses input-dependent attention parameters.
    """

    def __init__(
        self,
        n_qubits: int = 4,
        n_layers: int = 2,
        qsvt_degree: int = 3,
        d_model: int = 64,
        device: str = "cpu"
    ):
        super().__init__()

        self.n_qubits = n_qubits
        self.n_layers = n_layers
        self.d_model = d_model

        # Projections
        self.q_proj = nn.Linear(d_model, n_qubits)
        self.k_proj = nn.Linear(d_model, n_qubits)
        self.v_proj = nn.Linear(d_model, n_qubits)

        # Selective gate projection
        self.gate_proj = nn.Linear(d_model, n_qubits)

        # QSVT polynomial
        self.poly_coeffs = nn.Parameter(torch.randn(qsvt_degree + 1) * 0.1)

        # Variational parameters
        self.n_circuit_params = n_qubits * n_layers * 3
        self.circuit_params = nn.Parameter(torch.randn(self.n_circuit_params) * 0.1)

        self.dev = qml.device("lightning.qubit", wires=n_qubits)
        self._build_qnode()

    def _build_qnode(self):
        @qml.qnode(self.dev, interface="torch", diff_method="best")
        def selective_attention_circuit(q_angles, k_angles, v_angles, gate_angles,
                                        circuit_params, poly_coeffs):
            n_qubits = self.n_qubits

            # Encode with gating
            for i in range(n_qubits):
                qml.RY(q_angles[i] * gate_angles[i], wires=i)
                qml.RX(k_angles[i], wires=i)

            # QSVT with selective modulation
            for deg, coeff in enumerate(poly_coeffs):
                for i in range(n_qubits):
                    qml.RZ(coeff * gate_angles[i] * np.pi, wires=i)
                for i in range(n_qubits - 1):
                    qml.CNOT(wires=[i, i + 1])

            # Value integration
            for i in range(n_qubits):
                qml.RY(v_angles[i], wires=i)

            # Variational layer
            param_idx = 0
            for layer in range(self.n_layers):
                for i in range(n_qubits):
                    qml.RX(circuit_params[param_idx], wires=i)
                    qml.RY(circuit_params[param_idx + 1], wires=i)
                    qml.RZ(circuit_params[param_idx + 2], wires=i)
                    param_idx += 3

            return [qml.expval(qml.PauliZ(i)) for i in range(n_qubits)]

        self.qnode = selective_attention_circuit

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, seq_len, _ = x.shape
        device = x.device

        Q = torch.tanh(self.q_proj(x)) * np.pi
        K = torch.tanh(self.k_proj(x)) * np.pi
        V = torch.tanh(self.v_proj(x)) * np.pi
        G = torch.sigmoid(self.gate_proj(x))  # Selective gates

        outputs = []
        for t in range(seq_len):
            batch_out = []
            for b in range(batch_size):
                # Causal attention with selectivity
                context = torch.zeros(self.n_qubits, device=device)
                for s in range(t + 1):
                    attn = self.qnode(Q[b, t], K[b, s], V[b, s], G[b, t],
                                     self.circuit_params, self.poly_coeffs)
                    context = context + torch.stack(attn).to(device) / (t + 1)
                batch_out.append(context)
            outputs.append(torch.stack(batch_out))

        return torch.stack(outputs, dim=1)


class ClassicalFeaturesQuantumHydraAttention(nn.Module):
    """Model B2a: Classical + Quantum Hydra Attention"""

    def __init__(self, input_dim, num_classes, d_model=64, d_hidden=128,
                 n_qubits=4, n_layers=2, qsvt_degree=3, dropout=0.1, device="cpu"):
        super().__init__()

        self.encoder = SharedClassicalEncoder(input_dim, d_model, d_hidden, dropout)
        self.quantum_attention = QuantumHydraAttentionCore(
            n_qubits, n_layers, qsvt_degree, d_model, device
        )
        self.quantum_proj = nn.Linear(n_qubits, d_model)
        self.pooling = SequencePooling(d_model)
        self.output_head = SharedOutputHead(d_model, num_classes, dropout)

    def forward(self, x):
        features = self.encoder(x)
        quantum_out = self.quantum_attention(features)
        projected = self.quantum_proj(quantum_out)
        combined = features + projected
        pooled = self.pooling(combined)
        return self.output_head(pooled)


class ClassicalFeaturesQuantumMambaAttention(nn.Module):
    """Model B2b: Classical + Quantum Mamba Attention"""

    def __init__(self, input_dim, num_classes, d_model=64, d_hidden=128,
                 n_qubits=4, n_layers=2, qsvt_degree=3, dropout=0.1, device="cpu"):
        super().__init__()

        self.encoder = SharedClassicalEncoder(input_dim, d_model, d_hidden, dropout)
        self.quantum_attention = QuantumMambaAttentionCore(
            n_qubits, n_layers, qsvt_degree, d_model, device
        )
        self.quantum_proj = nn.Linear(n_qubits, d_model)
        self.pooling = SequencePooling(d_model)
        self.output_head = SharedOutputHead(d_model, num_classes, dropout)

    def forward(self, x):
        features = self.encoder(x)
        quantum_out = self.quantum_attention(features)
        projected = self.quantum_proj(quantum_out)
        combined = features + projected
        pooled = self.pooling(combined)
        return self.output_head(pooled)
```

### 5.4 Model B3: Classical + Quantum Gated (Hydra and Mamba Styles)

**File**: `models/ClassicalFeaturesQuantumGated.py`

```python
"""
Classical Features + Quantum Gated Recurrence (Models B3a and B3b)

B3a: Hydra-style bidirectional gating
B3b: Mamba-style selective gating
"""

import torch
import torch.nn as nn
import pennylane as qml
import numpy as np

from models.shared_components import (
    SharedClassicalEncoder,
    SharedOutputHead,
    SequencePooling
)


class QuantumHydraGatedCore(nn.Module):
    """
    Bidirectional Quantum Gated Recurrence (Hydra-style).

    Processes gating in both directions.
    """

    def __init__(
        self,
        n_qubits: int = 4,
        n_layers: int = 2,
        qsvt_degree: int = 3,
        d_model: int = 64,
        device: str = "cpu"
    ):
        super().__init__()

        self.n_qubits = n_qubits
        self.n_layers = n_layers
        self.d_model = d_model

        self.input_proj = nn.Linear(d_model, n_qubits)
        self.hidden_proj = nn.Linear(n_qubits, n_qubits)

        # Shared gate polynomials
        self.forget_poly = nn.Parameter(torch.randn(qsvt_degree + 1) * 0.1)
        self.input_poly = nn.Parameter(torch.randn(qsvt_degree + 1) * 0.1)
        self.output_poly = nn.Parameter(torch.randn(qsvt_degree + 1) * 0.1)

        # Direction combination
        self.fwd_coeff = nn.Parameter(torch.tensor([0.5]))
        self.bwd_coeff = nn.Parameter(torch.tensor([0.5]))

        n_params = n_qubits * n_layers * 3
        self.gate_params = nn.Parameter(torch.randn(n_params) * 0.1)

        self.dev = qml.device("lightning.qubit", wires=n_qubits)
        self._build_qnode()

    def _build_qnode(self):
        @qml.qnode(self.dev, interface="torch", diff_method="best")
        def gate_circuit(input_angles, hidden_angles, circuit_params, poly_coeffs):
            n_qubits = self.n_qubits

            for i in range(n_qubits):
                qml.RY(input_angles[i], wires=i)
                qml.RX(hidden_angles[i], wires=i)

            for deg, coeff in enumerate(poly_coeffs):
                for i in range(n_qubits):
                    qml.RZ(coeff * np.pi, wires=i)
                for i in range(n_qubits - 1):
                    qml.CNOT(wires=[i, i + 1])

            param_idx = 0
            for layer in range(self.n_layers):
                for i in range(n_qubits):
                    qml.RX(circuit_params[param_idx], wires=i)
                    qml.RY(circuit_params[param_idx + 1], wires=i)
                    qml.RZ(circuit_params[param_idx + 2], wires=i)
                    param_idx += 3

            return [qml.expval(qml.PauliZ(i)) for i in range(n_qubits)]

        self.gate_circuit = gate_circuit

    def _gated_pass(self, x_angles, reverse=False):
        batch_size, seq_len, _ = x_angles.shape
        device = x_angles.device

        if reverse:
            x_angles = x_angles.flip(dims=[1])

        h = torch.zeros(batch_size, self.n_qubits, device=device)
        c = torch.zeros(batch_size, self.n_qubits, device=device)

        outputs = []

        for t in range(seq_len):
            x_t = x_angles[:, t, :]
            h_proj = torch.tanh(self.hidden_proj(h)) * np.pi

            batch_h = []
            for b in range(batch_size):
                f = torch.sigmoid(torch.stack(
                    self.gate_circuit(x_t[b], h_proj[b], self.gate_params, self.forget_poly)
                ))
                i = torch.sigmoid(torch.stack(
                    self.gate_circuit(x_t[b], h_proj[b], self.gate_params, self.input_poly)
                ))
                o = torch.sigmoid(torch.stack(
                    self.gate_circuit(x_t[b], h_proj[b], self.gate_params, self.output_poly)
                ))
                c_cand = torch.tanh(torch.stack(
                    self.gate_circuit(x_t[b], h_proj[b], self.gate_params, self.forget_poly)
                ))

                c_new = f * c[b] + i * c_cand
                h_new = o * torch.tanh(c_new)
                batch_h.append(h_new)

            h = torch.stack(batch_h).to(device)
            outputs.append(h)

        output = torch.stack(outputs, dim=1)
        if reverse:
            output = output.flip(dims=[1])
        return output

    def forward(self, x):
        x_proj = self.input_proj(x)
        x_angles = torch.tanh(x_proj) * np.pi

        h_fwd = self._gated_pass(x_angles, reverse=False)
        h_bwd = self._gated_pass(x_angles, reverse=True)

        fwd_weight = torch.sigmoid(self.fwd_coeff)
        bwd_weight = torch.sigmoid(self.bwd_coeff)
        return fwd_weight * h_fwd + bwd_weight * h_bwd


class QuantumMambaGatedCore(nn.Module):
    """
    Selective Quantum Gated Recurrence (Mamba-style).

    Uses input-dependent gating parameters.
    """

    def __init__(
        self,
        n_qubits: int = 4,
        n_layers: int = 2,
        qsvt_degree: int = 3,
        d_model: int = 64,
        device: str = "cpu"
    ):
        super().__init__()

        self.n_qubits = n_qubits
        self.n_layers = n_layers
        self.d_model = d_model

        self.input_proj = nn.Linear(d_model, n_qubits)
        self.hidden_proj = nn.Linear(n_qubits, n_qubits)

        # Selective parameter projection
        self.selective_proj = nn.Linear(d_model, n_qubits)

        # Gate polynomials
        self.forget_poly = nn.Parameter(torch.randn(qsvt_degree + 1) * 0.1)
        self.input_poly = nn.Parameter(torch.randn(qsvt_degree + 1) * 0.1)
        self.output_poly = nn.Parameter(torch.randn(qsvt_degree + 1) * 0.1)

        n_params = n_qubits * n_layers * 3
        self.gate_params = nn.Parameter(torch.randn(n_params) * 0.1)

        self.dev = qml.device("lightning.qubit", wires=n_qubits)
        self._build_qnode()

    def _build_qnode(self):
        @qml.qnode(self.dev, interface="torch", diff_method="best")
        def selective_gate_circuit(input_angles, hidden_angles, select_angles,
                                   circuit_params, poly_coeffs):
            n_qubits = self.n_qubits

            for i in range(n_qubits):
                qml.RY(input_angles[i] * select_angles[i], wires=i)
                qml.RX(hidden_angles[i], wires=i)

            for deg, coeff in enumerate(poly_coeffs):
                for i in range(n_qubits):
                    qml.RZ(coeff * select_angles[i] * np.pi, wires=i)
                for i in range(n_qubits - 1):
                    qml.CNOT(wires=[i, i + 1])

            param_idx = 0
            for layer in range(self.n_layers):
                for i in range(n_qubits):
                    qml.RX(circuit_params[param_idx], wires=i)
                    qml.RY(circuit_params[param_idx + 1], wires=i)
                    qml.RZ(circuit_params[param_idx + 2], wires=i)
                    param_idx += 3

            return [qml.expval(qml.PauliZ(i)) for i in range(n_qubits)]

        self.gate_circuit = selective_gate_circuit

    def forward(self, x):
        batch_size, seq_len, _ = x.shape
        device = x.device

        x_proj = self.input_proj(x)
        x_angles = torch.tanh(x_proj) * np.pi
        select = torch.sigmoid(self.selective_proj(x))

        h = torch.zeros(batch_size, self.n_qubits, device=device)
        c = torch.zeros(batch_size, self.n_qubits, device=device)

        outputs = []

        for t in range(seq_len):
            x_t = x_angles[:, t, :]
            s_t = select[:, t, :]
            h_proj = torch.tanh(self.hidden_proj(h)) * np.pi

            batch_h = []
            for b in range(batch_size):
                f = torch.sigmoid(torch.stack(
                    self.gate_circuit(x_t[b], h_proj[b], s_t[b], self.gate_params, self.forget_poly)
                ))
                i = torch.sigmoid(torch.stack(
                    self.gate_circuit(x_t[b], h_proj[b], s_t[b], self.gate_params, self.input_poly)
                ))
                o = torch.sigmoid(torch.stack(
                    self.gate_circuit(x_t[b], h_proj[b], s_t[b], self.gate_params, self.output_poly)
                ))
                c_cand = torch.tanh(torch.stack(
                    self.gate_circuit(x_t[b], h_proj[b], s_t[b], self.gate_params, self.forget_poly)
                ))

                c_new = f * c[b] + i * c_cand
                h_new = o * torch.tanh(c_new)
                batch_h.append(h_new)

            h = torch.stack(batch_h).to(device)
            outputs.append(h)

        return torch.stack(outputs, dim=1)


class ClassicalFeaturesQuantumHydraGated(nn.Module):
    """Model B3a: Classical + Quantum Hydra Gated"""

    def __init__(self, input_dim, num_classes, d_model=64, d_hidden=128,
                 n_qubits=4, n_layers=2, qsvt_degree=3, dropout=0.1, device="cpu"):
        super().__init__()

        self.encoder = SharedClassicalEncoder(input_dim, d_model, d_hidden, dropout)
        self.quantum_gated = QuantumHydraGatedCore(n_qubits, n_layers, qsvt_degree, d_model, device)
        self.quantum_proj = nn.Linear(n_qubits, d_model)
        self.pooling = SequencePooling(d_model)
        self.output_head = SharedOutputHead(d_model, num_classes, dropout)

    def forward(self, x):
        features = self.encoder(x)
        quantum_out = self.quantum_gated(features)
        projected = self.quantum_proj(quantum_out)
        combined = features + projected
        pooled = self.pooling(combined)
        return self.output_head(pooled)


class ClassicalFeaturesQuantumMambaGated(nn.Module):
    """Model B3b: Classical + Quantum Mamba Gated"""

    def __init__(self, input_dim, num_classes, d_model=64, d_hidden=128,
                 n_qubits=4, n_layers=2, qsvt_degree=3, dropout=0.1, device="cpu"):
        super().__init__()

        self.encoder = SharedClassicalEncoder(input_dim, d_model, d_hidden, dropout)
        self.quantum_gated = QuantumMambaGatedCore(n_qubits, n_layers, qsvt_degree, d_model, device)
        self.quantum_proj = nn.Linear(n_qubits, d_model)
        self.pooling = SequencePooling(d_model)
        self.output_head = SharedOutputHead(d_model, num_classes, dropout)

    def forward(self, x):
        features = self.encoder(x)
        quantum_out = self.quantum_gated(features)
        projected = self.quantum_proj(quantum_out)
        combined = features + projected
        pooled = self.pooling(combined)
        return self.output_head(pooled)
```

---

## 6. Ablation 3: SSM vs Attention Comparison (Extended)

### Purpose
Compare complete architectures including both Hydra and Mamba variants.

### 6.1 Model List

| Model | Architecture | Style | Description |
|-------|-------------|-------|-------------|
| **C1a** | Classical + Quantum Hydra SSM | Bidir | Our primary (Hydra) |
| **C1b** | Classical + Quantum Mamba SSM | Select | Our primary (Mamba) |
| C2 | QTSTransformer (Original) | N/A | Existing quantum attention |
| C3a | Classical Hydra | Bidir | Classical baseline |
| C3b | Classical Mamba | Select | Classical baseline |
| C4 | Classical Transformer | N/A | Classical attention |

### 6.2 Comparison Matrix

```
                        Quantum                         Classical
              ┌─────────────────────────────┐   ┌─────────────────────────────┐
              │                             │   │                             │
   SSM        │  C1a (Q-Hydra)  C1b (Q-Mamba)│   │  C3a (Hydra)   C3b (Mamba)  │
              │     Bidir          Select   │   │     Bidir          Select   │
              │                             │   │                             │
              ├─────────────────────────────┤   ├─────────────────────────────┤
              │                             │   │                             │
   Attention  │  C2 (QTSTransformer)        │   │  C4 (Transformer)           │
              │                             │   │                             │
              └─────────────────────────────┘   └─────────────────────────────┘
```

### 6.3 Key Comparisons

| Comparison | Question Answered |
|------------|-------------------|
| C1a vs C1b | Is bidirectional or selective quantum SSM better? |
| C1a vs C2 | Is Quantum Hydra SSM better than QTSTransformer? |
| C1b vs C2 | Is Quantum Mamba SSM better than QTSTransformer? |
| C1a vs C3a | Does quantum help Hydra? |
| C1b vs C3b | Does quantum help Mamba? |
| C1 vs C4 | SSM vs Attention (overall) |

---

## 7. Quantum SSM Variants: Hydra vs Mamba

### 7.1 Key Algorithmic Differences

| Aspect | Quantum Hydra SSM | Quantum Mamba SSM |
|--------|-------------------|-------------------|
| **Direction** | Bidirectional (fwd + bwd) | Unidirectional (fwd only) |
| **Parameters** | Fixed A, B, C matrices | Input-dependent Δ, B, C |
| **QSVT Target** | Fixed exp(A) | Variable exp(Δ·A) per timestep |
| **Selectivity** | From bidirectional context | From input-dependent gates |
| **Quantum Calls** | 2× (forward + backward) | 1× but more complex |

### 7.2 When to Use Each

| Task Type | Recommended | Reason |
|-----------|-------------|--------|
| Global context needed | Hydra (C1a) | Bidirectional sees full sequence |
| Selective memory | Mamba (C1b) | Input-dependent forgetting |
| Long sequences | Mamba (C1b) | Fewer quantum calls |
| Classification | Hydra (C1a) | Global context helps |
| Prediction | Mamba (C1b) | Causal structure |

### 7.3 QSVT Polynomial Design

**Hydra**: Same polynomial for all timesteps
```python
# Fixed polynomial approximating exp(A)
poly_coeffs = [c_0, c_1, c_2, c_3]  # Learned once
for t in range(seq_len):
    h[t] = QSVT(poly_coeffs) @ h[t-1] + B @ x[t]
```

**Mamba**: Polynomial modulated by input-dependent Δ
```python
# Base polynomial, modulated by Δ[t]
poly_coeffs_base = [c_0, c_1, c_2, c_3]  # Learned
for t in range(seq_len):
    delta_t = softplus(proj(x[t]))  # Input-dependent
    # Effective polynomial: [c_0*Δ, c_1*Δ, c_2*Δ, c_3*Δ]
    h[t] = QSVT(poly_coeffs_base * delta_t) @ h[t-1] + delta_t * B[t] @ x[t]
```

---

## 8. Experimental Protocol

### 8.1 Datasets

| Dataset | Type | Seq Len | Classes | Purpose |
|---------|------|---------|---------|---------|
| demo_human_or_worm | Genomic | 200 | 2 | Short sequence |
| demo_coding_vs_intergenomic | Genomic | 200 | 2 | Classification |
| PhysioNet EEG | Time-series | 160 | 2 | Temporal dynamics |
| Forrelation (n=6) | Synthetic | 64 | 2 | Quantum advantage |
| MNIST (sequential) | Image | 784 | 10 | Standard benchmark |

### 8.2 Training Configuration

```python
TRAINING_CONFIG = {
    "batch_size": 32,
    "learning_rate": 1e-3,
    "weight_decay": 1e-4,
    "n_epochs": 30,
    "optimizer": "AdamW",
    "scheduler": "CosineAnnealingLR",
    "seeds": [2024, 2025, 2026, 2027, 2028],
    "early_stopping_patience": 10,
    "gradient_clip": 1.0
}
```

### 8.3 Statistical Analysis

```python
STATISTICAL_TESTS = {
    "paired_comparison": "paired_t_test",
    "non_parametric": "wilcoxon_signed_rank",
    "multiple_comparison": "bonferroni_correction",
    "effect_size": "cohens_d",
    "confidence_interval": 0.95,
    "significance_level": 0.05
}
```

### 8.4 Execution Phases

```
Phase 1: Baseline Establishment
├── Run A1 (Classical Hydra) on all datasets
├── Run A2a, A2b (Current Quantum models) on all datasets
└── Verify reproducibility

Phase 2: Ablation 1 - Where to Apply Quantum
├── Implement A3a (Classical + Quantum Hydra SSM)
├── Implement A3b (Classical + Quantum Mamba SSM)
├── Implement A4a, A4b (End-to-End variants)
├── Run all A1-A4 variants on all datasets
└── Compare: A3 vs A2 (main), A3a vs A3b (Hydra vs Mamba)

Phase 3: Ablation 2 - Which Quantum Mixing
├── Implement B2a, B2b (Quantum Attention variants)
├── Implement B3a, B3b (Quantum Gated variants)
├── Implement B4 (Classical control)
├── Run all B1-B4 on all datasets
└── Compare: B1 vs B2 vs B3, and Hydra vs Mamba within each

Phase 4: Ablation 3 - Architecture Comparison
├── Run C1a, C1b, C2, C3a, C3b, C4 on all datasets
└── Compare complete architectures

Phase 5: Analysis
├── Aggregate all results
├── Statistical significance tests
├── Generate figures and tables
└── Write findings
```

---

## 9. Implementation Checklist

### Files to Create

```
models/
├── shared_components.py                      # Shared encoder, output, pooling
├── ClassicalFeaturesQuantumHydraSSM.py       # A3a/B1a/C1a
├── ClassicalFeaturesQuantumMambaSSM.py       # A3b/B1b/C1b
├── EndToEndQuantumHydraSSM.py                # A4a
├── EndToEndQuantumMambaSSM.py                # A4b
├── ClassicalFeaturesQuantumHydraAttention.py # B2a
├── ClassicalFeaturesQuantumMambaAttention.py # B2b
├── ClassicalFeaturesQuantumHydraGated.py     # B3a
├── ClassicalFeaturesQuantumMambaGated.py     # B3b
├── ClassicalFeaturesClassicalSSM.py          # B4
├── TrueClassicalMamba.py                     # C3b
└── ClassicalTransformer.py                   # C4

scripts/
├── run_ablation_study.py
├── run_ablation1.py
├── run_ablation2.py
├── run_ablation3.py
└── aggregate_ablation_results.py
```

### Implementation Priority

1. **Phase 1 (Core):**
   - [ ] `shared_components.py`
   - [ ] `ClassicalFeaturesQuantumMambaSSM.py` (A3b) - Most novel
   - [ ] `ClassicalFeaturesQuantumHydraSSM.py` (A3a)

2. **Phase 2 (Ablation 1):**
   - [ ] `EndToEndQuantumMambaSSM.py` (A4b)
   - [ ] `EndToEndQuantumHydraSSM.py` (A4a)

3. **Phase 3 (Ablation 2):**
   - [ ] `ClassicalFeaturesQuantumMambaAttention.py` (B2b)
   - [ ] `ClassicalFeaturesQuantumHydraAttention.py` (B2a)
   - [ ] `ClassicalFeaturesQuantumMambaGated.py` (B3b)
   - [ ] `ClassicalFeaturesQuantumHydraGated.py` (B3a)
   - [ ] `ClassicalFeaturesClassicalSSM.py` (B4)

4. **Phase 4 (Ablation 3):**
   - [ ] `TrueClassicalMamba.py` (C3b)
   - [ ] `ClassicalTransformer.py` (C4)

---

## Appendix: Complete Model ID Reference

| ID | Full Name | Features | Mixing | Style |
|----|-----------|----------|--------|-------|
| A1 | Classical Hydra | Classical | Classical SSM | Bidir |
| A2a | QuantumHydraSSM | Quantum VQC | Classical Hydra | Bidir |
| A2b | QuantumMambaSSM | Quantum VQC | Classical Mamba | Select |
| **A3a** | Classical + Quantum Hydra SSM | Classical | Quantum SSM | Bidir |
| **A3b** | Classical + Quantum Mamba SSM | Classical | Quantum SSM | Select |
| A4a | End-to-End Quantum Hydra | Quantum VQC | Quantum Hydra | Bidir |
| A4b | End-to-End Quantum Mamba | Quantum VQC | Quantum Mamba | Select |
| B1a | = A3a | Shared | Quantum SSM | Bidir |
| B1b | = A3b | Shared | Quantum SSM | Select |
| B2a | Classical + Q-Hydra Attention | Shared | Quantum Attn | Bidir |
| B2b | Classical + Q-Mamba Attention | Shared | Quantum Attn | Select |
| B3a | Classical + Q-Hydra Gated | Shared | Quantum Gated | Bidir |
| B3b | Classical + Q-Mamba Gated | Shared | Quantum Gated | Select |
| B4 | Classical + Classical SSM | Shared | Classical SSM | Baseline |
| C1a | = A3a | Classical | Quantum SSM | Bidir |
| C1b | = A3b | Classical | Quantum SSM | Select |
| C2 | QTSTransformer | Built-in | Quantum Attn | N/A |
| C3a | Classical Hydra | Classical | Classical SSM | Bidir |
| C3b | Classical Mamba | Classical | Classical SSM | Select |
| C4 | Classical Transformer | Classical | Classical Attn | N/A |

---

## Document Information

- **Author**: Junghoon Park
- **Created**: December 2024
- **Last Updated**: December 2024
- **Purpose**: Extended implementation guide for ablation study
- **Key Addition**: Hydra (bidirectional) vs Mamba (selective) variants
- **Related Documents**:
  - `FOUR_MODEL_ARCHITECTURE_COMPARISON.md`
  - `COMPUTATIONAL_COMPLEXITY_COMPARISON.md`
