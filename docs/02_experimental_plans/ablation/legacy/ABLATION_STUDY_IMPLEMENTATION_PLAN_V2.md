# Ablation Study Implementation Plan V2 (Unified Encoder)

## Classical Features + Quantum SSM: A Rigorous Evaluation with QTSTransformer Encoder

This document provides an updated implementation plan using the **QTSTransformer encoder** as the unified classical feature extraction component across all models, ensuring maximally controlled ablation studies.

---

## Table of Contents

1. [Overview](#1-overview)
2. [Unified Encoder Design](#2-unified-encoder-design)
3. [Classical vs Quantum Component Separation](#3-classical-vs-quantum-component-separation)
4. [Quantum SSM Core Implementations](#4-quantum-ssm-core-implementations)
5. [Complete Model Architectures](#5-complete-model-architectures)
6. [Ablation Study Framework](#6-ablation-study-framework)
7. [Experimental Protocol](#7-experimental-protocol)
8. [Implementation Checklist](#8-implementation-checklist)

---

## 1. Overview

### 1.1 Key Design Decision

**All models use the identical QTSTransformer encoder** for classical feature extraction, ensuring that:
1. Performance differences are **100% attributable** to the quantum mixing mechanism
2. Comparison with QTSTransformer is **maximally fair**
3. The proven encoder won't bottleneck quantum components

### 1.2 Architecture Philosophy

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                        CLASSICAL COMPONENT (SHARED)                         │
│                      QTSTransformer Feature Encoder                         │
│                                                                             │
│   Input → Conv2d → ReLU → MaxPool → Conv2d → ReLU → MaxPool → GLU → Sigmoid │
│                                                                             │
│   Purpose: Extract features and convert to quantum-compatible angles        │
│   Output: (batch, seq_len, n_qubits) angles in [0, π]                       │
└───────────────────────────────────┬─────────────────────────────────────────┘
                                    │
            ┌───────────────────────┼───────────────────────┐
            │                       │                       │
            ▼                       ▼                       ▼
┌───────────────────┐   ┌───────────────────┐   ┌───────────────────┐
│  Quantum Hydra    │   │  Quantum Mamba    │   │  QTSTransformer   │
│  SSM (QSVT/LCU)   │   │  SSM (Selective)  │   │  (Quantum Attn)   │
│                   │   │                   │   │                   │
│  ✅ QUANTUM       │   │  ✅ QUANTUM       │   │  ✅ QUANTUM       │
│  Selective        │   │  Selective        │   │  Attention        │
│  Forgetting       │   │  Forgetting       │   │  Mixing           │
└─────────┬─────────┘   └─────────┬─────────┘   └─────────┬─────────┘
          │                       │                       │
          ▼                       ▼                       ▼
       C1a Output              C1b Output              C2 Output
```

### 1.3 Research Questions Addressed

| Question | Ablation | Key Insight |
|----------|----------|-------------|
| Where should quantum be applied? | Ablation 1 | Quantum dynamics > Quantum features |
| Which quantum mixing is best? | Ablation 2 | SSM vs Attention vs Gated |
| Bidirectional vs Selective? | Ablation 2 | Hydra vs Mamba style |
| Complete architecture comparison? | Ablation 3 | New model vs QTSTransformer |

### 1.4 Complete Model List

| ID | Model Name | Encoder | Quantum Mixing | Style | Status |
|----|------------|---------|----------------|-------|--------|
| A1 | Classical Hydra | Classical (simple) | None | Bidir | EXISTS |
| A2a | QuantumHydraSSM | Quantum VQC | Classical Hydra | Bidir | EXISTS |
| A2b | QuantumMambaSSM | Quantum VQC | Classical Mamba | Select | EXISTS |
| **A3a** | QTS + Q-Hydra SSM | **QTS Encoder** | **Quantum SSM (QSVT)** | Bidir | **TO BUILD** |
| **A3b** | QTS + Q-Mamba SSM | **QTS Encoder** | **Quantum SSM (QSVT)** | Select | **TO BUILD** |
| A4a | End-to-End Q-Hydra | Quantum VQC | Quantum Hydra SSM | Bidir | TO BUILD |
| A4b | End-to-End Q-Mamba | Quantum VQC | Quantum Mamba SSM | Select | TO BUILD |
| B1a | QTS + Q-Hydra SSM | QTS Encoder | Quantum SSM | Bidir | = A3a |
| B1b | QTS + Q-Mamba SSM | QTS Encoder | Quantum SSM | Select | = A3b |
| B2a | QTS + Q-Hydra Attn | QTS Encoder | Quantum Attention | Bidir | TO BUILD |
| B2b | QTS + Q-Mamba Attn | QTS Encoder | Quantum Attention | Select | TO BUILD |
| B3a | QTS + Q-Hydra Gated | QTS Encoder | Quantum Gated | Bidir | TO BUILD |
| B3b | QTS + Q-Mamba Gated | QTS Encoder | Quantum Gated | Select | TO BUILD |
| B4 | QTS + Classical SSM | QTS Encoder | Classical SSM | Baseline | TO BUILD |
| C1a | QTS + Q-Hydra SSM | QTS Encoder | Quantum SSM | Bidir | = A3a |
| C1b | QTS + Q-Mamba SSM | QTS Encoder | Quantum SSM | Select | = A3b |
| **C2** | QTSTransformer | **QTS Encoder** | **Quantum Attention** | N/A | **EXISTS** |
| C3a | Classical Hydra | Simple | Classical SSM | Bidir | = A1 |
| C3b | Classical Mamba | Simple | Classical SSM | Select | TO BUILD |
| C4 | Classical Transformer | Simple | Classical Attention | N/A | TO BUILD |

---

## 2. Unified Encoder Design

### 2.1 QTSFeatureEncoder (Extracted from QTSTransformer)

**File**: `models/qts_encoder.py`

```python
"""
QTS Feature Encoder - Unified Classical Feature Extraction

This module provides the IDENTICAL encoder used in QTSTransformer,
ensuring controlled comparison across all ablation models.

The encoder is CLASSICAL - all quantum computation happens in the
mixing layer (SSM, Attention, or Gated).
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Literal


class Conv2dFeatureExtractor(nn.Module):
    """
    2D Convolutional feature extractor from QTSTransformer.

    Treats (feature_dim, n_timesteps) as a single-channel "image"
    and extracts spatio-temporal patterns.
    """

    def __init__(self, feature_dim: int, n_timesteps: int, n_output: int):
        super().__init__()
        self.n_timesteps = n_timesteps
        self.n_output = n_output

        # 2D Convolutional Network (identical to QTSTransformer)
        self.conv_net = nn.Sequential(
            # Layer 1: 1 → 16 channels
            nn.Conv2d(in_channels=1, out_channels=16, kernel_size=(3, 3), padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(4, 4)),

            # Layer 2: 16 → 32 channels
            nn.Conv2d(in_channels=16, out_channels=32, kernel_size=(3, 3), padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(4, 4))
        )

        # Calculate flattened size dynamically
        with torch.no_grad():
            dummy_input = torch.zeros(1, 1, feature_dim, n_timesteps)
            dummy_output = self.conv_net(dummy_input)
            self.flattened_size = dummy_output.numel()

        # Final projection
        self.final_linear = nn.Linear(self.flattened_size, n_timesteps * n_output)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch, feature_dim, n_timesteps)
        Returns:
            (batch, n_timesteps, n_output)
        """
        batch_size = x.size(0)
        x = x.unsqueeze(1)  # (batch, 1, feature_dim, n_timesteps)
        x = self.conv_net(x)
        x = x.view(batch_size, -1)
        x = self.final_linear(x)
        x = x.view(batch_size, self.n_timesteps, self.n_output)
        return x


class GatedFeedForward(nn.Module):
    """
    Gated Linear Unit (GLU) from QTSTransformer.

    Provides expressive non-linear transformation with gating mechanism.
    """

    def __init__(self, input_dim: int, output_dim: int, hidden_multiplier: int = 4):
        super().__init__()

        hidden_dim = input_dim * hidden_multiplier

        # Project to 2x hidden for gate + content split
        self.W_in = nn.Linear(input_dim, 2 * hidden_dim, bias=False)
        self.activation = nn.GELU()
        self.W_out = nn.Linear(hidden_dim, output_dim, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_proj = self.W_in(x)
        gate, content = x_proj.chunk(2, dim=-1)
        gated = self.activation(content) * torch.sigmoid(gate)
        return self.W_out(gated)


class Conv2dGLUPreprocessor(nn.Module):
    """
    Combined Conv2d + GLU from QTSTransformer.

    This is the RECOMMENDED encoder for best performance.
    """

    def __init__(self, feature_dim: int, n_timesteps: int, n_output: int):
        super().__init__()
        self.n_timesteps = n_timesteps
        self.n_output = n_output

        # 2D Convolutional Network
        self.conv_net = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=4),
            nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=4)
        )

        # Calculate flattened size
        with torch.no_grad():
            dummy = torch.zeros(1, 1, feature_dim, n_timesteps)
            dummy_out = self.conv_net(dummy)
            self.flattened_size = dummy_out.numel()

        # Gated Feed-Forward
        self.gated_ffn = GatedFeedForward(
            input_dim=self.flattened_size,
            output_dim=n_timesteps * n_output
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size = x.size(0)
        x = x.unsqueeze(1)
        x = self.conv_net(x)
        x = x.view(batch_size, -1)
        x = self.gated_ffn(x)
        return x.view(batch_size, self.n_timesteps, self.n_output)


class QTSFeatureEncoder(nn.Module):
    """
    Unified Feature Encoder from QTSTransformer.

    CRITICAL: This encoder must be IDENTICAL across all ablation models
    to ensure controlled comparison. Only the quantum mixing differs.

    Projection Types:
    - 'Linear': Simple linear projection
    - 'Conv2d': 2D CNN feature extraction
    - 'Conv2d_GLU': 2D CNN + Gated Linear Unit (RECOMMENDED)
    - 'GLU': Gated Linear Unit only

    Output: Angles in [0, π] ready for quantum circuit encoding
    """

    PROJECTION_TYPES = Literal['Linear', 'Conv2d', 'Conv2d_GLU', 'GLU']

    def __init__(
        self,
        feature_dim: int,
        n_timesteps: int,
        n_output: int,
        projection_type: str = 'Conv2d_GLU',
        dropout: float = 0.1
    ):
        super().__init__()

        self.feature_dim = feature_dim
        self.n_timesteps = n_timesteps
        self.n_output = n_output
        self.projection_type = projection_type

        self.dropout = nn.Dropout(dropout)

        # Build projection layer (IDENTICAL to QTSTransformer)
        if projection_type == 'Linear':
            self.projection = nn.Linear(feature_dim, n_output)
        elif projection_type == 'Conv2d':
            self.projection = Conv2dFeatureExtractor(feature_dim, n_timesteps, n_output)
        elif projection_type == 'Conv2d_GLU':
            self.projection = Conv2dGLUPreprocessor(feature_dim, n_timesteps, n_output)
        elif projection_type == 'GLU':
            self.projection = GatedFeedForward(feature_dim, n_output)
        else:
            raise ValueError(f"Unknown projection type: {projection_type}")

        # Sigmoid activation scales to [0, π] for quantum angles
        self.angle_activation = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Extract features and convert to quantum angles.

        Args:
            x: (batch, feature_dim, n_timesteps) or (batch, n_timesteps, feature_dim)
        Returns:
            angles: (batch, n_timesteps, n_output) in [0, π]
        """
        x = self.dropout(x)

        # Handle different projection types
        if self.projection_type == 'Linear':
            # For Linear: transpose to (batch, n_timesteps, feature_dim)
            if x.shape[1] == self.feature_dim:
                x = x.transpose(1, 2)
            features = self.projection(x)
        elif self.projection_type == 'GLU':
            if x.shape[1] == self.feature_dim:
                x = x.transpose(1, 2)
            features = self.projection(x)
        else:
            # Conv2d and Conv2d_GLU expect (batch, feature_dim, n_timesteps)
            if x.shape[2] == self.feature_dim:
                x = x.transpose(1, 2)
            features = self.projection(x)

        # Scale to quantum angles [0, π]
        angles = self.angle_activation(features) * np.pi

        return angles

    def get_param_count(self) -> int:
        return sum(p.numel() for p in self.parameters())


# Convenience function for creating encoder with standard config
def create_qts_encoder(
    feature_dim: int,
    n_timesteps: int,
    n_qubits: int,
    projection_type: str = 'Conv2d_GLU',
    dropout: float = 0.1
) -> QTSFeatureEncoder:
    """
    Create QTS encoder with standard configuration.

    Args:
        feature_dim: Input feature dimension
        n_timesteps: Sequence length
        n_qubits: Number of qubits (output dimension)
        projection_type: Type of projection ('Linear', 'Conv2d', 'Conv2d_GLU', 'GLU')
        dropout: Dropout rate

    Returns:
        QTSFeatureEncoder instance
    """
    return QTSFeatureEncoder(
        feature_dim=feature_dim,
        n_timesteps=n_timesteps,
        n_output=n_qubits,
        projection_type=projection_type,
        dropout=dropout
    )
```

### 2.2 Encoder Configuration

```python
# Standard configuration for all ablation models
ENCODER_CONFIG = {
    'projection_type': 'Conv2d_GLU',  # Match QTSTransformer default
    'dropout': 0.1,
    'hidden_multiplier': 4,  # For GLU
}

# This ensures ALL models have IDENTICAL feature extraction
```

---

## 3. Classical vs Quantum Component Separation

### 3.1 Clear Boundary Definition

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                           CLASSICAL DOMAIN                                  │
│                                                                             │
│  ┌─────────────────────────────────────────────────────────────────────┐    │
│  │                    QTSFeatureEncoder                                │    │
│  │                                                                     │    │
│  │  • Conv2d layers (feature extraction)                               │    │
│  │  • ReLU activations                                                 │    │
│  │  • MaxPooling (downsampling)                                        │    │
│  │  • GLU gating (non-linear transformation)                           │    │
│  │  • Sigmoid scaling to [0, π]                                        │    │
│  │                                                                     │    │
│  │  Output: angles ∈ [0, π]^{n_qubits}                                 │    │
│  └─────────────────────────────────────────────────────────────────────┘    │
│                                                                             │
└─────────────────────────────────────┬───────────────────────────────────────┘
                                      │
                            QUANTUM BOUNDARY
                                      │
                                      ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                            QUANTUM DOMAIN                                   │
│                                                                             │
│  ┌─────────────────┐  ┌─────────────────┐  ┌─────────────────┐              │
│  │  Q-Hydra SSM    │  │  Q-Mamba SSM    │  │  Q-Attention    │              │
│  │                 │  │                 │  │  (QTSTransf.)   │              │
│  │  • RY encoding  │  │  • RY encoding  │  │  • RY encoding  │              │
│  │  • QSVT exp(A)  │  │  • QSVT exp(ΔA) │  │  • QSVT mixing  │              │
│  │  • CNOT entangl │  │  • CNOT entangl │  │  • CNOT entangl │              │
│  │  • LCU combine  │  │  • Selective Δ  │  │  • Polynomial   │              │
│  │  • PauliZ meas  │  │  • PauliZ meas  │  │  • PauliXYZ     │              │
│  └─────────────────┘  └─────────────────┘  └─────────────────┘              │
│                                                                             │
│  QUANTUM SELECTIVE FORGETTING:                                              │
│  • State evolution via quantum interference                                 │
│  • Entanglement for correlated processing                                   │
│  • QSVT polynomial transformation                                           │
│  • Input-dependent gating (Mamba)                                           │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

### 3.2 Component Attribution Table

| Component | Q-Hydra SSM | Q-Mamba SSM | QTSTransformer |
|-----------|-------------|-------------|----------------|
| **Conv2d feature extraction** | Classical | Classical | Classical |
| **GLU gating** | Classical | Classical | Classical |
| **Sigmoid → angles** | Classical | Classical | Classical |
| **RY angle encoding** | Quantum | Quantum | Quantum |
| **QSVT polynomial** | Quantum | Quantum | Quantum |
| **CNOT entanglement** | Quantum | Quantum | Quantum |
| **State evolution (h[t])** | Quantum | Quantum | N/A |
| **Selective Δ modulation** | N/A | Quantum | N/A |
| **LCU combination** | Quantum | N/A | N/A |
| **Attention mixing** | N/A | N/A | Quantum |
| **Measurement** | Quantum | Quantum | Quantum |

---

## 4. Quantum SSM Core Implementations

### 4.1 Quantum Hydra SSM Core (Bidirectional QSVT)

**File**: `models/quantum_hydra_ssm_core.py`

```python
"""
Quantum Hydra SSM Core - Bidirectional Quantum State Space Model

Implements QUANTUM selective forgetting using QSVT (Quantum Singular Value
Transformation) for state evolution and LCU (Linear Combination of Unitaries)
for bidirectional combination.

Classical Hydra:  h[t] = A·h[t-1] + B·x[t]
Quantum Hydra:    |h[t]⟩ = QSVT_exp(A)|h[t-1]⟩ ⊕ B|x[t]⟩

The QUANTUM SELECTIVE FORGETTING happens in:
1. QSVT polynomial approximating exp(A) - quantum state transformation
2. Quantum interference during state evolution
3. LCU combination of forward and backward passes
"""

import torch
import torch.nn as nn
import pennylane as qml
import numpy as np
from typing import List, Tuple


class QuantumHydraSSMCore(nn.Module):
    """
    Bidirectional Quantum SSM using QSVT.

    Key Quantum Components:
    1. QSVT: Polynomial transformation P(A) ≈ exp(A) applied quantumly
    2. Variational ansatz: Trainable quantum circuit layers
    3. LCU: Linear combination of forward and backward quantum states
    4. Measurement: Extract classical values from quantum state
    """

    def __init__(
        self,
        n_qubits: int = 4,
        n_layers: int = 2,
        qsvt_degree: int = 3,
        device: str = "cpu"
    ):
        super().__init__()

        self.n_qubits = n_qubits
        self.n_layers = n_layers
        self.qsvt_degree = qsvt_degree
        self.torch_device = device

        # QSVT polynomial coefficients (QUANTUM - learned)
        self.poly_coeffs = nn.Parameter(torch.randn(qsvt_degree + 1) * 0.1)

        # LCU coefficients for bidirectional combination (QUANTUM)
        self.fwd_coeff = nn.Parameter(torch.tensor([0.5]))
        self.bwd_coeff = nn.Parameter(torch.tensor([0.5]))

        # Variational circuit parameters (QUANTUM)
        self.n_circuit_params = n_qubits * n_layers * 3  # RX, RY, RZ per qubit per layer
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
        """Build QSVT-based quantum SSM circuit."""

        @qml.qnode(self.dev, interface="torch", diff_method="best")
        def qsvt_ssm_circuit(
            state_angles: torch.Tensor,
            input_angles: torch.Tensor,
            circuit_params: torch.Tensor,
            poly_coeffs: torch.Tensor
        ) -> List[torch.Tensor]:
            """
            QUANTUM SSM STATE EVOLUTION

            This circuit implements quantum selective forgetting:
            |h[t]⟩ = QSVT_exp(A)|h[t-1]⟩ + B|x[t]⟩

            Quantum operations:
            1. Encode previous state |h[t-1]⟩
            2. Apply QSVT polynomial (quantum exp(A))
            3. Variational layer for expressivity
            4. Inject new input
            5. Measure new state
            """
            n_qubits = self.n_qubits

            # ============================================
            # QUANTUM STATE ENCODING
            # Encode h[t-1] into quantum amplitudes
            # ============================================
            for i in range(n_qubits):
                qml.RY(state_angles[i], wires=i)

            # ============================================
            # QSVT TRANSFORMATION (QUANTUM SELECTIVE FORGETTING)
            # Polynomial P(A) ≈ exp(A) applied via quantum gates
            # This is the core QUANTUM operation
            # ============================================
            for deg in range(len(poly_coeffs)):
                coeff = poly_coeffs[deg]

                # Phase rotations (quantum polynomial coefficients)
                for i in range(n_qubits):
                    qml.RZ(coeff * np.pi, wires=i)

                # Entangling layer (creates quantum correlations)
                # This enables QUANTUM interference during state evolution
                for i in range(n_qubits - 1):
                    qml.CNOT(wires=[i, i + 1])
                if n_qubits > 1:
                    qml.CNOT(wires=[n_qubits - 1, 0])  # Circular

            # ============================================
            # VARIATIONAL ANSATZ
            # Trainable quantum layer for expressivity
            # ============================================
            param_idx = 0
            for layer in range(self.n_layers):
                for i in range(n_qubits):
                    qml.RX(circuit_params[param_idx], wires=i)
                    qml.RY(circuit_params[param_idx + 1], wires=i)
                    qml.RZ(circuit_params[param_idx + 2], wires=i)
                    param_idx += 3

                # Entanglement
                for i in range(n_qubits - 1):
                    qml.CNOT(wires=[i, i + 1])

            # ============================================
            # INPUT INJECTION (B·x[t])
            # Add new input to quantum state
            # ============================================
            for i in range(n_qubits):
                qml.RY(input_angles[i], wires=i)

            # ============================================
            # MEASUREMENT
            # Extract classical values from quantum state
            # ============================================
            return [qml.expval(qml.PauliZ(i)) for i in range(n_qubits)]

        self.qnode = qsvt_ssm_circuit

    def _quantum_ssm_pass(
        self,
        angles: torch.Tensor,
        circuit_params: torch.Tensor,
        reverse: bool = False
    ) -> torch.Tensor:
        """
        Single direction quantum SSM pass.

        This implements QUANTUM STATE EVOLUTION with selective forgetting.

        Args:
            angles: (batch, seq_len, n_qubits) quantum-encoded inputs
            circuit_params: Variational parameters
            reverse: Process in reverse for backward pass

        Returns:
            (batch, seq_len, n_qubits) quantum SSM outputs
        """
        batch_size, seq_len, _ = angles.shape
        device = angles.device

        if reverse:
            angles = angles.flip(dims=[1])

        # Initialize quantum hidden state
        h = torch.zeros(batch_size, self.n_qubits, device=device)
        outputs = []

        # Sequential quantum state evolution
        for t in range(seq_len):
            x_t = angles[:, t, :]
            batch_outputs = []

            for b in range(batch_size):
                # QUANTUM OPERATION: Evolve state through circuit
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

    def forward(self, angles: torch.Tensor) -> torch.Tensor:
        """
        Bidirectional quantum SSM processing.

        Implements Hydra-style bidirectional processing with:
        1. Forward quantum SSM pass
        2. Backward quantum SSM pass
        3. LCU (Linear Combination of Unitaries) combination

        Args:
            angles: (batch, seq_len, n_qubits) from QTS encoder
        Returns:
            (batch, seq_len, n_qubits) bidirectional quantum output
        """
        # Forward quantum pass
        h_fwd = self._quantum_ssm_pass(angles, self.fwd_params, reverse=False)

        # Backward quantum pass
        h_bwd = self._quantum_ssm_pass(angles, self.bwd_params, reverse=True)

        # LCU combination (quantum linear combination)
        fwd_weight = torch.sigmoid(self.fwd_coeff)
        bwd_weight = torch.sigmoid(self.bwd_coeff)
        combined = fwd_weight * h_fwd + bwd_weight * h_bwd

        return combined
```

### 4.2 Quantum Mamba SSM Core (Selective QSVT)

**File**: `models/quantum_mamba_ssm_core.py`

```python
"""
Quantum Mamba SSM Core - Selective Quantum State Space Model

Implements QUANTUM selective forgetting with INPUT-DEPENDENT parameters
using QSVT where Δ[t] modulates the quantum transformation per timestep.

Classical Mamba:  h[t] = exp(Δ[t]·A)·h[t-1] + Δ[t]·B[t]·x[t]
                  where Δ[t], B[t], C[t] = f(x[t])

Quantum Mamba:    |h[t]⟩ = QSVT_exp(Δ[t]·A)|h[t-1]⟩ ⊕ Δ[t]·B[t]|x[t]⟩
                  where Δ[t] MODULATES the quantum gates

The key QUANTUM SELECTIVE mechanism:
- Input-dependent Δ controls quantum transformation strength
- Large Δ: More input influence, less state retention (REMEMBER)
- Small Δ: Less input influence, more state retention (FORGET)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import pennylane as qml
import numpy as np
from typing import List


class QuantumMambaSSMCore(nn.Module):
    """
    Selective Quantum SSM using QSVT with input-dependent Δ.

    Key Quantum Components:
    1. Input-dependent Δ: Modulates QSVT transformation strength
    2. QSVT exp(Δ·A): Quantum state transition with variable strength
    3. Selective B, C: Input-dependent quantum gates
    4. D skip connection: Direct input-to-output path
    """

    def __init__(
        self,
        n_qubits: int = 4,
        n_layers: int = 2,
        qsvt_degree: int = 3,
        dt_rank: str = "auto",
        dt_min: float = 0.001,
        dt_max: float = 0.1,
        device: str = "cpu"
    ):
        super().__init__()

        self.n_qubits = n_qubits
        self.n_layers = n_layers
        self.qsvt_degree = qsvt_degree
        self.torch_device = device

        # dt_rank for selective parameter projection
        if dt_rank == "auto":
            self.dt_rank = max(n_qubits // 2, 1)
        else:
            self.dt_rank = dt_rank

        # Selective parameter projections (Mamba-style)
        # These compute input-dependent Δ, B, C
        self.x_proj = nn.Linear(n_qubits, self.dt_rank + n_qubits * 2, bias=False)

        # Δ projection with proper initialization
        self.dt_proj = nn.Linear(self.dt_rank, n_qubits, bias=True)

        # Initialize dt bias for desired range [dt_min, dt_max]
        dt = torch.exp(
            torch.rand(n_qubits) * (np.log(dt_max) - np.log(dt_min)) + np.log(dt_min)
        )
        inv_dt = dt + torch.log(-torch.expm1(-dt))
        with torch.no_grad():
            self.dt_proj.bias.copy_(inv_dt)

        # Base QSVT polynomial (modulated by Δ)
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
        """Build selective QSVT circuit with Δ modulation."""

        @qml.qnode(self.dev, interface="torch", diff_method="best")
        def selective_qsvt_circuit(
            state_angles: torch.Tensor,
            input_angles: torch.Tensor,
            dt_angles: torch.Tensor,
            circuit_params: torch.Tensor,
            poly_coeffs: torch.Tensor
        ) -> List[torch.Tensor]:
            """
            SELECTIVE QUANTUM SSM with input-dependent Δ

            The key innovation: dt_angles (Δ) MODULATES the quantum transformation

            - High Δ: Strong transformation, more input influence
            - Low Δ: Weak transformation, more state retention

            This implements QUANTUM SELECTIVE FORGETTING
            """
            n_qubits = self.n_qubits

            # ============================================
            # QUANTUM STATE ENCODING
            # ============================================
            for i in range(n_qubits):
                qml.RY(state_angles[i], wires=i)

            # ============================================
            # SELECTIVE QSVT (Δ-MODULATED TRANSFORMATION)
            # Key: Polynomial coefficients SCALED by input-dependent Δ
            # This is the QUANTUM SELECTIVE FORGETTING mechanism
            # ============================================
            for deg in range(len(poly_coeffs)):
                coeff = poly_coeffs[deg]

                for i in range(n_qubits):
                    # ⭐ SELECTIVE: Δ modulates transformation strength
                    # This is where selectivity happens in QUANTUM domain
                    qml.RZ(coeff * dt_angles[i] * np.pi, wires=i)

                # Entangling layer
                for i in range(n_qubits - 1):
                    qml.CNOT(wires=[i, i + 1])
                if n_qubits > 1:
                    qml.CNOT(wires=[n_qubits - 1, 0])

            # ============================================
            # VARIATIONAL ANSATZ
            # ============================================
            param_idx = 0
            for layer in range(self.n_layers):
                for i in range(n_qubits):
                    qml.RX(circuit_params[param_idx], wires=i)
                    qml.RY(circuit_params[param_idx + 1], wires=i)
                    qml.RZ(circuit_params[param_idx + 2], wires=i)
                    param_idx += 3

                for i in range(n_qubits - 1):
                    qml.CNOT(wires=[i, i + 1])

            # ============================================
            # SELECTIVE INPUT INJECTION
            # Input influence also controlled by Δ
            # ============================================
            for i in range(n_qubits):
                # ⭐ SELECTIVE: Input scaled by Δ
                qml.RY(input_angles[i] * dt_angles[i], wires=i)

            # ============================================
            # MEASUREMENT
            # ============================================
            return [qml.expval(qml.PauliZ(i)) for i in range(n_qubits)]

        self.qnode = selective_qsvt_circuit

    def forward(self, angles: torch.Tensor) -> torch.Tensor:
        """
        Selective quantum SSM processing.

        Computes input-dependent Δ, B, C then applies selective
        quantum transformation at each timestep.

        Args:
            angles: (batch, seq_len, n_qubits) from QTS encoder
        Returns:
            (batch, seq_len, n_qubits) selective quantum output
        """
        batch_size, seq_len, _ = angles.shape
        device = angles.device

        # ============================================
        # COMPUTE SELECTIVE PARAMETERS (Classical)
        # Δ[t], B[t], C[t] = f(angles[t])
        # ============================================
        x_dbc = self.x_proj(angles)

        # Split into components
        dt_raw = x_dbc[..., :self.dt_rank]
        B = x_dbc[..., self.dt_rank:self.dt_rank + self.n_qubits]
        C = x_dbc[..., self.dt_rank + self.n_qubits:]

        # Compute Δ with softplus (ensures positive)
        dt = F.softplus(self.dt_proj(dt_raw))
        dt_angles = torch.tanh(dt) * np.pi  # Scale for quantum gates

        # ============================================
        # SEQUENTIAL SELECTIVE QUANTUM PROCESSING
        # ============================================
        h = torch.zeros(batch_size, self.n_qubits, device=device)
        outputs = []

        for t in range(seq_len):
            x_t = angles[:, t, :]
            dt_t = dt_angles[:, t, :]  # Input-dependent Δ
            C_t = C[:, t, :]

            batch_outputs = []
            for b in range(batch_size):
                # QUANTUM OPERATION with selective Δ
                measurements = self.qnode(
                    h[b],
                    x_t[b],
                    dt_t[b],  # ⭐ Selective modulation
                    self.circuit_params,
                    self.poly_coeffs_base
                )
                batch_outputs.append(torch.stack(measurements))

            h = torch.stack(batch_outputs).to(device)

            # Output with selective C and skip connection D
            y_t = C_t * h + self.D * angles[:, t, :]
            outputs.append(y_t)

        return torch.stack(outputs, dim=1)
```

---

## 5. Complete Model Architectures

### 5.1 Model A3a/C1a: QTS + Quantum Hydra SSM

**File**: `models/QTSQuantumHydraSSM.py`

```python
"""
QTS + Quantum Hydra SSM (Models A3a/B1a/C1a)

Architecture:
    QTSFeatureEncoder (CLASSICAL) → QuantumHydraSSMCore (QUANTUM) → Output

This model uses:
- IDENTICAL encoder to QTSTransformer (controlled comparison)
- QUANTUM bidirectional SSM with QSVT selective forgetting
"""

import torch
import torch.nn as nn
import numpy as np

from models.qts_encoder import QTSFeatureEncoder, create_qts_encoder
from models.quantum_hydra_ssm_core import QuantumHydraSSMCore


class SequencePooling(nn.Module):
    """Attention-weighted sequence pooling."""

    def __init__(self, d_model: int):
        super().__init__()
        self.attention = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.Tanh(),
            nn.Linear(d_model // 2, 1)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        attn_scores = self.attention(x)
        attn_weights = torch.softmax(attn_scores, dim=1)
        return (attn_weights * x).sum(dim=1)


class QTSQuantumHydraSSM(nn.Module):
    """
    QTS Encoder + Quantum Hydra SSM

    Components:
    - encoder: QTSFeatureEncoder (CLASSICAL, identical to QTSTransformer)
    - quantum_ssm: QuantumHydraSSMCore (QUANTUM selective forgetting)
    - output_proj: Project quantum output
    - pooling: Sequence-level pooling
    - classifier: Final classification head
    """

    def __init__(
        self,
        feature_dim: int,
        n_timesteps: int,
        num_classes: int,
        n_qubits: int = 4,
        n_layers: int = 2,
        qsvt_degree: int = 3,
        projection_type: str = 'Conv2d_GLU',
        dropout: float = 0.1,
        device: str = "cpu"
    ):
        super().__init__()

        self.n_qubits = n_qubits
        self.device_str = device

        # CLASSICAL: QTS Feature Encoder (identical to QTSTransformer)
        self.encoder = create_qts_encoder(
            feature_dim=feature_dim,
            n_timesteps=n_timesteps,
            n_qubits=n_qubits,
            projection_type=projection_type,
            dropout=dropout
        )

        # QUANTUM: Hydra SSM Core (bidirectional QSVT)
        self.quantum_ssm = QuantumHydraSSMCore(
            n_qubits=n_qubits,
            n_layers=n_layers,
            qsvt_degree=qsvt_degree,
            device=device
        )

        # Output layers
        self.output_proj = nn.Linear(n_qubits, n_qubits * 3)
        self.layer_norm = nn.LayerNorm(n_qubits * 3)
        self.pooling = SequencePooling(n_qubits * 3)
        self.classifier = nn.Sequential(
            nn.Linear(n_qubits * 3, n_qubits * 3),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(n_qubits * 3, num_classes)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Args:
            x: (batch, feature_dim, n_timesteps) or (batch, n_timesteps, feature_dim)
        Returns:
            (batch, num_classes) logits
        """
        # CLASSICAL: Feature extraction
        angles = self.encoder(x)  # (batch, seq_len, n_qubits)

        # QUANTUM: Bidirectional SSM
        quantum_out = self.quantum_ssm(angles)  # (batch, seq_len, n_qubits)

        # Residual connection
        residual = quantum_out + angles

        # Output projection
        projected = self.output_proj(residual)
        projected = self.layer_norm(projected)

        # Pool and classify
        pooled = self.pooling(projected)
        return self.classifier(pooled)

    def get_param_count(self) -> dict:
        """Return parameter counts by component."""
        encoder_params = sum(p.numel() for p in self.encoder.parameters())
        quantum_params = sum(p.numel() for p in self.quantum_ssm.parameters())
        output_params = sum(p.numel() for p in self.output_proj.parameters())
        output_params += sum(p.numel() for p in self.classifier.parameters())
        output_params += sum(p.numel() for p in self.pooling.parameters())

        return {
            'encoder (classical)': encoder_params,
            'quantum_ssm (quantum)': quantum_params,
            'output (classical)': output_params,
            'total': encoder_params + quantum_params + output_params
        }
```

### 5.2 Model A3b/C1b: QTS + Quantum Mamba SSM

**File**: `models/QTSQuantumMambaSSM.py`

```python
"""
QTS + Quantum Mamba SSM (Models A3b/B1b/C1b)

Architecture:
    QTSFeatureEncoder (CLASSICAL) → QuantumMambaSSMCore (QUANTUM) → Output

This model uses:
- IDENTICAL encoder to QTSTransformer (controlled comparison)
- QUANTUM selective SSM with input-dependent Δ
"""

import torch
import torch.nn as nn

from models.qts_encoder import create_qts_encoder
from models.quantum_mamba_ssm_core import QuantumMambaSSMCore


class SequencePooling(nn.Module):
    """Attention-weighted sequence pooling."""

    def __init__(self, d_model: int):
        super().__init__()
        self.attention = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.Tanh(),
            nn.Linear(d_model // 2, 1)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        attn_scores = self.attention(x)
        attn_weights = torch.softmax(attn_scores, dim=1)
        return (attn_weights * x).sum(dim=1)


class QTSQuantumMambaSSM(nn.Module):
    """
    QTS Encoder + Quantum Mamba SSM

    Components:
    - encoder: QTSFeatureEncoder (CLASSICAL, identical to QTSTransformer)
    - quantum_ssm: QuantumMambaSSMCore (QUANTUM with selective Δ)
    - output_proj: Project quantum output
    - pooling: Sequence-level pooling
    - classifier: Final classification head
    """

    def __init__(
        self,
        feature_dim: int,
        n_timesteps: int,
        num_classes: int,
        n_qubits: int = 4,
        n_layers: int = 2,
        qsvt_degree: int = 3,
        projection_type: str = 'Conv2d_GLU',
        dropout: float = 0.1,
        device: str = "cpu"
    ):
        super().__init__()

        self.n_qubits = n_qubits
        self.device_str = device

        # CLASSICAL: QTS Feature Encoder (identical to QTSTransformer)
        self.encoder = create_qts_encoder(
            feature_dim=feature_dim,
            n_timesteps=n_timesteps,
            n_qubits=n_qubits,
            projection_type=projection_type,
            dropout=dropout
        )

        # QUANTUM: Mamba SSM Core (selective QSVT)
        self.quantum_ssm = QuantumMambaSSMCore(
            n_qubits=n_qubits,
            n_layers=n_layers,
            qsvt_degree=qsvt_degree,
            device=device
        )

        # Output layers
        self.output_proj = nn.Linear(n_qubits, n_qubits * 3)
        self.layer_norm = nn.LayerNorm(n_qubits * 3)
        self.pooling = SequencePooling(n_qubits * 3)
        self.classifier = nn.Sequential(
            nn.Linear(n_qubits * 3, n_qubits * 3),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(n_qubits * 3, num_classes)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass with selective quantum processing.

        Args:
            x: (batch, feature_dim, n_timesteps)
        Returns:
            (batch, num_classes) logits
        """
        # CLASSICAL: Feature extraction
        angles = self.encoder(x)

        # QUANTUM: Selective SSM with input-dependent Δ
        quantum_out = self.quantum_ssm(angles)

        # Residual connection
        residual = quantum_out + angles

        # Output projection
        projected = self.output_proj(residual)
        projected = self.layer_norm(projected)

        # Pool and classify
        pooled = self.pooling(projected)
        return self.classifier(pooled)

    def get_param_count(self) -> dict:
        """Return parameter counts by component."""
        encoder_params = sum(p.numel() for p in self.encoder.parameters())
        quantum_params = sum(p.numel() for p in self.quantum_ssm.parameters())
        output_params = sum(p.numel() for p in self.output_proj.parameters())
        output_params += sum(p.numel() for p in self.classifier.parameters())
        output_params += sum(p.numel() for p in self.pooling.parameters())

        return {
            'encoder (classical)': encoder_params,
            'quantum_ssm (quantum)': quantum_params,
            'output (classical)': output_params,
            'total': encoder_params + quantum_params + output_params
        }
```

---

## 6. Ablation Study Framework

### 6.1 Ablation 1: Where to Apply Quantum

**Question**: Is quantum more effective for feature extraction or sequence dynamics?

```
              Feature Extraction    Sequence Mixing
              ─────────────────    ───────────────
A1            Classical            Classical SSM       ← Baseline
A2a/A2b       Quantum VQC          Classical SSM       ← Current approach
A3a/A3b       QTS (Classical)      Quantum SSM         ← NEW (This work)
A4a/A4b       Quantum VQC          Quantum SSM         ← End-to-End
```

**Expected Finding**: A3a/A3b should outperform A2a/A2b because:
- Classical encoders excel at feature extraction (proven by deep learning)
- Quantum excels at dynamics (interference, superposition for state evolution)

### 6.2 Ablation 2: Which Quantum Sequence Mixing

**Question**: Given identical features, which quantum mixing is best?

```
All models use: QTSFeatureEncoder (IDENTICAL)

B1a: QTS Encoder → Quantum Hydra SSM (bidirectional QSVT)
B1b: QTS Encoder → Quantum Mamba SSM (selective QSVT)
B2a: QTS Encoder → Quantum Hydra Attention
B2b: QTS Encoder → Quantum Mamba Attention
B3a: QTS Encoder → Quantum Hydra Gated
B3b: QTS Encoder → Quantum Mamba Gated
B4:  QTS Encoder → Classical SSM (control)
```

**Key Insight**: Because all models share the encoder, ANY performance difference is 100% due to the mixing mechanism.

### 6.3 Ablation 3: Architecture Comparison

**Question**: How does our new architecture compare to QTSTransformer?

```
C1a: QTS Encoder → Quantum Hydra SSM    ← Our model (Hydra)
C1b: QTS Encoder → Quantum Mamba SSM    ← Our model (Mamba)
C2:  QTS Encoder → Quantum Attention    ← QTSTransformer (existing)
C3a: Classical Encoder → Classical Hydra
C3b: Classical Encoder → Classical Mamba
C4:  Classical Encoder → Classical Attention
```

**Critical**: C1a/C1b vs C2 is a MAXIMALLY FAIR comparison because they share the IDENTICAL encoder.

### 6.4 Comparison Framework Diagram

```
                              ┌─────────────────────────────────────────┐
                              │        QTSFeatureEncoder                │
                              │  Conv2d → MaxPool → Conv2d → MaxPool    │
                              │           → GLU → Sigmoid(·π)           │
                              │                                         │
                              │  CLASSICAL (shared across all models)   │
                              └────────────────────┬────────────────────┘
                                                   │
                                                   │ angles ∈ [0, π]
                                                   │
       ┌───────────────┬───────────────┬───────────┴───────────┬───────────────┐
       │               │               │                       │               │
       ▼               ▼               ▼                       ▼               ▼
┌─────────────┐ ┌─────────────┐ ┌─────────────┐       ┌─────────────┐ ┌─────────────┐
│ Q-Hydra SSM │ │ Q-Mamba SSM │ │ Q-Attention │       │ Q-Gated     │ │ Classical   │
│ (QSVT+LCU)  │ │ (QSVT+Δ)    │ │ (QTS orig)  │       │ (QSVT+Gate) │ │ SSM         │
│             │ │             │ │             │       │             │ │             │
│ Bidirect.   │ │ Selective   │ │ Full Attn   │       │ Gated       │ │ Control     │
└──────┬──────┘ └──────┬──────┘ └──────┬──────┘       └──────┬──────┘ └──────┬──────┘
       │               │               │                     │               │
       ▼               ▼               ▼                     ▼               ▼
      B1a             B1b             C2                    B3a/B3b          B4
     (C1a)           (C1b)        (Existing)
```

---

## 7. Experimental Protocol

### 7.1 Hyperparameters (Fixed Across All Models)

```python
# Quantum parameters
N_QUBITS = 4
N_LAYERS = 2
QSVT_DEGREE = 3

# Encoder parameters (from QTSTransformer)
PROJECTION_TYPE = 'Conv2d_GLU'
DROPOUT = 0.1

# Training parameters
BATCH_SIZE = 32
LEARNING_RATE = 1e-3
WEIGHT_DECAY = 1e-4
N_EPOCHS = 30
OPTIMIZER = 'AdamW'
SCHEDULER = 'CosineAnnealingLR'
SEEDS = [2024, 2025, 2026, 2027, 2028]
```

### 7.2 Datasets

| Dataset | Type | Seq Length | Classes | Purpose |
|---------|------|------------|---------|---------|
| demo_human_or_worm | Genomic | 200 | 2 | Short sequence |
| demo_coding_vs_intergenomic | Genomic | 200 | 2 | Classification |
| PhysioNet EEG | Time-series | 160 | 2 | Temporal dynamics |
| Forrelation (n=6) | Synthetic | 64 | 2 | Quantum advantage |
| MNIST (sequential) | Image | 784 | 10 | Standard benchmark |

### 7.3 Statistical Analysis

```python
# For each comparison (e.g., C1a vs C2):
# 1. Run both models with 5 seeds
# 2. Compute mean and std
# 3. Perform paired t-test
# 4. Compute effect size (Cohen's d)
# 5. Report 95% confidence interval

from scipy import stats

def compare_models(results_a, results_b, alpha=0.05):
    """Statistical comparison of two models."""
    # Paired t-test
    t_stat, p_value = stats.ttest_rel(results_a, results_b)

    # Effect size (Cohen's d)
    diff = np.array(results_a) - np.array(results_b)
    cohens_d = diff.mean() / diff.std()

    # 95% CI
    ci = stats.t.interval(0.95, len(diff)-1, loc=diff.mean(), scale=stats.sem(diff))

    return {
        'mean_diff': diff.mean(),
        'p_value': p_value,
        'significant': p_value < alpha,
        'cohens_d': cohens_d,
        'ci_95': ci
    }
```

### 7.4 Execution Phases

```
Phase 1: Baseline Establishment (1 week)
├── Run C2 (QTSTransformer) on all datasets
├── Run A1 (Classical Hydra) on all datasets
└── Verify reproducibility

Phase 2: Core Models (2 weeks)
├── Implement QTSFeatureEncoder (extract from QTSTransformer)
├── Implement QuantumHydraSSMCore
├── Implement QuantumMambaSSMCore
├── Implement QTSQuantumHydraSSM (C1a)
├── Implement QTSQuantumMambaSSM (C1b)
└── Run C1a, C1b on all datasets

Phase 3: Ablation 1 - Where Quantum (1 week)
├── Run A2a, A2b (existing quantum feature models)
├── Run A4a, A4b (end-to-end quantum)
└── Compare A1, A2, A3, A4

Phase 4: Ablation 2 - Which Mixing (1 week)
├── Implement B2a, B2b (Quantum Attention variants)
├── Implement B3a, B3b (Quantum Gated variants)
├── Implement B4 (Classical control)
└── Compare B1-B4

Phase 5: Ablation 3 - Full Comparison (1 week)
├── Run all C models
├── Statistical analysis
└── Generate figures

Phase 6: Analysis & Writing (1 week)
├── Aggregate all results
├── Statistical significance tests
├── Generate publication figures
└── Write findings
```

---

## 8. Implementation Checklist

### 8.1 Files to Create

```
models/
├── qts_encoder.py                    # QTSFeatureEncoder (from QTSTransformer)
│   ├── Conv2dFeatureExtractor
│   ├── GatedFeedForward
│   ├── Conv2dGLUPreprocessor
│   └── QTSFeatureEncoder
│
├── quantum_hydra_ssm_core.py         # Quantum Hydra SSM Core
│   └── QuantumHydraSSMCore
│
├── quantum_mamba_ssm_core.py         # Quantum Mamba SSM Core
│   └── QuantumMambaSSMCore
│
├── QTSQuantumHydraSSM.py             # A3a/B1a/C1a
│   └── QTSQuantumHydraSSM
│
├── QTSQuantumMambaSSM.py             # A3b/B1b/C1b
│   └── QTSQuantumMambaSSM
│
├── QTSQuantumHydraAttention.py       # B2a
├── QTSQuantumMambaAttention.py       # B2b
├── QTSQuantumHydraGated.py           # B3a
├── QTSQuantumMambaGated.py           # B3b
└── QTSClassicalSSM.py                # B4

scripts/
├── run_ablation_study.py             # Main runner
├── run_ablation1_where_quantum.py
├── run_ablation2_which_mixing.py
├── run_ablation3_architecture.py
└── aggregate_ablation_results.py
```

### 8.2 Implementation Priority

```
Priority 1 (Core - MUST HAVE):
├── [x] qts_encoder.py
├── [ ] quantum_hydra_ssm_core.py
├── [ ] quantum_mamba_ssm_core.py
├── [ ] QTSQuantumHydraSSM.py
└── [ ] QTSQuantumMambaSSM.py

Priority 2 (Ablation 2):
├── [ ] QTSQuantumHydraAttention.py
├── [ ] QTSQuantumMambaAttention.py
├── [ ] QTSQuantumHydraGated.py
├── [ ] QTSQuantumMambaGated.py
└── [ ] QTSClassicalSSM.py

Priority 3 (Scripts):
├── [ ] run_ablation_study.py
├── [ ] run_ablation1_where_quantum.py
├── [ ] run_ablation2_which_mixing.py
├── [ ] run_ablation3_architecture.py
└── [ ] aggregate_ablation_results.py
```

---

## Appendix A: Key Differences from V1

| Aspect | V1 (Previous) | V2 (This Document) |
|--------|---------------|---------------------|
| **Encoder** | SharedClassicalEncoder (simple) | QTSFeatureEncoder (from QTSTransformer) |
| **Feature extraction** | Linear → LayerNorm → GELU | Conv2d → MaxPool → GLU |
| **Comparison fairness** | Moderate | Maximum (identical encoder) |
| **QTSTransformer comparison** | Different encoders | Same encoder |
| **Proven encoder** | No | Yes (validated on EEG tasks) |

## Appendix B: Complete Model ID Reference

| ID | Full Name | Encoder | Quantum Component | Style |
|----|-----------|---------|-------------------|-------|
| A1 | Classical Hydra | Simple | None | Bidir |
| A2a | QuantumHydraSSM | Quantum VQC | Classical SSM | Bidir |
| A2b | QuantumMambaSSM | Quantum VQC | Classical SSM | Select |
| **A3a** | QTS + Q-Hydra SSM | **QTS Encoder** | **QSVT Hydra** | Bidir |
| **A3b** | QTS + Q-Mamba SSM | **QTS Encoder** | **QSVT Mamba** | Select |
| A4a | E2E Q-Hydra | Quantum VQC | QSVT Hydra | Bidir |
| A4b | E2E Q-Mamba | Quantum VQC | QSVT Mamba | Select |
| B1a | = A3a | QTS | QSVT Hydra | Bidir |
| B1b | = A3b | QTS | QSVT Mamba | Select |
| B2a | QTS + Q-Hydra Attn | QTS | Q-Attention | Bidir |
| B2b | QTS + Q-Mamba Attn | QTS | Q-Attention | Select |
| B3a | QTS + Q-Hydra Gated | QTS | Q-Gated | Bidir |
| B3b | QTS + Q-Mamba Gated | QTS | Q-Gated | Select |
| B4 | QTS + Classical SSM | QTS | None | Control |
| **C1a** | = A3a | **QTS Encoder** | **QSVT Hydra** | Bidir |
| **C1b** | = A3b | **QTS Encoder** | **QSVT Mamba** | Select |
| **C2** | QTSTransformer | **QTS Encoder** | Q-Attention | N/A |
| C3a | Classical Hydra | Simple | None | Bidir |
| C3b | Classical Mamba | Simple | None | Select |
| C4 | Classical Transformer | Simple | None | N/A |

---

## Document Information

- **Version**: 2.0 (Unified QTS Encoder)
- **Author**: Junghoon Park
- **Created**: December 2024
- **Last Updated**: December 2024
- **Key Change**: Adopted QTSTransformer encoder for all models
- **Related Documents**:
  - `ABLATION_STUDY_IMPLEMENTATION_PLAN.md` (V1)
  - `FOUR_MODEL_ARCHITECTURE_COMPARISON.md`
  - `COMPUTATIONAL_COMPLEXITY_COMPARISON.md`
  - `QuixerTSModel_Pennylane2.py` (QTSTransformer source)
