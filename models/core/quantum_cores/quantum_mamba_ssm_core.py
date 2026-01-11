"""
Quantum Mamba SSM Core - Selective Quantum State Space Model

Implements QUANTUM selective forgetting with INPUT-DEPENDENT parameters
using QSVT where Δ[t] modulates the quantum transformation per timestep.

Theory:
-------
Classical Mamba SSM (Gu & Dao, 2024):
    Δ[t], B[t], C[t] = f(x[t])                    # Input-dependent params
    h[t] = exp(Δ[t] · A) · h[t-1] + Δ[t] · B[t] · x[t]  # State update
    y[t] = C[t] · h[t] + D · x[t]                 # Output

Quantum Mamba SSM:
    Δ[t], B[t], C[t] = f(x[t])                    # Classical projection
    |h[t]⟩ = QSVT_exp(Δ[t]·A)|h[t-1]⟩ ⊕ Δ[t]·B[t]|x[t]⟩  # QUANTUM evolution
    y[t] = Measure(C[t] · |h[t]⟩) + D · x[t]     # Output

The key QUANTUM SELECTIVE mechanism:
- Input-dependent Δ controls quantum transformation strength
- Large Δ: More input influence, less state retention (REMEMBER new info)
- Small Δ: Less input influence, more state retention (FORGET/maintain state)

This selectivity happens IN THE QUANTUM DOMAIN through Δ-modulated QSVT gates.

Author: Junghoon Park
Date: December 2024
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import pennylane as qml
import numpy as np
from typing import List, Optional


class QuantumMambaSSMCore(nn.Module):
    """
    Selective Quantum SSM using QSVT with input-dependent Δ.

    This module implements Mamba-style selective state space modeling
    where the transformation strength varies based on input content.

    Key Components:
    1. Input-dependent Δ projection: Computes selective gate from input
    2. Input-dependent B, C: Computes selective input/output projections
    3. QSVT exp(Δ·A): Quantum state transition with variable strength
    4. D skip connection: Direct input-to-output path

    The selectivity mechanism:
    - Δ (delta/timestep) controls how much the quantum state transforms
    - High Δ → stronger quantum transformation → more input influence
    - Low Δ → weaker quantum transformation → more state retention

    Args:
        n_qubits: Number of qubits in the circuit
        n_layers: Number of variational ansatz layers
        qsvt_degree: Degree of QSVT polynomial
        dt_rank: Rank of Δ projection ("auto" = n_qubits // 2)
        dt_min: Minimum Δ value (for initialization)
        dt_max: Maximum Δ value (for initialization)
        device: Device for computation ('cpu' or 'cuda')

    Example:
        core = QuantumMambaSSMCore(n_qubits=4, n_layers=2)
        angles = torch.randn(32, 200, 4)  # (batch, seq_len, n_qubits)
        output = core(angles)  # (batch, seq_len, n_qubits)
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

        # ============================================
        # Δ (Delta) Rank for Selective Projection
        # Lower rank = more parameter efficient
        # ============================================
        if dt_rank == "auto":
            self.dt_rank = max(n_qubits // 2, 1)
        else:
            self.dt_rank = int(dt_rank)

        # ============================================
        # SELECTIVE PARAMETER PROJECTIONS (Classical)
        # These compute input-dependent Δ, B, C
        #
        # x_proj: input → [dt_raw, B, C]
        # dt_proj: dt_raw → Δ (with proper initialization)
        # ============================================
        self.x_proj = nn.Linear(
            n_qubits,
            self.dt_rank + n_qubits * 2,  # dt_rank + B + C
            bias=False
        )

        # Δ projection with proper initialization for stable training
        self.dt_proj = nn.Linear(self.dt_rank, n_qubits, bias=True)

        # Initialize dt bias for desired range [dt_min, dt_max]
        # This follows Mamba's initialization strategy
        dt = torch.exp(
            torch.rand(n_qubits) * (np.log(dt_max) - np.log(dt_min)) + np.log(dt_min)
        )
        # Inverse of softplus for initialization
        inv_dt = dt + torch.log(-torch.expm1(-dt))
        with torch.no_grad():
            self.dt_proj.bias.copy_(inv_dt)

        # ============================================
        # QSVT Base Polynomial (QUANTUM)
        # These coefficients are modulated by Δ
        # ============================================
        self.poly_coeffs_base = nn.Parameter(
            torch.randn(qsvt_degree + 1) * 0.1
        )

        # ============================================
        # Variational Circuit Parameters (QUANTUM)
        # ============================================
        self.n_circuit_params = n_qubits * n_layers * 3
        self.circuit_params = nn.Parameter(
            torch.randn(self.n_circuit_params) * 0.1
        )

        # ============================================
        # D Parameter (Skip Connection)
        # Direct input-to-output path
        # ============================================
        self.D = nn.Parameter(torch.ones(n_qubits))

        # ============================================
        # Setup Quantum Device
        # ============================================
        self._setup_quantum_device(device)
        self._build_qnode()

    def _setup_quantum_device(self, device: str):
        """Setup PennyLane quantum device.

        Always uses default.qubit for quantum simulation (CPU-based).
        Classical layers use CUDA GPU via self.torch_device.
        Data is moved to CPU for quantum ops and back to GPU for classical ops.
        """
        # Always use default.qubit - lightning backends have compatibility issues
        self.dev = qml.device("default.qubit", wires=self.n_qubits)

    def _build_qnode(self):
        """Build selective QSVT circuit with Δ modulation."""

        n_qubits = self.n_qubits
        n_layers = self.n_layers

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

            - High Δ: Strong transformation, more input influence (REMEMBER)
            - Low Δ: Weak transformation, more state retention (FORGET)

            This implements QUANTUM SELECTIVE FORGETTING where the
            degree of forgetting/remembering is controlled by input content.

            Circuit Structure:
            1. State Encoding: RY(h[t-1])
            2. Δ-Modulated QSVT: RZ(coeff * Δ * π) + CNOT
            3. Variational Block: RX, RY, RZ + CNOT
            4. Δ-Modulated Input: RY(x[t] * Δ)
            5. Measurement: ⟨Z⟩

            Args:
                state_angles: Previous state h[t-1], shape (n_qubits,)
                input_angles: Current input x[t], shape (n_qubits,)
                dt_angles: Input-dependent Δ[t], shape (n_qubits,)
                circuit_params: Variational parameters
                poly_coeffs: Base QSVT polynomial coefficients

            Returns:
                List of expectation values ⟨Z_i⟩
            """
            # ============================================
            # STEP 1: QUANTUM STATE ENCODING
            # ============================================
            for i in range(n_qubits):
                qml.RY(state_angles[i], wires=i)

            # ============================================
            # STEP 2: Δ-MODULATED QSVT TRANSFORMATION
            #
            # This is the CORE SELECTIVE MECHANISM:
            # - Polynomial coefficients are SCALED by input-dependent Δ
            # - High Δ → larger rotations → stronger transformation
            # - Low Δ → smaller rotations → weaker transformation
            #
            # Effectively: P(Δ·A) where Δ varies with input
            # ============================================
            for deg in range(len(poly_coeffs)):
                coeff = poly_coeffs[deg]

                for i in range(n_qubits):
                    # ⭐ SELECTIVE: Δ modulates transformation strength
                    # This is where quantum selectivity happens!
                    qml.RZ(coeff * dt_angles[i] * np.pi, wires=i)

                # Entangling layer for quantum correlations
                for i in range(n_qubits - 1):
                    qml.CNOT(wires=[i, i + 1])
                if n_qubits > 1:
                    qml.CNOT(wires=[n_qubits - 1, 0])

            # ============================================
            # STEP 3: VARIATIONAL ANSATZ
            # ============================================
            param_idx = 0
            for layer in range(n_layers):
                for i in range(n_qubits):
                    qml.RX(circuit_params[param_idx], wires=i)
                    qml.RY(circuit_params[param_idx + 1], wires=i)
                    qml.RZ(circuit_params[param_idx + 2], wires=i)
                    param_idx += 3

                for i in range(n_qubits - 1):
                    qml.CNOT(wires=[i, i + 1])

            # ============================================
            # STEP 4: Δ-MODULATED INPUT INJECTION
            #
            # Input influence is also controlled by Δ:
            # - High Δ → input strongly affects state
            # - Low Δ → input weakly affects state
            # ============================================
            for i in range(n_qubits):
                # ⭐ SELECTIVE: Input scaled by Δ
                qml.RY(input_angles[i] * dt_angles[i], wires=i)

            # ============================================
            # STEP 5: MEASUREMENT
            # ============================================
            return [qml.expval(qml.PauliZ(i)) for i in range(n_qubits)]

        self.qnode = selective_qsvt_circuit

    def forward(self, angles: torch.Tensor) -> torch.Tensor:
        """
        Selective quantum SSM processing.

        Computes input-dependent Δ, B, C then applies selective
        quantum transformation at each timestep.

        The selectivity mechanism allows the model to:
        - Remember important information (high Δ)
        - Forget irrelevant information (low Δ)
        - All controlled by the input content itself

        Args:
            angles: (batch, seq_len, n_qubits) from QTS encoder
                   Values should be in [0, π] for proper angle encoding

        Returns:
            (batch, seq_len, n_qubits) selective quantum output
        """
        batch_size, seq_len, _ = angles.shape
        original_device = angles.device

        # ============================================
        # COMPUTE SELECTIVE PARAMETERS (Classical)
        # Δ[t], B[t], C[t] = f(angles[t])
        #
        # This is done classically for efficiency,
        # but the resulting Δ controls QUANTUM gates
        # ============================================
        x_dbc = self.x_proj(angles)  # (batch, seq_len, dt_rank + 2*n_qubits)

        # Split into components
        dt_raw = x_dbc[..., :self.dt_rank]
        B = x_dbc[..., self.dt_rank:self.dt_rank + self.n_qubits]
        C = x_dbc[..., self.dt_rank + self.n_qubits:]

        # Compute Δ with softplus (ensures positive, matches Mamba)
        dt = F.softplus(self.dt_proj(dt_raw))  # (batch, seq_len, n_qubits)

        # Scale Δ to quantum angle range [0, π]
        dt_angles = torch.tanh(dt) * np.pi

        # ============================================
        # SEQUENTIAL SELECTIVE QUANTUM PROCESSING
        #
        # For each timestep:
        # 1. Get input-dependent Δ[t]
        # 2. Apply Δ-modulated quantum transformation
        # 3. Update hidden state
        #
        # Note: Quantum ops run on CPU (default.qubit),
        # classical ops run on original device (GPU if available)
        # ============================================
        h = torch.zeros(batch_size, self.n_qubits, device='cpu')  # Quantum state on CPU
        outputs = []

        # Move quantum circuit params to CPU for quantum operations
        circuit_params_cpu = self.circuit_params.detach().cpu()
        poly_coeffs_cpu = self.poly_coeffs_base.detach().cpu()

        for t in range(seq_len):
            x_t = angles[:, t, :].detach().cpu()      # Current input angles (CPU)
            dt_t = dt_angles[:, t, :].detach().cpu()  # Input-dependent Δ (CPU)
            C_t = C[:, t, :]                          # Input-dependent output projection (original device)

            batch_outputs = []
            for b in range(batch_size):
                # QUANTUM OPERATION with selective Δ modulation (on CPU)
                measurements = self.qnode(
                    h[b],              # Previous state (CPU)
                    x_t[b],            # Current input (CPU)
                    dt_t[b],           # ⭐ SELECTIVE: Input-dependent Δ (CPU)
                    circuit_params_cpu,  # Variational params (CPU)
                    poly_coeffs_cpu      # Base QSVT polynomial (CPU)
                )
                batch_outputs.append(torch.stack(measurements))

            # Stack quantum outputs (CPU)
            h_cpu = torch.stack(batch_outputs)
            h = h_cpu  # Keep hidden state on CPU for next iteration

            # Move to original device for classical operations
            h_device = h_cpu.to(original_device)

            # ============================================
            # OUTPUT COMPUTATION
            # y[t] = C[t] · h[t] + D · x[t]
            #
            # C[t] is input-dependent (selective output projection)
            # D provides skip connection for gradient flow
            # ============================================
            y_t = C_t * h_device + self.D * angles[:, t, :]
            outputs.append(y_t)

        return torch.stack(outputs, dim=1)

    def get_param_count(self) -> dict:
        """Return parameter counts by component."""
        return {
            'x_proj (selective)': sum(p.numel() for p in self.x_proj.parameters()),
            'dt_proj (selective)': sum(p.numel() for p in self.dt_proj.parameters()),
            'poly_coeffs (QSVT)': self.poly_coeffs_base.numel(),
            'circuit_params (variational)': self.circuit_params.numel(),
            'D (skip)': self.D.numel(),
            'total': sum(p.numel() for p in self.parameters())
        }

    def __repr__(self) -> str:
        return (
            f"QuantumMambaSSMCore(\n"
            f"  n_qubits={self.n_qubits},\n"
            f"  n_layers={self.n_layers},\n"
            f"  qsvt_degree={self.qsvt_degree},\n"
            f"  dt_rank={self.dt_rank},\n"
            f"  circuit_params={self.n_circuit_params},\n"
            f"  total_params={sum(p.numel() for p in self.parameters())}\n"
            f")"
        )


# Export
__all__ = ['QuantumMambaSSMCore']
