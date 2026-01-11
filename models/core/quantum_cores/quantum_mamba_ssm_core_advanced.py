"""
Advanced Quantum Mamba SSM Core - Selective Quantum State Space Model

ADVANCED VERSION with:
1. UNIFIED ANSATZ (RX+RY+RZ+CRX) - consistent with QuantumFeatureExtractor
2. Multi-observable measurement (PauliX, PauliY, PauliZ) - 3x more information
3. Improved QSVT block with Δ-modulated bidirectional entanglement
4. INPUT-DEPENDENT Δ, B, C parameters (Mamba-style selectivity)

Theory:
-------
Classical Mamba SSM (Gu & Dao, 2024):
    Δ[t], B[t], C[t] = f(x[t])                    # Input-dependent params
    h[t] = exp(Δ[t] · A) · h[t-1] + Δ[t] · B[t] · x[t]  # State update
    y[t] = C[t] · h[t] + D · x[t]                 # Output

Advanced Quantum Mamba SSM:
    Δ[t], B[t], C[t] = f(x[t])                    # Classical projection
    |h[t]⟩ = unified_QSVT_exp(Δ[t]·A)|h[t-1]⟩ ⊕ Δ[t]·B[t]|x[t]⟩  # QUANTUM evolution
    y[t] = [⟨X⟩, ⟨Y⟩, ⟨Z⟩] @ (C[t] · |h[t]⟩) + D · x[t]     # Multi-observable

UNIFIED ANSATZ (consistent with QuantumFeatureExtractor):
    Structure per layer:
    1. RX, RY, RZ on all qubits (full single-qubit expressivity)
    2. CRX forward entanglement (i → i+1, circular)
    3. CRX backward entanglement (i → i-1, circular)
    Parameters per layer: 5 × n_qubits

The key QUANTUM SELECTIVE mechanism:
- Input-dependent Δ controls quantum transformation strength
- Large Δ: More input influence, less state retention (REMEMBER new info)
- Small Δ: Less input influence, more state retention (FORGET/maintain state)
- Multi-observable extracts 3× more information from quantum state

Author: Junghoon Park
Date: December 2024
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import pennylane as qml
import numpy as np
from typing import List, Optional


class QuantumMambaSSMCoreAdvanced(nn.Module):
    """
    Advanced Selective Quantum SSM using UNIFIED ANSATZ with input-dependent Δ.

    This module implements Mamba-style selective state space modeling
    with the UNIFIED ANSATZ for fair ablation comparison.

    Key Components:
    1. Input-dependent Δ projection: Computes selective gate from input
    2. Input-dependent B, C: Computes selective input/output projections
    3. Unified QSVT exp(Δ·A): Quantum state transition with variable strength
    4. Multi-observable measurement: PauliX, Y, Z (3 × n_qubits outputs)
    5. D skip connection: Direct input-to-output path

    UNIFIED ANSATZ (5 × n_qubits params per layer):
    - RX, RY, RZ rotations (3 × n_qubits)
    - Forward CRX entanglement (n_qubits)
    - Backward CRX entanglement (n_qubits)

    The selectivity mechanism:
    - Δ (delta/timestep) controls how much the quantum state transforms
    - High Δ → stronger quantum transformation → more input influence
    - Low Δ → weaker quantum transformation → more state retention

    Args:
        n_qubits: Number of qubits in the circuit
        n_layers: Number of unified ansatz layers
        qsvt_degree: Degree of QSVT polynomial
        dt_rank: Rank of Δ projection ("auto" = n_qubits // 2)
        dt_min: Minimum Δ value (for initialization)
        dt_max: Maximum Δ value (for initialization)
        device: Device for computation ('cpu' or 'cuda')

    Output Dimension: 3 × n_qubits (from PauliX, PauliY, PauliZ measurements)

    Example:
        core = QuantumMambaSSMCoreAdvanced(n_qubits=4, n_layers=2)
        angles = torch.randn(32, 200, 4)  # (batch, seq_len, n_qubits)
        output = core(angles)  # (batch, seq_len, 12) = 3 × 4 qubits
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

        # Output dimension: 3 × n_qubits (X, Y, Z measurements)
        self.output_dim = 3 * n_qubits

        # ============================================
        # Δ (Delta) Rank for Selective Projection
        # Lower rank = more parameter efficient
        # ============================================
        if dt_rank == "auto":
            self.dt_rank = max(n_qubits // 2, 1)
        else:
            self.dt_rank = int(dt_rank)

        # ============================================
        # UNIFIED Ansatz Parameters
        # Each layer has 5 × n_qubits parameters:
        # - 3 × n_qubits for RX, RY, RZ rotations
        # - 2 × n_qubits for forward + backward CRX
        # ============================================
        self.n_params_per_layer = 5 * n_qubits
        self.n_circuit_params = self.n_params_per_layer * n_layers

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
        # QSVT Block Parameters (separate from ansatz)
        # For polynomial coefficient application
        # ============================================
        self.qsvt_params = nn.Parameter(
            torch.randn(n_qubits * qsvt_degree) * 0.1
        )

        # ============================================
        # Variational Circuit Parameters (QUANTUM)
        # Using unified ansatz: 5 × n_qubits × n_layers
        # ============================================
        self.circuit_params = nn.Parameter(
            torch.randn(self.n_circuit_params) * 0.1
        )

        # ============================================
        # D Parameter (Skip Connection)
        # Direct input-to-output path
        # Expanded for multi-observable output
        # ============================================
        self.D = nn.Parameter(torch.ones(self.output_dim))

        # ============================================
        # C output projection (expands to match multi-observable)
        # Maps 3*n_qubits to 3*n_qubits with input-dependent modulation
        # ============================================
        self.c_expand = nn.Linear(n_qubits, self.output_dim, bias=False)

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
        """Build selective QSVT circuit with UNIFIED ANSATZ and Δ modulation."""

        n_qubits = self.n_qubits
        n_layers = self.n_layers

        @qml.qnode(self.dev, interface="torch", diff_method="best")
        def selective_advanced_qsvt_circuit(
            state_angles: torch.Tensor,
            input_angles: torch.Tensor,
            dt_angles: torch.Tensor,
            circuit_params: torch.Tensor,
            poly_coeffs: torch.Tensor,
            qsvt_params: torch.Tensor
        ) -> List[torch.Tensor]:
            """
            ADVANCED SELECTIVE QUANTUM SSM with input-dependent Δ and UNIFIED ANSATZ

            The key innovation: dt_angles (Δ) MODULATES the quantum transformation
            with unified ansatz's bidirectional entanglement for consistency.

            Circuit Structure:
            1. State Encoding: RY(h[t-1])
            2. Δ-Modulated QSVT: coeff * Δ * π rotations + bidirectional CNOT
            3. UNIFIED Variational Ansatz: RX,RY,RZ → CRX(fwd) → CRX(bwd)
            4. Δ-Modulated Input: RY(x[t] * Δ)
            5. Multi-Observable Measurement: ⟨X⟩, ⟨Y⟩, ⟨Z⟩

            Args:
                state_angles: Previous state h[t-1], shape (n_qubits,)
                input_angles: Current input x[t], shape (n_qubits,)
                dt_angles: Input-dependent Δ[t], shape (n_qubits,)
                circuit_params: unified ansatz variational parameters
                poly_coeffs: Base QSVT polynomial coefficients
                qsvt_params: QSVT rotation parameters

            Returns:
                List of 3 × n_qubits expectation values [⟨X⟩, ⟨Y⟩, ⟨Z⟩]
            """
            # ============================================
            # STEP 1: QUANTUM STATE ENCODING
            # ============================================
            for i in range(n_qubits):
                qml.RY(state_angles[i], wires=i)

            # ============================================
            # STEP 2: Δ-MODULATED QSVT TRANSFORMATION (Advanced)
            #
            # This is the CORE SELECTIVE MECHANISM:
            # - Polynomial coefficients are SCALED by input-dependent Δ
            # - High Δ → larger rotations → stronger transformation
            # - Low Δ → smaller rotations → weaker transformation
            # - Bidirectional entanglement for better quantum correlations
            # ============================================
            qsvt_idx = 0
            for deg in range(len(poly_coeffs)):
                coeff = poly_coeffs[deg]

                for i in range(n_qubits):
                    # ⭐ SELECTIVE: Δ modulates transformation strength
                    # with learned QSVT parameters
                    if qsvt_idx < len(qsvt_params):
                        angle = coeff * dt_angles[i] * qsvt_params[qsvt_idx] * np.pi
                        qsvt_idx += 1
                    else:
                        angle = coeff * dt_angles[i] * np.pi
                    qml.RZ(angle, wires=i)

                # Bidirectional entangling layer (forward)
                for i in range(n_qubits - 1):
                    qml.CNOT(wires=[i, i + 1])

                # Bidirectional entangling layer (reverse/circular)
                if n_qubits > 1:
                    qml.CNOT(wires=[n_qubits - 1, 0])
                for i in range(n_qubits - 1, 0, -1):
                    qml.CNOT(wires=[i, i - 1])

            # ============================================
            # STEP 3: UNIFIED VARIATIONAL ANSATZ
            # Consistent with QuantumFeatureExtractor
            # ============================================
            param_idx = 0
            for layer in range(n_layers):
                # 1. Full single-qubit rotations: RX, RY, RZ
                for i in range(n_qubits):
                    qml.RX(circuit_params[param_idx], wires=i)
                    qml.RY(circuit_params[param_idx + 1], wires=i)
                    qml.RZ(circuit_params[param_idx + 2], wires=i)
                    param_idx += 3

                # 2. Forward CRX entanglement (i → i+1, circular)
                for i in range(n_qubits):
                    qml.CRX(circuit_params[param_idx], wires=[i, (i + 1) % n_qubits])
                    param_idx += 1

                # 3. Backward CRX entanglement (i → i-1, circular)
                for i in range(n_qubits - 1, -1, -1):
                    qml.CRX(circuit_params[param_idx], wires=[i, (i - 1) % n_qubits])
                    param_idx += 1

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
            # STEP 5: MULTI-OBSERVABLE MEASUREMENT
            # Extract 3 × n_qubits values via Pauli X, Y, Z
            # ============================================
            observables = []
            # PauliX measurements
            for i in range(n_qubits):
                observables.append(qml.expval(qml.PauliX(i)))
            # PauliY measurements
            for i in range(n_qubits):
                observables.append(qml.expval(qml.PauliY(i)))
            # PauliZ measurements
            for i in range(n_qubits):
                observables.append(qml.expval(qml.PauliZ(i)))

            return observables

        self.qnode = selective_advanced_qsvt_circuit

    def forward(self, angles: torch.Tensor) -> torch.Tensor:
        """
        Advanced selective quantum SSM processing.

        Computes input-dependent Δ, B, C then applies selective
        quantum transformation at each timestep with sim14 ansatz.

        The selectivity mechanism allows the model to:
        - Remember important information (high Δ)
        - Forget irrelevant information (low Δ)
        - All controlled by the input content itself
        - With 3× more output information via multi-observable

        Args:
            angles: (batch, seq_len, n_qubits) from QTS encoder
                   Values should be in [0, π] for proper angle encoding

        Returns:
            (batch, seq_len, 3 * n_qubits) selective quantum output
            where 3 * n_qubits = [X_measurements, Y_measurements, Z_measurements]
        """
        batch_size, seq_len, _ = angles.shape
        original_device = angles.device

        # ============================================
        # COMPUTE SELECTIVE PARAMETERS (Classical - on GPU)
        # Δ[t], B[t], C[t] = f(angles[t])
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
        # Quantum ops on CPU (default.qubit), classical ops on original device (GPU)
        # ============================================
        h = torch.zeros(batch_size, self.n_qubits, device='cpu')  # Quantum state on CPU
        outputs = []

        # Move quantum circuit params to CPU for quantum operations
        circuit_params_cpu = self.circuit_params.detach().cpu()
        poly_coeffs_cpu = self.poly_coeffs_base.detach().cpu()
        qsvt_params_cpu = self.qsvt_params.detach().cpu()

        for t in range(seq_len):
            x_t = angles[:, t, :].detach().cpu()      # Current input angles (CPU)
            dt_t = dt_angles[:, t, :].detach().cpu()  # Input-dependent Δ (CPU)
            C_t = C[:, t, :]                          # Keep on original device for classical ops

            batch_outputs = []
            for b in range(batch_size):
                # QUANTUM OPERATION with selective Δ modulation (on CPU)
                measurements = self.qnode(
                    h[b],                    # Previous state (CPU)
                    x_t[b],                  # Current input (CPU)
                    dt_t[b],                 # ⭐ SELECTIVE: Input-dependent Δ (CPU)
                    circuit_params_cpu,      # sim14 variational params (CPU)
                    poly_coeffs_cpu,         # Base QSVT polynomial (CPU)
                    qsvt_params_cpu          # QSVT rotation params (CPU)
                )
                batch_outputs.append(torch.stack(measurements))

            # Stack: (batch, 3 * n_qubits) - still on CPU
            h_full_cpu = torch.stack(batch_outputs)
            # Move to original device (GPU) for classical operations
            h_full = h_full_cpu.to(original_device)

            # Update hidden state for next iteration (use Z measurements, keep on CPU)
            h = h_full_cpu[:, 2 * self.n_qubits:]  # Take Z measurements as state (CPU)

            # ============================================
            # OUTPUT COMPUTATION (Multi-Observable)
            # y[t] = C[t] · h[t] + D · x[t]
            #
            # C[t] is input-dependent, expanded to match output dim
            # D provides skip connection for gradient flow
            # ============================================
            C_expanded = self.c_expand(C_t)  # (batch, 3*n_qubits)
            x_expanded = torch.cat([angles[:, t, :]] * 3, dim=-1)  # Repeat input
            y_t = C_expanded * h_full + self.D * x_expanded
            outputs.append(y_t)

        return torch.stack(outputs, dim=1)

    def get_param_count(self) -> dict:
        """Return parameter counts by component."""
        return {
            'x_proj (selective)': sum(p.numel() for p in self.x_proj.parameters()),
            'dt_proj (selective)': sum(p.numel() for p in self.dt_proj.parameters()),
            'poly_coeffs (QSVT)': self.poly_coeffs_base.numel(),
            'qsvt_params (QSVT rotations)': self.qsvt_params.numel(),
            'circuit_params (unified ansatz)': self.circuit_params.numel(),
            'c_expand (output projection)': sum(p.numel() for p in self.c_expand.parameters()),
            'D (skip)': self.D.numel(),
            'total': sum(p.numel() for p in self.parameters())
        }

    def __repr__(self) -> str:
        return (
            f"QuantumMambaSSMCoreAdvanced(\n"
            f"  n_qubits={self.n_qubits},\n"
            f"  n_layers={self.n_layers},\n"
            f"  qsvt_degree={self.qsvt_degree},\n"
            f"  dt_rank={self.dt_rank},\n"
            f"  ansatz='unified (RX+RY+RZ+CRX)',\n"
            f"  measurement='multi-observable (X,Y,Z)',\n"
            f"  selective='input-dependent Δ',\n"
            f"  output_dim={self.output_dim},\n"
            f"  circuit_params={self.n_circuit_params},\n"
            f"  total_params={sum(p.numel() for p in self.parameters())}\n"
            f")"
        )


# Export
__all__ = ['QuantumMambaSSMCoreAdvanced']
