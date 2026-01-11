"""
Advanced Quantum Hydra SSM Core - Bidirectional Quantum State Space Model

ADVANCED VERSION with:
1. UNIFIED ANSATZ (RX+RY+RZ+CRX) - consistent with QuantumFeatureExtractor
2. Multi-observable measurement (PauliX, PauliY, PauliZ) - 3x more information
3. Improved QSVT block with bidirectional entanglement
4. Proper state normalization and continuous state evolution

Theory:
-------
Classical Hydra SSM:
    h_fwd[t] = A · h_fwd[t-1] + B · x[t]  (forward)
    h_bwd[t] = A · h_bwd[t+1] + B · x[t]  (backward)
    h[t] = combine(h_fwd[t], h_bwd[t])

Quantum Hydra SSM (Advanced):
    |h_fwd[t]⟩ = unified_QSVT(A)|h_fwd[t-1]⟩ ⊕ B|x[t]⟩  (quantum forward)
    |h_bwd[t]⟩ = unified_QSVT(A)|h_bwd[t+1]⟩ ⊕ B|x[t]⟩  (quantum backward)
    |h[t]⟩ = LCU(|h_fwd[t]⟩, |h_bwd[t]⟩)                 (quantum combination)
    y[t] = [⟨X⟩, ⟨Y⟩, ⟨Z⟩]                              (multi-observable)

UNIFIED ANSATZ (consistent with QuantumFeatureExtractor):
    Structure per layer:
    1. RX, RY, RZ on all qubits (full single-qubit expressivity)
    2. CRX forward entanglement (i → i+1, circular)
    3. CRX backward entanglement (i → i-1, circular)
    Parameters per layer: 5 × n_qubits

Key Features:
1. Multi-observable: 3 × n_qubits outputs instead of n_qubits
2. Bidirectional entanglement in QSVT block
3. Continuous state evolution (not reset each timestep)

Author: Junghoon Park
Date: December 2024
"""

import torch
import torch.nn as nn
import pennylane as qml
import numpy as np
from typing import List, Tuple, Optional, Literal


class QuantumHydraSSMCoreAdvanced(nn.Module):
    """
    Advanced Bidirectional Quantum SSM using UNIFIED ANSATZ and multi-observable.

    This module implements quantum selective forgetting through:
    1. UNIFIED ansatz: RX+RY+RZ+CRX (consistent with QuantumFeatureExtractor)
    2. QSVT: Polynomial transformation P(A) ≈ exp(A) with bidirectional entanglement
    3. LCU: Linear combination of forward and backward quantum states
    4. Multi-observable: PauliX, PauliY, PauliZ measurements (3 × n_qubits outputs)

    UNIFIED ANSATZ (5 × n_qubits params per layer):
    - RX, RY, RZ rotations (3 × n_qubits)
    - Forward CRX entanglement (n_qubits)
    - Backward CRX entanglement (n_qubits)

    Args:
        n_qubits: Number of qubits in the circuit
        n_layers: Number of unified ansatz layers (each has 5 × n_qubits params)
        qsvt_degree: Degree of QSVT polynomial (higher = better approximation)
        device: Device for computation ('cpu' or 'cuda')

    Output Dimension: 3 × n_qubits (from PauliX, PauliY, PauliZ measurements)

    Example:
        core = QuantumHydraSSMCoreAdvanced(n_qubits=4, n_layers=2, qsvt_degree=3)
        angles = torch.randn(32, 200, 4)  # (batch, seq_len, n_qubits)
        output = core(angles)  # (batch, seq_len, 12) = 3 × 4 qubits
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

        # Output dimension: 3 × n_qubits (X, Y, Z measurements)
        self.output_dim = 3 * n_qubits

        # ============================================
        # UNIFIED Ansatz Parameters
        # Each layer has 5 × n_qubits parameters:
        # - 3 × n_qubits for RX, RY, RZ rotations
        # - 2 × n_qubits for forward + backward CRX
        # ============================================
        self.n_params_per_layer = 5 * n_qubits
        self.n_circuit_params = self.n_params_per_layer * n_layers

        # ============================================
        # QSVT Polynomial Coefficients (QUANTUM)
        # These define the polynomial P(A) ≈ exp(A)
        # Learned during training
        # ============================================
        self.poly_coeffs = nn.Parameter(
            torch.randn(qsvt_degree + 1) * 0.1
        )

        # ============================================
        # LCU Coefficients for Bidirectional Combination
        # α|h_fwd⟩ + β|h_bwd⟩
        # ============================================
        self.fwd_coeff = nn.Parameter(torch.tensor([0.5]))
        self.bwd_coeff = nn.Parameter(torch.tensor([0.5]))

        # ============================================
        # Variational Circuit Parameters (QUANTUM)
        # Separate parameters for forward and backward passes
        # Using unified ansatz: 5 × n_qubits × n_layers per direction
        # ============================================
        self.fwd_params = nn.Parameter(
            torch.randn(self.n_circuit_params) * 0.1
        )
        self.bwd_params = nn.Parameter(
            torch.randn(self.n_circuit_params) * 0.1
        )

        # ============================================
        # QSVT Block Parameters (separate from ansatz)
        # For polynomial coefficient application
        # ============================================
        self.qsvt_params = nn.Parameter(
            torch.randn(n_qubits * qsvt_degree) * 0.1
        )

        # ============================================
        # Setup Quantum Device
        # Use lightning.kokkos for CUDA to avoid hang issues
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
        """Build advanced QSVT-based quantum SSM circuit with UNIFIED ANSATZ."""

        n_qubits = self.n_qubits
        n_layers = self.n_layers

        @qml.qnode(self.dev, interface="torch", diff_method="best")
        def advanced_qsvt_ssm_circuit(
            state_angles: torch.Tensor,
            input_angles: torch.Tensor,
            circuit_params: torch.Tensor,
            poly_coeffs: torch.Tensor,
            qsvt_params: torch.Tensor
        ) -> List[torch.Tensor]:
            """
            ADVANCED QUANTUM SSM STATE EVOLUTION via UNIFIED ANSATZ + QSVT

            Circuit Structure:
            1. State Encoding: RY(h[t-1]) on each qubit
            2. QSVT Block: Polynomial transformation with bidirectional entanglement
            3. UNIFIED Variational Ansatz: RX,RY,RZ → CRX(fwd) → CRX(bwd) per layer
            4. Input Injection: RY(x[t]) on each qubit
            5. Multi-Observable Measurement: ⟨X⟩, ⟨Y⟩, ⟨Z⟩ on each qubit

            Args:
                state_angles: Previous hidden state h[t-1], shape (n_qubits,)
                input_angles: Current input x[t], shape (n_qubits,)
                circuit_params: unified ansatz variational parameters
                poly_coeffs: QSVT polynomial coefficients
                qsvt_params: Additional QSVT rotation parameters

            Returns:
                List of 3 × n_qubits expectation values [⟨X⟩, ⟨Y⟩, ⟨Z⟩]
            """
            # ============================================
            # STEP 1: QUANTUM STATE ENCODING
            # Encode h[t-1] into quantum amplitudes via RY rotations
            # |0⟩ → cos(θ/2)|0⟩ + sin(θ/2)|1⟩
            # ============================================
            for i in range(n_qubits):
                qml.RY(state_angles[i], wires=i)

            # ============================================
            # STEP 2: QSVT TRANSFORMATION (Advanced)
            # Apply polynomial P(A) ≈ exp(A) with bidirectional entanglement
            #
            # For each polynomial degree:
            # - Apply coefficient-scaled phase rotations
            # - Add bidirectional entanglement (forward + reverse CNOT)
            # ============================================
            qsvt_idx = 0
            for deg in range(len(poly_coeffs)):
                coeff = poly_coeffs[deg]

                # Phase rotations with learned QSVT params
                for i in range(n_qubits):
                    if qsvt_idx < len(qsvt_params):
                        qml.RZ(coeff * qsvt_params[qsvt_idx] * np.pi, wires=i)
                        qsvt_idx += 1
                    else:
                        qml.RZ(coeff * np.pi, wires=i)

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
            # STEP 4: INPUT INJECTION (B · x[t])
            # Add new input information to the quantum state
            # ============================================
            for i in range(n_qubits):
                qml.RY(input_angles[i], wires=i)

            # ============================================
            # STEP 5: MULTI-OBSERVABLE MEASUREMENT
            # Extract 3 × n_qubits values via Pauli X, Y, Z expectations
            # This provides much richer information than Z-only
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

        self.qnode = advanced_qsvt_ssm_circuit

    def _quantum_ssm_pass(
        self,
        angles: torch.Tensor,
        circuit_params: torch.Tensor,
        reverse: bool = False
    ) -> torch.Tensor:
        """
        Single direction quantum SSM pass with advanced circuit.

        Quantum operations run on CPU (default.qubit), results moved to original device.

        Args:
            angles: (batch, seq_len, n_qubits) quantum-encoded inputs
            circuit_params: sim14 variational parameters for this direction
            reverse: If True, process sequence in reverse (for backward pass)

        Returns:
            (batch, seq_len, 3 * n_qubits) quantum SSM outputs (X, Y, Z measurements)
        """
        batch_size, seq_len, _ = angles.shape
        original_device = angles.device

        # Reverse sequence for backward pass
        if reverse:
            angles = angles.flip(dims=[1])

        # Initialize quantum hidden state on CPU (for quantum ops)
        h = torch.zeros(batch_size, self.n_qubits, device='cpu')
        outputs = []

        # Move quantum params to CPU
        circuit_params_cpu = circuit_params.detach().cpu()
        poly_coeffs_cpu = self.poly_coeffs.detach().cpu()
        qsvt_params_cpu = self.qsvt_params.detach().cpu()

        # Sequential quantum state evolution
        for t in range(seq_len):
            x_t = angles[:, t, :].detach().cpu()  # Current input (CPU)
            batch_outputs = []

            # Process each sample in batch
            for b in range(batch_size):
                # QUANTUM OPERATION: Evolve state through advanced circuit (CPU)
                measurements = self.qnode(
                    h[b],                  # Previous state (CPU)
                    x_t[b],                # Current input (CPU)
                    circuit_params_cpu,    # sim14 variational params (CPU)
                    poly_coeffs_cpu,       # QSVT polynomial (CPU)
                    qsvt_params_cpu        # QSVT rotation params (CPU)
                )
                batch_outputs.append(torch.stack(measurements))

            # Stack outputs: (batch, 3 * n_qubits) - on CPU
            h_full_cpu = torch.stack(batch_outputs)
            # Move to original device for output
            h_full = h_full_cpu.to(original_device)
            outputs.append(h_full)

            # Update hidden state for next iteration (use Z measurements, keep on CPU)
            h = h_full_cpu[:, 2 * self.n_qubits:]  # Take Z measurements as state

        # Stack outputs: (seq_len, batch, 3*n_qubits) → (batch, seq_len, 3*n_qubits)
        output = torch.stack(outputs, dim=1)

        # Reverse back for backward pass
        if reverse:
            output = output.flip(dims=[1])

        return output

    def forward(self, angles: torch.Tensor) -> torch.Tensor:
        """
        Bidirectional quantum SSM processing with advanced circuits.

        Implements Hydra-style bidirectional processing:
        1. Forward quantum SSM pass: left-to-right with unified ansatz
        2. Backward quantum SSM pass: right-to-left with unified ansatz
        3. LCU (Linear Combination of Unitaries) combination

        Args:
            angles: (batch, seq_len, n_qubits) from QTS encoder
                   Values should be in [0, π] for proper angle encoding

        Returns:
            (batch, seq_len, 3 * n_qubits) bidirectional quantum output
            where 3 * n_qubits = [X_measurements, Y_measurements, Z_measurements]
        """
        # ============================================
        # FORWARD QUANTUM SSM PASS
        # Process sequence left-to-right with unified ansatz
        # ============================================
        h_fwd = self._quantum_ssm_pass(
            angles,
            self.fwd_params,
            reverse=False
        )

        # ============================================
        # BACKWARD QUANTUM SSM PASS
        # Process sequence right-to-left with unified ansatz
        # ============================================
        h_bwd = self._quantum_ssm_pass(
            angles,
            self.bwd_params,
            reverse=True
        )

        # ============================================
        # LCU COMBINATION
        # Linear Combination of Unitaries:
        # |h[t]⟩ = α|h_fwd[t]⟩ + β|h_bwd[t]⟩
        # ============================================
        fwd_weight = torch.sigmoid(self.fwd_coeff)
        bwd_weight = torch.sigmoid(self.bwd_coeff)
        combined = fwd_weight * h_fwd + bwd_weight * h_bwd

        return combined

    def get_param_count(self) -> dict:
        """Return parameter counts by component."""
        return {
            'poly_coeffs (QSVT)': self.poly_coeffs.numel(),
            'qsvt_params (QSVT rotations)': self.qsvt_params.numel(),
            'fwd_params (unified ansatz)': self.fwd_params.numel(),
            'bwd_params (unified ansatz)': self.bwd_params.numel(),
            'lcu_coeffs': 2,
            'total': sum(p.numel() for p in self.parameters())
        }

    def __repr__(self) -> str:
        return (
            f"QuantumHydraSSMCoreAdvanced(\n"
            f"  n_qubits={self.n_qubits},\n"
            f"  n_layers={self.n_layers},\n"
            f"  qsvt_degree={self.qsvt_degree},\n"
            f"  ansatz='unified (RX+RY+RZ+CRX)',\n"
            f"  measurement='multi-observable (X,Y,Z)',\n"
            f"  output_dim={self.output_dim},\n"
            f"  circuit_params_per_layer={self.n_params_per_layer},\n"
            f"  total_params={sum(p.numel() for p in self.parameters())}\n"
            f")"
        )


# Export
__all__ = ['QuantumHydraSSMCoreAdvanced']
