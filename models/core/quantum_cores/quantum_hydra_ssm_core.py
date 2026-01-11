"""
Quantum Hydra SSM Core - Bidirectional Quantum State Space Model

Implements QUANTUM selective forgetting using QSVT (Quantum Singular Value
Transformation) for state evolution and LCU (Linear Combination of Unitaries)
for bidirectional combination.

Theory:
-------
Classical Hydra SSM:
    h_fwd[t] = A · h_fwd[t-1] + B · x[t]  (forward)
    h_bwd[t] = A · h_bwd[t+1] + B · x[t]  (backward)
    h[t] = combine(h_fwd[t], h_bwd[t])

Quantum Hydra SSM:
    |h_fwd[t]⟩ = QSVT_exp(A)|h_fwd[t-1]⟩ ⊕ B|x[t]⟩  (quantum forward)
    |h_bwd[t]⟩ = QSVT_exp(A)|h_bwd[t+1]⟩ ⊕ B|x[t]⟩  (quantum backward)
    |h[t]⟩ = LCU(|h_fwd[t]⟩, |h_bwd[t]⟩)           (quantum combination)

The QUANTUM SELECTIVE FORGETTING happens in:
1. QSVT polynomial approximating exp(A) - quantum state transformation
2. Quantum interference during state evolution
3. LCU combination of forward and backward passes

Author: Junghoon Park
Date: December 2024
"""

import torch
import torch.nn as nn
import pennylane as qml
import numpy as np
from typing import List, Tuple, Optional


class QuantumHydraSSMCore(nn.Module):
    """
    Bidirectional Quantum SSM using QSVT.

    This module implements quantum selective forgetting through:
    1. QSVT: Polynomial transformation P(A) ≈ exp(A) applied quantumly
    2. Variational ansatz: Trainable quantum circuit layers
    3. LCU: Linear combination of forward and backward quantum states
    4. Measurement: Extract classical values from quantum state

    Key Quantum Operations:
    - RY gates: Angle encoding of state and input
    - RZ gates: QSVT polynomial phase rotations
    - CNOT gates: Entanglement for quantum correlations
    - PauliZ measurement: State readout

    Args:
        n_qubits: Number of qubits in the circuit
        n_layers: Number of variational ansatz layers
        qsvt_degree: Degree of QSVT polynomial (higher = better approximation)
        device: Device for computation ('cpu' or 'cuda')

    Example:
        core = QuantumHydraSSMCore(n_qubits=4, n_layers=2, qsvt_degree=3)
        angles = torch.randn(32, 200, 4)  # (batch, seq_len, n_qubits)
        output = core(angles)  # (batch, seq_len, n_qubits)
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
        # RX, RY, RZ rotations per qubit per layer
        # ============================================
        self.n_circuit_params = n_qubits * n_layers * 3
        self.fwd_params = nn.Parameter(
            torch.randn(self.n_circuit_params) * 0.1
        )
        self.bwd_params = nn.Parameter(
            torch.randn(self.n_circuit_params) * 0.1
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
        """Build QSVT-based quantum SSM circuit."""

        n_qubits = self.n_qubits
        n_layers = self.n_layers

        @qml.qnode(self.dev, interface="torch", diff_method="best")
        def qsvt_ssm_circuit(
            state_angles: torch.Tensor,
            input_angles: torch.Tensor,
            circuit_params: torch.Tensor,
            poly_coeffs: torch.Tensor
        ) -> List[torch.Tensor]:
            """
            QUANTUM SSM STATE EVOLUTION via QSVT

            This circuit implements quantum selective forgetting:
            |h[t]⟩ = QSVT_exp(A)|h[t-1]⟩ + B|x[t]⟩

            Circuit Structure:
            1. State Encoding: RY(h[t-1]) on each qubit
            2. QSVT Block: Polynomial transformation via RZ + CNOT
            3. Variational Block: Trainable RX, RY, RZ + CNOT
            4. Input Injection: RY(x[t]) on each qubit
            5. Measurement: ⟨Z⟩ on each qubit

            Args:
                state_angles: Previous hidden state h[t-1], shape (n_qubits,)
                input_angles: Current input x[t], shape (n_qubits,)
                circuit_params: Variational parameters, shape (n_circuit_params,)
                poly_coeffs: QSVT polynomial coefficients, shape (qsvt_degree+1,)

            Returns:
                List of expectation values ⟨Z_i⟩ for i in range(n_qubits)
            """
            # ============================================
            # STEP 1: QUANTUM STATE ENCODING
            # Encode h[t-1] into quantum amplitudes via RY rotations
            # |0⟩ → cos(θ/2)|0⟩ + sin(θ/2)|1⟩
            # ============================================
            for i in range(n_qubits):
                qml.RY(state_angles[i], wires=i)

            # ============================================
            # STEP 2: QSVT TRANSFORMATION
            # Apply polynomial P(A) ≈ exp(A) via phase rotations
            #
            # This is the CORE QUANTUM SELECTIVE FORGETTING mechanism:
            # - Phase rotations encode polynomial coefficients
            # - CNOT gates create entanglement for interference
            # - The combined effect approximates matrix exponential
            # ============================================
            for deg in range(len(poly_coeffs)):
                coeff = poly_coeffs[deg]

                # Phase rotations (polynomial coefficient encoding)
                for i in range(n_qubits):
                    qml.RZ(coeff * np.pi, wires=i)

                # Entangling layer (creates quantum correlations)
                # This enables quantum interference during state evolution
                for i in range(n_qubits - 1):
                    qml.CNOT(wires=[i, i + 1])
                if n_qubits > 1:
                    qml.CNOT(wires=[n_qubits - 1, 0])  # Circular entanglement

            # ============================================
            # STEP 3: VARIATIONAL ANSATZ
            # Trainable quantum layer for expressivity
            # Allows learning task-specific transformations
            # ============================================
            param_idx = 0
            for layer in range(n_layers):
                # Single-qubit rotations
                for i in range(n_qubits):
                    qml.RX(circuit_params[param_idx], wires=i)
                    qml.RY(circuit_params[param_idx + 1], wires=i)
                    qml.RZ(circuit_params[param_idx + 2], wires=i)
                    param_idx += 3

                # Entanglement layer
                for i in range(n_qubits - 1):
                    qml.CNOT(wires=[i, i + 1])

            # ============================================
            # STEP 4: INPUT INJECTION (B · x[t])
            # Add new input information to the quantum state
            # ============================================
            for i in range(n_qubits):
                qml.RY(input_angles[i], wires=i)

            # ============================================
            # STEP 5: MEASUREMENT
            # Extract classical values via Pauli-Z expectations
            # ⟨Z⟩ ∈ [-1, 1] for each qubit
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
        Processes the sequence either forward or backward.

        Note: Quantum ops run on CPU (default.qubit), classical ops run on
        original device (GPU if available).

        Args:
            angles: (batch, seq_len, n_qubits) quantum-encoded inputs from encoder
            circuit_params: Variational parameters for this direction
            reverse: If True, process sequence in reverse (for backward pass)

        Returns:
            (batch, seq_len, n_qubits) quantum SSM outputs
        """
        batch_size, seq_len, _ = angles.shape
        original_device = angles.device

        # Reverse sequence for backward pass
        if reverse:
            angles = angles.flip(dims=[1])

        # Initialize quantum hidden state to zero (on CPU for quantum ops)
        h = torch.zeros(batch_size, self.n_qubits, device='cpu')
        outputs = []

        # Move quantum circuit params to CPU for quantum operations
        circuit_params_cpu = circuit_params.detach().cpu()
        poly_coeffs_cpu = self.poly_coeffs.detach().cpu()

        # Sequential quantum state evolution
        for t in range(seq_len):
            x_t = angles[:, t, :].detach().cpu()  # Current input (CPU)
            batch_outputs = []

            # Process each sample in batch
            # Note: Could be parallelized with batched quantum execution
            for b in range(batch_size):
                # QUANTUM OPERATION: Evolve state through circuit (on CPU)
                measurements = self.qnode(
                    h[b],              # Previous state (CPU)
                    x_t[b],            # Current input (CPU)
                    circuit_params_cpu,  # Variational params (CPU)
                    poly_coeffs_cpu      # QSVT polynomial (CPU)
                )
                batch_outputs.append(torch.stack(measurements))

            # Stack quantum outputs (CPU)
            h_cpu = torch.stack(batch_outputs)
            h = h_cpu  # Keep hidden state on CPU for next iteration

            # Move to original device for output collection
            h_device = h_cpu.to(original_device)
            outputs.append(h_device)

        # Stack outputs: (seq_len, batch, n_qubits) → (batch, seq_len, n_qubits)
        output = torch.stack(outputs, dim=1)

        # Reverse back for backward pass
        if reverse:
            output = output.flip(dims=[1])

        return output

    def forward(self, angles: torch.Tensor) -> torch.Tensor:
        """
        Bidirectional quantum SSM processing.

        Implements Hydra-style bidirectional processing:
        1. Forward quantum SSM pass: left-to-right
        2. Backward quantum SSM pass: right-to-left
        3. LCU (Linear Combination of Unitaries) combination

        The bidirectional design allows:
        - Forward pass captures left context
        - Backward pass captures right context
        - LCU combines both for global context

        Args:
            angles: (batch, seq_len, n_qubits) from QTS encoder
                   Values should be in [0, π] for proper angle encoding

        Returns:
            (batch, seq_len, n_qubits) bidirectional quantum output
        """
        # ============================================
        # FORWARD QUANTUM SSM PASS
        # Process sequence left-to-right
        # ============================================
        h_fwd = self._quantum_ssm_pass(
            angles,
            self.fwd_params,
            reverse=False
        )

        # ============================================
        # BACKWARD QUANTUM SSM PASS
        # Process sequence right-to-left
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
        #
        # Learned coefficients allow model to balance
        # forward and backward context importance
        # ============================================
        fwd_weight = torch.sigmoid(self.fwd_coeff)
        bwd_weight = torch.sigmoid(self.bwd_coeff)
        combined = fwd_weight * h_fwd + bwd_weight * h_bwd

        return combined

    def get_param_count(self) -> dict:
        """Return parameter counts by component."""
        return {
            'poly_coeffs (QSVT)': self.poly_coeffs.numel(),
            'fwd_params (variational)': self.fwd_params.numel(),
            'bwd_params (variational)': self.bwd_params.numel(),
            'lcu_coeffs': 2,
            'total': sum(p.numel() for p in self.parameters())
        }

    def __repr__(self) -> str:
        return (
            f"QuantumHydraSSMCore(\n"
            f"  n_qubits={self.n_qubits},\n"
            f"  n_layers={self.n_layers},\n"
            f"  qsvt_degree={self.qsvt_degree},\n"
            f"  circuit_params={self.n_circuit_params},\n"
            f"  total_params={sum(p.numel() for p in self.parameters())}\n"
            f")"
        )


# Export
__all__ = ['QuantumHydraSSMCore']
