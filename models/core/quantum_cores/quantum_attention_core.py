"""
Quantum Attention Core - QSVT + LCU Timestep Mixing

Implements the quantum attention mechanism using UNIFIED ANSATZ for fair ablation.

Theory:
-------
Quantum Attention Mechanism:
    1. Per-timestep quantum circuit: unified ansatz transforms |ψ⟩ → U(θ_t)|ψ⟩
    2. LCU (Linear Combination of Unitaries): Mixes timesteps with learned coefficients
    3. QSVT polynomial: P(U) = Σ_k c_k U^k for polynomial state transformation
    4. Final measurement: Multi-observable ⟨X⟩, ⟨Y⟩, ⟨Z⟩

The key mechanism is TIMESTEP ATTENTION via LCU:
    |ψ_mixed⟩ = Σ_t α_t U_t|ψ⟩

Where α_t are learned complex coefficients that weight each timestep's contribution.
This is analogous to attention weights in classical transformers.

UNIFIED ANSATZ (consistent with QuantumFeatureExtractor):
    Structure per layer:
    1. RX, RY, RZ on all qubits (full single-qubit expressivity)
    2. CRX forward entanglement (i → i+1, circular)
    3. CRX backward entanglement (i → i-1, circular)

    Parameters per layer: 5 × n_qubits
    - 3 × n_qubits for RX, RY, RZ rotations
    - 2 × n_qubits for forward and backward CRX

Key Differences from SSM:
- SSM: Sequential state evolution h[t] = f(h[t-1], x[t])
- Attention: Global timestep mixing via LCU |ψ_mixed⟩ = Σ_t α_t U_t|ψ⟩

Author: Junghoon Park
Date: December 2024
"""

import torch
import torch.nn as nn
import pennylane as qml
import numpy as np
from math import log2
from typing import List, Optional


def unified_ansatz_circuit(params: torch.Tensor, wires: int, layers: int = 1):
    """
    UNIFIED ANSATZ circuit consistent with QuantumFeatureExtractor.

    This ensures all quantum models use the SAME circuit structure for fair ablation.

    Structure per layer:
    1. RX, RY, RZ on all qubits (full single-qubit expressivity)
    2. CRX forward entanglement (i → i+1, circular)
    3. CRX backward entanglement (i → i-1, circular)

    Parameters per layer: 5 × n_qubits
    - 3 × n_qubits for RX, RY, RZ rotations
    - 2 × n_qubits for forward and backward CRX

    Args:
        params: Circuit parameters, shape (n_params,) or (batch, n_params)
        wires: Number of qubits
        layers: Number of ansatz layers
    """
    is_batched = params.ndim == 2
    param_idx = 0

    for _ in range(layers):
        # 1. Full single-qubit rotations: RX, RY, RZ
        for i in range(wires):
            angle_x = params[:, param_idx] if is_batched else params[param_idx]
            angle_y = params[:, param_idx + 1] if is_batched else params[param_idx + 1]
            angle_z = params[:, param_idx + 2] if is_batched else params[param_idx + 2]
            qml.RX(angle_x, wires=i)
            qml.RY(angle_y, wires=i)
            qml.RZ(angle_z, wires=i)
            param_idx += 3

        # 2. Forward CRX entanglement (i → i+1, circular)
        for i in range(wires):
            angle = params[:, param_idx] if is_batched else params[param_idx]
            qml.CRX(angle, wires=[i, (i + 1) % wires])
            param_idx += 1

        # 3. Backward CRX entanglement (i → i-1, circular)
        for i in range(wires - 1, -1, -1):
            angle = params[:, param_idx] if is_batched else params[param_idx]
            qml.CRX(angle, wires=[i, (i - 1) % wires])
            param_idx += 1


# Alias for backward compatibility
sim14_circuit = unified_ansatz_circuit


class QuantumAttentionCore(nn.Module):
    """
    Quantum Attention Core using QSVT + LCU for timestep mixing.

    This module implements the quantum attention mechanism using the
    UNIFIED ANSATZ for fair ablation comparison.

    Key Components:
    1. Timestep QNode: Applies unified ansatz circuit to transform quantum state
    2. LCU Mixing: Linear Combination of Unitaries across timesteps
    3. QSVT Polynomial: Polynomial state preparation P(U)|ψ⟩
    4. QFF QNode: Final quantum feed-forward with multi-observable measurement

    The attention mechanism:
    - Each timestep t has parameters θ_t from the encoder
    - LCU coefficients α_t weight each timestep's contribution
    - QSVT polynomial coefficients c_k define the transformation polynomial
    - Final state: P(Σ_t α_t U(θ_t))|0⟩

    UNIFIED ANSATZ (5 × n_qubits params per layer):
    - RX, RY, RZ rotations (3 × n_qubits)
    - Forward CRX entanglement (n_qubits)
    - Backward CRX entanglement (n_qubits)

    Args:
        n_qubits: Number of qubits in the circuit
        n_timesteps: Sequence length (for LCU coefficients)
        n_layers: Number of unified ansatz layers
        qsvt_degree: Degree of QSVT polynomial
        device: Device for computation

    Output Dimension: 3 × n_qubits (from PauliX, PauliY, PauliZ measurements)

    Example:
        core = QuantumAttentionCore(n_qubits=4, n_timesteps=200, n_layers=2)
        timestep_params = torch.randn(32, 200, 40)  # (batch, seq_len, 5*4*2=40)
        output = core(timestep_params)  # (batch, 12) = 3 × 4 qubits
    """

    def __init__(
        self,
        n_qubits: int = 4,
        n_timesteps: int = 200,
        n_layers: int = 2,
        qsvt_degree: int = 3,
        device: str = "cpu"
    ):
        super().__init__()

        self.n_qubits = n_qubits
        self.n_timesteps = n_timesteps
        self.n_layers = n_layers
        self.qsvt_degree = qsvt_degree
        self.torch_device = device

        # Output dimension: 3 × n_qubits (X, Y, Z measurements)
        self.output_dim = 3 * n_qubits

        # Number of rotations per timestep: 5 × n_qubits × n_layers (unified ansatz)
        # 3 × n_qubits (RX, RY, RZ) + 2 × n_qubits (forward + backward CRX)
        self.n_rots = 5 * n_qubits * n_layers

        # QFF (Quantum Feed-Forward) parameters: 5 × n_qubits × 1 layer
        self.qff_n_rots = 5 * n_qubits * 1

        # ============================================
        # Trainable Quantum Parameters
        # ============================================

        # QSVT polynomial coefficients
        # P(U) = Σ_k c_k U^k
        self.n_poly_coeffs = qsvt_degree + 1
        self.poly_coeffs = nn.Parameter(torch.rand(self.n_poly_coeffs))

        # LCU mixing coefficients (complex for quantum superposition)
        # |ψ_mixed⟩ = Σ_t α_t U_t|ψ⟩
        self.mix_coeffs = nn.Parameter(
            torch.rand(n_timesteps, dtype=torch.complex64)
        )

        # QFF (Quantum Feed-Forward) parameters for final transformation
        self.qff_params = nn.Parameter(torch.rand(self.qff_n_rots))

        # ============================================
        # Setup Quantum Device and QNodes
        # ============================================
        self._setup_quantum_device(device)
        self._build_qnodes()

    def _setup_quantum_device(self, device: str):
        """Setup PennyLane quantum device."""
        # Use default.qubit for state vector manipulation (required for LCU)
        # This matches the original QuantumTSTransformer
        self.dev = qml.device("default.qubit", wires=self.n_qubits)

    def _build_qnodes(self):
        """Build quantum nodes for attention mechanism."""

        n_qubits = self.n_qubits
        n_layers = self.n_layers

        # ============================================
        # Timestep State QNode
        # Transforms initial state using unified ansatz circuit
        # Returns full quantum state for LCU manipulation
        # ============================================
        @qml.qnode(self.dev, interface="torch", diff_method="backprop")
        def timestep_state_qnode(initial_state, params):
            """
            Apply unified ansatz circuit to initial state.

            Args:
                initial_state: (2^n_qubits,) complex state vector
                params: (n_rots,) or (batch, n_rots) circuit parameters

            Returns:
                Evolved quantum state vector
            """
            qml.StatePrep(initial_state, wires=range(n_qubits))
            unified_ansatz_circuit(params, wires=n_qubits, layers=n_layers)
            return qml.state()

        self.timestep_state_qnode = timestep_state_qnode

        # ============================================
        # QFF (Quantum Feed-Forward) QNode
        # Final transformation with multi-observable measurement
        # ============================================
        @qml.qnode(self.dev, interface="torch", diff_method="backprop")
        def qff_qnode_expval(initial_state, params):
            """
            Apply final quantum feed-forward and measure.

            Args:
                initial_state: (2^n_qubits,) normalized state vector
                params: (qff_n_rots,) QFF circuit parameters

            Returns:
                List of 3 × n_qubits expectation values [⟨X⟩, ⟨Y⟩, ⟨Z⟩]
            """
            qml.StatePrep(initial_state, wires=range(n_qubits))
            unified_ansatz_circuit(params, wires=n_qubits, layers=1)

            # Multi-observable measurement (same as original)
            observables = []
            for i in range(n_qubits):
                observables.append(qml.expval(qml.PauliX(i)))
            for i in range(n_qubits):
                observables.append(qml.expval(qml.PauliY(i)))
            for i in range(n_qubits):
                observables.append(qml.expval(qml.PauliZ(i)))

            return observables

        self.qff_qnode_expval = qff_qnode_expval

    def _apply_unitaries_lcu(
        self,
        base_states: torch.Tensor,
        unitary_params: torch.Tensor,
        coeffs: torch.Tensor
    ) -> torch.Tensor:
        """
        Apply Linear Combination of Unitaries (LCU) across timesteps.

        This is the CORE ATTENTION MECHANISM:
        |ψ_mixed⟩ = Σ_t α_t U(θ_t)|ψ⟩

        Vectorized implementation for efficiency.

        Args:
            base_states: (batch, 2^n_qubits) initial quantum states
            unitary_params: (batch, n_timesteps, n_rots) per-timestep parameters
            coeffs: (batch, n_timesteps) complex LCU coefficients

        Returns:
            (batch, 2^n_qubits) LCU-mixed quantum states
        """
        bsz, n_timesteps, n_rots = unitary_params.shape
        n_qbs = int(log2(base_states.shape[1]))

        # Flatten timesteps into batch dimension for efficient QNode execution
        flat_params = unitary_params.reshape(bsz * n_timesteps, n_rots)
        repeated_base_states = base_states.repeat_interleave(n_timesteps, dim=0)

        # Execute QNode ONCE with entire batch (vectorized)
        evolved_states = self.timestep_state_qnode(
            initial_state=repeated_base_states,
            params=flat_params
        )

        # Reshape back to (batch, n_timesteps, 2^n_qubits)
        evolved_states_reshaped = evolved_states.reshape(bsz, n_timesteps, 2**n_qbs)

        # Ensure complex dtype for quantum state manipulation
        evolved_states_reshaped = evolved_states_reshaped.to(torch.complex64)
        coeffs = coeffs.to(torch.complex64)

        # Apply LCU: Σ_t α_t |ψ_t⟩
        # einsum: for each batch, sum over timesteps weighted by coefficients
        lcu_states = torch.einsum('bti,bt->bi', evolved_states_reshaped, coeffs)

        return lcu_states

    def _evaluate_qsvt_polynomial(
        self,
        base_states: torch.Tensor,
        unitary_params: torch.Tensor,
        lcu_coeffs: torch.Tensor,
        poly_coeffs: torch.Tensor
    ) -> torch.Tensor:
        """
        Evaluate QSVT polynomial state preparation.

        P(U)|ψ⟩ = Σ_k c_k U^k |ψ⟩

        where U represents the LCU operation.

        Args:
            base_states: (batch, 2^n_qubits) initial states
            unitary_params: (batch, n_timesteps, n_rots) circuit parameters
            lcu_coeffs: (batch, n_timesteps) LCU mixing coefficients
            poly_coeffs: (n_poly_coeffs,) polynomial coefficients

        Returns:
            (batch, 2^n_qubits) polynomial-transformed states
        """
        # Initialize accumulator with c_0 * |ψ⟩
        acc = poly_coeffs[0] * base_states
        working_register = base_states

        # Accumulate polynomial terms: c_k * U^k |ψ⟩
        for c in poly_coeffs[1:]:
            working_register = self._apply_unitaries_lcu(
                working_register, unitary_params, lcu_coeffs
            )
            acc = acc + c * working_register

        # Normalize by polynomial coefficient norm
        return acc / torch.linalg.vector_norm(poly_coeffs, ord=1)

    def forward(self, timestep_params: torch.Tensor) -> torch.Tensor:
        """
        Forward pass: Quantum attention via QSVT + LCU.

        Flow:
        1. Initialize |0⟩^⊗n base state
        2. Apply QSVT polynomial with LCU timestep mixing
        3. Normalize the mixed state
        4. Apply QFF (Quantum Feed-Forward) transformation
        5. Measure multi-observable ⟨X⟩, ⟨Y⟩, ⟨Z⟩

        Args:
            timestep_params: (batch, n_timesteps, n_rots) from encoder
                            These are the per-timestep circuit parameters
                            scaled to [0, 1] by sigmoid

        Returns:
            (batch, 3 * n_qubits) multi-observable measurements
        """
        bsz = timestep_params.shape[0]
        device = timestep_params.device

        # ============================================
        # STEP 1: Initialize |0⟩^⊗n base state
        # ============================================
        base_states = torch.zeros(
            bsz, 2 ** self.n_qubits,
            dtype=torch.complex64,
            device=device
        )
        base_states[:, 0] = 1.0  # |00...0⟩

        # ============================================
        # STEP 2: QSVT Polynomial with LCU Mixing
        # This is the QUANTUM ATTENTION mechanism
        # |ψ_mixed⟩ = P(Σ_t α_t U(θ_t))|0⟩
        # ============================================
        mixed_state = self._evaluate_qsvt_polynomial(
            base_states,
            timestep_params,
            self.mix_coeffs.repeat(bsz, 1),
            self.poly_coeffs
        )

        # ============================================
        # STEP 3: Normalize the mixed state
        # ============================================
        norm = torch.linalg.vector_norm(mixed_state, dim=1, keepdim=True)
        normalized_mixed_state = mixed_state / (norm + 1e-9)

        # ============================================
        # STEP 4: QFF (Quantum Feed-Forward)
        # Final transformation before measurement
        # ============================================
        # ============================================
        # STEP 5: Multi-Observable Measurement
        # ⟨X⟩, ⟨Y⟩, ⟨Z⟩ on each qubit
        # ============================================
        expvals = self.qff_qnode_expval(
            initial_state=normalized_mixed_state,
            params=self.qff_params
        )

        # Stack and convert to float
        output = torch.stack(expvals, dim=1)
        output = output.float()

        return output

    def get_param_count(self) -> dict:
        """Return parameter counts by component."""
        return {
            'poly_coeffs (QSVT)': self.poly_coeffs.numel(),
            'mix_coeffs (LCU attention)': self.mix_coeffs.numel() * 2,  # complex
            'qff_params (feed-forward)': self.qff_params.numel(),
            'total': sum(p.numel() for p in self.parameters())
        }

    def __repr__(self) -> str:
        return (
            f"QuantumAttentionCore(\n"
            f"  n_qubits={self.n_qubits},\n"
            f"  n_timesteps={self.n_timesteps},\n"
            f"  n_layers={self.n_layers},\n"
            f"  qsvt_degree={self.qsvt_degree},\n"
            f"  n_rots_per_timestep={self.n_rots},\n"
            f"  mechanism='QSVT + LCU timestep mixing',\n"
            f"  measurement='multi-observable (X,Y,Z)',\n"
            f"  output_dim={self.output_dim},\n"
            f"  total_params={sum(p.numel() for p in self.parameters())}\n"
            f")"
        )


# Export
__all__ = ['QuantumAttentionCore', 'unified_ansatz_circuit', 'sim14_circuit']
