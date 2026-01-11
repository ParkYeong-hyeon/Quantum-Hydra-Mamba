import torch
import torch.nn as nn
import pennylane as qml
from math import log2
from typing import Optional, List

# ================================================================================
# Quantum Circuit Building Blocks
# ================================================================================

def basic_ansatz_circuit(params, wires, layers=1):
    """
    Basic parametrized quantum circuit for encoding and processing.
    Similar to sim14_circuit but simplified for modular use.

    Args:
        params: Tensor of rotation angles (batch_size, n_params) or (n_params,)
        wires: Number of qubits
        layers: Number of circuit repetitions
    """
    is_batched = params.ndim == 2
    param_idx = 0

    for _ in range(layers):
        # Layer 1: Single-qubit rotations (RY)
        for i in range(wires):
            angle = params[:, param_idx] if is_batched else params[param_idx]
            qml.RY(angle, wires=i)
            param_idx += 1

        # Layer 2: Entangling gates (CRX forward)
        for i in range(wires):
            angle = params[:, param_idx] if is_batched else params[param_idx]
            qml.CRX(angle, wires=[i, (i + 1) % wires])
            param_idx += 1

        # Layer 3: Single-qubit rotations (RY)
        for i in range(wires):
            angle = params[:, param_idx] if is_batched else params[param_idx]
            qml.RY(angle, wires=i)
            param_idx += 1

        # Layer 4: Entangling gates (CRX backward)
        for i in range(wires - 1, -1, -1):
            angle = params[:, param_idx] if is_batched else params[param_idx]
            qml.CRX(angle, wires=[i, (i - 1) % wires])
            param_idx += 1


def quantum_shift_circuit(wires, shift_amount=1):
    """
    Implements quantum cyclic shift operation.
    Shifts quantum state amplitudes by 'shift_amount' positions to the right.

    Args:
        wires: Number of qubits
        shift_amount: How many positions to shift (default: 1)
    """
    # For computational basis states |x⟩, this shifts x → (x + shift_amount) mod 2^n
    # Implemented via SWAP gates in a cyclic pattern
    for _ in range(shift_amount):
        for i in range(wires - 1):
            qml.SWAP(wires=[i, i + 1])


def quantum_flip_circuit(wires):
    """
    Implements quantum flip (reversal) operation.
    Reverses the order of qubits, effectively flipping the sequence.

    Args:
        wires: Number of qubits
    """
    # Reverse qubit ordering: |q0 q1 q2 ... qn⟩ → |qn ... q2 q1 q0⟩
    n = wires
    for i in range(n // 2):
        qml.SWAP(wires=[i, n - 1 - i])


# ================================================================================
# QLCU: Quantum Semi-Separable Matrix Simulation via Parametrized Circuit
# ================================================================================

def qlcu_circuit(params, wires, layers=1):
    """
    Quantum Linear Combination of Unitaries circuit.
    This simulates the semi-separable matrix SS(X) from classical Hydra.

    Uses a strongly entangling parametrized ansatz to create complex
    matrix-like transformations on the quantum state.

    Args:
        params: Tensor of rotation angles
        wires: Number of qubits
        layers: Circuit depth
    """
    is_batched = params.ndim == 2
    param_idx = 0

    for _ in range(layers):
        # Block 1: Y-rotations
        for i in range(wires):
            angle = params[:, param_idx] if is_batched else params[param_idx]
            qml.RY(angle, wires=i)
            param_idx += 1

        # Block 2: Ising XX interactions (strong entanglement)
        for i in range(wires):
            angle = params[:, param_idx] if is_batched else params[param_idx]
            qml.IsingXX(angle, wires=[i, (i + 1) % wires])
            param_idx += 1

        # Block 3: Y-rotations
        for i in range(wires):
            angle = params[:, param_idx] if is_batched else params[param_idx]
            qml.RY(angle, wires=i)
            param_idx += 1

        # Block 4: Ising YY interactions
        for i in range(wires):
            angle = params[:, param_idx] if is_batched else params[param_idx]
            qml.IsingYY(angle, wires=[i, (i + 1) % wires])
            param_idx += 1


# ================================================================================
# QD: Quantum Diagonal Operation
# ================================================================================

def qd_circuit(params, wires):
    """
    Quantum diagonal matrix operation QD.
    Applies independent single-qubit rotations (no entanglement).
    Corresponds to diagonal matrix D in classical Hydra.

    Args:
        params: Tensor of rotation angles (3 * wires parameters)
        wires: Number of qubits
    """
    is_batched = params.ndim == 2
    param_idx = 0

    # Apply RX, RY, RZ to each qubit (fully general single-qubit operation)
    for i in range(wires):
        angle_x = params[:, param_idx] if is_batched else params[param_idx]
        qml.RX(angle_x, wires=i)
        param_idx += 1

    for i in range(wires):
        angle_y = params[:, param_idx] if is_batched else params[param_idx]
        qml.RY(angle_y, wires=i)
        param_idx += 1

    for i in range(wires):
        angle_z = params[:, param_idx] if is_batched else params[param_idx]
        qml.RZ(angle_z, wires=i)
        param_idx += 1


# ================================================================================
# Helper Functions for State Evolution
# ================================================================================

def apply_circuit_to_state(initial_states, params, qnode_func):
    """
    Apply a parametrized quantum circuit to a batch of initial states.

    Args:
        initial_states: Tensor (batch_size, 2^n_qubits) - quantum state vectors
        params: Tensor (batch_size, n_params) - circuit parameters
        qnode_func: PennyLane QNode that applies the circuit

    Returns:
        evolved_states: Tensor (batch_size, 2^n_qubits) - output states
    """
    evolved_states = qnode_func(initial_state=initial_states, params=params)
    return evolved_states


# ================================================================================
# Visualization Functions
# ================================================================================

def draw_qshift_circuit(n_qubits=4, shift_amount=1, filename="qshift_circuit.pdf"):
    """
    Draws the quantum shift circuit and saves to PDF.

    Args:
        n_qubits: Number of qubits
        shift_amount: Number of positions to shift
        filename: Output filename
    """
    import matplotlib.pyplot as plt

    print(f"Generating Qshift circuit diagram ({n_qubits} qubits, shift={shift_amount})...")

    dev = qml.device("default.qubit", wires=n_qubits)

    @qml.qnode(dev, interface="torch")
    def circuit_to_draw():
        # Initialize to some superposition for visualization
        for i in range(n_qubits):
            qml.Hadamard(wires=i)
        quantum_shift_circuit(wires=n_qubits, shift_amount=shift_amount)
        return qml.state()

    fig, ax = qml.draw_mpl(circuit_to_draw, style="black_white")()
    fig.suptitle(f"Quantum Shift Circuit ({n_qubits} Qubits, Shift={shift_amount})", fontsize=14)

    try:
        plt.savefig(filename, format="pdf", bbox_inches="tight")
        print(f"Successfully saved to '{filename}'")
    except Exception as e:
        print(f"Error saving file: {e}")
    plt.close(fig)


def draw_qflip_circuit(n_qubits=4, filename="qflip_circuit.pdf"):
    """
    Draws the quantum flip circuit and saves to PDF.

    Args:
        n_qubits: Number of qubits
        filename: Output filename
    """
    import matplotlib.pyplot as plt

    print(f"Generating Qflip circuit diagram ({n_qubits} qubits)...")

    dev = qml.device("default.qubit", wires=n_qubits)

    @qml.qnode(dev, interface="torch")
    def circuit_to_draw():
        # Initialize to some superposition
        for i in range(n_qubits):
            qml.Hadamard(wires=i)
        quantum_flip_circuit(wires=n_qubits)
        return qml.state()

    fig, ax = qml.draw_mpl(circuit_to_draw, style="black_white")()
    fig.suptitle(f"Quantum Flip Circuit ({n_qubits} Qubits)", fontsize=14)

    try:
        plt.savefig(filename, format="pdf", bbox_inches="tight")
        print(f"Successfully saved to '{filename}'")
    except Exception as e:
        print(f"Error saving file: {e}")
    plt.close(fig)


def draw_qlcu_circuit(n_qubits=4, qlcu_layers=2, filename="qlcu_circuit.pdf"):
    """
    Draws the QLCU (semi-separable simulation) circuit and saves to PDF.

    Args:
        n_qubits: Number of qubits
        qlcu_layers: Circuit depth
        filename: Output filename
    """
    import matplotlib.pyplot as plt

    print(f"Generating QLCU circuit diagram ({n_qubits} qubits, {qlcu_layers} layers)...")

    dev = qml.device("default.qubit", wires=n_qubits)
    n_params = 4 * n_qubits * qlcu_layers

    @qml.qnode(dev, interface="torch")
    def circuit_to_draw(params):
        qlcu_circuit(params, wires=n_qubits, layers=qlcu_layers)
        return qml.state()

    dummy_params = torch.randn(n_params)

    fig, ax = qml.draw_mpl(circuit_to_draw, style="black_white")(dummy_params)
    fig.suptitle(f"QLCU Circuit ({n_qubits} Qubits, {qlcu_layers} Layers)", fontsize=14)

    try:
        plt.savefig(filename, format="pdf", bbox_inches="tight")
        print(f"Successfully saved to '{filename}'")
    except Exception as e:
        print(f"Error saving file: {e}")
    plt.close(fig)


def draw_qd_circuit(n_qubits=4, filename="qd_circuit.pdf"):
    """
    Draws the QD (diagonal operation) circuit and saves to PDF.

    Args:
        n_qubits: Number of qubits
        filename: Output filename
    """
    import matplotlib.pyplot as plt

    print(f"Generating QD circuit diagram ({n_qubits} qubits)...")

    dev = qml.device("default.qubit", wires=n_qubits)
    n_params = 3 * n_qubits

    @qml.qnode(dev, interface="torch")
    def circuit_to_draw(params):
        qd_circuit(params, wires=n_qubits)
        return qml.state()

    dummy_params = torch.randn(n_params)

    fig, ax = qml.draw_mpl(circuit_to_draw, style="black_white")(dummy_params)
    fig.suptitle(f"QD (Diagonal) Circuit ({n_qubits} Qubits)", fontsize=14)

    try:
        plt.savefig(filename, format="pdf", bbox_inches="tight")
        print(f"Successfully saved to '{filename}'")
    except Exception as e:
        print(f"Error saving file: {e}")
    plt.close(fig)


def draw_full_hydra_branch(branch_type="psi1", n_qubits=4, qlcu_layers=2,
                          shift_amount=1, filename="hydra_branch.pdf"):
    """
    Draws one complete branch of the Quantum Hydra circuit.

    Args:
        branch_type: "psi1", "psi2", or "psi3"
        n_qubits: Number of qubits
        qlcu_layers: QLCU circuit depth
        shift_amount: Shift amount for Qshift
        filename: Output filename
    """
    import matplotlib.pyplot as plt

    print(f"Generating Quantum Hydra {branch_type} circuit...")

    dev = qml.device("default.qubit", wires=n_qubits)

    if branch_type == "psi1":
        # |ψ₁⟩ = Qshift(QLCU|X⟩)
        n_params = 4 * n_qubits * qlcu_layers

        @qml.qnode(dev, interface="torch")
        def circuit_to_draw(params):
            qlcu_circuit(params, wires=n_qubits, layers=qlcu_layers)
            quantum_shift_circuit(wires=n_qubits, shift_amount=shift_amount)
            return qml.state()

        dummy_params = torch.randn(n_params)
        fig, ax = qml.draw_mpl(circuit_to_draw, style="black_white")(dummy_params)
        title = f"Hydra Branch ψ₁: Qshift(QLCU|X⟩)"

    elif branch_type == "psi2":
        # |ψ₂⟩ = Qflip(Qshift(QLCU(Qflip|X⟩)))
        n_params = 4 * n_qubits * qlcu_layers

        @qml.qnode(dev, interface="torch")
        def circuit_to_draw(params):
            quantum_flip_circuit(wires=n_qubits)
            qlcu_circuit(params, wires=n_qubits, layers=qlcu_layers)
            quantum_shift_circuit(wires=n_qubits, shift_amount=shift_amount)
            quantum_flip_circuit(wires=n_qubits)
            return qml.state()

        dummy_params = torch.randn(n_params)
        fig, ax = qml.draw_mpl(circuit_to_draw, style="black_white")(dummy_params)
        title = f"Hydra Branch ψ₂: Qflip(Qshift(QLCU(Qflip|X⟩)))"

    elif branch_type == "psi3":
        # |ψ₃⟩ = QD|X⟩
        n_params = 3 * n_qubits

        @qml.qnode(dev, interface="torch")
        def circuit_to_draw(params):
            qd_circuit(params, wires=n_qubits)
            return qml.state()

        dummy_params = torch.randn(n_params)
        fig, ax = qml.draw_mpl(circuit_to_draw, style="black_white")(dummy_params)
        title = f"Hydra Branch ψ₃: QD|X⟩"

    else:
        raise ValueError(f"Unknown branch_type: {branch_type}")

    fig.suptitle(title, fontsize=14)

    try:
        plt.savefig(filename, format="pdf", bbox_inches="tight")
        print(f"Successfully saved to '{filename}'")
    except Exception as e:
        print(f"Error saving file: {e}")
    plt.close(fig)


def visualize_all_circuits(n_qubits=4, qlcu_layers=2, shift_amount=1,
                          output_dir="./quantum_hydra_circuits"):
    """
    Generate all circuit diagrams for Quantum Hydra and save to directory.

    Args:
        n_qubits: Number of qubits
        qlcu_layers: QLCU circuit depth
        shift_amount: Shift amount
        output_dir: Directory to save PDFs
    """
    import os

    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    print("=" * 80)
    print("Generating all Quantum Hydra circuit diagrams...")
    print("=" * 80)

    # Individual components
    draw_qshift_circuit(n_qubits, shift_amount,
                       f"{output_dir}/01_qshift_circuit.pdf")
    draw_qflip_circuit(n_qubits,
                      f"{output_dir}/02_qflip_circuit.pdf")
    draw_qlcu_circuit(n_qubits, qlcu_layers,
                     f"{output_dir}/03_qlcu_circuit.pdf")
    draw_qd_circuit(n_qubits,
                   f"{output_dir}/04_qd_circuit.pdf")

    # Full branches
    draw_full_hydra_branch("psi1", n_qubits, qlcu_layers, shift_amount,
                          f"{output_dir}/05_branch_psi1.pdf")
    draw_full_hydra_branch("psi2", n_qubits, qlcu_layers, shift_amount,
                          f"{output_dir}/06_branch_psi2.pdf")
    draw_full_hydra_branch("psi3", n_qubits, qlcu_layers, shift_amount,
                          f"{output_dir}/07_branch_psi3.pdf")

    print("=" * 80)
    print(f"All circuit diagrams saved to: {output_dir}/")
    print("=" * 80)


# ================================================================================
# Main Quantum Hydra Model
# ================================================================================

class QuantumHydraLayer(nn.Module):
    """
    Quantum Hydra Layer - Quantum version of Hydra state-space model.

    Implements the equation:
        |ψ⟩ = α|ψ₁⟩ + β|ψ₂⟩ + γ|ψ₃⟩
    where:
        |ψ₁⟩ = Qshift(QLCU|X⟩)
        |ψ₂⟩ = Qflip(Qshift(QLCU(Qflip|X⟩)))
        |ψ₃⟩ = QD|X⟩

    Output: Y_i = ⟨ψ|M_i|ψ⟩ for observables M_i (PauliX, PauliY, PauliZ)

    Args:
        n_qubits: Number of qubits
        qlcu_layers: Circuit depth for QLCU (semi-separable simulation)
        shift_amount: How many positions to shift in Qshift
        feature_dim: Input feature dimension
        output_dim: Output dimension
        dropout: Dropout probability
        device: torch device
    """

    def __init__(
        self,
        n_qubits: int,
        qlcu_layers: int = 2,
        shift_amount: int = 1,
        feature_dim: int = 128,
        output_dim: int = 1,
        dropout: float = 0.1,
        device: str = "cpu"
    ):
        super().__init__()

        self.n_qubits = n_qubits
        self.qlcu_layers = qlcu_layers
        self.shift_amount = shift_amount
        self.device = device

        # Calculate parameter counts for each circuit type
        # QLCU: 4 gates per layer, each with n_qubits parameters
        self.n_qlcu_params = 4 * n_qubits * qlcu_layers

        # QD: 3 rotations (RX, RY, RZ) per qubit
        self.n_qd_params = 3 * n_qubits

        # Classical layers
        self.feature_projection = nn.Linear(feature_dim, self.n_qlcu_params)
        self.dropout = nn.Dropout(dropout)
        self.activation = nn.Sigmoid()  # Bound parameters to [0, 1], then scale to [0, 2π]

        # Output layer: 3*n_qubits measurements → output_dim
        self.output_ff = nn.Linear(3 * n_qubits, output_dim)

        # Trainable quantum parameters
        # Complex coefficients for LCU combination
        self.alpha = nn.Parameter(torch.rand(1, dtype=torch.complex64))
        self.beta = nn.Parameter(torch.rand(1, dtype=torch.complex64))
        self.gamma = nn.Parameter(torch.rand(1, dtype=torch.complex64))

        # Fixed quantum circuit parameters (could also be made learnable)
        self.qlcu_base_params = nn.Parameter(torch.rand(self.n_qlcu_params))
        self.qd_params = nn.Parameter(torch.rand(self.n_qd_params))

        # PennyLane quantum device - use GPU if available and requested
        if device == "cuda" and torch.cuda.is_available():
            try:
                self.dev = qml.device("lightning.gpu", wires=self.n_qubits)
            except:
                # Fallback if lightning.gpu not installed
                import warnings
                warnings.warn("lightning.gpu not available, using default.qubit (CPU)")
                self.dev = qml.device("default.qubit", wires=self.n_qubits)
        else:
            self.dev = qml.device("default.qubit", wires=self.n_qubits)

        # Define QNodes for each operation
        self._setup_qnodes()

        # Move all parameters to specified device
        self.to(device)

    def _setup_qnodes(self):
        """Setup PennyLane QNodes for quantum operations."""

        # QNode for QLCU operation
        @qml.qnode(self.dev, interface="torch", diff_method="backprop")
        def _qlcu_qnode(initial_state, params):
            qml.StatePrep(initial_state, wires=range(self.n_qubits))
            qlcu_circuit(params, wires=self.n_qubits, layers=self.qlcu_layers)
            return qml.state()

        # QNode for QLCU + Shift
        @qml.qnode(self.dev, interface="torch", diff_method="backprop")
        def _qlcu_shift_qnode(initial_state, params):
            qml.StatePrep(initial_state, wires=range(self.n_qubits))
            qlcu_circuit(params, wires=self.n_qubits, layers=self.qlcu_layers)
            quantum_shift_circuit(wires=self.n_qubits, shift_amount=self.shift_amount)
            return qml.state()

        # QNode for Flip operation
        @qml.qnode(self.dev, interface="torch", diff_method="backprop")
        def _flip_qnode(initial_state):
            qml.StatePrep(initial_state, wires=range(self.n_qubits))
            quantum_flip_circuit(wires=self.n_qubits)
            return qml.state()

        # QNode for Flip + QLCU + Shift + Flip
        @qml.qnode(self.dev, interface="torch", diff_method="backprop")
        def _flip_qlcu_shift_flip_qnode(initial_state, params):
            qml.StatePrep(initial_state, wires=range(self.n_qubits))
            quantum_flip_circuit(wires=self.n_qubits)
            qlcu_circuit(params, wires=self.n_qubits, layers=self.qlcu_layers)
            quantum_shift_circuit(wires=self.n_qubits, shift_amount=self.shift_amount)
            quantum_flip_circuit(wires=self.n_qubits)
            return qml.state()

        # QNode for QD (diagonal) operation
        @qml.qnode(self.dev, interface="torch", diff_method="backprop")
        def _qd_qnode(initial_state, params):
            qml.StatePrep(initial_state, wires=range(self.n_qubits))
            qd_circuit(params, wires=self.n_qubits)
            return qml.state()

        # QNode for measuring observables
        @qml.qnode(self.dev, interface="torch", diff_method="backprop")
        def _measurement_qnode(initial_state):
            qml.StatePrep(initial_state, wires=range(self.n_qubits))
            # Measure PauliX, PauliY, PauliZ on each qubit
            observables = [qml.PauliX(i) for i in range(self.n_qubits)] + \
                          [qml.PauliY(i) for i in range(self.n_qubits)] + \
                          [qml.PauliZ(i) for i in range(self.n_qubits)]
            return [qml.expval(op) for op in observables]

        # Store QNodes
        self.qlcu_qnode = _qlcu_qnode
        self.qlcu_shift_qnode = _qlcu_shift_qnode
        self.flip_qnode = _flip_qnode
        self.flip_qlcu_shift_flip_qnode = _flip_qlcu_shift_flip_qnode
        self.qd_qnode = _qd_qnode
        self.measurement_qnode = _measurement_qnode

    def forward(self, x):
        """
        Forward pass of Quantum Hydra.

        Args:
            x: Input tensor (batch_size, feature_dim)

        Returns:
            output: Tensor (batch_size, output_dim)
        """
        batch_size = x.shape[0]

        # Project input features to quantum circuit parameters
        x_projected = self.feature_projection(self.dropout(x))
        qlcu_params = self.activation(x_projected) * 2 * 3.14159  # Scale to [0, 2π]

        # Add base parameters (learnable bias)
        qlcu_params = qlcu_params + self.qlcu_base_params.unsqueeze(0)

        # Initialize quantum state to |0...0⟩
        base_states = torch.zeros(batch_size, 2 ** self.n_qubits,
                                   dtype=torch.complex64, device=self.device)
        base_states[:, 0] = 1.0  # Set |0⟩ amplitude to 1

        # ====================================================================
        # Compute three quantum branches
        # ====================================================================

        # Branch 1: |ψ₁⟩ = Qshift(QLCU|X⟩)
        psi1 = self.qlcu_shift_qnode(initial_state=base_states, params=qlcu_params)

        # Branch 2: |ψ₂⟩ = Qflip(Qshift(QLCU(Qflip|X⟩)))
        psi2 = self.flip_qlcu_shift_flip_qnode(initial_state=base_states, params=qlcu_params)

        # Branch 3: |ψ₃⟩ = QD|X⟩
        qd_params_expanded = self.qd_params.unsqueeze(0).expand(batch_size, -1)
        psi3 = self.qd_qnode(initial_state=base_states, params=qd_params_expanded)

        # ====================================================================
        # Linear Combination of Unitaries (LCU) with trainable coefficients
        # ====================================================================

        # Ensure complex dtype
        psi1 = psi1.to(torch.complex64)
        psi2 = psi2.to(torch.complex64)
        psi3 = psi3.to(torch.complex64)

        # Combine with trainable coefficients
        psi_combined = self.alpha * psi1 + self.beta * psi2 + self.gamma * psi3

        # Normalize the combined state
        norms = torch.linalg.vector_norm(psi_combined, dim=1, keepdim=True)
        psi_normalized = psi_combined / (norms + 1e-9)

        # ====================================================================
        # Measurement: Compute expectation values
        # ====================================================================

        measurements = self.measurement_qnode(initial_state=psi_normalized)
        measurements = torch.stack(measurements, dim=1)  # (batch_size, 3*n_qubits)
        measurements = measurements.float()  # Convert to real values

        # ====================================================================
        # Classical output layer
        # ====================================================================

        output = self.output_ff(measurements)

        return output


# ================================================================================
# Multi-Timestep Quantum Hydra (for time-series data)
# ================================================================================

class QuantumHydraTS(nn.Module):
    """
    Quantum Hydra for Time-Series data.
    Processes sequential data by treating each timestep with Quantum Hydra,
    similar to QTSTransformer.

    Args:
        n_qubits: Number of qubits
        n_timesteps: Sequence length
        qlcu_layers: QLCU circuit depth
        shift_amount: Shift amount for Qshift
        feature_dim: Feature dimension per timestep
        output_dim: Final output dimension
        dropout: Dropout probability
        device: torch device
    """

    def __init__(
        self,
        n_qubits: int = 6,
        n_timesteps: int = 200,
        qlcu_layers: int = 2,
        shift_amount: int = 1,
        feature_dim: int = 129,  # e.g., EEG channels
        output_dim: int = 1,
        dropout: float = 0.1,
        device: str = "cpu"
    ):
        super().__init__()

        self.n_timesteps = n_timesteps
        self.n_qubits = n_qubits
        self.device = device

        # Feature projection: map each timestep to quantum parameters
        self.n_qlcu_params = 4 * n_qubits * qlcu_layers
        self.feature_projection = nn.Linear(feature_dim, self.n_qlcu_params)
        self.dropout = nn.Dropout(dropout)

        # Single Quantum Hydra layer (shared across timesteps)
        self.quantum_hydra = QuantumHydraLayer(
            n_qubits=n_qubits,
            qlcu_layers=qlcu_layers,
            shift_amount=shift_amount,
            feature_dim=self.n_qlcu_params,
            output_dim=3 * n_qubits,  # Output measurements
            dropout=0.0,  # Already applied dropout
            device=device
        )

        # Temporal aggregation: combine outputs across timesteps
        self.temporal_weights = nn.Parameter(torch.rand(n_timesteps, dtype=torch.complex64))

        # Final output layer
        self.output_layer = nn.Linear(3 * n_qubits, output_dim)

        # Move all parameters to specified device
        self.to(device)

    def forward(self, x):
        """
        Forward pass for time-series input.

        Args:
            x: Input tensor (batch_size, feature_dim, n_timesteps) or
               (batch_size, n_timesteps, feature_dim)

        Returns:
            output: Tensor (batch_size, output_dim)
        """
        batch_size = x.shape[0]

        # Handle different input formats - ensure (batch_size, n_timesteps, feature_dim)
        if x.shape[1] != self.n_timesteps and x.shape[2] == self.n_timesteps:
            # Input is (batch_size, feature_dim, n_timesteps)
            x = x.permute(0, 2, 1)  # Convert to (batch_size, n_timesteps, feature_dim)

        # VECTORIZED PROCESSING: Process all timesteps as a single batch
        # Reshape: (batch_size, n_timesteps, feature_dim) -> (batch_size * n_timesteps, feature_dim)
        x_flat = x.reshape(batch_size * self.n_timesteps, -1)

        # Apply dropout and feature projection to all timesteps at once
        x_proj = self.feature_projection(self.dropout(x_flat))

        # Run quantum layer on all timesteps as a large batch
        out_flat = self.quantum_hydra(x_proj)  # (batch_size * n_timesteps, 3*n_qubits)

        # Reshape back: (batch_size * n_timesteps, 3*n_qubits) -> (batch_size, n_timesteps, 3*n_qubits)
        timestep_outputs = out_flat.reshape(batch_size, self.n_timesteps, -1)

        # Weighted aggregation across timesteps
        weights = self.temporal_weights / torch.sum(torch.abs(self.temporal_weights))
        weights = weights.unsqueeze(0).unsqueeze(-1)  # (1, n_timesteps, 1)

        # Complex-valued weighted sum
        timestep_outputs_complex = timestep_outputs.to(torch.complex64)
        aggregated = torch.sum(timestep_outputs_complex * weights, dim=1)  # (batch_size, 3*n_qubits)
        aggregated = torch.abs(aggregated)  # Take magnitude

        # Final prediction
        output = self.output_layer(aggregated.float())

        return output


# ================================================================================
# Testing and Visualization
# ================================================================================

if __name__ == "__main__":
    print("=" * 80)
    print("Quantum Hydra - Testing")
    print("=" * 80)

    # Test single-sample forward pass
    device = "cpu"
    n_qubits = 4
    feature_dim = 10
    output_dim = 2
    batch_size = 8

    print("\n[1] Testing QuantumHydraLayer...")
    model = QuantumHydraLayer(
        n_qubits=n_qubits,
        qlcu_layers=2,
        shift_amount=1,
        feature_dim=feature_dim,
        output_dim=output_dim,
        dropout=0.1,
        device=device
    )

    x = torch.randn(batch_size, feature_dim)
    output = model(x)
    print(f"  Input shape: {x.shape}")
    print(f"  Output shape: {output.shape}")
    print(f"  Output sample: {output[0]}")

    print("\n[2] Testing QuantumHydraTS (time-series)...")
    n_timesteps = 20
    feature_dim_ts = 64  # e.g., EEG channels

    model_ts = QuantumHydraTS(
        n_qubits=4,
        n_timesteps=n_timesteps,
        qlcu_layers=2,
        shift_amount=1,
        feature_dim=feature_dim_ts,
        output_dim=1,
        dropout=0.1,
        device=device
    )

    x_ts = torch.randn(batch_size, feature_dim_ts, n_timesteps)
    output_ts = model_ts(x_ts)
    print(f"  Input shape: {x_ts.shape}")
    print(f"  Output shape: {output_ts.shape}")
    print(f"  Output sample: {output_ts[0]}")

    print("\n[3] Checking trainable parameters...")
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Total trainable parameters: {total_params:,}")

    print("\n[4] Verifying quantum coefficients...")
    print(f"  Alpha: {model.alpha.data}")
    print(f"  Beta: {model.beta.data}")
    print(f"  Gamma: {model.gamma.data}")

    print("\n[5] Testing visualization functions...")
    print("  Generating all circuit diagrams (this may take a minute)...")
    try:
        visualize_all_circuits(n_qubits=4, qlcu_layers=2, shift_amount=1,
                              output_dir="./quantum_hydra_circuits")
        print("  ✓ Circuit diagrams generated successfully!")
    except Exception as e:
        print(f"  ✗ Error generating circuit diagrams: {e}")

    print("\n" + "=" * 80)
    print("Testing complete!")
    print("=" * 80)
