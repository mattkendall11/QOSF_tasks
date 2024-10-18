import random
from qiskit import QuantumCircuit,\
    transpile
import numpy as np
def apply_random_pauli(circuit, qubit):
    """Applies a random Pauli operator (X, Y, or Z) to the specified qubit."""
    pauli_gate = random.choice(['X', 'Y', 'Z'])
    if pauli_gate == 'X':
        circuit.x(qubit)
    elif pauli_gate == 'Y':
        circuit.y(qubit)
    elif pauli_gate == 'Z':
        circuit.z(qubit)
    return circuit

def add_pauli_noise(circuit, p1, p2):
    """Adds random Pauli noise after each gate in the quantum circuit.

    Args:
        circuit (QuantumCircuit): Quantum circuit where noise will be added.
        p1 (float): Probability of noise after a single-qubit gate.
        p2 (float): Probability of noise after a two-qubit gate.

    Returns:
        QuantumCircuit: Noisy quantum circuit.
    """
    # Create a new circuit with the same qubit and classical bit registers
    noisy_circuit = QuantumCircuit(circuit.num_qubits, circuit.num_clbits)

    # Iterate over the instructions (gates) in the original circuit
    for instr, qargs, cargs in circuit.data:
        noisy_circuit.append(instr, qargs, cargs)

        # Check if it's a single-qubit gate
        if len(qargs) == 1:
            if random.random() < p1:
                noisy_circuit = apply_random_pauli(noisy_circuit, qargs[0])

        # Check if it's a two-qubit gate
        elif len(qargs) == 2:
            if random.random() < p2:
                noisy_circuit = apply_random_pauli(noisy_circuit, qargs[0])
                noisy_circuit = apply_random_pauli(noisy_circuit, qargs[1])
    noisy_circuit.measure(range(circuit.num_qubits), range(circuit.num_qubits))
    return noisy_circuit


def qft(circuit, n):
    """Apply Quantum Fourier Transform (QFT) to the first n qubits in the circuit, with debugging output."""
    print("Applying QFT to the circuit")
    for i in range(n):
        # Apply Hadamard to the i-th qubit
        print(f"Applying Hadamard gate to qubit {i}")
        circuit.h(i)

        # Apply controlled-phase rotations
        for j in range(i + 1, n):
            angle = np.pi / 2 ** (j - i)
            print(f"Applying controlled-phase rotation (CP) with angle {angle} between qubits {j} and {i}")
            circuit.cp(angle, j, i)

    # Reverse the qubits at the end to match standard QFT order
    print(f"Applying SWAP gates to reverse the qubit order for QFT")
    for i in range(n // 2):
        circuit.swap(i, n - i - 1)


def inverse_qft(circuit, n):
    """Apply inverse Quantum Fourier Transform (QFT†) to the first n qubits in the circuit, with debugging output."""
    print("Applying Inverse QFT to the circuit")

    # Reverse the swap operations
    print(f"Reversing SWAP gates applied during QFT")
    for i in range(n // 2):
        circuit.swap(i, n - i - 1)

    # Apply the inverse QFT (same as QFT but with inverse phases)
    for i in range(n):
        for j in range(i + 1, n):
            angle = -np.pi / 2 ** (j - i)
            print(f"Applying inverse controlled-phase rotation (CP†) with angle {angle} between qubits {j} and {i}")
            circuit.cp(angle, j, i)
        print(f"Applying Hadamard gate to qubit {i}")
        circuit.h(i)

def decompose_to_basis(circuit: QuantumCircuit) -> QuantumCircuit:
    """
    Decomposes a quantum circuit into a specified basis set of gates.

    Args:
        circuit (QuantumCircuit): The input quantum circuit to decompose.

    Returns:
        QuantumCircuit: The decomposed circuit in the target gate basis.
    """
    # Define the target gate basis
    basis_gates = {'cx', 'id', 'rz', 'sx', 'x', 'cp', 'p', 'measure'}  # Added cp, p, measure

    # Initialize a new circuit with the same registers
    decomposed_circuit = QuantumCircuit(*circuit.qregs, *circuit.cregs)

    # Mapping of gate names to their manual decompositions
    decomposition_map = {
        'h': lambda qc, q: [qc.rz(np.pi / 2, q), qc.sx(q), qc.rz(np.pi / 2, q)],
        'z': lambda qc, q: [qc.rz(np.pi, q)],
        'y': lambda qc, q: [qc.rz(np.pi, q), qc.sx(q), qc.rz(np.pi, q)],
        't': lambda qc, q: [qc.rz(np.pi / 4, q)],
        'tdg': lambda qc, q: [qc.rz(-np.pi / 4, q)],
        'swap': lambda qc, q: [qc.cx(q[0], q[1]), qc.cx(q[1], q[0]), qc.cx(q[0], q[1])]
    }

    # Iterate over each gate in the original circuit
    for gate, qargs, cargs in circuit.data:
        # If the gate is already in the basis, append it directly
        if gate.name in basis_gates:
            decomposed_circuit.append(gate, qargs, cargs)

        # If the gate is in the manual decomposition map, apply the decomposition
        elif gate.name in decomposition_map:
            for op in decomposition_map[gate.name](decomposed_circuit, qargs):
                pass  # Each operation is appended within the lambda function

        # Otherwise, try Qiskit's built-in decomposition
        else:
            try:
                decomposed_gate = gate.decompose()
                decomposed_circuit.append(decomposed_gate, qargs, cargs)
            except AttributeError:
                print(f"Warning: Gate '{gate.name}' could not be decomposed.")

    return decomposed_circuit



def quantum_sum(a, b, num_qubits):
    """
    Add two numbers `a` and `b` using the Draper adder algorithm.
    This function assumes that the binary representations of `a` and `b`
    fit within `num_qubits` qubits.
    """
    # Ensure we have enough qubits for both numbers and carry bits
    total_qubits = num_qubits  # You only need num_qubits for both numbers in this case
    circuit = QuantumCircuit(total_qubits, num_qubits)  # Only num_qubits classical bits for measurement

    # Load the first number 'a' into the circuit
    for i in range(num_qubits):
        if (a >> i) & 1:
            circuit.x(i)

    # Apply QFT to prepare for addition
    qft(circuit, num_qubits)

    # Add the second number 'b' using controlled phase rotations
    for i in range(num_qubits):
        if (b >> i) & 1:
            for j in range(i, num_qubits):
                angle = np.pi / 2**(j - i)
                circuit.p(angle, j)  # Apply phase to qubits holding the result (not shifted by num_qubits)

    # Apply inverse QFT
    inverse_qft(circuit, num_qubits)

    # Measure the output qubits
    circuit.measure(range(num_qubits), range(num_qubits))

    return circuit
