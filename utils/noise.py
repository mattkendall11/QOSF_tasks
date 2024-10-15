import random
from qiskit import QuantumCircuit, Aer, transpile
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

    return noisy_circuit



def qft(circuit, n):
    """Apply Quantum Fourier Transform (QFT) to the first n qubits in the circuit."""
    for i in range(n):
        # Apply Hadamard to the i-th qubit
        circuit.h(i)
        # Apply controlled-phase rotations
        for j in range(i + 1, n):
            angle = np.pi / 2 ** (j - i)
            circuit.cp(angle, j, i)

    # Reverse the qubits at the end to match standard QFT order
    for i in range(n // 2):
        circuit.swap(i, n - i - 1)


def inverse_qft(circuit, n):
    """Apply inverse Quantum Fourier Transform (QFT†) to the first n qubits in the circuit."""
    # Reverse the swap operations
    for i in range(n // 2):
        circuit.swap(i, n - i - 1)

    # Apply the inverse QFT (same as QFT but with inverse phases)
    for i in range(n):
        for j in range(i + 1, n):
            angle = -np.pi / 2 ** (j - i)
            circuit.cp(angle, j, i)
        circuit.h(i)


def decompose_to_basis(circuit: QuantumCircuit) -> QuantumCircuit:
    basis_gates = ['cx', 'id', 'rz', 'sx', 'x']  # The target gate basis
    decomposed_circuit = QuantumCircuit(*circuit.qregs, *circuit.cregs)

    for gate, qargs, cargs in circuit.data:
        if gate.name == 'h':  # Hadamard gate H
            # Decompose H into RZ, SX, X
            decomposed_circuit.rz(np.pi / 2, qargs[0])
            decomposed_circuit.sx(qargs[0])
            decomposed_circuit.rz(np.pi / 2, qargs[0])

        elif gate.name == 'z':  # Pauli-Z gate
            # Z = RZ(pi)
            decomposed_circuit.rz(np.pi, qargs[0])

        elif gate.name == 'y':  # Pauli-Y gate
            # Y = Z.X.Z
            decomposed_circuit.rz(np.pi, qargs[0])
            decomposed_circuit.sx(qargs[0])
            decomposed_circuit.rz(np.pi, qargs[0])

        elif gate.name == 't':  # T gate
            # T = RZ(pi/4)
            decomposed_circuit.rz(np.pi / 4, qargs[0])

        elif gate.name == 'tdg':  # Tdg gate (inverse of T)
            # Tdg = RZ(-pi/4)
            decomposed_circuit.rz(-np.pi / 4, qargs[0])

        elif gate.name not in basis_gates:
            # Try using Qiskit's built-in decomposition if possible
            try:
                decomposed_gate = gate.decompose()  # Decompose into the basic gate set
                decomposed_circuit.append(decomposed_gate, qargs, cargs)
            except AttributeError:
                print(f"Warning: Could not decompose gate {gate.name}")
        else:
            # If the gate is already in the basis, append it directly
            decomposed_circuit.append(gate, qargs, cargs)

    return decomposed_circuit

def quantum_sum(a, b, num_qubits):
    """
    Add two numbers `a` and `b` using the Draper adder algorithm.
    This function assumes that the binary representations of `a` and `b`
    fit within `num_qubits` qubits.
    """
    # Initialize quantum circuit with enough qubits to hold both numbers
    circuit = QuantumCircuit(num_qubits)

    # Step 1: Initialize qubits in the state representing the number `a`
    # The number `a` is encoded directly in the computational basis
    for i in range(num_qubits):
        if (a >> i) & 1:
            circuit.x(i)

    # Step 2: Apply QFT to the qubits (we'll sum the number `b` in the phase of these qubits)
    qft(circuit, num_qubits)

    # Step 3: Add the number `b` by modifying the phase of the qubits
    # This is done by applying controlled-RZ rotations corresponding to the bits of `b`
    for i in range(num_qubits):
        if (b >> i) & 1:
            # Apply phase shift to the qubits corresponding to 2^i * b
            for j in range(i, num_qubits):
                angle = np.pi / 2**(j - i)
                circuit.p(angle, j)

    # Step 4: Apply inverse QFT to transform the result back to the computational basis
    inverse_qft(circuit, num_qubits)

    return circuit


