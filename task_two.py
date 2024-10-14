import random
from qiskit import QuantumCircuit, Aer, transpile

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

