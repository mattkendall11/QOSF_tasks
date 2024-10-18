import random
from qiskit import QuantumCircuit, transpile
from qiskit_aer import Aer
import numpy as np
from utils.noise import *

# Your functions here (apply_random_pauli, add_pauli_noise, qft, inverse_qft, decompose_to_basis, quantum_sum)

def main():
    # Create a simple quantum circuit for testing
    num_qubits = 4
    circuit = QuantumCircuit(num_qubits, num_qubits)

    # Add a few gates to the circuit
    circuit.h(0)
    circuit.cx(0, 1)
    circuit.rz(np.pi / 2, 2)
    circuit.measure(range(num_qubits), range(num_qubits))

    print("Original Circuit:")
    print(circuit)

    # Apply random Pauli noise
    p1 = 0.1  # Probability of single-qubit noise
    p2 = 0.2  # Probability of two-qubit noise
    noisy_circuit = add_pauli_noise(circuit, p1, p2)
    print("\nCircuit after Adding Pauli Noise:")
    print(noisy_circuit)

    # Apply QFT
    qft_circuit = QuantumCircuit(num_qubits)
    qft(qft_circuit, num_qubits)
    print("\nQFT Circuit:")
    print(qft_circuit)

    # Apply Inverse QFT
    inverse_qft_circuit = QuantumCircuit(num_qubits)
    inverse_qft(inverse_qft_circuit, num_qubits)
    print("\nInverse QFT Circuit:")
    print(inverse_qft_circuit)

    # Decompose the original circuit
    decomposed_circuit = decompose_to_basis(circuit)
    print("\nDecomposed Circuit:")
    print(decomposed_circuit)

    # Test the quantum sum function
    a = 0  # Example number 1
    b = 0  # Example number 2
    sum_circuit = quantum_sum(a, b, num_qubits)
    print("\nQuantum Sum Circuit:")
    print(sum_circuit)

    # Execute the circuit using Aer simulator
    backend = Aer.get_backend('aer_simulator')
    circ = transpile(circuit, backend)

    job = backend.run(circ, shots=1000)  # Run with multiple shots to get a distribution
    result = job.result()
    counts = result.get_counts()
    print("\nResult of Quantum Sum:")
    print(counts)


if __name__ == "__main__":
    main()
