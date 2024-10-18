from qiskit import QuantumCircuit, transpile
from qiskit_aer import Aer
from qiskit.visualization import plot_histogram
import numpy as np


from utils.noise import add_pauli_noise, decompose_to_basis, quantum_sum

'''
script that takes a draper adder circuit ,decomposes to basis then adds noise
'''

circuit = quantum_sum(2,3,5)

basis_circuit = decompose_to_basis(circuit)

noisy_circuit = add_pauli_noise(basis_circuit, 0,0)

simulator = Aer.get_backend('qasm_simulator')

circ = transpile(circuit, simulator)

job = simulator.run(circ, shots=1000)  # Run with multiple shots to get a distribution
result = job.result()

# Get the measurement results
counts = result.get_counts()

# Plot the histogram of results
plot_histogram(counts)


# Interpret the results
def interpret_results(counts, num_qubits):
    """Interpret the noisy results and compare with the ideal case."""
    total_shots = sum(counts.values())
    expected_result = bin(2 + 3)[2:].zfill(num_qubits)  # Expected sum (2+3=5) in binary

    correct_counts = counts.get(expected_result, 0)  # Get the count of the correct result
    accuracy = correct_counts / total_shots  # Proportion of correct results
    error_rate = 1 - accuracy  # Proportion of incorrect results

    print(f"Expected result (binary): {expected_result}")
    print(f"Correct counts: {correct_counts}")
    print(f"Total shots: {total_shots}")
    print(f"Accuracy: {accuracy:.2f}")
    print(f"Error rate: {error_rate:.2f}")
    return accuracy, error_rate


# Analyze the noisy results
num_qubits = 5
interpret_results(counts, num_qubits)