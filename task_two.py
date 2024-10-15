from qiskit import QuantumCircuit, transpile, Aer
import numpy as np
#from qiskit.circuit.library import SXGate, XGate, RZGate, CXGate, IGate
from utils.noise import add_pauli_noise, decompose_to_basis, quantum_sum
#from qiskit_aer import AerSimulator
'''
script that takes a draper adder circuit ,decomposes to basis then adds noise
'''

#define adder cicruit

circuit = quantum_sum(2,3,5)

basis_circuit = decompose_to_basis(circuit)

noisy_circuit = add_pauli_noise(basis_circuit, 0.05,0.3)

simulator = Aer.get_backend()

circ = transpile(noisy_circuit, simulator)

# Run and get counts
result = simulator.run(circ).result()
counts = result.get_counts(circ)


