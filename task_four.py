import numpy as np
from qiskit import QuantumCircuit, transpile


# Input values
size = 5
state_values = [22, 17, 27, 12]
state_vector = [0] * 2**size
for s in state_values:
    state_vector[s] = 0.5

# Create a quantum circuit for 5 qubits
qc = QuantumCircuit(size)

# Step 1: Apply Hadamard or Pauli-X to initialize the qubits
# Binary representations of the states: 10110 (22), 10001 (17), 11011 (27), 01100 (12)
# Initialize these states using X and H gates

# Initialize |22> (10110)
qc.x(0)
qc.x(2)
qc.x(4)

# Initialize |17> (10001)
qc.x(0)
qc.x(4)

# Initialize |27> (11011)
qc.x(0)
qc.x(1)
qc.x(2)
qc.x(4)

# Initialize |12> (01100)
qc.x(2)
qc.x(3)

# Step 2: Apply entangling operations (CNOT gates) based on the architecture
qc.cx(0, 1)  # Connection (0,1)
qc.cx(1, 4)  # Connection (1,4)
qc.cx(2, 4)  # Connection (2,4)
qc.cx(3, 4)  # Connection (3,4)
qc.cx(2, 3)  # Connection (2,3)

# Step 3: Apply rotation gates (Rz) to encode the amplitudes
for idx, s in enumerate(state_values):
    qc.rz(np.pi/2, idx)  # Adjust the angle as needed to reflect amplitude phases

# Step 4: Transpile the circuit with basis gates [x,h,rz,cx]
basis_gates = ['x', 'h', 'rz', 'cx']
qc = transpile(qc, basis_gates=basis_gates, optimization_level=3)

# Print the circuit
print(qc)

# Optionally visualize the circuit
qc.draw(output='mpl')
