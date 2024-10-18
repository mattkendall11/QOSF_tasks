import numpy as np
from qiskit import QuantumCircuit, transpile
'''
setup initial values
'''
size = 5
state_values = [22, 17, 27, 12]
state_vector = [0] * 2**size
for s in state_values:
    state_vector[s] = 0.5
'''
build circuit
'''

qc = QuantumCircuit(size)

qc.x(0)
qc.x(2)
qc.x(4)

qc.x(0)
qc.x(4)

qc.x(0)
qc.x(1)
qc.x(2)
qc.x(4)

qc.x(2)
qc.x(3)


qc.cx(0, 1)
qc.cx(1, 4)
qc.cx(2, 4)
qc.cx(3, 4)
qc.cx(2, 3)

for idx, s in enumerate(state_values):
    qc.rz(np.pi/2, idx)

basis_gates = ['x', 'h', 'rz', 'cx']
qc = transpile(qc, basis_gates=basis_gates, optimization_level=3)


print(qc)

qc.draw(output='mpl')
