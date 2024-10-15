import time
from tqdm.auto import tqdm
import matplotlib.pyplot as plt

from utils.matrix import matrix_circuit
from utils.tensor import tensor_circuit

'''
script to compare the runtime of state-vector and tensor simulations of quantum circuits for the exact same circuit
'''
x = list(range(2,25))
matrix_runtimes = []
tensor_runtimes = []
for qubit in tqdm(x):

    start_m = time.time()
    matrix_circuit(qubit)
    end_m = time.time()
    matrix_runtimes.append(end_m-start_m)

    start_t = time.time()
    tensor_circuit(qubit)
    end_t = time.time()
    tensor_runtimes.append(end_t-start_t)

plt.plot(x, matrix_runtimes, label = 'state vector')
plt.plot(x, tensor_runtimes, label = 'tensor')
plt.xlabel('no of qubits')
plt.ylabel('runtime (s)')
plt.legend()
plt.show()

