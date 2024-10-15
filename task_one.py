import time

import numpy as np
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
    if qubit<15:
        start_m = time.time()
        matrix_circuit(qubit)
        end_m = time.time()
        matrix_runtimes.append(end_m-start_m)

    start_t = time.time()
    tensor_circuit(qubit)
    end_t = time.time()
    tensor_runtimes.append(end_t-start_t)
np.savetxt('matrix_runtimes.txt', matrix_runtimes)
np.savetxt('tensor_runtimes.txt', tensor_runtimes)
matrix_data = np.loadtxt('matrix_runtimes.txt')
tensor_data = np.loadtxt('tensor_runtimes.txt')
fig, ax = plt.subplots(1, 2, figsize=(10, 4))  # 1 row, 2 columns

# First plot (sin curve)
ax[0].plot(x[0:15], matrix_data, color='b')
ax[0].set_title('State vector')
ax[0].set_xlabel('qubit no')
ax[0].set_ylabel('runtime')

# Second plot (cos curve)
ax[1].plot(x, tensor_data, color='r')
ax[1].set_title('Tensor')
ax[1].set_xlabel('qubit no')
ax[1].set_ylabel('runtime')

# Adjust the layout
plt.tight_layout()

# Show the plot
plt.show()

