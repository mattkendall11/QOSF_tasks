import numpy as np

def quantum_register(number_of_qubits):
    """Creates a linear register of qubits, initialized to |000...0〉.
    Returns a complex tensor of shape (2, 2, ..., 2)."""
    shape = (2,) * number_of_qubits
    first = (0,) * number_of_qubits
    register = np.zeros(shape, dtype=np.complex128)
    register[first] = 1+0j
    return register


def get_transposition(n, indices):

    transpose = [0] * n
    k = len(indices)
    ptr = 0
    for i in range(n):
        if i in indices:
            transpose[i] = n - k + indices.index(i)
        else:
            transpose[i] =  ptr
            ptr += 1
    return transpose


def apply_gate(gate, *indices):

    axes = (indices, range(len(indices)))
    def op(register):
        return np.tensordot(register, gate, axes=axes).transpose(
               get_transposition(register.ndim, indices))
    return op


def X(index):

    gate = np.array([[0, 1], [1, 0]], dtype=np.complex128)
    return apply_gate(gate, index)


def H(index):
    """Generates a Hadamard gate. It returns a function that can be applied to a register."""
    gate = np.array([[1, 1], [1, -1]]) / np.sqrt(2)
    return apply_gate(gate, index)


def CNOT(i, j):

    gate = np.array([1, 0, 0, 0,
                     0, 1, 0, 0,
                     0, 0, 0, 1,
                     0, 0, 1, 0]
                     ).reshape((2, 2, 2, 2))
    return apply_gate(gate,i, j)

def tensor_circuit(n):
    register = quantum_register(n)
    for i in range(1,n):

        register = X(i)(register)
        register = H(i)(register)
        register = CNOT(i,0)(register)
