import numpy as np
from functools import reduce


def qubit_register(num_qubits):
    '''

    :param num_qubits: number of qubits in |0> state to initalise
    :return: np array of qubit states
    '''
    one = np.array([1, 0j])
    register = reduce(np.kron, [one] * num_qubits)

    return register

def X(index, num_qubits):
    '''

    :param index: index of qubit to act on
    :param num_qubits: number of qubits in register
    :return:
    '''

    x = np.array([[0, 1], [1, 0]], dtype=np.complex128)


    I = np.eye(2, dtype=np.complex128)


    gate = [I] * num_qubits
    gate[index] = x


    total_gate = reduce(np.kron, gate)

    return total_gate


def H(index, num_qubits):
    '''

    :param index: index of qubit to act on
    :param num_qubits: number of qubits in the register
    :return:
    '''

    h = np.array([[1, 1], [1, -1]], dtype=np.complex128) / np.sqrt(2)


    I = np.eye(2, dtype=np.complex128)


    gate = [I] * num_qubits
    gate[index] = h  # Apply the H gate to the specified qubit index


    total_gate = reduce(np.kron, gate)

    return total_gate


def CNOT(n, control, target):
    """returns the matrix representation of a CNOT gate acting on qubit `control` and `target`
       in an `n`-qubit quantum system.

    :param:
        n (int): The total number of qubits in the quantum system.
        control (int): The index of the control qubit.
        target (int): The index of the target qubit.

    :return:
        np.ndarray: A 2^n x 2^n matrix representing the CNOT gate.
    """

    X = np.array([[0, 1],
                  [1, 0]])


    I = np.eye(2)

    cnot = np.zeros((2 ** n, 2 ** n))

    for k in range(2 ** n):
        binary = f'{k:0{n}b}'
        next_state = list(binary)

        if binary[-(control + 1)] == '1':  # control qubit is 1 (reverse index due to endianness)
            # Flip target qubit
            next_state[-(target + 1)] = '1' if binary[-(target + 1)] == '0' else '0'

        next_state_idx = int("".join(next_state), 2)
        cnot[next_state_idx, k] = 1

    return cnot



def matrix_circuit(n):
    '''
    :param n: number of qubits
    :return: output of circuit
    defines a circuit consisting of X,H and CNOT gates on all qubits
    '''
    register = qubit_register(n)
    for i in range(1,n):
        register = np.dot(register, X(i,n))
        register = np.dot(register, H(i,n))
        register = np.dot(register, CNOT(n,i,0))
    return register



