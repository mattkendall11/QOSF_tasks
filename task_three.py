from docplex.mp.model import Model
from dwave.system import DWaveSampler, EmbeddingComposite
import numpy as np
import pennylane as qml
from pennylane import numpy as pnp
from dwave.system import DWaveSampler, EmbeddingComposite
from dwave.embedding import embed_qubo
from dwave_qbsolv import QBSolv
from scipy.optimize import minimize
def create_ilp(weights, bin_capacity):
    n = len(weights)
    m = n
    model = Model('bin_packing')

    x = model.binary_var_matrix(n, m, name='x')
    y = model.binary_var_list(m, name='y')


    model.minimize(model.sum(y[j] for j in range(m)))

    for i in range(n):
        model.add_constraint(model.sum(x[i, j] for j in range(m)) )
    for j in range(m):
        model.add_constraint(model.sum(weights[i] * x[i, j] for i in range(n)) <= bin_capacity * y[j])

    return model

def ilp_to_qubo(weights, bin_capacity):
    n = len(weights)
    m = n

    Q = {}

    for j in range(m):
        Q[(j, j)] = Q.get((j, j), 0) + 1  # Penalty for using bin j


    for i in range(n):
        for j in range(m):
            for k in range(m):
                if j != k:
                    Q[(i * m + j, i * m + k)] = Q.get((i * m + j, i * m + k), 0) + 2

    for j in range(m):
        for i in range(n):
            Q[(j, j)] += weights[i]

    return Q


import itertools


def evaluate_qubo(Q, binary_vector):
    """Evaluate the QUBO objective function for a given binary vector."""
    score = 0
    n = len(binary_vector)

    for i in range(n):
        for j in range(n):
            if (i, j) in Q:
                score += Q[(i, j)] * binary_vector[i] * binary_vector[j]

    return score


def brute_force_solve(Q, n):
    """Brute force solver for the QUBO problem."""
    best_score = float('inf')
    best_solution = None

    # Generate all possible binary vectors of length n
    for binary_vector in itertools.product([0, 1], repeat=n):
        score = evaluate_qubo(Q, binary_vector)
        if score < best_score:
            best_score = score
            best_solution = binary_vector

    return best_solution, best_score


def solve_qubo_with_dwave(Q):
    # Convert the QUBO to the required format for D-Wave
    linear = {k: Q.get((k, k), 0) for k in range(len(Q))}
    quadratic = {(i, j): Q[(i, j)] for i in range(len(Q)) for j in range(len(Q)) if (i, j) in Q and i != j}

    # Create a D-Wave sampler
    sampler = EmbeddingComposite(DWaveSampler(token = 'DEV-7750a673bb430aa31dc5f1fcec7ecbdcfca09411'))

    # Solve the QUBO
    sampleset = sampler.sample_qubo(Q, num_reads=100)

    # Extract the best solution and its energy
    best_solution = sampleset.first.sample
    best_energy = sampleset.first.energy

    return best_solution, best_energy


# Define the Quantum Circuit
def create_ansatz(num_qubits, ansatz_type):
    if ansatz_type == "ry":
        @qml.qnode(dev)
        def circuit(params):
            for i in range(num_qubits):
                qml.RY(params[i], wires=i)
            for i in range(num_qubits - 1):
                qml.CNOT(wires=[i, i + 1])
            return qml.expval(qml.PauliZ(0))

    elif ansatz_type == "rx":
        @qml.qnode(dev)
        def circuit(params):
            for i in range(num_qubits):
                qml.RX(params[i], wires=i)
            for i in range(num_qubits - 1):
                qml.CNOT(wires=[i, i + 1])
            return qml.expval(qml.PauliZ(0))

    # Add more ansätze as needed...

    return circuit


def solve_qubo_with_ansatz(Q, ansatz_type, num_reads=100):
    n = len(Q)

    # Prepare a device
    dev = qml.device('default.qubit', wires=n)

    # Create the ansatz circuit
    circuit = create_ansatz(n, ansatz_type)

    # Initialize parameters
    params = pnp.random.uniform(0, 2 * np.pi, size=n)

    # Define cost function
    @qml.qnode(dev)
    def cost_function(params):
        # Apply the ansatz
        circuit(params)

        energy = 0
        for (i, j), coeff in Q.items():
            energy += coeff * qml.expval(qml.PauliZ(i)) * qml.expval(qml.PauliZ(j))
        return energy


    optimizer = qml.NesterovMomentumOptimizer(0.1)
    for _ in range(100):
        params = optimizer.step(cost_function, params)

    sampleset = QBSolv().sample_qubo(Q, num_reads=num_reads)
    best_solution = sampleset.first.sample
    best_energy = sampleset.first.energy

    return best_solution, best_energy


def qaoa_circuit(angles, Q, p):
    """Define the QAOA circuit."""
    n = len(Q)

    # Initialize the quantum circuit
    for i in range(n):
        qml.Hadamard(wires=i)  # Initialize qubits to superposition

    for layer in range(p):
        # Apply the cost Hamiltonian
        for (i, j), coeff in Q.items():
            qml.PauliZ(i) @ qml.PauliZ(j)  # Cost Hamiltonian term
            qml.phase_shift(-coeff * angles[layer], wires=[i, j])

        # Apply the mixing Hamiltonian
        for i in range(n):
            qml.RX(2 * angles[p + layer], wires=i)  # Mixing operator


def objective_function(angles, Q, p):
    """Objective function to be minimized."""

    @qml.qnode(dev)
    def circuit():
        qaoa_circuit(angles, Q, p)
        return [qml.expval(qml.PauliZ(i)) for i in range(len(Q))]

    return circuit()


def solve_qubo_qaoa(Q, p, shots=100):
    """Solve a QUBO using QAOA."""
    n = len(Q)
    global dev
    dev = qml.device('default.qubit', wires=n)

    initial_angles = np.random.rand(2 * p)

    result = minimize(objective_function, initial_angles, args=(Q, p), method='BFGS', options={'maxiter': 100})

    optimized_angles = result.x

    @qml.qnode(dev)
    def circuit():
        qaoa_circuit(optimized_angles, Q, p)
        return qml.sample(wires=range(n), shots=shots)

    samples = circuit()
    return samples

