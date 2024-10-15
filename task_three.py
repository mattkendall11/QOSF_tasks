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
    m = n  # initially assume maximum bins needed equals number of items
    model = Model('bin_packing')

    # Decision variables: binary matrix for items in bins
    x = model.binary_var_matrix(n, m, name='x')  # x[i][j] = 1 if item i is in bin j
    y = model.binary_var_list(m, name='y')        # y[j] = 1 if bin j is used

    # Objective: minimize the number of bins used
    model.minimize(model.sum(y[j] for j in range(m)))

    # Constraints: each item must be in exactly one bin
    for i in range(n):
        model.add_constraint(model.sum(x[i, j] for j in range(m)) == 1)

    # Constraints: total weight in each bin must not exceed capacity
    for j in range(m):
        model.add_constraint(model.sum(weights[i] * x[i, j] for i in range(n)) <= bin_capacity * y[j])

    return model

def ilp_to_qubo(weights, bin_capacity):
    n = len(weights)
    m = n  # Number of bins can be at most the number of items

    # Initialize QUBO matrix
    Q = {}

    # Objective part: minimize number of bins used
    for j in range(m):
        Q[(j, j)] = Q.get((j, j), 0) + 1  # Penalty for using bin j

    # Adding constraints to QUBO
    for i in range(n):
        for j in range(m):
            for k in range(m):
                if j != k:
                    Q[(i * m + j, i * m + k)] = Q.get((i * m + j, i * m + k), 0) + 2

    for j in range(m):
        for i in range(n):
            Q[(j, j)] += weights[i]  # Weight contribution

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

        # Calculate the expectation value
        energy = 0
        for (i, j), coeff in Q.items():
            energy += coeff * qml.expval(qml.PauliZ(i)) * qml.expval(qml.PauliZ(j))
        return energy

    # Use an optimizer to find the best parameters
    optimizer = qml.NesterovMomentumOptimizer(0.1)
    for _ in range(100):  # Optimization iterations
        params = optimizer.step(cost_function, params)

    # Extract the solution using qbsolv
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

    # Initial parameters
    initial_angles = np.random.rand(2 * p)  # 2 angles per layer

    # Optimize the parameters
    result = minimize(objective_function, initial_angles, args=(Q, p), method='BFGS', options={'maxiter': 100})

    # Get the optimized angles
    optimized_angles = result.x

    # Run the QAOA circuit with optimized parameters
    @qml.qnode(dev)
    def circuit():
        qaoa_circuit(optimized_angles, Q, p)
        return qml.sample(wires=range(n), shots=shots)

    samples = circuit()
    return samples


'''
To compare and analyze the results of QAOA, Quantum Annealing, and Quantum Variational approaches (with different ansätze), as well as the brute force approach, let's break down each method's characteristics, strengths, and weaknesses. This will help clarify how they perform in solving the QUBO problem.

1. Quantum Approximate Optimization Algorithm (QAOA)
Description: QAOA is a hybrid quantum-classical algorithm designed specifically for combinatorial optimization problems. It uses parameterized quantum circuits with alternating layers of problem-specific and mixing operators.
Strengths:
The ability to adaptively tune parameters to minimize the objective function.
Flexibility in circuit design allows for potentially better performance with increased layers (p).
Weaknesses:
Performance can be sensitive to the choice of ansatz and the number of layers.
Requires efficient classical optimization, which can be challenging for larger problems.
2. Quantum Annealing
Description: Quantum Annealing is a process that finds the global minimum of a function by exploiting 
quantum tunneling and superposition. It is implemented on quantum hardware like D-Wave's systems.
Strengths:
Particularly effective for specific types of problems (e.g., QUBOs).
Leverages quantum phenomena to potentially escape local minima.
Weaknesses:
Results can be probabilistic, requiring multiple runs to obtain a good solution.
The quality of the solution can vary depending on the annealing time and system noise.
3. Quantum Variational Approaches (with different Ansätze)
Description: These approaches use parameterized quantum circuits and optimize their parameters to 
minimize the cost function. Different ansätze (like RY, RX, etc.) can be implemented.
Strengths:
Flexibility in designing circuits tailored to specific problems.
Ability to combine quantum and classical optimization techniques.
Weaknesses:
Similar to QAOA, the choice of ansatz affects performance.
May require a significant number of optimization iterations to converge to a good solution.
4. Brute Force Approach
Description: This classical method evaluates all possible combinations of solutions, making it an exhaustive search method.
Strengths:
Guarantees finding the optimal solution if enough computational resources are available.
Simple to implement for small instances.
Weaknesses:
Computationally infeasible for larger problem sizes due to exponential growth in the solution space (
𝑂
(
2
𝑛
)
O(2 
n
 )).
Takes much longer than quantum approaches for larger instances.
Comparative Analysis
Performance:

For small instances, the brute force approach guarantees optimal solutions, while quantum approaches may 
provide good approximations depending on their configuration.
As problem size increases, quantum methods (especially QAOA and Quantum Annealing) become more viable due 
to their ability to explore large solution spaces more efficiently than brute force.
Scalability:

Brute force is impractical for larger problems, whereas QAOA and quantum annealing can handle larger 
instances due to their probabilistic nature and the ability to utilize quantum parallelism.
Accuracy:

QAOA and variational approaches might not always reach the exact optimal solution, especially with 
limited layers or iterations.
Quantum annealing can yield variable results, depending on how well the quantum system is tuned.
Speed:

Quantum approaches can potentially find good solutions much faster than brute force, particularly as problem sizes grow.
Summary of Results
QAOA: Generally provides good approximations, especially with a carefully chosen ansatz and sufficient layers. 
Performance may vary based on the optimization method used.

Quantum Annealing: Can yield competitive results, particularly for well-suited problems. The results can be 
variable due to stochastic nature.

Variational Approaches: Results depend heavily on the ansatz and optimization strategy. They can be 
fine-tuned to improve accuracy but may require significant computational resources for larger problems.

Brute Force: Provides exact solutions for small instances but fails to scale efficiently. It serves as a 
benchmark for evaluating the performance of quantum approaches.

Conclusion
Ultimately, the choice between these methods depends on the specific problem instance, size, and resource 
availability. Quantum methods have the potential to outperform classical methods for large-scale optimization 
problems, but their effectiveness can vary widely based on their implementation and the nature of the problem. 
The brute force approach remains a reliable method for small instances or for validating results from quantum methods.
'''