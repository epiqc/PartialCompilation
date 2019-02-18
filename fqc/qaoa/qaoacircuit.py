"""
qaoacircuit.py - Functions for generating QAOA circuit for MAXCUT problem
                 on 3-regular graphs.
"""

import numpy as np
import pickle
import networkx as nx

#lib from Qiskit Terra
from qiskit import QuantumCircuit, QuantumRegister
from qiskit.extensions.standard import *


### BUILD CIRCUTS AND UNITARIES ###

def get_qaoa_circuit(n, p=1, theta_vector=None):
    """Produce the full UCCSD circuit.
    Args:
    n :: int - number of vertices, also the number of qubits in circuit
    p :: int - number of repetitions of QAOA iterations
    theta_vector :: array - [gamma_1, beta_1, gamma_2, beta_2, ..., gamma_p, beta_p]
                            If not provided, random angles are set.

    Returns:
    circuit :: qiskit.QuantumCircuit - the UCCSD circuit parameterized
                                       by theta_vector
    """
    if theta_vector is None:
        theta_vector = []
        for _ in range(p):
            # gamma is [0, 2pi), beta is [0, pi)
            gamma = np.random.random() * 2 * np.pi
            beta = np.random.random() * np.pi
            theta_vector.extend([gamma, beta])

    graph = nx.generators.random_graphs.random_regular_graph(d=3, n=n)
    q = QuantumRegister(n, 'q')
    circuit = QuantumCircuit(q)

    # Start in ground state (actually, most excited state, but both ways works)
    for i in range(n):
        circuit.h(q[i])

    for _ in range(p):
        # Cost Hamiltonian
        gamma = theta_vector.pop(0)
        for edge in graph.edges:
            i, j = edge[0], edge[1]
            circuit.cx(q[i], q[j])
            circuit.rz(-gamma, q[j])
            circuit.cx(q[i], q[j])

        # Mixing Hamiltonian
        beta = theta_vector.pop(0)
        for i in range(n):
            circuit.rx(beta, q[i])

    return circuit


def _is_theta_dependent(gate):
    """Return true if a gate is dependent on the theta vector,
    false otherwise.

    The only theta dependent gate in QAOA MAXCUT circuit is RZ (as in UCCSD)
    so this function is easy.

    Args:
    gate :: qiskit.QuantumGate - an arbitrary quantum gate
    """
    return isinstance(gate, RZGate)


# TODO: write get_qaoa_slices (maybe the same as uccsd?)

if __name__ == "__main__":
    _tests()
