"""
pass.py - A module for testing qiskit's transpiler.
"""

import numpy as np
from qiskit import QuantumCircuit, QuantumRegister, Aer
from qiskit.mapper import CouplingMap, swap_mapper
from qiskit.converters import circuit_to_dag, dag_to_circuit
from qiskit.transpiler import PassManager, transpile
from qiskit.transpiler.passes import (BasicSwap, CXCancellation,
                                      CommutationTransformation, Optimize1qGates,
                                      Unroller)
from qiskit.extensions.standard import *

from uccsd_unitary import get_uccsd_circuit

backend = Aer.get_backend('statevector_simulator')

def main():
    # Build uccsd circuit
    theta = [np.random.random() for _ in range(8)]
    initial_circuit = get_uccsd_circuit(theta, use_basis_gates=True)
    
    # Define topology
    coupling_list = [[0, 1], [1, 2], [2, 3]]
    coupling_map = CouplingMap(coupling_list)

    # Define transformations
    pm = PassManager()
    pm.append(Optimize1qGates())
    pm.append(CXCancellation())
    pm.append(CommutationTransformation())
    pm.append(BasicSwap(coupling_map))
    
    circuit = transpile(initial_circuit, backend=backend,
                        coupling_map=coupling_list, pass_manager=pm)

    print(initial_circuit)
    print(circuit)

    print(initial_circuit.count_ops())
    print(circuit.count_ops())

if __name__ == "__main__":
    main()
