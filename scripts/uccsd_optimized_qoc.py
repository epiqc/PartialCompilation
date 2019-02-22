"""
uccsd_optimized_qoc.py - A script for running QOC on the optimized UCCSD circuit.
"""

import numpy as np
from quantum_optimal_control.main_grape.grape import Grape
from quantum_optimal_control.core.hamiltonian import (get_H0,
                                                      get_Hops_and_Hnames,
                                                      get_full_states_concerned_list,
                                                      get_maxA)
from fqc.uccsd import get_uccsd_circuit
from fqc.util import (optimize_circuit, get_unitary,
                      get_nearest_neighbor_coupling_list)

def main():
    # Define output path.
    data_path = "/project/ftchong/qoc/thomas/uccsd_optimized_qoc"
    file_name = "pulse"
    
    # Define hardware specific parameters.
    num_qubits = 4
    num_states = 2
    connected_qubit_pairs = get_nearest_neighbor_coupling_list(2, 2, directed=False)
    H0 = get_H0(num_qubits, num_states, connected_qubit_pairs)
    Hops, Hnames = get_Hops_and_Hnames(num_qubits, num_states)
    states_concerned_list = get_full_states_concerned_list(num_qubits, num_states)
    maxA = get_maxA(num_qubits, num_states)

    # Define unitary.
    theta = [np.random.random() for _ in range(8)]
    circuit = get_uccsd_circuit('LiH', theta)
    optimized_circuit = optimize_circuit(circuit, connected_qubit_pairs)
    U = get_unitary(circuit)

    # Define convergence parameters and penalties.
    max_iterations = 1000
    decay = max_iterations / 2
    convergence = {'rate':0.01, 'conv_target': 1e-3,
                   'max_iterations': max_iterations, 'learning_rate_decay':decay}
    reg_coeffs = {}

    # Define time scale in nanoseconds.
    total_time = 50
    steps = total_time * 100
    
    # Perform GRAPE.
    uks, U_f = Grape(H0, Hops, Hnames, U, total_time, steps,
                     states_concerned_list, convergence = convergence,
                     method = 'ADAM', maxA = maxA, reg_coeffs = reg_coeffs,
                     use_gpu = False, sparse_H = False, show_plots = False,
                     file_name = file_name, data_path = data_path)
    return

if __name__ == "__main__":
    main()
