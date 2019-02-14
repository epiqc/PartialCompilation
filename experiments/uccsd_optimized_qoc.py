"""
uccsd_optimized_qoc.py - A script for running QOC on the optimized UCCSD circuit.
"""

import numpy as np
from quantum_optimal_control.main_grape.grape import Grape
from quantum_optimal_control.core.hamiltonian import (get_H0,
                                                      get_Hops_and_Hnames,
                                                      get_full_states_concerned_list)
from fqc.uccsd import get_uccsd_circuit
from fqc.util import optimize_circuit, get_unitary

def main():
    # Get CLI args and define constants.
    DATA_PATH = "/project/ftchong/qoc/thomas/uccsd_optimized_qoc"
    OUTPUT_FILE_NAME = "pulse"

    MAX_ITERATIONS = 1000
    NUM_QUBITS = 4
    NUM_STATES = 2
    CONNECTED_QUBIT_PAIRS = [(0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3)]
    TAYLOR_TERMS = (20, 0)
    UNITARY_ERROR = 1e-8
    CONV_TARGET = 1e-3
    MAX_AMPLITUDE = 2 * np.pi * 0.3
    
    theta = [np.random.random() for _ in range(8)]
    circuit = get_uccsd_circuit(theta)
    optimized_circuit = optimize_circuit(circuit, CONNECTED_QUBIT_PAIRS)
    U = get_unitary(circuit)

    # Define drift hamiltonian.
    H0 = get_H0(NUM_QUBITS, NUM_STATES, CONNECTED_QUBIT_PAIRS)

    # Define controls.
    Hops, Hnames = get_Hops_and_Hnames(NUM_QUBITS, NUM_STATES)

    # Define concerned states (starting states).
    psi0 = get_full_states_concerned_list(NUM_QUBITS, NUM_STATES)

    # Define time scale in nanoseconds.
    TOTAL_TIME = 50
    STEPS = TOTAL_TIME * 100

    # Define convergence parameters
    convergence = {'rate':0.01, 'update_step':10, 'max_iterations':MAX_ITERATIONS, \
                   'conv_target':CONV_TARGET, 'learning_rate_decay':MAX_ITERATIONS/2}

    reg_coeffs = {'speed_up': 0.001}

    uks,U_f = Grape(H0, Hops, Hnames, U, TOTAL_TIME, STEPS, psi0, 
                    convergence = convergence, method = 'L-BFGS-B', 
                    maxA = [MAX_AMPLITUDE] * len(Hops), use_gpu = False, sparse_H = False,
                    reg_coeffs = reg_coeffs, unitary_error = UNITARY_ERROR,
                    show_plots = False, file_name = OUTPUT_FILE_NAME,
                    Taylor_terms = TAYLOR_TERMS, data_path = DATA_PATH)
    return

if __name__ == "__main__":
    main()
