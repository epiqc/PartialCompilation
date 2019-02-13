"""A module for running quantum optimal control on unitaries."""
import argparse, os, pickle

import numpy as np
from quantum_optimal_control.main_grape.grape import Grape
from quantum_optimal_control.core.hamiltonian import (get_H0,
                                                      get_Hops_and_Hnames,
                                                      get_full_states_concerned_list)
def main():
    # Define Constants
    DATA_PATH = '../out/test/'
    OUTPUT_FILE_NAME = "pulse"

    MAX_ITERATIONS = 1000
    NUM_STATES = 2
    CONNECTED_QUBIT_PAIRS = []
    UNITARY_ERROR = 1e-08
    CONV_TARGET = 1e-3
    SPEED_UP = 0.01
    TAYLOR_TERMS = (20, 0)
    
    with open("out.pickle", "rb") as f:
        unitaries = pickle.load(f)

    U = unitaries[0]
    CIRCUIT_DEPTH = 3
    NUM_QUBITS = int(np.log2(U.shape[0]))
    print("UNITARY", U)
    print("NUM_QUBITS", NUM_QUBITS)

    # Define drift hamiltonian.
    H0 = get_H0(NUM_QUBITS, NUM_STATES, CONNECTED_QUBIT_PAIRS)

    # Define controls.
    Hops, Hnames = get_Hops_and_Hnames(NUM_QUBITS, NUM_STATES)

    # Define concerned states (starting states).
    psi0 = get_full_states_concerned_list(NUM_QUBITS, NUM_STATES)

    # Define time scale in nanoseconds.
    TOTAL_TIME = CIRCUIT_DEPTH * 100
    STEPS = int(TOTAL_TIME * 100)

    # Define convergence parameters
    convergence = {'rate':0.01, 'update_step':10, 'max_iterations':MAX_ITERATIONS, \
                   'conv_target':CONV_TARGET, 'learning_rate_decay': MAX_ITERATIONS/2}

    # Define penalties
    reg_coeffs = {'speed_up': SPEED_UP}

    uks,U_f = Grape(H0, Hops, Hnames, U, TOTAL_TIME, STEPS, psi0, 
                    convergence = convergence, method = 'L-BFGS-B', 
                    maxA = [np.pi] * len(Hops), use_gpu = False, sparse_H = False,
                    reg_coeffs = reg_coeffs, unitary_error = UNITARY_ERROR,
                    save_plots = False, file_name = OUTPUT_FILE_NAME,
                    Taylor_terms = [20, 0], data_path = DATA_PATH)

    return

if __name__ == "__main__":
    main()
