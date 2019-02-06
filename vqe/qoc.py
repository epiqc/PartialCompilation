"""A module for running quantum optimal control on arbitrary unitaries"""
import argparse, os, pickle

import numpy as np
from quantum_optimal_control.main_grape.grape import Grape
from quantum_optimal_control.core.hamiltonian import (get_H0,
                                                      get_Hops_and_Hnames,
                                                      get_full_states_concerned_list)
def main(args):
    # Get CLI args and define constants.
    DATA_PATH = args["out"]
    UNITARY_FILE_NAME = args["unitaries"]
    MAX_ITERATIONS = args["iterations"]
    NUM_STATES = args["states"]
    CONNECTED_QUBIT_PAIRS = []
    OUTPUT_FILE_NAME = "pulse"
    
    # Load unitaries.
    unitaries = list()
    with open(UNITARY_FILE_NAME, "rb") as unitary_file:
        unitaries = pickle.load(unitary_file)

    for U, CIRCUIT_DEPTH in unitaries:
        # Display information about the unitary.
        NUM_QUBITS = np.log2(U.shape()[0])
        print("UNITARY", U)
        print("CIRCUIT_DEPTH", CIRCUIT_DEPTH)
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
        print("TOTAL_TIME", TOTAL_TIME)

        # Define convergence parameters
        convergence = {'rate':0.01, 'update_step':10, 'max_iterations':MAX_ITERATIONS, \
                       'conv_target':1e-3, 'learning_rate_decay':MAX_ITERATIONS/2}

        # Define states to include in the drawing of occupation
        # TODO: Unsure what the correct states_draw_list is, but
        # the NUM_QUBITS > 1 branch does not work for NUM_QUBITS == 1.
        if (NUM_QUBITS > 1):
            states_draw_list = [0, 1, NUM_STATES, NUM_STATES+1]
            states_draw_names = ['00','01','10','11']
        else:
            states_draw_list = [0, 1]
            states_draw_names = ['0', '1']

        # Define reg coeffs
        # TODO: Unsure what the correct states_forbidden_list is, but
        # the NUM_QUBITS > 1 branch does not work for NUM_QUBITS == 1.
        states_forbidden_list = []
        if (NUM_QUBITS > 1):
            for ii in range(NUM_STATES):
                forbid_state = (NUM_STATES-1)*NUM_STATES+ii
                if not forbid_state in states_forbidden_list:
                    states_forbidden_list.append(forbid_state)

                forbid_state = (NUM_STATES-2)*NUM_STATES+ii
                if not forbid_state in states_forbidden_list:
                    states_forbidden_list.append(forbid_state)


            for ii in range(NUM_STATES):
                forbid_state = ii*NUM_STATES + (NUM_STATES-1)
                if not forbid_state in states_forbidden_list:
                    states_forbidden_list.append(forbid_state)

                forbid_state = ii*NUM_STATES + (NUM_STATES-2)
                if not forbid_state in states_forbidden_list:
                    states_forbidden_list.append(forbid_state)

        # Define penalties
        ops_max_amp = [np.pi] * len(Hops)

        # nothing
        #reg_coeffs = {'envelope' : 0.0, 'dwdt':0.0,'d2wdt2':0.0,'forbidden':0.0,
        #             'states_forbidden_list': states_forbidden_list,'forbid_dressed':False}

        # forbid
        #reg_coeffs = {'envelope' : 0.0, 'dwdt':0.0,'d2wdt2':0.0, 'forbidden':50.0,
        #              'states_forbidden_list': states_forbidden_list,'forbid_dressed':False}

        # forbid + pulse reg + speedup
        reg_coeffs = {'amplitude':0.01,'dwdt':0.00007,'d2wdt2':0.0, 
                      'forbidden_coeff_list':[10] * len(states_forbidden_list),
                      'states_forbidden_list': states_forbidden_list,'forbid_dressed':False,
                      'speed_up': 1.0}

        uks,U_f = Grape(H0, Hops, Hnames, U, TOTAL_TIME, STEPS, psi0, 
                        convergence = convergence, method = 'L-BFGS-B', 
                        draw = [states_draw_list, states_draw_names] , 
                        maxA = ops_max_amp, use_gpu = False, sparse_H = False,
                        reg_coeffs = reg_coeffs, unitary_error = 1e-08, 
                        save_plots = False, file_name = FILE_NAME, 
                        Taylor_terms = [20, 0], data_path = DATA_PATH)
    return

if __name__ == "__main__":
    # Parse CLI args.

    # TODO: add CLI for the following arguments:
    # * use gpu?
    # * save plots?
    # * unitary error ?
    # * sparse H ?
    # * taylor terms
    # * optimization method

    parser = argparse.ArgumentParser()
    parser.add_argument("unitaries", type=str,
                        help="The .npz file containing the unitaries to perform "
                             "qoc on.")
    parser.add_argument("out", type=str,
                        help="the directory to write output to")
    parser.add_argument("-i", "--iterations", type=int, default=1000,
                        help="The maximum number of iterations to perform "
                             "optimization for.")
    parser.add_argument("-s", "--states", type=int, default=2,
                        help="The number of qubit states to perform "
                             "qoc for.")
    args = vars(parser.parse_args())

    main(args)
