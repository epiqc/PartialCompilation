"""A module for running Quantum Optimal Control on the 
Unitary Coupled Cluster Single-Double ansatz.
This module was adapted from 
GRAPE-Tensorflow-Examples/paper-examples/Transmon_Transmon_CNOT.ipynb
"""
import argparse, os, time

import numpy as np
from quantum_optimal_control.main_grape.grape import Grape
from quantum_optimal_control.core.hamiltonian import (get_H0,
                                                      get_full_states_concerned_list,
                                                      get_H0, get_Hops_and_Hnames)
from uccsd_unitary import get_uccsd_slices

# Parse CLI and define constants.
parser = argparse.ArgumentParser()
parser.add_argument("-i", "--iterations", type=int, default=1000,
                    help="The maximum number of iterations to perform "
                         "optimization for.")
parser.add_argument("-s", "--states", type=int, default=2,
                    help="The number of qubit states to perform "
                         "optimization for.")
args = vars(parser.parse_args())

# TODO: What connected qubit pairs do we want?
# An empty connected_qubit_pairs list will return a zero matrix
# for the dirft hamiltonian.
CONNECTED_QUBIT_PAIRS = []
MAX_ITERATIONS = args["iterations"]
NUM_STATES = args["states"]
FILE_NAME = 'slice_pulse'

def main():
    # Create output directory.
    # https://stackoverflow.com/questions/12517451/automatically-creating-directories-with-file-output

    time_stamp = int(time.time())
    DATA_PATH = '../out/uccsd_qoc_slice_{}/'.format(time_stamp)
    if not os.path.exists(DATA_PATH):
        try:
            os.makedirs(DATA_PATH)
        except OSError as exc: # Guard against race condition
            if exc.errno != errno.EEXIST:
                raise

    # Get uccsd partial circuits / unitaries.
    theta = [np.random.random() * 2 * np.pi for _ in range(8)]
    print("THETA", theta)
    slices = get_uccsd_slices(theta)
    
    # Run QOC on each slice
    for uccsdslice in slices:
        # Skip redundant slices.
        if uccsdslice.redundant:
            continue

        # Define qubit parameters
        NUM_QUBITS = uccsdslice.circuit.width()
        print("NUM_QUBITS", NUM_QUBITS)
        print("CIRCUIT", uccsdslice.circuit)

        # Define target unitary
        U = uccsdslice.unitary
        print("UNITARY", U)

        # Define drift hamiltonian
        H0 = get_H0(NUM_QUBITS, NUM_STATES, CONNECTED_QUBIT_PAIRS)

        # Define controls
        Hops, Hnames = get_Hops_and_Hnames(NUM_QUBITS, NUM_STATES)

        # Define concerned states (starting states)
        psi0 = get_full_states_concerned_list(NUM_QUBITS, NUM_STATES)

        # Define time scales
        TOTAL_TIME = 80.0 * len(uccsdslice.circuit)
        STEPS = int(TOTAL_TIME * 100)

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
        ops_max_amp = [np.pi for _ in range(len(Hops))]

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
                        Taylor_terms = [20,0], data_path = DATA_PATH)
    return

if __name__ == "__main__":
    main()


