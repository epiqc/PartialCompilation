"""A module for running Quantum Optimal Control on the 
Unitary Coupled Cluster Single-Double ansatz.
This module was adapted from 
GRAPE-Tensorflow-Examples/paper-examples/Transmon_Transmon_CNOT.ipynb
"""
from functools import reduce
import argparse, inspect, os, random as rd, sys, time, warnings

import h5py
import numpy as np
from IPython import display
import scipy.linalg as la
from quantum_optimal_control.helper_functions.grape_functions import *
from quantum_optimal_control.main_grape.grape import Grape
from quantum_optimal_control.core.hamiltonian import (get_H0,
                                                      get_full_states_concerned_list,
                                                      get_H0, get_Hops_and_Hnames)

from uccsd_unitary import get_uccsd_circuit, get_unitary

# Parse CLI and define constants.
# TODO: add "states" parameter and adapt file to handle qudits.
parser = argparse.ArgumentParser()
parser.add_argument("-i", "--iterations", type=int, default=1000,
                    help="The maximum number of iterations to perform "
                         "optimization for.")
args = vars(parser.parse_args())

# TODO: Handle uccsd ansatz for arbitrary state number and qubit count.
# We will want to make a module for generating arbitrary UCCSD 
# circuits via qiskit.
NUM_QUBITS = 4
NUM_STATES = 2
# TODO: What connected qubit pairs do we want?
# An empty list will return a zero matrix for the dirft hamiltonian.
CONNECTED_QUBIT_PAIRS = []
MAX_ITERATIONS = args["iterations"]

DATA_PATH = '../out/'
FILE_NAME = 'uccsd_qoc'

# Define time scales
TOTAL_TIME = 10.0
STEPS = 1000

# Define target unitary
theta = [1.0, 1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.7]
circuit = get_uccsd_circuit(theta)
U = get_unitary(circuit)

# Define drift hamiltonian
H0 = get_H0(NUM_QUBITS, NUM_STATES, CONNECTED_QUBIT_PAIRS)

# Define controls
Hops, Hnames = get_Hops_and_Hnames(NUM_QUBITS, NUM_STATES)

# Define concerned states (starting states)
psi0 = get_full_states_concerned_list(NUM_QUBITS, NUM_STATES)

# Define states to include in the drawing of occupation
states_draw_list = [0, 1, NUM_STATES, NUM_STATES+1]
states_draw_names = ['00','01','10','11']

# Define convergence parameters
decay = MAX_ITERATIONS/2

convergence = {'rate':0.01, 'update_step':10, 'max_iterations':MAX_ITERATIONS, \
               'conv_target':1e-3, 'learning_rate_decay':decay}

# Define reg coeffs
states_forbidden_list = []
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

# forbid + pulse reg
reg_coeffs = {'amplitude':0.01,'dwdt':0.00007,'d2wdt2':0.0, 
              'forbidden_coeff_list':[10] * len(states_forbidden_list),
              'states_forbidden_list': states_forbidden_list,'forbid_dressed':False,
              'speed_up': 1.0}

if __name__ == "__main__":
    uks,U_f = Grape(H0, Hops, Hnames, U, TOTAL_TIME, STEPS, psi0, 
                    convergence = convergence, method = 'L-BFGS-B', 
                    draw = [states_draw_list, states_draw_names] , 
                    maxA = ops_max_amp, use_gpu = False, sparse_H = False,
                    reg_coeffs = reg_coeffs, unitary_error = 1e-08, 
                    save_plots = True, file_name = FILE_NAME, 
                    Taylor_terms = [20,0], data_path = DATA_PATH)

