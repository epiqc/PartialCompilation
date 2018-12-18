"""A module for running Quantum Optimal Control on the 
Unitary Coupled Cluster Single-Double ansatz.
This module was adapted from 
GRAPE-Tensorflow-Examples/paper-examples/Transmon_Transmon_CNOT.ipynb
"""
# TODO: implement for multiple states.
# This task is mostly dependent on making uccsd_unitary able to 
# handle multiple states.

from functools import reduce
import argparse, inspect, os, random as rd, sys, time, warnings

import h5py
import numpy as np
from IPython import display
import scipy.linalg as la
from quantum_optimal_control.helper_functions.grape_functions import *
from quantum_optimal_control.main_grape.grape import Grape
from quantum_optimal_control.core.util import kron_many, print_matrix
from uccsd_unitary import uccsd_unitary

# Parse CLI and define constants.
parser = argparse.ArgumentParser()
parser.add_argument("-i", "--iterations", type=int, default=1000,
                    help="The maximum number of iterations to perform "
                         "optimization for.")
parser.add_argument("-o", "--ops", type=str, default="HHHH",
                    help="The four single qubit operators of the circuit.")
parser.add_argument("-q", "--qubits", type=int, default=4,
                    help="The number of qubits to build the circuit for.")
parser.add_argument("-t", "--theta", type=int, default=0,
                    help="The angle for the Rz gate in the uccsd circuit.")
args = vars(parser.parse_args())

NUM_STATES = 2
MAX_ITERATIONS = args["iterations"]
SQOPS = args["ops"]
NUM_QUBITS = args["qubits"]
THETA = args["theta"]

DATA_PATH = '../out/'
FILE_NAME = 'uccsd_qoc'

# Define time scales
TOTAL_TIME = 10.0
STEPS = 1000

# Define H0 (drift hamiltonian)
ens = lambda freq, alpha: [2*np.pi*ii*(freq - 0.5*(ii-1)*alpha) 
                               for ii in np.arange(NUM_STATES)]

# Frequencies in GHz, "ens" may refer to "eigens"
alpha_list = [0.255 for i in range(NUM_QUBITS)]
freq_ge_list = [3.9] + [3.5 for i in range(NUM_QUBITS - 1)]
H0_list = [np.diag(ens(freq_ge_list[i], alpha_list[i])) for i in range(NUM_QUBITS)]

G = 2*np.pi*0.1

Q_x   = np.diag(np.sqrt(np.arange(1,NUM_STATES)),1)+np.diag(np.sqrt(np.arange(1,NUM_STATES)),-1)
Q_y   = (0+1j)*(np.diag(np.sqrt(np.arange(1,NUM_STATES)),1)-np.diag(np.sqrt(np.arange(1,NUM_STATES)),-1))
Q_z   = np.diag(np.arange(0,NUM_STATES))
I = np.identity(NUM_STATES)

# To obtain the drift hamiltonian, by symmetry with examples, we take the sum of 
# the matrices which are the (NUM_QUBITS) permutations of taking the kronecker
# product of a gate from H0_list with (NUM_QUBITS - 1) identity matrices, plus a constant.
base = [I for i in range(NUM_QUBITS - 1)]
H0_permutations = []
for i in range(NUM_QUBITS):
    H0_gates = base[:]
    H0_gates.insert(i, H0_list[i])
    H0_permutations.append(kron_many(*H0_gates))

H0 = reduce(lambda x, y: x + y, H0_permutations) + G * kron_many(*[Q_x for i in range(NUM_QUBITS)])

# Define concerned states (starting states)
psi0 = [0,1,NUM_STATES,NUM_STATES+1] #[gg,ge,eg,ee]

# Define states to include in the drawing of occupation
states_draw_list = [0, 1, NUM_STATES, NUM_STATES+1]
states_draw_names = ['00','01','10','11']


# Define U (target unitary)
U = uccsd_unitary(num_qubits=NUM_QUBITS, sqops_str=SQOPS, theta=THETA)

# Define controls
# We want, by symmetry with examples, Q_xi for i in {1..NUM_QUBITS} and Q_zn 
# where n = NUM_QUBITS. Q_xi is the (NUM_QUBITS) permutations of (NUM_QUBITS - 1)
# identity matrices with Q_x in the ith position.
Hops = []
Hnames = []
for i in range(NUM_QUBITS):
    gates = base[:]
    gates.insert(i, Q_x)
    Hops.append(kron_many(*gates))
    Hnames.append('x' + str(i))
gates = base[:]
gates.append(Q_z)
Hops.append(kron_many(*gates))
Hnames.append('z' + str(NUM_QUBITS))
ops_max_amp = [np.pi for _ in range(NUM_QUBITS + 1)]

# Define convergence parameters
DECAY = MAX_ITERATIONS/2
convergence = {'rate':0.01, 'update_step':10, 'max_iterations':MAX_ITERATIONS,\
               'conv_target':1e-3, 'learning_rate_decay':DECAY}

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
        
# print ("states_forbidden_list", states_forbidden_list)

## nothing
#reg_coeffs = {'envelope' : 0.0, 'dwdt':0.0,'d2wdt2':0.0,'forbidden':0.0,
#             'states_forbidden_list': states_forbidden_list,'forbid_dressed':False}

## forbid
#reg_coeffs = {'envelope' : 0.0, 'dwdt':0.0,'d2wdt2':0.0, 'forbidden':50.0,
#              'states_forbidden_list': states_forbidden_list,'forbid_dressed':False}

# forbid + pulse reg
reg_coeffs = {'amplitude':0.01,'dwdt':0.00007,'d2wdt2':0.0, 
              'forbidden_coeff_list':[10] * len(states_forbidden_list),
              'states_forbidden_list': states_forbidden_list,'forbid_dressed':False}


if __name__ == "__main__":
    uks,U_f =Grape(H0, Hops, Hnames, U, TOTAL_TIME, STEPS, psi0, convergence=convergence, 
               method = 'L-BFGS-B', draw = [states_draw_list,states_draw_names] ,
               maxA = ops_max_amp, use_gpu=False, sparse_H = False, reg_coeffs=reg_coeffs, 
               unitary_error = 1e-08, save_plots=True, file_name=FILE_NAME, 
               Taylor_terms = [20,0], data_path = DATA_PATH)
