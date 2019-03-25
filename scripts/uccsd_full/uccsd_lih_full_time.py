"""
uccsd_lih_full_time.py - A script for computing the appropriate pulse_time for
                         the full UCCSD LIH circuit.
"""
# Set random seeds for reasonable reproducability.
import random
random.seed(0)
import numpy as np
np.random.seed(1)
import tensorflow as tf
tf.set_random_seed(2)

import os
import sys
import time

from fqc.uccsd import get_uccsd_circuit, get_uccsd_slices
from fqc.util import (optimize_circuit, get_unitary,
                      get_nearest_neighbor_coupling_list, get_max_pulse_time)
from fqc.data import UCCSD_LIH_THETA
from hyperopt import fmin, tpe, hp, STATUS_OK, Trials
from mpi4py.futures import MPIPoolExecutor
from quantum_optimal_control.main_grape.grape import Grape
from quantum_optimal_control.core.hamiltonian import (get_H0, 
        get_Hops_and_Hnames, get_full_states_concerned_list, get_maxA)


### CONSTANTS ###


DATA_PATH = "/project/ftchong/qoc/thomas/uccsd_full_time/lih"
FILE_NAME = "uccsd_lih_full"
# Define GRAPE parameters.
NUM_QUBITS = 4
NUM_STATES = 2
CONNECTED_QUBIT_PAIRS = get_nearest_neighbor_coupling_list(2, 2, directed=False)
H0 = np.zeros((NUM_STATES ** NUM_QUBITS, NUM_STATES ** NUM_QUBITS))
Hops, Hnames = get_Hops_and_Hnames(NUM_QUBITS, NUM_STATES, CONNECTED_QUBIT_PAIRS)
STATES_CONCERNED_LIST = get_full_states_concerned_list(NUM_QUBITS, NUM_STATES)
MAX_PULSE_AMPLITUDE = get_maxA(NUM_QUBITS, NUM_STATES, CONNECTED_QUBIT_PAIRS)
# Define convergence parameters and penalties.
CONVERGENCE = {'rate': 2e-2, 'conv_target': 1e-5,
               'max_iterations': 1e3,
               'learning_rate_decay': 1e3}
REG_COEFFS = {}
USE_GPU = False
SPARSE_H = False
SHOW_PLOTS = False
METHOD = 'ADAM'

# Define binary search parameters.
# binary search granularity, how many nanoseconds of precision do you need?
BSG = 10.0 
# steps per nanosecond
SPN = 20.0 
# nanoseconds per step
NPS = 1 / SPN

# Constrcut the circuit.
# We use a pregenerated theta for experimental consistency.
UCCSD_LIH_FULL_CIRCUIT = optimize_circuit(get_uccsd_circuit('LiH', UCCSD_LIH_THETA),
                                          CONNECTED_QUBIT_PAIRS)
# Get target unitary.
U = get_unitary(UCCSD_LIH_FULL_CIRCUIT)


### MAIN METHODS ###


def main():
    log_file = FILE_NAME + '.log'
    log_file_path = os.path.join(DATA_PATH, log_file)
    with open(log_file_path, "w") as log:
        # Redirect everything to a log file.
        sys.stdout = sys.stderr = log

        # Display pid, time, slice, circuit.
        print("PID={}\nWALL_TIME={}"
              "".format(os.getpid(), time.time()))

        # Define search space.
        max_pulse_time = get_max_pulse_time(UCCSD_LIH_FULL_CIRCUIT)
        min_steps = 0
        max_steps = int(max_pulse_time * SPN)
        print("MAX_PULSE_TIME={}\nMIN_STEPS={}\nMAX_STEPS={}"
              "".format(max_pulse_time, min_steps, max_steps))

        # Run binary search.
        binary_search_for_shortest_pulse_time(min_steps, max_steps)


def binary_search_for_shortest_pulse_time(min_steps, max_steps):
    """Search between [min_steps, max_steps] (inclusive).
    Args:
    min_steps :: int - the minimum number of steps to consider
    max_steps :: int - the maximum number of steps to consider
    """
    # mid_steps is the number of steps we try for the pulse on each
    # iteration of binary search. It is in the "middle" of max_steps
    # and min_steps.
    # The most recent mid_steps that achieves convergence is the best.
    # If no mid_steps converge, display -1.
    prev_converged_mid_steps = -1

    while min_steps + BSG < max_steps:
        print("\n")
        mid_steps = int((min_steps + max_steps) / 2)
        pulse_time = mid_steps * NPS
        print("MAX_STEPS={}\nMIN_STEPS={}\nMID_STEPS={}\nTRIAL_PULSE_TIME={}"
              "\nGRAPE_START_TIME={}"
              "".format(max_steps, min_steps, mid_steps, pulse_time,
                        time.time()))
        SS = Grape(H0, Hops, Hnames, U, pulse_time, mid_steps,
                   STATES_CONCERNED_LIST, CONVERGENCE, reg_coeffs=REG_COEFFS,
                   use_gpu=USE_GPU, sparse_H=SPARSE_H, method=METHOD, maxA=MAX_PULSE_AMPLITUDE,
                   show_plots=SHOW_PLOTS, file_name=FILE_NAME, data_path=DATA_PATH)
        print("GRAPE_END_TIME={}".format(time.time()))
        converged = SS.l < SS.conv.conv_target
        print("CONVERGED={}".format(converged))
        # If the tiral converged, lower the ceiling.
        # If the tiral did not converge, raise the floor.
        if converged:
            max_steps = mid_steps
            prev_converged_mid_steps = mid_steps
        else:
            min_steps = mid_steps
    # ENDWHILE
    
    # Display results.
    print("BEST_STEPS={}, BEST_TIME={}"
          "".format(prev_converged_mid_steps,
                    prev_converged_mid_steps * NPS))


if __name__ == "__main__":
    main()
