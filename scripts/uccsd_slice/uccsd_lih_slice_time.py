"""
uccsd_lih_slice_time.py - A script for computing the appropriate pulse_times for
                          each UCCSD slice via binary search.
"""
# Set random seeds for reasonable reproducability.
import random
random.seed(0)
import numpy as np
np.random.seed(1)
import tensorflow as tf
tf.set_random_seed(2)

import argparse
import os
import sys
import time

from fqc.uccsd import get_uccsd_circuit, get_uccsd_slices
from fqc.util import (optimize_circuit, get_unitary,
                      get_nearest_neighbor_coupling_list, get_max_pulse_time)
from fqc.data import UCCSD_LIH_THETA, UCCSD_LIH_SLICE_HYPERPARAMETERS
from hyperopt import fmin, tpe, hp, STATUS_OK, Trials
from mpi4py.futures import MPIPoolExecutor
from quantum_optimal_control.main_grape.grape import Grape
from quantum_optimal_control.core.hamiltonian import (get_H0,
                                                      get_Hops_and_Hnames,
                                                      get_full_states_concerned_list,
                                                      get_maxA)


### CONSTANTS ###


DATA_PATH = ""
# Define GRAPE parameters.
NUM_QUBITS = 4
NUM_STATES = 2
CONNECTED_QUBIT_PAIRS = get_nearest_neighbor_coupling_list(2, 2, directed=False)
H0 = np.zeros((NUM_STATES ** NUM_QUBITS, NUM_STATES ** NUM_QUBITS))
Hops, Hnames = get_Hops_and_Hnames(NUM_QUBITS, NUM_STATES, CONNECTED_QUBIT_PAIRS)
STATES_CONCERNED_LIST = get_full_states_concerned_list(NUM_QUBITS, NUM_STATES)
MAX_PULSE_AMPLITUDE = get_maxA(NUM_QUBITS, NUM_STATES, CONNECTED_QUBIT_PAIRS)
# Define convergence parameters and penalties.
CONVERGENCE = {'conv_target': 1e-5,
               'max_iterations': 1e3}
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

# Get UCCSD LIH slices such that each slice is dependent on one
# theta value but may have multiple gates dependent on that value.
SLICE_GRANULARITY = 2
# We use the pregenerated theta value UCCSD_LIH_THETA
# for experimental consistency.
UCCSD_LIH_FULL_CIRCUIT = optimize_circuit(get_uccsd_circuit('LiH', UCCSD_LIH_THETA),
                                          CONNECTED_QUBIT_PAIRS)
UCCSD_LIH_SLICES = get_uccsd_slices(UCCSD_LIH_FULL_CIRCUIT,
                                    granularity=SLICE_GRANULARITY,
                                    dependence_grouping=True)


### ADTs ###


class ProcessState(object):
    """A class to encapsulate the state of one process.
    Fields:
    uccsdslice :: fqc.uccsd.uccsdslice.UCCSDSlice - the slice that is
                                                    being binary searched on
    slice_index :: int - the index of the uccsdslice
    file_name :: string - a unique identifier for the slice
    lr :: float - the learning rate to use for the optimization
    decay :: float - the learning rate decay to use for the optimization
    """

    def __init__(self, uccsdslice, slice_index):
        """
        See class fields for parameter definitions.
        """
        super()
        self.uccsdslice = uccsdslice
        self.slice_index = slice_index
        self.file_name = "s{}".format(slice_index)
        self.lr = UCCSD_LIH_SLICE_HYPERPARAMETERS[slice_index]['lr']
        self.decay = UCCSD_LIH_SLICE_HYPERPARAMETERS[slice_index]['decay']


### MAIN METHODS ###


def main():
    # Handle CLI.
    parser = argparse.ArgumentParser()
    parser.add_argument("--slice-start", type=int, default=0, help="the "
                        "inclusive lower bound of slice indices to include "
                        "(0-7)")
    parser.add_argument("--slice-stop", type=int, default=0, help="the "
                        "inclusive upper bound of slice indices to include "
                        "(0-7)")
    args = vars(parser.parse_args())
    slice_start = args["slice_start"]
    slice_stop = args["slice_stop"]

    # Trim the slices to match those specified.
    slices = UCCSD_LIH_SLICES[slice_start:slice_stop + 1]
    slice_count = len(slices)

    # Binary search for the optimal time on each slice specified.
    state_iter = [ProcessState(uccsdslice, i + slice_start)
                  for i, uccsdslice in enumerate(slices)]
    with MPIPoolExecutor(slice_count) as executor:
        executor.map(process_init, state_iter)


def process_init(state):
    """Initialize a time optimization loop for a single slice.
    Args:
    state :: ProcessState - the state encapsulating the slice
                            to binary search on

    Returns: nothing
    """

    log_file = state.file_name + '.log'
    log_file_path = os.path.join(DATA_PATH, log_file)
    with open(log_file_path, "w") as log:
        # Redirect everything to a log file.
        sys.stdout = sys.stderr = log

        # Display pid, time, slice, circuit.
        print("PID={}\nWALL_TIME={}\nSLICE_INDEX={}"
              "".format(os.getpid(), time.time(), state.slice_index))

        # Define search space.
        max_pulse_time = get_max_pulse_time(state.uccsdslice.circuit)
        min_steps = 0
        max_steps = int(max_pulse_time * SPN)
        print("MAX_PULSE_TIME={}\nMIN_STEPS={}\nMAX_STEPS={}"
              "".format(max_pulse_time, min_steps, max_steps))

        # Run binary search.
        binary_search_for_shortest_pulse_time(state, min_steps, max_steps)


def binary_search_for_shortest_pulse_time(state, min_steps, max_steps):
    """Search between [min_steps, max_steps] (inclusive).
    Args:
    state :: ProcessState - the state encapsulating the slice to
                            binary search on
    min_steps :: int - the minimum number of steps to consider
    max_steps :: int - the maximum number of steps to consider
    """
    # Get grape arguments.
    U = state.uccsdslice.unitary()
    convergence = CONVERGENCE
    convergence.update({
        'rate': state.lr,
        'learning_rate_decay': state.decay,
    })
    
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
        sess = Grape(H0, Hops, Hnames, U, pulse_time, mid_steps,
                     STATES_CONCERNED_LIST, convergence, reg_coeffs = REG_COEFFS,
                     use_gpu = USE_GPU, sparse_H = SPARSE_H, method = METHOD,
                     maxA = MAX_PULSE_AMPLITUDE,
                     show_plots = SHOW_PLOTS,
                     file_name = state.file_name,
                     data_path = DATA_PATH)
        print("GRAPE_END_TIME={}".format(time.time()))
        converged = sess.l < sess.conv.conv_target
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
