"""
uccsd_slice_hpo_v2.py - A script for computing the appropriate hyperparameters
                        for GRAPE on a UCCSD slice. We optimize over learning rate
                        and decay for a fixed time and fixed angle on one slice.
"""
# Set random seeds for reasonable reproducibility.
import random
random.seed(0)
import numpy as np
np.random.seed(1)
import tensorflow as tf
tf.set_random_seed(2)

import argparse
from itertools import product
import json
import os
import sys
import time

from fqc.data import UCCSD_LIH_THETA
from fqc.uccsd import get_uccsd_circuit, get_uccsd_slices
from fqc.util import (optimize_circuit, get_unitary,
                      get_nearest_neighbor_coupling_list,
                      get_max_pulse_time, CustomJSONEncoder)
from hyperopt import fmin, tpe, hp, STATUS_OK, Trials
from mpi4py.futures import MPIPoolExecutor
from quantum_optimal_control.main_grape.grape import Grape
from quantum_optimal_control.core.hamiltonian import (get_H0, 
        get_Hops_and_Hnames, get_full_states_concerned_list, get_maxA)


### CONSTANTS ###


# Grape constants.
DATA_PATH = "/project/ftchong/qoc/thomas/uccsd_slice_hpo/lih_v2"
# Define hardware specific parameters.
NUM_QUBITS = 4
NUM_STATES = 2
CONNECTED_QUBIT_PAIRS = get_nearest_neighbor_coupling_list(2, 2, directed=False)
H0 = np.zeros((NUM_STATES ** NUM_QUBITS, NUM_STATES ** NUM_QUBITS))
Hops, Hnames = get_Hops_and_Hnames(NUM_QUBITS, NUM_STATES, CONNECTED_QUBIT_PAIRS)
STATES_CONCERNED_LIST = get_full_states_concerned_list(NUM_QUBITS, NUM_STATES)
MAX_AMPLITUDE = get_maxA(NUM_QUBITS, NUM_STATES, CONNECTED_QUBIT_PAIRS)
METHOD = 'ADAM'
MAX_GRAPE_ITERATIONS = 1e3
DECAY = 1e3
REG_COEFFS = {}
USE_GPU = False
SPARSE_H = False
SHOW_PLOTS = False

# Get all UCCSD lih slices.
SLICE_GRANULARITY = 2
UCCSD_LIH_FULL_CIRCUIT = optimize_circuit(get_uccsd_circuit('LiH', UCCSD_LIH_THETA),
                                          CONNECTED_QUBIT_PAIRS)
UCCSD_LIH_SLICES = get_uccsd_slices(UCCSD_LIH_FULL_CIRCUIT,
                                    granularity = SLICE_GRANULARITY,
                                    dependence_grouping=True)

# Hyperparmeter optimization constants and search space.
MAX_HPO_ITERATIONS = 50
LR_LB = 1e-5
LR_UB = 1
DECAY_LB = 1
DECAY_UB = 1e5

BROADWELL_CORE_COUNT = 14
TMP_CORE_COUNT = 8


### OBJECTS ###


class ProcessState(object):
    """A class to track the state of an optimization loop.
    Fields:
    file_name :: string - the identifier of the optimization
    iteration_count :: int - tracks how many iterations the optimization
                             has performed
    pulse_time :: float - the pulse time to optimize the slice for (in nanoseconds)
    slice_index :: int - the index of the slice that is being optimized
    trials :: [dict] - list of dictionaries, each detailing an iteration
                       of optimization
    uccsdslice :: fqc.uccsd.uccsdslice.UCCSDSlice - the slice that
                                                    is being optimized
    """

    def __init__(self, uccsdslice, slice_index, pulse_time):
        """See corresponding class field declarations above for other arguments.
        """
        super(ProcessState, self).__init__()
        self.uccsdslice = uccsdslice
        self.slice_index = slice_index
        self.pulse_time = pulse_time
        self.file_name = ("s{}_t{}"
                          "".format(slice_index, int(pulse_time)))
        self.trials = list()
        self.iteration_count = 0


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

    # Trim slices to only include start thru stop.
    slices = UCCSD_LIH_SLICES[slice_start:slice_stop + 1]
    slice_count = len(slices)

    # Generate the state objects to encapsulate the optimization for each slice.
    # We will run one hpo for the slice at 10% of its gate-based time
    # and one for the slice at 20% of its gate-based time.
    state_iter = list()
    for i, uccsdslice in enumerate(slices):
        slice_index = i + slice_start
        max_pulse_time = get_max_pulse_time(uccsdslice.circuit)
        state_iter.append(ProcessState(uccsdslice, slice_index, max_pulse_time * 0.1))
        state_iter.append(ProcessState(uccsdslice, slice_index, max_pulse_time * 0.2))

    # Run optimization on the slices.
    with MPIPoolExecutor(TMP_CORE_COUNT) as executor:
        executor.map(process_init, state_iter)


def process_init(state):
    """Initialize a hyperparameter optimization loop.
    Args:
    state :: ProcessState - the state that encapsulates the pending
                                 optimization

    Returns: nothing
    """
    # Redirect everything to a log file.
    log_file = state.file_name + ".log"
    log_file_path = os.path.join(DATA_PATH, log_file)
    with open(log_file_path, "w") as log:
        sys.stdout = sys.stderr = log

        # Display pid, time, slice.
        print("PID={}\nTIME={}\nSLICE_INDEX={}\nPULSE_TIME={}"
              "".format(os.getpid(), time.time(), state.slice_index,
                        state.pulse_time))

        # Define the search space on the parameters: learning rate.
        print("LR_LB={}, LR_UB={}, DECAY_LB={}, DECAY_UB={}"
              "".format(LR_LB, LR_UB, DECAY_LB, DECAY_UB))
        space = {
            'lr': hp.loguniform('lr', np.log(LR_LB), np.log(LR_UB)),
            'decay': hp.loguniform('decay', np.log(DECAY_LB), np.log(DECAY_UB)),
        }

        # Run optimization.
        objective_wrapper = lambda params: objective(state,
                                                     params)
        trials = Trials()
        best = fmin(objective_wrapper, space = space,
                    algo = tpe.suggest, max_evals = MAX_HPO_ITERATIONS,
                    trials = trials, show_progressbar = False)
        
        print("BEST={}".format(best))


def objective(state, params):
    """This is the function to minimize.
    Args:
    state :: ProcessState - the state that encapsulates the optimization
    params :: dict - the new parameters to run the objective for

    Returns: results :: dict - a results dictionary interpretable by hyperopt
    """
    # Grab and log parameters.
    lr = params['lr']
    decay = params['decay']
    print("\nITERATION={}\nLEARNING_RATE={}\nDECAY={}"
          "".format(state.iteration_count, lr, decay))

    # Build necessary grape arguments using parameters.
    U = state.uccsdslice.unitary()
    convergence = {'rate': lr,
                   'max_iterations': MAX_GRAPE_ITERATIONS,
                   'learning_rate_decay': decay}
    pulse_time = state.pulse_time
    steps = int(pulse_time * 100)
    
    # Run grape.
    grape_start_time = time.time()
    print("GRAPE_START_TIME={}".format(grape_start_time))
    grape_sess = Grape(H0, Hops, Hnames, U, pulse_time, steps,
                       STATES_CONCERNED_LIST, convergence = convergence,
                       reg_coeffs = REG_COEFFS, method = METHOD, maxA = MAX_AMPLITUDE,
                       use_gpu = USE_GPU, sparse_H = SPARSE_H,
                       show_plots = SHOW_PLOTS, file_name = state.file_name,
                       data_path = DATA_PATH)
    grape_end_time = time.time()
    print("GRAPE_END_TIME={}".format(grape_end_time))

    
    # Log results.
    print("LOSS={}".format(grape_sess.l))
    trial = {
        'iter': state.iteration_count,
        'lr': lr,
        'decay': decay,
        'loss': grape_sess.l,
        'wall_run_time': grape_end_time - grape_start_time,
    }
    trial_file = state.file_name + ".json"
    trial_file_path = os.path.join(DATA_PATH, trial_file)
    with open(trial_file_path, "a+") as trial_file:
        trial_file.write(json.dumps(trial, cls=CustomJSONEncoder)
                         + "\n")
    # Update state.
    state.trials.append(trial)
    state.iteration_count += 1

    return {
        'loss': grape_sess.l,
        'status': STATUS_OK,
    }


if __name__ == "__main__":
    main()




