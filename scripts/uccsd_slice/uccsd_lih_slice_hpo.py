"""
uccsd_lih_slice_hpo.py - A script for computing the appropriate hyperparameters
                         for GRAPE on a UCCSD slice.
"""

import argparse
from itertools import product
import os
import sys
import time

from fqc.data import UCCSD_LIH_THETA
from fqc.uccsd import get_uccsd_circuit, get_uccsd_slices
from fqc.util import (optimize_circuit, get_unitary,
                      get_nearest_neighbor_coupling_list,
                      get_max_pulse_time)
from hyperopt import fmin, tpe, hp, STATUS_OK, Trials
from mpi4py.futures import MPIPoolExecutor
import numpy as np
from quantum_optimal_control.main_grape.grape import Grape
from quantum_optimal_control.core.hamiltonian import (get_H0, 
        get_Hops_and_Hnames, get_full_states_concerned_list, get_maxA)


### CONSTANTS ###


# Grape constants.
DATA_PATH = "/project/ftchong/qoc/thomas/uccsd_slice_hpo/"
# Define hardware specific parameters.
NUM_QUBITS = 4
NUM_STATES = 2
CONNECTED_QUBIT_PAIRS = get_nearest_neighbor_coupling_list(2, 2, directed=False)
H0 = get_H0(NUM_QUBITS, NUM_STATES)
Hops, Hnames = get_Hops_and_Hnames(NUM_QUBITS, NUM_STATES, CONNECTED_QUBIT_PAIRS)
STATES_CONCERNED_LIST = get_full_states_concerned_list(NUM_QUBITS, NUM_STATES)
MAX_AMPLITUDE = get_maxA(NUM_QUBITS, NUM_STATES, CONNECTED_QUBIT_PAIRS)
METHOD = 'ADAM'
MAX_GRAPE_ITERATIONS = 1000
MIN_GRADS = 1e-12
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

# Hyperparmeter optimization constants.
MAX_HPO_ITERATIONS = 1000


### OBJECTS ###


class OptimizationState(object):
    """A class to track the state of an optimization loop.
    Fields:
    file_name :: string - the identifier of the optimization
    iteration_count :: int - tracks how many iterations the optimization
                             has performed
    slice_index :: int - the index of the slice that is being optimized
    uccsdslice :: fqc.uccsd.uccsdslice.UCCSDSlice - the slice that
                                                    is being optimized
    """

    def __init__(self, uccsdslice, slice_index):
        """
        Args:
        uccsdslice :: fqc.uccsd.uccsdslice.UCCSDSlice - the slice that
                                                        is being optimized
        slice_index :: int - the index of the slice that is being optimized
        """
        super(OptimizationState, self).__init__()
        self.uccsdslice = uccsdslice
        self.slice_index = slice_index
        self.file_name = "s{}".format(slice_index)
        self.iteration_count = 0
        

### MAIN METHODS ###


def main():
    # Generate the state objects to encapsulate the optimization of each slice.
    state_iter = [OptimizationState(uccsdslice, i)
                  for i, uccsdslice in enumerate(UCCSD_LIH_SLICES)]
        
    # Run optimization on the slices.
    # We currently only optimize on the first slice because this
    # method is in development. But you could imagine optimizing
    # over all the slices and multiple angles by using an MPIPool here.
    process_init(state_iter[0])


def process_init(state):
    """Initialize a hyperparameter optimization loop.
    Args:
    state :: OptimizationState - the state that encapsulates the pending
                                 optimization

    Returns: nothing
    """
    # Redirect everything to a log file.
    log_file = state.file_name + '.log'
    log_file_path = os.path.join(DATA_PATH, log_file)
    with open(log_file_path, "w") as log:
        # sys.stdout = sys.stderr = log

        # Display pid, time, slice.
        print("PID={}\nTIME={}\nSLICE_INDEX={}"
              "".format(os.getpid(), time.time(), state.slice_index))
        print(state.uccsdslice.circuit)

        # Define the search space on the parameters: pulse time, learning rate,
        # and learning rate decay.
        time_ub = get_max_pulse_time(state.uccsdslice.circuit)
        time_lb = 0
        lr_lb = 1e-5
        lr_ub = 1
        decay_lb = 0
        decay_ub = 1000
        print("TIME_UB={}, TIME_LB={}, LR_LB={}, LR_UB={}, DECAY_LB={}, DECAY_UB={}"
              "".format(time_ub, time_lb, lr_lb, lr_ub, decay_lb, decay_ub))
        space = {
            'time': hp.uniform('time', time_lb, time_ub),
            'lr': hp.loguniform('lr', np.log(lr_lb), np.log(lr_ub)),
            'decay': hp.uniform('decay', decay_lb, decay_ub),
        }

        # Run optimization.
        objective_wrapper = lambda params: objective(state,
                                                     params)
        trials = Trials()
        best = fmin(objective_wrapper, space = space,
                    algo = tpe.suggest, max_evals = MAX_HPO_ITERATIONS,
                    trials = trials)
        
        print("BEST={}".format(best))


def objective(state, params):
    """This is the function to minimize.
    Args:
    state :: OptimizationState - the state that encapsulates the optimization
    params :: dict - the new parameters to run the objective for

    Returns: results :: dict - a results dictionary interpretable by hyperopt
    """
    # Grab and log parameters.
    pulse_time = params['time']
    lr = params['lr']
    decay = params['decay']
    print("\nITERATION={}\nPULSE_TIME={}\nLEARNING_RATE={}\nDECAY={}"
          "".format(state.iteration_count, pulse_time, lr, decay))

    # Build necessary grape arguments using parameters.
    U = state.uccsdslice.unitary()
    convergence = {'rate': params['lr'],
                   'max_iterations': MAX_GRAPE_ITERATIONS,
                   'learning_rate_decay': params['decay'],
                   'min_grads': MIN_GRADS}
    steps = int(pulse_time * 100)
    
    # Run grape.
    print("GRAPE_START_TIME={}".format(time.time()))
    grape_sess = Grape(H0, Hops, Hnames, U, pulse_time, steps,
                       STATES_CONCERNED_LIST, convergence = convergence,
                       reg_coeffs = REG_COEFFS, method = METHOD, maxA = MAX_AMPLITUDE,
                       use_gpu = USE_GPU, sparse_H = SPARSE_H,
                       show_plots = SHOW_PLOTS, file_name = state.file_name,
                       data_path = DATA_PATH)
    print("GRAPE_END_TIME={}".format(time.time()))
    
    # Log results.
    print("LOSS={}".format(grape_sess.l))

    # Update state.
    state.iteration_count += 1

    return {
        'loss': grape_sess.l,
        'status': STATUS_OK,
    }


if __name__ == "__main__":
    main()
