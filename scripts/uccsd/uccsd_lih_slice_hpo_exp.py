"""
uccsd_lih_slice_hpo_exp.py - This is a script to find out how good hyperparameter (hp) choices
vary with pulse time and angle. Previous results form the hypothesis that good hp choices are agnostic
to pulse time and angle, and are dependent on the circuit. To test this hypothesis we will run
50 iterations of hyperparaeter optimization (hpo) on UCCSD LiH slices for 3 different
angles (90deg, 180deg, 270deg) at 50% of gate-based compilation time and 3 different pulse times
(25%, 50%, 75%) of gate based compilation time at 180degrees. We reuse the trial where angle = 180deg, and
time = 50% of gate-based compilation time in the plotting of results. We run our hpo in serial.
"""

# Set random seeds for reasonable reproducibility.
import random
random.seed(0)
import numpy as np
np.random.seed(1)
import tensorflow as tf
tf.set_random_seed(2)
import os
os.environ["HYPEROPT_FMIN_SEED"] = "23"

import argparse
import json
import sys
import time

from fqc.data import UCCSD_LIH_THETA
from fqc.uccsd import get_uccsd_circuit, get_uccsd_slices
from fqc.util import (optimize_circuit, get_unitary,
                      get_nearest_neighbor_coupling_list,
                      get_max_pulse_time, CustomJSONEncoder)
from hyperopt import hp, fmin, tpe, Trials, STATUS_OK
from mpi4py.futures import MPIPoolExecutor
from quantum_optimal_control.main_grape.grape import Grape
from quantum_optimal_control.core.hamiltonian import (get_H0, 
        get_Hops_and_Hnames, get_full_states_concerned_list, get_maxA)


### CONSTANTS ###


DATA_PATH = ""
ANGLES = [90, 180, 270]
ANGLE_CONSTANT = 180
TIME_MULTIPLIERS = [0.25, 0.5, 0.75]
TIME_MULTIPLIER_CONSTANT = 0.5
JOBS_PER_SLICE = 5
BROADWELL_CORE_COUNT = 14

# Grape constants.
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
# Wave steps per nanosecond of pulse time.
SPN = 20

# Get slices and information.
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
    angle :: float - the value of the parameterized gates of the slice
    """

    def __init__(self, uccsdslice, slice_index, angle_deg, pulse_time):
        """See corresponding class field declarations above for other arguments.
        """
        super(ProcessState, self).__init__()
        self.uccsdslice = uccsdslice
        self.slice_index = slice_index
        self.angle_deg = angle_deg
        self.angle = np.deg2rad(angle_deg)
        uccsdslice.update_angles([self.angle] * len(uccsdslice.angles))
        self.pulse_time = pulse_time
        self.file_name = ("s{}_a{}_t{}"
                          "".format(slice_index, angle_deg, int(pulse_time)))
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
    state_iter = list()
    for i, uccsdslice in enumerate(slices):
        slice_index = i + slice_start
        max_pulse_time = get_max_pulse_time(uccsdslice.circuit)
        for angle_deg in ANGLES:
            state_iter.append(ProcessState(uccsdslice, slice_index, angle_deg,
                                           max_pulse_time * TIME_MULTIPLIER_CONSTANT))
        for k in TIME_MULTIPLIERS:
            if not k == TIME_MULTIPLIER_CONSTANT:
                state_iter.append(ProcessState(uccsdslice, slice_index, ANGLE_CONSTANT,
                                               max_pulse_time * k))
    # ENDFOR
    
    # Run optimization on the slices.
    core_count = min(slice_count * JOBS_PER_SLICE, BROADWELL_CORE_COUNT)
    with MPIPoolExecutor(core_count) as executor:
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

        # Display details about this run.
        print("PID={}\nWALL_TIME={}\nSLICE_INDEX={}\nANGLE={}\nPULSE_TIME={}\n{}"
              "".format(os.getpid(), time.time(), state.slice_index,
                        state.angle_deg, state.pulse_time, state.uccsdslice.circuit))

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
    steps = int(pulse_time * SPN)
    
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




