"""
uccsd_beh2_full_hpo.py - A script attempting convergence for QOC
                         on the UCCSD BEH2 molecule.
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

import argparse
from fqc.data import UCCSD_BEH2_THETA
from fqc.uccsd import get_uccsd_circuit, get_uccsd_slices, MOLECULE_TO_INFO
from fqc.util import (optimize_circuit, get_unitary,
                      get_nearest_neighbor_coupling_list, get_max_pulse_time,
                      merge_rotation_gates)
from hyperopt import fmin, tpe, hp, STATUS_OK, Trials
from mpi4py.futures import MPIPoolExecutor
from quantum_optimal_control.main_grape.grape import Grape
from quantum_optimal_control.core.hamiltonian import (get_H0, 
        get_Hops_and_Hnames, get_full_states_concerned_list, get_maxA)


### CONSTANTS ###


# Grape constants.
DATA_PATH = "/project/ftchong/qoc/thomas/uccsd_full_hpo/beh2"
# Define hardware specific parameters.
NUM_QUBITS = 6
NUM_STATES = 2
CONNECTED_QUBIT_PAIRS = get_nearest_neighbor_coupling_list(3, 2, directed=False)
H0 = np.zeros((NUM_STATES ** NUM_QUBITS, NUM_STATES ** NUM_QUBITS))
Hops, Hnames = get_Hops_and_Hnames(NUM_QUBITS, NUM_STATES, CONNECTED_QUBIT_PAIRS)
STATES_CONCERNED_LIST = get_full_states_concerned_list(NUM_QUBITS, NUM_STATES)
MAX_AMPLITUDE = get_maxA(NUM_QUBITS, NUM_STATES, CONNECTED_QUBIT_PAIRS)
METHOD = 'ADAM'
MAX_GRAPE_ITERATIONS = 1000
DECAY = MAX_GRAPE_ITERATIONS
REG_COEFFS = {}
USE_GPU = False
SPARSE_H = False
SHOW_PLOTS = False
# Optimization granularity.
STEPS_PER_NANOSECOND = 20

# Construct the circuit.
UCCSD_BEH2_FULL_CIRCUIT = optimize_circuit(get_uccsd_circuit('BeH2', UCCSD_BEH2_THETA),
                                           CONNECTED_QUBIT_PAIRS)

# Hyperparmeter optimization constants.
MAX_HPO_ITERATIONS = 100
LR_LB = 1e-5
LR_UB = 1


### OBJECTS ###


class OptimizationState(object):
    """A class to track the state of an optimization loop.
    Fields:
    file_name :: string - the identifier of the optimization
    iteration_count :: int - tracks how many iterations the optimization
                             has performed
    pulse_time :: float - the pulse time to optimize the slice for (in nanoseconds)
    trials :: [dict] - list of dictionaries, each detailing an iteration
                       of optimization
    circuit :: qiskit.QuantumCircuit - the circuit being GRAPE'd
    """

    def __init__(self, circuit, pulse_time):
        """
        Args:
        angle_deg :: float - the angle to parameterize the theta dependent gates
                     of the slice (in degrees)
        See corresponding class field declarations above for other arguments.
        """
        super(OptimizationState, self).__init__()
        self.circuit = circuit
        self.pulse_time = pulse_time
        self.file_name = ("t{}"
                          "".format(pulse_time))
        self.trials = list()
        self.iteration_count = 0


### MAIN METHODS ###


def main():
    # Handle CLI.
    parser = argparse.ArgumentParser()
    parser.add_argument("--time-start", type=float, default=0.0, help="the "
                        "inclusive lower bound of time steps to optimize the "
                        "slice for (units in nanoseconds, behaves like "
                        "np.arange)")
    parser.add_argument("--time-stop", type=float, default=1.0, help="the "
                        "exclusive upper bound of time steps to optimize the "
                        "slice for (units in nanoseconds, behaves like "
                        "np.arange)")
    parser.add_argument("--time-step", type=float, default=0.5, help="the step size "
                        "between time values (units in nanoseconds, behaves "
                        "like np.arange)")
    args = vars(parser.parse_args())
    time_start = args["time_start"]
    time_stop = args["time_stop"]
    time_step = args["time_step"]

    # Get a list of time steps.
    pulse_time_list = list(np.arange(time_start, time_stop, time_step))
    pulse_time_count = len(pulse_time_list)

    # Generate the state objects to encapsulate the optimization of each slice.
    state_iter = list()
    for pulse_time in pulse_time_list:
        state_iter.append(OptimizationState(UCCSD_BEH2_FULL_CIRCUIT, pulse_time))

    # Run optimization on the slices.
    with MPIPoolExecutor(BROADWELL_CORE_COUNT) as executor:
        executor.map(process_init, state_iter)


def process_init(state):
    """Initialize a hyperparameter optimization loop.
    Args:
    state :: OptimizationState - the state that encapsulates the pending
                                 optimization

    Returns: nothing
    """
    # Redirect everything to a log file.
    log_file = state.file_name + ".log"
    log_file_path = os.path.join(DATA_PATH, log_file)
    with open(log_file_path, "w") as log:
        sys.stdout = sys.stderr = log

        # Display pid, time, slice.
        print("PID={}\nTIME={}\nPULSE_TIME={}"
              "".format(os.getpid(), time.time(), state.pulse_time))
        # print(state.circuit)

        # Define the search space on the parameters: learning rate.
        print("LR_LB={}, LR_UB={}"
              "".format(LR_LB, LR_UB))
        space = {
            'lr': hp.loguniform('lr', np.log(LR_LB), np.log(LR_UB)),
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
    state :: OptimizationState - the state that encapsulates the optimization
    params :: dict - the new parameters to run the objective for

    Returns: results :: dict - a results dictionary interpretable by hyperopt
    """
    # Grab and log parameters.
    lr = params['lr']
    print("\nITERATION={}\nLEARNING_RATE={}"
          "".format(state.iteration_count, lr))

    # Build necessary grape arguments using parameters.
    U = get_unitary(state.circuit)
    convergence = {'rate': params['lr'],
                   'max_iterations': MAX_GRAPE_ITERATIONS,
                   'learning_rate_decay': DECAY}
    pulse_time = state.pulse_time
    steps = int(pulse_time * STEPS_PER_NANOSECOND)
    
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




