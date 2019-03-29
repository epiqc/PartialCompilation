"""
uccsd_slice_hpo_v3.py - A script for computing the appropriate hyperparameters
                        for GRAPE on a UCCSD slice. We optimize over learning rate
                        and decay for a fixed angle and time on one slice.
                        Pulse time is set to 25% of the max pulse time.
                        Angle is pregenerated. We do HPO in parallel using ray.
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
from hyperopt import hp
from quantum_optimal_control.main_grape.grape import Grape
from quantum_optimal_control.core.hamiltonian import (get_H0, 
        get_Hops_and_Hnames, get_full_states_concerned_list, get_maxA)
import ray

### CONSTANTS ###


DATA_PATH = "/project/ftchong/qoc/thomas/uccsd_slice_hpo/lih_v3"
PULSE_TIME_MULTIPLIER = 0.25

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
SAVE = False
GRAPE_CONFIG = {
    "H0": H0,
    "Hops": Hops,
    "Hnames": Hnames,
    "states_concerned_list": STATES_CONCERNED_LIST,
    "reg_coeffs": REG_COEFFS,
    "maxA": MAX_AMPLITUDE,
    "use_gpu": USE_GPU,
    "show_plots": SHOW_PLOTS,
    "method": METHOD,
    "data_path": DATA_PATH,
    "save": SAVE,
}
# How many pulse steps per nanosecond?
SPN = 20

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

# How many cores do you want to give to this optimization?
# This is the same as the maximum number of active hyperparameter evaluations.
BROADWELL_CORE_COUNT = 14
CORE_COUNT = 2

### OBJECTS ###


# TODO: We do not know of a way to pass a shared object between process via ray,
# yet. But when we do we could include useful fileds like global_iteration_count,
# and giving each process its own stdout/err log file, etc.
# That means that this object is READ-ONLY for now.
# There may also be collision in writing to log files 
# that are dubbed with the file_name filed.
class ProcessState(object):
    """An object to encapsulate the HPO of a slice.
    Fields:
    file_name :: string - the identifier of the optimization
    pulse_time :: float - the pulse time to optimize the slice for (in nanoseconds)
    slice_index :: int - the index of the slice that is being optimized
    uccsdslice :: fqc.uccsd.uccsdslice.UCCSDSlice - the slice that
                                                    is being optimized
    """

    def __init__(self, uccsdslice, slice_index, pulse_time):
        """See corresponding class field declarations above for other arguments.
        """
        super()
        self.uccsdslice = uccsdslice
        self.slice_index = slice_index
        self.pulse_time = pulse_time
        self.file_name = ("s{}".format(slice_index))


### MAIN METHODS ###


# TODO: This module is currently equipped to run HPO for a single slice.
# That is because we are unsure what the best way to instantiate seperate
# ray instances on the same node is. One option would be to have an
# outer MPI loop that spawns a few beauracratic processes to instantiate
# ray instances and then we have one ray instance per HPO. This seems
# unlikely to be the best method. However, each HPO instance takes
# variable amount of time and the current implementation provides
# a nice modularity to running jobs. If we were to have multiple
# HPO jobs on the same node and one terminates earlier than the others,
# we have cores sitting idle. Unless we can figure out a way to let the
# other jobs know that there is more room to party.
def main():
    """The meat and potatoes, the central dogma, what you paid to see.
    """
    # Handle CLI.
    parser = argparse.ArgumentParser()
    parser.add_argument("--slice-index", type=int, default=0, help="the "
                        "slice to perform HPO on (0-7)")
    args = vars(parser.parse_args())
    slice_index = args["slice_index"]

    # Generate the state object that encapsulates the optimization for the slice.
    uccsdslice = UCCSD_LIH_SLICES[slice_index]
    pulse_time = PULSE_TIME_MULTIPLIER * get_max_pulse_time(uccsdslice.circuit)
    state = ProcessState(uccsdslice, slice_index, pulse_time)

    # Redirect everything the central process puts out to a log file.
    # By default, ray redirects the stdout of each worker process
    # to the central process.
    log_file = state.file_name + ".log"
    log_file_path = os.path.join(DATA_PATH, log_file)
    with open(log_file_path, "w") as log:
        sys.stdout = sys.stderr = log

        # Display run characteristics.
        print("PID={}\nWALL_TIME={}\nSLICE_INDEX={}\nPULSE_TIME={}\n{}"
              "".format(os.getpid(), time.time(), state.slice_index,
                        state.pulse_time, state.uccsdslice.circuit))

        # Define the search space on the parameters: learning rate,
        # learning rate decay.
        print("LR_LB={}, LR_UB={}, DECAY_LB={}, DECAY_UB={}"
              "".format(LR_LB, LR_UB, DECAY_LB, DECAY_UB))
        space = {
            "lr": hp.loguniform("lr", np.log(LR_LB), np.log(LR_UB)),
            "decay": hp.loguniform("decay", np.log(DECAY_LB), np.log(DECAY_UB)),
        }
        
        # We want to minimize QOC error/loss, i.e. we want to maximize
        # negative loss.
        algo = ray.tune.suggest.HyperOptSearch(space, max_concurrent=CORE_COUNT,
                                               reward_attr="neg_loss")

        run_config = {
            "num_samples": MAX_HPO_ITERATIONS,
            "name": state.file_name,
            "loggers": [ray.tune.logger.NoopLogger],
            "search_alg": algo,
            "verbose": 1,
            "local_dir": DATA_PATH,
            # This config is passed to the objective.
            "config": {
                "state": state,
            },
        }

        # Start ray and run HPO.
        ray.init(num_cpus=CORE_COUNT)
        ray.tune.run(objective, **run_config)


def objective(config, reporter):
    """This function takes hyperparameters and reports their loss.
    Args:
    config :: dict - contains the hyperparameters to evaluate and other
                     information ray or we specified
    reporter :: ray.tune.function_runner.StatusReporter - report the loss
                                                          to this object
    Returns: nothing
    """
    # Unpack config. Log parameters.
    lr = config["lr"]
    decay = config["decay"]
    state = config["state"]
    print("LEARNING_RATE={}\nDECAY={}"
          "".format(lr, decay))

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
    grape_sess = Grape(U=U, total_time=pulse_time, steps=steps,
                       convergence=convergence, **GRAPE_CONFIG)
    grape_end_time = time.time()
    print("GRAPE_END_TIME={}".format(grape_end_time))

    
    # Log results.
    loss = grape_sess.l
    print("LOSS={}".format(loss))
    trial = {
        'lr': lr,
        'decay': decay,
        'loss': loss,
        'wall_run_time': grape_end_time - grape_start_time,
    }
    trial_file = state.file_name + ".json"
    trial_file_path = os.path.join(DATA_PATH, trial_file)
    with open(trial_file_path, "a+") as trial_file:
        trial_file.write(json.dumps(trial, cls=CustomJSONEncoder)
                         + "\n")
    
    # Report results.
    reporter(neg_loss=-loss, done=True)


if __name__ == "__main__":
    main()




