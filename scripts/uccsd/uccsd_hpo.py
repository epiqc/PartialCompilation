"""
uccsd_hpo.py - A script for running hyperparameter optimization
               for GRAPE on a UCCSD circuit or slice. We optimize over
               learning rate and decay for a fixed angle and time.
               Pulse time is set to 25% of the circuit's max pulse time.
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

from fqc.data import UCCSD_DATA, SPN
from fqc.util import (get_unitary, get_max_pulse_time, CustomJSONEncoder)
from hyperopt import hp
from quantum_optimal_control.main_grape.grape import Grape
import ray
import ray.tune

### CONSTANTS ###

BASE_DATA_PATH = "/project/ftchong/qoc/thomas/hpo/"
PULSE_TIME_MULTIPLIER = 0.25

# Grape constants
GRAPE_MAX_ITERATIONS = int(1e3)
METHOD = "ADAM"
USE_GPU = False
SPARSE_H = False
SHOW_PLOTS = False
SAVE = False
GRAPE_TASK_CONFIG = {
    "method": METHOD,
    "use_gpu": USE_GPU,
    "sparse_H": SPARSE_H,
    "show_plots": SHOW_PLOTS,
    "save": SAVE,
}

# Hyperparmeter optimization constants and search space.
# MAX_HPO_ITERATIONS is high because we assumes that compute time
# is constrained and the optimization can always be resumed.
HPO_MAX_ITERATIONS = int(1e6)
LR_LB = 1e-5
LR_UB = 1
DECAY_LB = 1
DECAY_UB = 1e3

# Ray parameters, preallocate 5gb to obj store and redis shard respectively.
OBJECT_STORE_MEMORY = int(5e9)
REDIS_MAX_MEMORY = int(5e9)


### OBJECTS ###


# TODO: We do not know of a way to pass a shared object between process via ray,
# yet. But when we do we could include useful fileds like global_iteration_count,
# and giving each process its own stdout/err log file, etc.
# That means that this object is READ-ONLY for now.
# There may also be collision in writing to log files 
# that are dubbed with the file_name filed.
class ProcessState(object):
    """An object to encapsulate the HPO of a circuit.
    Fields:
    molecule :: string - identifies the uccsd molecule
    slice_index :: int - the index of the circuit slice to optimize over,
                   defaults to -1 if the full circuit is being optimized
    pulse_time :: float - the pulse time to optimize the circuit for
    circuit :: qiskit.QuantumCircuit - the circuit being optimized
    unitary :: np.matrix - the unitary that represents the circuit being optimized
    grape_config :: dict - molecule specific grape parameters
    data_path :: string
    file_name :: string - the identifier of the optimization
    """

    def __init__(self, molecule, slice_index=-1):
        """See corresponding class field declarations above for other arguments.
        """
        super()
        self.molecule = molecule
        self.slice_index = slice_index
        # If the slice index is -1, the full circuit is being optimized.
        if self.slice_index == -1:
            self.circuit = UCCSD_DATA[molecule]["CIRCUIT"]
            self.file_name = "full"
        else:
            self.circuit = UCCSD_DATA[molecule]["SLICES"][slice_index].circuit
            self.file_name = "s{}".format(slice_index)
        self.unitary = get_unitary(self.circuit)
        self.grape_config = UCCSD_DATA[molecule]["GRAPE_CONFIG"]
        self.grape_config.update(GRAPE_TASK_CONFIG)
        self.pulse_time = get_max_pulse_time(self.circuit) * PULSE_TIME_MULTIPLIER
        self.data_path = os.path.join(BASE_DATA_PATH,
                                      "uccsd_{}".format(molecule.lower()))
        # TODO: We assume BASE_DATA_PATH exists.
        if not os.path.exists(self.data_path):
            os.mkdir(self.data_path)
            

### MAIN METHODS ###

def main():
    """Run HPO on one circuit.
    """
    # Handle CLI.
    parser = argparse.ArgumentParser()
    parser.add_argument("--molecule", type=str, default="H2", help="the "
                        "UCCSD molecule to perform HPO on")
    parser.add_argument("--slice-index", type=int, default=-1, help="the "
                        "slice to perform HPO on, do not specify to run HPO "
                        "on the full circuit")
    parser.add_argument("--core-count", type=int, default=1, help="the "
                        "number of cpu cores this run may use")
    args = vars(parser.parse_args())
    molecule = args["molecule"]
    slice_index = args["slice_index"]
    core_count = args["core_count"]

    # Generate the state object that encapsulates the optimization for the circuit.
    state = ProcessState(molecule, slice_index)

    # Redirect everything the central process puts out to a log file.
    # By default, ray redirects the stdout of each worker process
    # to the central process.
    log_file = state.file_name + ".log"
    log_file_path = os.path.join(state.data_path, log_file)
    with open(log_file_path, "a+") as log:
        # sys.stdout = sys.stderr = log

        # Display run characteristics.
        print("PID={}\nWALL_TIME={}\nSLICE_INDEX={}\nPULSE_TIME={}\n"
              "(LR_LB, LR_UB)=({}, {})\n(DECAY_LB, DECAY_UB)=({}, {})\n"
              "CORE_COUNT={}\n{}"
              "".format(os.getpid(), time.time(), state.slice_index,
                        state.pulse_time, LR_LB, LR_UB, DECAY_LB, DECAY_UB, 
                        core_count, state.circuit))

        # Define the search space on the parameters: learning rate and
        # learning rate decay.
        space = {
            "lr": hp.loguniform("lr", np.log(LR_LB), np.log(LR_UB)),
            "decay": hp.uniform("decay", DECAY_LB, DECAY_UB),
        }
        
        # We want to minimize QOC error/loss, i.e. we want to maximize
        # negative loss.
        algo = ray.tune.suggest.HyperOptSearch(space, max_concurrent=core_count,
                                               reward_attr="neg_loss")
        run_config = {
            "num_samples": HPO_MAX_ITERATIONS,
            "name": state.file_name,
            "loggers": [ray.tune.logger.NoopLogger],
            "search_alg": algo,
            "verbose": 1,
            "local_dir": state.data_path,
            "resume": True,
        }
        
        # Ray cannot serialize python objects in its object store,
        # so we have to pass the state in a lambda wrapper.
        objective_wrapper = lambda config, reporter: objective(state, config,
                                                               reporter)
        
        # Start ray and run HPO.
        ray.init(num_cpus=core_count, object_store_memory=OBJECT_STORE_MEMORY,
                 redis_max_memory=REDIS_MAX_MEMORY)
        ray.tune.register_trainable("lambda_id", objective_wrapper)
        ray.tune.run("lambda_id", **run_config)


def objective(state, config, reporter):
    """This function takes hyperparameters and reports their loss.
    Args:
    state :: ProcessState - contains information about the slice
    config :: dict - contains the hyperparameters to evaluate and other
                     information ray or we specified
    reporter :: ray.tune.function_runner.StatusReporter - report the loss
                                                          to this object
    Returns: nothing
    """
    # Unpack config. Log parameters.
    lr = config["lr"]
    decay = config["decay"]
    print("LEARNING_RATE={}\nDECAY={}"
          "".format(lr, decay))

    # Build necessary grape arguments using parameters.
    U = state.unitary
    convergence = {'rate': lr,
                   'max_iterations': GRAPE_MAX_ITERATIONS,
                   'learning_rate_decay': decay}
    pulse_time = state.pulse_time
    steps = int(pulse_time * SPN)
    
    # Run grape.
    grape_start_time = time.time()
    print("GRAPE_START_TIME={}".format(grape_start_time))
    grape_sess = Grape(U=U, total_time=pulse_time, steps=steps,
                       convergence=convergence, file_name=state.file_name,
                       data_path=state.data_path, **state.grape_config)
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
    trial_file_path = os.path.join(state.data_path, trial_file)
    with open(trial_file_path, "a+") as trial_file:
        trial_file.write(json.dumps(trial, cls=CustomJSONEncoder)
                         + "\n")
    
    # Report results.
    reporter(neg_loss=-loss, done=True)


if __name__ == "__main__":
    main()




