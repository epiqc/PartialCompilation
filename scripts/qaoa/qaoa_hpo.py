"""
qaoa_hpo.py - A script for running hyperparameter optimization
              for GRAPE on our qaoa benchmarks.
              We block the circuits into 4 and 2-qubit chunks.
              We optimize over learning rate and decay for a fixed angle and time.
              Pulse time is set to 25% of the circuit's max pulse time if it's
              depth is greater than 10 gates on the critical path and 50% otherwise.
              Angle is randomly generated. We do HPO in parallel using ray.
"""
import random
random.seed(0)
import numpy as np
np.random.seed(1)
import tensorflow as tf
tf.set_random_seed(2)
import os
os.environ["HYPEROPT_FMIN_SEED"] = "23"

import argparse
from copy import deepcopy
import fcntl
import json
import pickle
import sys
import time

from fqc.uccsd import (get_uccsd_slices, get_uccsd_circuit)
from fqc.util import (get_unitary, get_max_pulse_time, CustomJSONEncoder,
                      squash_circuit, get_nearest_neighbor_coupling_list,
                      optimize_circuit)
from hyperopt import hp

from qiskit import QuantumCircuit, QuantumRegister
from quantum_optimal_control.main_grape.grape import Grape
from quantum_optimal_control.core.hamiltonian import (get_Hops_and_Hnames,
                                                      get_full_states_concerned_list,
                                                      get_maxA)
import ray
import ray.tune

### CONSTANTS ###

BASE_DATA_PATH = "/project/ftchong/qoc/thomas/hpo/"
PULSE_TIME_MULTIPLIER = 0.25
REDUCED_PULSE_TIME_MULTIPLIER = 0.5
REDUCED_CIRCUIT_DEPTH_CUTOFF = 10

NUM_STATES = 2
# Optimiztion time intervals (steps)
# per nanosecond of pulse time.
SPN = 20.

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

# Ray parameters, preallocate 1gb to obj store and redis shard respectively.
OBJECT_STORE_MEMORY = int(1e9)
REDIS_MAX_MEMORY = int(1e9)


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
    qaoa_id :: str - identifies the benchmark
    slice_index :: str - the circuit indexed into all circuits of the qaoa_id circuits
    rz_index :: str - the circuit indexed into the rz containing circuits of the qaoa_id circuits
    circuit :: qiskit.QuantumCircuit - the circuit being optimized
    connected_qubit_pairs :: list((int, int)) - represents physical
                                                constraints of qubits
    unitary :: np.matrix - the unitary that represents the circuit being optimized
    pulse_time :: float - the pulse time to optimize the circuit for
    grape_config :: dict - circuit specific grape parameters
    file_name :: string - the identifier of the optimization
    data_path :: string - output directory
    """

    def __init__(self, qaoa_id, slice_index, rz_index, circuit, connected_qubit_pairs):
        """See corresponding class field declarations above for other arguments.
        """
        super()
        self.qaoa_id = qaoa_id
        self.slice_index = slice_index
        self.rz_index = rz_index
        self.circuit = circuit
        self.connected_qubit_pairs = connected_qubit_pairs
        self.unitary = get_unitary(self.circuit)
        if self.circuit.depth() > REDUCED_CIRCUIT_DEPTH_CUTOFF:
            self.pulse_time = get_max_pulse_time(self.circuit) * PULSE_TIME_MULTIPLIER
        else:
            self.pulse_time = get_max_pulse_time(self.circuit) * REDUCED_PULSE_TIME_MULTIPLIER
        self.file_name = "s{}".format(self.slice_index)
        self.data_path = os.path.join(BASE_DATA_PATH,
                                      "qaoa_{}".format(qaoa_id.lower()))
        # TODO: We assume BASE_DATA_PATH exists.
        if not os.path.exists(self.data_path):
            os.mkdir(self.data_path)
        
        # Set grape parameters.
        num_qubits = self.circuit.width()
        H0 = np.zeros((NUM_STATES ** num_qubits, NUM_STATES ** num_qubits))
        Hops, Hnames = get_Hops_and_Hnames(num_qubits, NUM_STATES, self.connected_qubit_pairs)
        states_concerned_list = get_full_states_concerned_list(num_qubits, NUM_STATES)
        maxA = get_maxA(num_qubits, NUM_STATES, self.connected_qubit_pairs)
        reg_coeffs = {}
        self.grape_config = {
            "H0": H0,
            "Hops": Hops,
            "Hnames": Hnames,
            "states_concerned_list": states_concerned_list,
            "reg_coeffs": reg_coeffs,
            "maxA": maxA,
        }
        self.grape_config.update(GRAPE_TASK_CONFIG)
        

### MAIN METHODS ###

def main():
    """Run HPO on one circuit.
    """
    # Handle CLI.
    parser = argparse.ArgumentParser()
    parser.add_argument("--id", type=str, default="n6e", help="the "
                        "id of the qaoa benchmark")
    parser.add_argument("-p", type=int, default=1, help="the p value "
                        "of the qaoa benchmark")
    parser.add_argument("--index", type=int, default=0, help="the "
                        "index-th theta-dependent circuit")
    parser.add_argument("--core-count", type=int, default=1, help="the "
                        "number of cpu cores this run may use")
    args = vars(parser.parse_args())
    q_id = args["id"]
    rz_index = args["index"]
    p = args["p"]
    core_count = args["core_count"]
    
    # Grab appropriate circuit based on molecule and slice index.
    circuit_file = "{}_circuits.pickle".format(q_id.lower())
    circuit_file_path = os.path.join(BASE_DATA_PATH, circuit_file)
    with open(circuit_file_path, "rb") as f:
        rz_indices, circuits = pickle.load(f)[p - 1]
    slice_index = rz_indices[rz_index]
    circuit, connected_qubit_pairs = circuits[slice_index]

    # Generate the state object that encapsulates the optimization for the circuit.
    qaoa_id = "{}_p{}".format(q_id, p)
    state = ProcessState(qaoa_id, slice_index, rz_index, circuit, connected_qubit_pairs)

    # Redirect everything the central process puts out to a log file.
    # By default, ray redirects the stdout of each worker process
    # to the central process.
    log_file = "{}.log".format(state.file_name)
    log_file_path = os.path.join(state.data_path, log_file)
    with open(log_file_path, "a+") as log:
        sys.stdout = sys.stderr = log

        # Display run characteristics.
        print("PID={}\nWALL_TIME={}\nSLICE_INDEX={}\nRZ_INDEX={}\nPULSE_TIME={}\n"
              "(LR_LB, LR_UB)=({}, {})\n(DECAY_LB, DECAY_UB)=({}, {})\n"
              "CORE_COUNT={}\n{}"
              "".format(os.getpid(), time.time(), state.slice_index,
                        state.rz_index,
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
    trial_entry = "{}\n".format(json.dumps(trial, cls=CustomJSONEncoder))
    trial_file = "{}.json".format(state.file_name)
    trial_file_path = os.path.join(state.data_path, trial_file)
    with open(trial_file_path, "a+") as trial_file:
        fcntl.flock(trial_file, fcntl.LOCK_EX)
        trial_file.write(trial_entry)
        fcntl.flock(trial_file, fcntl.LOCK_UN)
    
    # Report results.
    reporter(neg_loss=-loss, done=True)


if __name__ == "__main__":
    main()




