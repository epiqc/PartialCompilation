"""
uccsd_hpo_blocking.py - A script for running hyperparameter optimization
                        for GRAPE on the (BeH2, NaH, and H2O) molecules.
                        We block the circuits into 4 and 2-qubit chunks.
                        We optimize over learning rate and decay for a fixed angle and time.
                        Pulse time is set to 25% of the circuit's max pulse time.
                        Angle is randomly generated. We do HPO in parallel using ray.
"""
import argparse
from copy import deepcopy
import fcntl
import json
import os
import pickle
import random
import sys
import time

from fqc.uccsd import (get_uccsd_slices, get_uccsd_circuit)
from fqc.util import (get_unitary, get_max_pulse_time, CustomJSONEncoder,
                      squash_circuit, get_nearest_neighbor_coupling_list,
                      optimize_circuit)
from hyperopt import hp
import numpy as np
from qiskit import QuantumCircuit, QuantumRegister
from quantum_optimal_control.main_grape.grape import Grape
from quantum_optimal_control.core.hamiltonian import (get_Hops_and_Hnames,
                                                      get_full_states_concerned_list,
                                                      get_maxA)
import ray
import ray.tune
import tensorflow as tf

### CONSTANTS ###

BASE_DATA_PATH = "/project/ftchong/qoc/thomas/hpo/"
PULSE_TIME_MULTIPLIER = 0.25
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
    molecule :: string - identifies the uccsd molecule
    slice_index :: int - identifies the circuit in the list of
                         get_<molecule>_circuits_to_compile
    circuit :: qiskit.QuantumCircuit - the circuit being optimized
    connected_qubit_pairs :: list((int, int)) - represents physical
                                                constraints of qubits
    unitary :: np.matrix - the unitary that represents the circuit being optimized
    pulse_time :: float - the pulse time to optimize the circuit for
    grape_config :: dict - circuit specific grape parameters
    file_name :: string - the identifier of the optimization
    data_path :: string - output directory
    """

    def __init__(self, molecule, slice_index, circuit, connected_qubit_pairs):
        """See corresponding class field declarations above for other arguments.
        """
        super()
        self.molecule = molecule
        self.slice_index = slice_index
        self.circuit = circuit
        self.connected_qubit_pairs = connected_qubit_pairs
        self.unitary = get_unitary(self.circuit)
        self.pulse_time = get_max_pulse_time(self.circuit) * PULSE_TIME_MULTIPLIER
        self.file_name = "s{}".format(self.slice_index)
        self.data_path = os.path.join(BASE_DATA_PATH,
                                      "uccsd_{}".format(molecule.lower()))
        # TODO: We assume BASE_DATA_PATH exists.
        if not os.path.exists(self.data_path):
            os.mkdir(self.data_path)
        
        # Set grape parameters.
        num_qubits = self.circuit.width()
        H0 = np.zeros((NUM_STATES ** num_qubits, NUM_STATES ** num_qubits))
        Hops, Hnames = get_Hops_and_Hnames(num_qubits, NUM_STATES, self.connected_qubit_pairs)
        states_concerned_list = get_full_states_concerned_list(num_qubits, NUM_STATES)
        maxA = get_maxA(num_qubits, NUM_STATES, connected_qubit_pairs)
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
    
    # Grab appropriate circuit based on molecule and slice index.
    circuit_file = "{}_circuits.pickle".format(molecule.lower())
    circuit_file_path = os.path.join(BASE_DATA_PATH, circuit_file)
    with open(circuit_file_path, "rb") as f:
        circuit, connected_qubit_pairs = pickle.load(f)[slice_index]

    # Generate the state object that encapsulates the optimization for the circuit.
    state = ProcessState(molecule, slice_index, circuit, connected_qubit_pairs)

    # Redirect everything the central process puts out to a log file.
    # By default, ray redirects the stdout of each worker process
    # to the central process.
    log_file = "{}.log".format(state.file_name)
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
    trial_entry = "{}\n".format(json.dumps(trial, cls=CustomJSONEncoder))
    trial_file = "{}.json".format(state.file_name)
    trial_file_path = os.path.join(state.data_path, trial_file)
    with open(trial_file_path, "a+") as trial_file:
        fcntl.flock(trial_file, fcntl.LOCK_EX)
        trial_file.write(trial_entry)
        fcntl.flock(trial_file, fcntl.LOCK_UN)
    
    # Report results.
    reporter(neg_loss=-loss, done=True)


### PRANAV'S CODE ###
# See fqc/experiments/CompileTimes.ipynb

def set_seeds():
    random.seed(0)
    np.random.seed(1)
    tf.set_random_seed(2)
    os.environ["HYPEROPT_FMIN_SEED"] = "23"

def _get_slice_circuits_list(circuit, blockings):
    set_seeds()

    def indices(gate):
        return [qarg[1] for qarg in gate.qargs]

    def gate_block_index(gate, blocking):
        if len(indices(gate)) == 1:
            return [indices(gate)[0] in grouping for grouping in blocking].index(True)
        else:
            # check which block each qubit is in
            control_index, target_index = indices(gate)
            control_block_index = [control_index in grouping for grouping in blocking].index(True)
            target_block_index = [target_index in grouping for grouping in blocking].index(True)
            if control_block_index != target_block_index:
                return -1
            else:
                return control_block_index

    blockings_index = 0

    gates = circuit.data
    width = circuit.width()

    slice_circuits_list = []

    while len(gates) > 0:
        blocking = blockings[blockings_index]
        remaining_gates = []; contaminated_indices = set()

        slice_circuits = [deepcopy(circuit) for _ in range(len(blocking.blocks))]
        for slice_circuit in slice_circuits:
            slice_circuit.data = []

        for i, gate in enumerate(gates):
            if len(contaminated_indices) == width:
                remaining_gates.extend(gates[i:])
                break

            block_index = gate_block_index(gate, blocking.blocks)
            if block_index == -1:
                contaminated_indices.add(indices(gate)[0]); contaminated_indices.add(indices(gate)[1]);
                remaining_gates.append(gate)
            else:
                if len(indices(gate)) == 1:
                    if indices(gate)[0] in contaminated_indices:
                        remaining_gates.append(gate)
                    else:
                        slice_circuits[block_index].data.append(gate)
                else:
                    if indices(gate)[0] in contaminated_indices or indices(gate)[1] in contaminated_indices:
                        contaminated_indices.add(indices(gate)[0]); contaminated_indices.add(indices(gate)[1]);
                        remaining_gates.append(gate)
                    else:
                        slice_circuits[block_index].data.append(gate)

        slice_circuits_list.append((slice_circuits, blocking))
        gates = remaining_gates
        blockings_index = (blockings_index + 1) % len(blockings)


    return slice_circuits_list


def _get_circuits_to_compile(slice_circuits_list):
    circuits_to_compile = []
    for slice_circuits, blocking in slice_circuits_list:
        for slice_circuit, block, connected_qubit_pairs in zip(
            slice_circuits, blocking.blocks, blocking.connected_qubit_pairs_list):
            for index in block:
                assert len(slice_circuit.qregs) == 1
                slice_circuit.iden(slice_circuit.qregs[0][index])
            slice_circuit = squash_circuit(slice_circuit)
            
            for subslice in get_uccsd_slices(slice_circuit, granularity=2, dependence_grouping=True):
                circuits_to_compile.append((subslice.circuit, connected_qubit_pairs))
    return circuits_to_compile

def get_beh2_circuits_to_compile():
    set_seeds()
    
    circuit = get_uccsd_circuit('BeH2')
    circuit = optimize_circuit(circuit)
    coupling_list = get_nearest_neighbor_coupling_list(2, 3)
    circuit = optimize_circuit(circuit, coupling_list)

    # layout is 0 2 4
    #           1 3 5
    
    class Blocking1(object):
        blocks = [{0, 1, 2, 3}, {4, 5}]
        connected_qubit_pairs_list = [[(0, 1), (1, 3), (2, 3), (0, 2)], [(0, 1)]]

    class Blocking2(object):
        blocks = [{0, 1}, {2, 3, 4, 5}]
        connected_qubit_pairs_list = [[(0, 1)], [(0, 1), (1, 3), (2, 3), (0, 2)]]

    blockings = [Blocking1, Blocking2]
    
    slice_circuits_list = _get_slice_circuits_list(circuit, blockings)
    return _get_circuits_to_compile(slice_circuits_list)

def get_nah_circuits_to_compile():
    set_seeds()
    
    circuit = get_uccsd_circuit('NaH')
    circuit = optimize_circuit(circuit)
    coupling_list = get_nearest_neighbor_coupling_list(2, 4)
    circuit = optimize_circuit(circuit, coupling_list)

    # layout is 0 2 4 6
    #           1 3 5 7
    
    class Blocking1(object):
        blocks = [{0, 1, 2, 3}, {4, 5, 6, 7}]
        connected_qubit_pairs_list = [[(0, 1), (1, 3), (2, 3), (0, 2)], [(0, 1), (1, 3), (2, 3), (0, 2)]]

    class Blocking2(object):
        blocks = [{0, 1}, {2, 3, 4, 5}, {6, 7}]
        connected_qubit_pairs_list = [[(0, 1)], [(0, 1), (1, 3), (2, 3), (0, 2)], [(0, 1)]]

    blockings = [Blocking1, Blocking2]
    
    slice_circuits_list = _get_slice_circuits_list(circuit, blockings)
    return _get_circuits_to_compile(slice_circuits_list)

def get_h2o_circuits_to_compile():
    set_seeds()
    
    circuit = get_uccsd_circuit('H2O')
    circuit = optimize_circuit(circuit)
    coupling_list = get_nearest_neighbor_coupling_list(2, 5)
    circuit = optimize_circuit(circuit, coupling_list)

    # layout is 0 2 4 6 8
    #           1 3 5 7 9
    
    class Blocking1(object):
        blocks = [{0, 1, 2, 3}, {4, 5, 6, 7}, {8, 9}]
        connected_qubit_pairs_list = [[(0, 1), (1, 3), (2, 3), (0, 2)], [(0, 1), (1, 3), (2, 3), (0, 2)], [(0, 1)]]

    class Blocking2(object):
        blocks = [{0, 1}, {2, 3, 4, 5}, {6, 7, 8, 9}]
        connected_qubit_pairs_list = [[(0, 1)], [(0, 1), (1, 3), (2, 3), (0, 2)], [(0, 1), (1, 3), (2, 3), (0, 2)]]

    blockings = [Blocking1, Blocking2]
    
    slice_circuits_list = _get_slice_circuits_list(circuit, blockings)
    return _get_circuits_to_compile(slice_circuits_list)


if __name__ == "__main__":
    main()




