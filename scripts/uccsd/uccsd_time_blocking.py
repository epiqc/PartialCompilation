"""
uccsd_time_blocking.py - A script for computing the appropriate pulse_times for
                         UCCSD circuits and slices via binary search.
"""
# Set random seeds for reasonable reproducability.
import random
random.seed(0)
import numpy as np
np.random.seed(1)
import tensorflow as tf
tf.set_random_seed(2)

import argparse
import json
import os
import pickle
import sys
import time

from fqc.uccsd import get_uccsd_circuit, get_uccsd_slices
from fqc.util import (optimize_circuit, get_unitary,
                      get_nearest_neighbor_coupling_list, get_max_pulse_time,
                      CustomJSONEncoder)
from quantum_optimal_control.main_grape.grape import Grape
from quantum_optimal_control.core.hamiltonian import (get_Hops_and_Hnames,
                                                      get_full_states_concerned_list,
                                                      get_maxA)


### CONSTANTS ###

HPO_DATA_PATH = "/project/ftchong/qoc/thomas/hpo"
TIME_DATA_PATH = "/project/ftchong/qoc/thomas/time"

NUM_STATES = 2
SPN = 20.

# Grape constants
MAX_ITERATIONS = int(1e3)
CONV_TARGET = 1e-5
GRAPE_CONVERGENCE = {'conv_target': CONV_TARGET,
                     'max_iterations': MAX_ITERATIONS}
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

# Define binary search parameters.
# binary search granularity, how many nanoseconds of precision do you need?
BSG = 1.0


### OBJECTS ###

class ProcessState(object):
    """A class to encapsulate the binary search of a circuit.
    Fields:
    molecule :: string - identifies the uccsd molecule
    slice_index :: int - the circuit indexed into all circuits of the molecule
    rz_index :: int the circuit indexed into all rz containing circuits of the molecule
    circuit :: qiskit.QuantumCircuit - the circuit being searched on
    unitary :: np.matrix - the unitary that represents the circuit being optimized
    connected_qubit_pairs :: [(int, int)] - represents physical
                                            constraints of qubits
    grape_config :: dict - molecule specific grape parameters
    data_path :: string - where to store output
    file_name :: string - the identifier of the search
    lr :: float - the learning rate to use for the optimization
    decay :: float - the learning rate decay to use for the optimization
    """

    def __init__(self, molecule, rz_index):
        """See class fields for parameter definitions.
        """
        super()
        self.molecule = molecule
        self.rz_index = rz_index

        # Get circuit, cqp and slice_index.
        pickle_file_name = "{}_circuits.pickle".format(molecule.lower())
        pickle_file_path = os.path.join(HPO_DATA_PATH, pickle_file_name)
        with open(pickle_file_path, "rb") as f:
            rz_indices, circuit_list = pickle.load(f)
        slice_index = rz_indices[self.rz_index]
        circuit, connected_qubit_pairs = circuit_list[slice_index]

        self.slice_index = slice_index
        self.circuit = circuit
        self.connected_qubit_pairs = connected_qubit_pairs
        self.unitary = get_unitary(circuit)
        self.file_name = "s{}".format(slice_index)
        datadir_name = "uccsd_{}".format(molecule.lower())
        self.data_path = os.path.join(TIME_DATA_PATH, datadir_name)
        # TODO: We assume TIME_DATA_PATH exists.
        if not os.path.exists(self.data_path):
            os.mkdir(self.data_path)
        # Get grape config.
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
        
        # Get hyperparameters by choosing the configuration with the lowest loss.
        best_data = {"loss": 1}
        hpo_file_name = "{}.json".format(self.file_name)
        hpo_file_path = os.path.join(HPO_DATA_PATH, datadir_name, hpo_file_name)
        with open(hpo_file_path) as f:
            line = f.readline()
            while line:
                data = json.loads(line)
                if data["loss"] < best_data["loss"]:
                    best_data = data
                line = f.readline()
        #ENDWITH
        self.lr = best_data["lr"]
        self.decay = best_data["decay"]


### MAIN METHODS ###

def main():
    """Binary search for the optimal pulse time for a single circuit.
    """
    # Handle CLI.
    parser = argparse.ArgumentParser()
    parser.add_argument("--molecule", type=str, default="BeH2", help="the "
                        "UCCSD molecule to perform binary search on")
    parser.add_argument("--rz-index", type=int, default=0, help="the "
                        "circuit indexed into the rz containing circuits "
                        "of the molecule")
    args = vars(parser.parse_args())
    molecule = args["molecule"]
    rz_index = args["rz_index"]

    # Binary search for the optimal pulse time for the circuit specified.
    state = ProcessState(molecule, rz_index)
    process_init(state)


def process_init(state):
    """Initialize a time optimization loop for a single circuit.
    Args:
    state :: ProcessState - the state encapsulating the circuit
                            to binary search on

    Returns: nothing
    """
    log_file = state.file_name + '.log'
    log_file_path = os.path.join(state.data_path, log_file)
    with open(log_file_path, "w") as log:
        # Redirect everything to a log file.
        sys.stdout = sys.stderr = log

        # Display run characteristics.
        print("PID={}\nWALL_TIME={}\nSLICE_INDEX={}\nRZ_INDEX={}\n"
              "LEARNING_RATE={}\nLEARNING_RATE_DECAY={}\n{}"
              "".format(os.getpid(), time.time(), state.slice_index,
                        state.rz_index, state.lr, state.decay,
                        state.circuit))

        # Define search space.
        max_pulse_time = get_max_pulse_time(state.circuit)
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
    U = state.unitary
    convergence = GRAPE_CONVERGENCE
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
        pulse_time = mid_steps / SPN
        print("MAX_STEPS={}\nMIN_STEPS={}\nMID_STEPS={}\nTRIAL_PULSE_TIME={}"
              "\nGRAPE_START_TIME={}"
              "".format(max_steps, min_steps, mid_steps, pulse_time,
                        time.time()))
        sess = Grape(U=U, total_time=pulse_time, steps=mid_steps,
                     convergence=convergence, data_path=state.data_path,
                     file_name=state.file_name, **state.grape_config)
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
    best_time = prev_converged_mid_steps / SPN
    best_steps = prev_converged_mid_steps
    print("BEST_STEPS={}, BEST_TIME={}"
          "".format(best_steps, best_time))

    # Log results.
    result_entry = {"time": best_time, "steps": best_steps,
                    "lr": state.lr, "decay": state.decay}
    result_file = "{}.json".format(state.file_name)
    result_file_path = os.path.join(state.data_path, result_file)
    with open(result_file_path, "w") as f:
        f.write("{}\n".format(json.dumps(result_entry ,cls=CustomJSONEncoder)))


if __name__ == "__main__":
    main()
        

