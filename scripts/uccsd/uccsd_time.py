"""
uccsd_time.py - A script for computing the appropriate pulse_times for
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
import os
import sys
import time

from fqc.uccsd import get_uccsd_circuit, get_uccsd_slices
from fqc.util import (optimize_circuit, get_unitary,
                      get_nearest_neighbor_coupling_list, get_max_pulse_time)
from fqc.data import UCCSD_DATA, SPN
from quantum_optimal_control.main_grape.grape import Grape


### CONSTANTS ###

BASE_DATA_PATH = "/project/ftchong/qoc/thomas/time"

# Grape constants
MAX_ITERATIONS = int(1e3)
CONV_TARGET = 1e-5
GRAPE_CONVERGENCE = {'conv_target': CONV_TARGET,
                     'max_iterations': MAX_ITERATIONS}
METHOD = "ADAM"
USE_GPU = False
SPARSE_H = False
SHOW_PLOTS = False
SAVE = True
GRAPE_TASK_CONFIG = {
    "method": METHOD,
    "use_gpu": USE_GPU,
    "sparse_H": SPARSE_H,
    "show_plots": SHOW_PLOTS,
    "save": SAVE,
}

# Define binary search parameters.
# binary search granularity, how many nanoseconds of precision do you need?
BSG = 10.0


### Objects ###


class ProcessState(object):
    """A class to encapsulate the binary search of a circuit.
    Fields:
    molecule :: string - identifies the uccsd molecule
    slice_index :: int - the index of the circuit slice to search on,
                         defaults to -1 if the full circuit is being searched
    circuit :: qiskit.QuantumCircuit - the circuit being searched on
    unitary :: np.matrix - the unitary that represents the circuit being optimized
    grape_config :: dict - molecule specific grape parameters
    data_path :: string - where to store output
    file_name :: string - the identifier of the search
    lr :: float - the learning rate to use for the optimization
    decay :: float - the learning rate decay to use for the optimization
    """

    def __init__(self, molecule, slice_index=-1):
        """See class fields for parameter definitions.
        """
        super()
        self.molecule = molecule
        self.slice_index = slice_index
        # If the slice index is -1, the full circuit is being optimized.
        if self.slice_index == -1:
            self.circuit = UCCSD_DATA[molecule]["CIRCUIT"]
            self.file_name = "full"
            self.lr = UCCSD_DATA[molecule]["FULL_DATA"]["HP"]["lr"]
            self.decay = UCCSD_DATA[molecule]["FULL_DATA"]["HP"]["decay"]
        else:
            self.circuit = UCCSD_DATA[molecule]["SLICES"][slice_index].circuit
            self.file_name = "s{}".format(slice_index)
            self.lr = UCCSD_DATA[molecule]["SLICE_DATA"]["HP"][slice_index]["lr"]
            self.decay = UCCSD_DATA[molecule]["SLICE_DATA"]["HP"][slice_index]["decay"]
        self.unitary = get_unitary(self.circuit)
        self.grape_config = UCCSD_DATA[molecule]["GRAPE_CONFIG"]
        self.grape_config.update(GRAPE_TASK_CONFIG)
        self.data_path = os.path.join(BASE_DATA_PATH,
                                      "uccsd_{}".format(molecule.lower()))
        # TODO: We assume BASE_DAT_PATH exists.
        if not os.path.exists(self.data_path):
            os.mkdir(self.data_path)


### MAIN METHODS ###

def main():
    """Binary search for the optimal pulse time for a single circuit.
    """
    # Handle CLI.
    parser = argparse.ArgumentParser()
    parser.add_argument("--molecule", type=str, default="H2", help="the "
                        "UCCSD molecule to perform binary search on")
    parser.add_argument("--slice-index", type=int, default=-1, help="the "
                        "slice to perform binary search on, do not specify to run "
                        "binary search on the full circuit")
    args = vars(parser.parse_args())
    molecule = args["molecule"]
    slice_index = args["slice_index"]

    # Binary search for the optimal pulse time for the circuit specified.
    state = ProcessState(molecule, slice_index)
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
        print("PID={}\nWALL_TIME={}\nSLICE_INDEX={}\nLEARNING_RATE={}\n"
              "LEARNING_RATE_DECAY={}\n{}"
              "".format(os.getpid(), time.time(), state.slice_index,
                        state.lr, state.decay, state.circuit))

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
    print("BEST_STEPS={}, BEST_TIME={}"
          "".format(prev_converged_mid_steps,
                    prev_converged_mid_steps / SPN))


if __name__ == "__main__":
    main()
