"""
uccsd_lih_slice_qoc.py - A module for running quantum optimal control on
                         UCCSD LIH slices.
"""
# Set random seeds for reasonable reproducibility.
import random
random.seed(0)
import numpy as np
np.random.seed(1)
import tensorflow as tf
tf.set_random_seed(2)

from copy import deepcopy
from itertools import product
import os
import sys
import time
import argparse

from fqc.data import (UCCSD_LIH_THETA, UCCSD_LIH_SLICE_TIMES,
                      UCCSD_LIH_HYPERPARAMETERS)
from fqc.uccsd import get_uccsd_circuit, get_uccsd_slices
from fqc.util import (optimize_circuit, get_unitary,
                      get_nearest_neighbor_coupling_list)
from mpi4py.futures import MPIPoolExecutor
from quantum_optimal_control.main_grape.grape import Grape
from quantum_optimal_control.core.hamiltonian import (get_H0,
                                                      get_Hops_and_Hnames,
                                                      get_full_states_concerned_list,
                                                      get_maxA)


### CONSTANTS ###


DATA_PATH = ""

# Define Grape parameters.
NUM_QUBITS = 4
NUM_STATES = 2
CONNECTED_QUBIT_PAIRS = get_nearest_neighbor_coupling_list(2, 2, directed=False)
H0 = np.zeros((NUM_STATES ** NUM_QUBITS, NUM_STATES ** NUM_QUBITS))
Hops, Hnames = get_Hops_and_Hnames(NUM_QUBITS, NUM_STATES,
                                   CONNECTED_QUBIT_PAIRS)
STATES_CONCERNED_LIST = get_full_states_concerned_list(NUM_QUBITS, NUM_STATES)
MAX_PULSE_AMPLITUDE = get_maxA(NUM_QUBITS, NUM_STATES, CONNECTED_QUBIT_PAIRS)
MAX_ITERATIONS = 1e3
CONV_TARGET = 1e-3
CONVERGENCE = {'conv_target': CONV_TARGET,
               'max_iterations': MAX_ITERATIONS,
}
REG_COEFFS = {}
USE_GPU = False
SPARSE_H = False
SHOW_PLOTS = False
METHOD = "ADAM"
SAVE = True
GRAPE_CONFIG = {
    "H0": H0,
    "Hops": Hops,
    "Hnames": Hnames,
    "states_concerned_list": STATES_CONCERNED_LIST,
    "reg_coeffs": REG_COEFFS,
    "maxA": MAX_PULSE_AMPLITUDE,
    "use_gpu": USE_GPU,
    "sparse_H": SPARSE_H,
    "show_plots": SHOW_PLOTS,
    "method": METHOD,
    "data_path": DATA_PATH,
    "save": SAVE,
}
# Pulse granularity (steps per nanosecond)
SPN = 20.0 

# Get slices to perform qoc on.
# We will work with g2 slices which each have 1 unique angle per slice.
SLICE_GRANULARITY = 2
UCCSD_LIH_FULL_CIRCUIT = optimize_circuit(get_uccsd_circuit('LiH', UCCSD_LIH_THETA),
                                          CONNECTED_QUBIT_PAIRS)
UCCSD_LIH_SLICES = get_uccsd_slices(UCCSD_LIH_FULL_CIRCUIT, granularity=SLICE_GRANULARITY,
                                    dependence_grouping=True)

# How many cores are we running on?
# https://ark.intel.com/products/91754/Intel-Xeon-Processor-E5-2680-v4-35M-Cache-2-40-GHz-
BROADWELL_CORE_COUNT = 14


### ADTs ###


class ProcessState(object):
    """A class to encapsulate the optimization performed by a process.
    Fields:
    uccsdslice :: fqc.uccsd.uccsdslice.UCCSDSlice - the slice being QOC'd
    slice_index :: int - the index of the slice
    angle :: float - the angle of the theta dependent gates in the slice
    angle_deg :: float - the angle of the theta dependent gates in the slice
                         in degrees
    file_name :: str - a unique identifier for the slice
    pulse_time :: float - the time length to optimize the pulse for
    lr :: float - the learning rate to specify to grape for the slice
    decay :: float - the learning rate decay to specify to grape for the slice
    """
    
    def __init__(self, uccsdslice, slice_index, angle_deg):
        """See class fields for argument definitions.
        """
        super()
        # IMPLEMENTATION NOTE: This probably doesn't need to be a deepcopy since
        # the state is passed to seperate processes.
        self.uccsdslice = deepcopy(uccsdslice)
        self.slice_index = slice_index
        self.angle_deg = angle_deg
        self.angle = np.deg2rad(angle_deg)
        self.uccsdslice.update_angles([self.angle] * len(self.uccsdslice.angles))
        self.file_name = "s{}_a{}".format(slice_index, angle_deg)
        self.pulse_time = UCCSD_LIH_SLICE_TIMES[slice_index]
        self.lr = UCCSD_LIH_HYPERPARAMETERS[slice_index]["lr"]
        self.decay = UCCSD_LIH_HYPERPARAMETERS[slice_index]["decay"]
        

### MAIN METHODS ###


def main():
    # Handle CLI
    parser = argparse.ArgumentParser()
    parser.add_argument("--angle-start", type=float, default=0.0, help="the "
                        "inclusive lower bound of angles to optimize the "
                        "slice for (units in degrees, behaves like np.arange)")
    parser.add_argument("--angle-stop", type=float, default=1.0, help="the "
                        "exclusive upper bound of angles to optimize the "
                        "slice for (units in degrees, behaves like np.arange)")
    parser.add_argument("--angle-step", type=float, default=5.0, help="the step size "
                        "between angle values (units in degrees, behaves "
                        "like np.arange)")
    parser.add_argument("--slice-index", type=int, default=0, help="the "
                        "slice to perform QOC on (0-7)")
    args = vars(parser.parse_args())
    angle_start = args["angle_start"]
    angle_stop = args["angle_stop"]
    angle_step = args["angle_step"]
    slice_index = args["slice_index"]

    # Trim slices to only include start thru stop.
    uccsdslice = UCCSD_LIH_SLICES[slice_index]
    # Get the angles to optimize for.
    angle_deg_list = list(np.arange(angle_start, angle_stop, angle_step))
    
    # Construct a state for each job. I.e. each angle from the angle
    # list on the slice specified.
    state_iter = [ProcessState(uccsdslice, slice_index, angle_deg)
                  for angle_deg in angle_deg_list]
    
    # Run QOC for each process state.
    with MPIPoolExecutor(BROADWELL_CORE_COUNT) as executor:
        executor.map(process_init, state_iter)


def process_init(state):
    """Do all necessary process specific tasks before running grape.
    Args: 
    state :: ProcessState - encapulates the task of one process

    Returns: nothing
    """
    log_file = state.file_name + ".log"
    log_file_path = os.path.join(DATA_PATH, log_file)
    with open(log_file_path, "w") as log:
        # Redirect everything to a log file.
        sys.stdout = sys.stderr = log
        
        # Display pid, time, slice id, angle, circuit.
        print("PID={}\nWALL_TIME={}\nSLICE_ID={}\nANGLE={}\n{}"
              "".format(os.getpid(), time.time(), state.slice_index,
                        state.angle, state.uccsdslice.circuit))
        
        # Gather necessary grape parameters.
        U = state.uccsdslice.unitary()
        pulse_time = state.pulse_time
        steps = int(pulse_time * SPN)
        convergence = CONVERGENCE
        convergence.update({
            "rate": state.lr,
            "learning_rate_decay": state.decay,
        })

        # Run grape.
        print("GRAPE_START_TIME={}".format(time.time()))
        grape_sess = Grape(U=U, total_time=pulse_time, steps=steps,
                           convergence=convergence, **GRAPE_CONFIG)
        print("GRAPE_END_TIME={}".format(time.time()))

if __name__ == "__main__":
    main()
