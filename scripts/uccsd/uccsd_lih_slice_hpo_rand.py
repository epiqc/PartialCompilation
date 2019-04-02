"""
uccsd_lih_slice_hpo_rand.py - A script for running random hyperparameter search on
                              a single UCCSD LiH slice. We search over 30 configurations
                              of pulse time and angles. We use the pulse times [0.25, 0.5, 0.75]
                              each as fractions of the maximum circuit pulse time. We choose
                              10 randomly generated angles. The idea here is that there could
                              be a relationship between the angles that we do not know.
                              E.g. an evenly space interval pi / 2, pi, 3pi / 2 
                              generate similar circuits. However, we do want to see
                              how the hyperparameters behave in over-constrained and
                              under-constrained pulse time situations. The hyperparameters
                              we optimize over are learning rate and learning rate decay for
                              ADAM on Grape. We choose 100 hyperparameter configurations
                              per pulse time and angle configuration.
"""
# Set random seeds for reasonable reproducibility.
import random
random.seed(0)
import numpy as np
np.random.seed(1)
import tensorflow as tf
tf.set_random_seed(2)

import argparse
import fcntl
import json
import os
import sys
import time

from fqc.data import UCCSD_DATA, SPN
from fqc.util import (get_max_pulse_time, CustomJSONEncoder)
from mpi4py.futures import MPIPoolExecutor
from quantum_optimal_control.main_grape.grape import Grape

### CONSTANTS ###

BASE_DATA_PATH = "/project/ftchong/qoc/thomas/hpo/uccsd_lih_slice_hpo_rand/"

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
GRAPE_CONFIG = UCCSD_DATA["LiH"]["GRAPE_CONFIG"]
GRAPE_CONFIG.update(GRAPE_TASK_CONFIG)

# Hyperparmeter optimization constants and search space.
# We use a loguniform distribution on learning rate
# and a uniform distribution on decay.
NUM_HP_SAMPLES = 100
LR_LB = 1e-5
LR_UB = 1
DECAY_LB = 1
DECAY_UB = 1e3
LR_SAMPLES = np.exp(np.random.uniform(np.log(LR_LB), np.log(LR_UB), NUM_HP_SAMPLES))
DECAY_SAMPLES = np.random.uniform(DECAY_LB, DECAY_UB, NUM_HP_SAMPLES)
HP_SAMPLES = list(np.column_stack((LR_SAMPLES, DECAY_SAMPLES)))

NUM_ANGLES = 10
ANGLES_DEG = np.random.uniform(0, 360, NUM_ANGLES)
PULSE_TIME_MULTIPLIERS = [0.25, 0.5, 0.75]


### OBJECTS ###

class ProcessState(object):
    """An object to encapsulate the work computed by one process.
    Fields:
    uccsdslice :: fqc.uccsd.uccsdslice.UCCSDSlice - the slice that is being optimized
    slice_index :: int - the index of the circuit slice that is being optimized
    angle_deg :: float -t he arugment in degress of the slice's theta-dpendent gates
    angle :: float - the argument in radians of the slice's theta-dependent gates
    pulse_time_multiplier :: float - the fraction of the circuit's maximum pulse time
                                     to optimize for
    pulse_time :: float - the pulse time to optimize the circuit for
    lr :: float - the learning rate to use for the optimization
    decay :: float - the learning rate decay to use for the optimization
    log_file_name/path :: string - where all output of the optimization will be stored
    trial_file_name/path :: string - where the results of the optimization will be stored,
                                this file is shared by multiple processes
    data_path :: string - the output directory
    """
    def __init__(self, uccsdslice, slice_index, angle_deg, pulse_time_multiplier,
                 lr, decay, data_path):
        """See corresponding class field declarations for arguments.
        """
        super()
        self.uccsdslice = uccsdslice
        self.slice_index = slice_index
        self.angle = np.deg2rad(angle_deg)
        uccsdslice.update_angles([self.angle] * len(uccsdslice.angles))
        self.pulse_time_multiplier = pulse_time_multiplier
        self.pulse_time = (get_max_pulse_time(uccsdslice.circuit)
                           * pulse_time_multiplier)
        self.lr = lr
        self.decay = decay
        self.data_path = data_path
        self.log_file_name = ("s{}_a{:.2f}_t{:.2f}_l{:.4f}_d{:.4f}.log"
                              "".format(self.slice_index, self.angle,
                                        self.pulse_time, self.lr, self.decay))
        self.log_file_path = os.path.join(data_path, self.log_file_name)
        self.trial_file_name = ("s{}_a{:.2f}_t{:.2f}.json"
                                "".format(self.slice_index,
                                          self.angle, self.pulse_time))
        self.trial_file_path = os.path.join(data_path, self.trial_file_name)


### MAIN METHODS ###

def main():
    """Run random hyperparameter search on the predetermined angle and pulse time
    combinations for the predetermined hyperparameter configurations on one UCCSD LiH slice.
    """
    # Handle CLI.
    parser = argparse.ArgumentParser()
    parser.add_argument("--slice-index", type=int, default=0, help="the "
                       "slice to search on.")
    parser.add_argument("--core-count", type=int, default=0, help="the "
                       "number of cores available for computation.")
    args = vars(parser.parse_args())
    slice_index = args["slice_index"]
    core_count = args["core_count"]

    # Log run characteristics.
    run_entry = ("PID={}\nWALL_TIME={}\nCORE_COUNT={}\nSLICE_INDEX={}\n"
                 "PULSE_TIME_MULTIPLIERS={}\nANGLES_DEG={}\nLR_SAMPLES={}"
                 "\nDECAY_SAMPLES={}"
                 "".format(os.getpid(), time.time(), core_count, slice_index,
                           PULSE_TIME_MULTIPLIERS, ANGLES_DEG, LR_SAMPLES,
                           DECAY_SAMPLES))
    data_path = os.path.join(BASE_DATA_PATH, "s{}".format(slice_index))
    # TODO: We assume BASE_DATA_PATH exists.
    if not os.path.exists(data_path):
        os.mkdir(data_path)
    run_log_path = os.path.join(data_path, "run.log")
    with open(run_log_path, "w+") as f:
        f.write(run_entry)

    # Generate states for search.
    uccsdslice = UCCSD_DATA["LiH"]["SLICES"][slice_index]
    state_iter = list()
    # We traverse pulse time in the middle because we would rather have a complete set on a few
    # angles than an incomplete set on many angles.
    for angle_deg in ANGLES_DEG:
        for pulse_time_multiplier in PULSE_TIME_MULTIPLIERS:
            for lr, decay in HP_SAMPLES:
                state_iter.append(ProcessState(uccsdslice, slice_index,
                                               angle_deg, pulse_time_multiplier,
                                               lr, decay, data_path))
    
    # Run search.
    with MPIPoolExecutor(core_count) as executor:
        executor.map(process_init, state_iter)


def process_init(state):
    """Carry out a computation specified by state.
    Args:
    state :: ProcessState - encapsulates the computation to perform
    Returns: nothing
    """
    # Redirect everything to a log file.
    with open(state.log_file_path, "w+") as log:
        sys.stdout = sys.stderr = log
        
        print("PID={}\nWALL_TIME={}\nSLICE_INDEX={}\nPULSE_TIME={}\nANGLE={}"
              "\nLR={}\nDECAY={}\n{}"
              "".format(os.getpid(), time.time(), state.slice_index,
                        state.pulse_time, state.angle, state.lr, state.decay,
                        state.uccsdslice.circuit))

        # Build necessary grape arguments using parameters.
        U = state.uccsdslice.unitary()
        convergence = {'rate': state.lr,
                       'max_iterations': GRAPE_MAX_ITERATIONS,
                       'learning_rate_decay': state.decay}
        pulse_time = state.pulse_time
        steps = int(pulse_time * SPN)

        # Run grape.
        print("GRAPE_START_TIME={}".format(time.time()))
        grape_sess = Grape(U=U, total_time=pulse_time, steps=steps,
                           convergence=convergence, file_name=state.log_file_name,
                           data_path=state.data_path, **GRAPE_CONFIG)
        print("GRAPE_END_TIME={}".format(time.time()))

        # Log results.
        loss = grape_sess.l
        print("LOSS={}".format(loss))
        trial_entry = {
            "loss": loss,
            "lr": state.lr,
            "decay": state.decay,
        }
        with open(state.trial_file_path, "a+") as f:
            fcntl.flock(f, fcntl.LOCK_EX)
            f.write(json.dumps(trial_entry, cls=CustomJSONEncoder)
                             + "\n")
            fcntl.flock(f, fcntl.LOCK_UN)


if __name__ == "__main__":
    main()




