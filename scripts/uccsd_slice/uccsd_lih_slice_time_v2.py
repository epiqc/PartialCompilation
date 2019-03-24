"""
uccsd_slice_time_v2.py - A module for searching for the optimal slice time
                         for UCCSD slices via binary search. This version
                         implements binary search on the pulse time for
                         the same slice at different intervals.
"""
# Set random seeds for reasonable reproducibility.
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

from fqc.data import UCCSD_LIH_THETA
from fqc.uccsd import get_uccsd_circuit, get_uccsd_slices
from fqc.util import (optimize_circuit, get_max_pulse_time,
                      get_nearest_neighbor_coupling_list)
from mpi4py.futures import MPIPoolExecutor
from quantum_optimal_control.main_grape.grape import Grape
from quantum_optimal_control.core.hamiltonian import (get_H0,
                                                      get_Hops_and_Hnames,
                                                      get_full_states_concerned_list,
                                                      get_maxA)

### CONSTANTS ###

# We will work with g2 slices which each have 1 unique angle per slice.
SLICE_GRANULARITY = 2

# Binary search nanosecond granularity.
BNS_GRANULARITY = 10

# Grape args.
DATA_PATH = "/project/ftchong/qoc/thomas/uccsd_slice_time/lih_v2"

# Define hardware specific parameters.
NUM_QUBITS = 4
NUM_STATES = 2
CONNECTED_QUBIT_PAIRS = get_nearest_neighbor_coupling_list(2, 2, directed=False)
H0 = get_H0(NUM_QUBITS, NUM_STATES)
Hops, Hnames = get_Hops_and_Hnames(NUM_QUBITS, NUM_STATES,
                                   CONNECTED_QUBIT_PAIRS)
STATES_CONCERNED_LIST = get_full_states_concerned_list(NUM_QUBITS, NUM_STATES)
MAXA = get_maxA(NUM_QUBITS, NUM_STATES, CONNECTED_QUBIT_PAIRS)

# Define convergence parameters and penalties.
MAX_ITERATIONS = 1000
DECAY = MAX_ITERATIONS / 2
CONVERGENCE = {'rate':0.01, 'conv_target': 1e-3,
               'max_iterations': MAX_ITERATIONS, 'learning_rate_decay':DECAY,
               'min_grads': 1e-5}
REG_COEFFS = {}
USE_GPU = False
SPARSE_H = False
SHOW_PLOTS = False
METHOD = 'ADAM'
# Steps per nanosecond and nanoseconds per step
spn = 20.0
nps = 1 / spn

# Get slices to perform qoc on. The initial angle of each RZ
# gate does not matter.
THETA = UCCSD_LIH_THETA
FULL_UCCSD_LIH_CIRCUIT = optimize_circuit(get_uccsd_circuit('LiH', THETA),
                           CONNECTED_QUBIT_PAIRS)
UCCSD_SLICES = get_uccsd_slices(FULL_UCCSD_LIH_CIRCUIT,
                                granularity = SLICE_GRANULARITY,
                                dependence_grouping = True)

# https://ark.intel.com/products/91754/Intel-Xeon-Processor-E5-2680-v4-35M-Cache-2-40-GHz-
BROADWELL_CORE_COUNT = 14


### MAIN ###

def main():
    # Handle CLI.
    parser = argparse.ArgumentParser()
    parser.add_argument("--angle-start", type=float, default=0.0, help="the "
                        "inclusive lower bound of angles to time the "
                        "slice for (units in degrees, behaves like np.arange)")
    parser.add_argument("--angle-stop", type=float, default=1.0, help="the "
                        "exclusive upper bound of angles to time the "
                        "slice for (units in degrees, behaves like np.arange)")
    parser.add_argument("--angle-step", type=float, default=5.0, help="the step size "
                        "between angle values (units in degrees, behaves "
                        "like np.arange)")
    parser.add_argument("--slice-start", type=int, default=0, help="the "
                        "inclusive lower bound of slice indices to include "
                        "(0-7)")
    parser.add_argument("--slice-stop", type=int, default=0, help="the "
                        "inclusive upper bound of slice indices to include "
                        "(0-7)")
    args = vars(parser.parse_args())
    angle_start = args["angle_start"]
    angle_stop = args["angle_stop"]
    angle_step = args["angle_step"]
    slice_start = args["slice_start"]
    slice_stop = args["slice_stop"]
    slice_count = slice_stop - slice_start + 1
    
    # Trim slices to only include start thru stop.
    slices = UCCSD_SLICES[slice_start:slice_stop + 1]

    # Get a list of the angles that each slice should be timed for.
    angle_list_deg = list(np.arange(angle_start, angle_stop, angle_step))
    angle_list_rad = list(np.deg2rad(angle_list_deg))
    angle_count = len(angle_list_rad)
    
    # Build argument iterators to map to each process.
    job_count = slice_count * angle_count
    slice_iter = list()
    slice_index_iter = list()
    angle_iter = angle_list_rad * slice_count
    file_name_iter = list()
    for i, slice_index in enumerate(range(slice_start, slice_start + slice_count)):
        slice_iter += [slices[i]] * angle_count
        slice_index_iter += [slice_index] * angle_count
        file_name_iter += ["s{}_{}".format(slice_index, angle)
                           for angle in angle_list_deg]
    
    # Run binary search for each slice on each angle list.
    with MPIPoolExecutor(BROADWELL_CORE_COUNT) as executor:
        executor.map(process_init, slice_iter, slice_index_iter,
                     angle_iter, file_name_iter)


def process_init(uccsdslice, slice_index, angle,
                 file_name):
    """Do all necessary process specific tasks before running binary search.
    Args:
    uccsdslice :: fqc.uccsd.uccsdslice.UCCSDSlice - The slice to binary search on.
    slice_index :: int - The index of uccsdslice in the slice list.
    angle :: float - The angle in radians to update the slice with.
    file_name :: string - The file name to associate with this run.

    Returns: nothing
    """
    # Redirect output to a log file.
    log_file = "{}.log".format(file_name)
    log_file_path = os.path.join(DATA_PATH, log_file)
    with open(log_file_path, "w") as log:
        sys.stdout = sys.stderr = log
        # Display pid, time, slice id, and circuit.
        print("PID={}\nTIME={}\nSLICE_ID={}\nANGLE={}"
              "".format(os.getpid(), time.time(), slice_index, angle))
        # Update slice theta dependent gates to angle.
        uccsdslice.update_angles([angle] * len(uccsdslice.angles))
        U = uccsdslice.unitary()
        print(uccsdslice.circuit)
        
        # Define search space.
        # time_upper_bound is the pulse time for a trivial
        # gate lookup that we should always beat.
        time_upper_bound = get_max_pulse_time(uccsdslice.circuit)
        print("TIME_UPPER_BOUND={}".format(time_upper_bound))
        # min_steps and max_steps are the min/max steps for the
        # the search on the current angle. mid_steps is the steps
        # we will try for the current search.
        min_steps = 0
        max_steps = time_upper_bound * spn
        mid_steps = int((min_steps + max_steps) / 2)
        prev_converged_mid_steps = -1


        # Binary search for the minimum pulse time on the current angle.
        # Search in the search space until we have a convergence window
        # of BNS_GRANULARITY
        while min_steps + BNS_GRANULARITY < max_steps:
            mid_steps = int((min_steps + max_steps) / 2)
            total_time = mid_steps * nps
            print("\nMAX_STEPS={}\nMIN_STEPS={}\nMID_STEPS={}\nTIME={}"
                  "\nGRAPE_START_TIME={}"
                  "".format(max_steps, min_steps, mid_steps, total_time,
                            time.time()))
            grape_sess = Grape(H0, Hops, Hnames, U, total_time, mid_steps,
                          convergence = CONVERGENCE, reg_coeffs = REG_COEFFS,
                          use_gpu = USE_GPU, sparse_H = SPARSE_H,
                          method = METHOD, maxA = MAXA,
                          states_concerned_list = STATES_CONCERNED_LIST,
                          show_plots = SHOW_PLOTS, file_name = file_name,
                          data_path = DATA_PATH)
            print("GRAPE_END_TIME={}".format(time.time()))
            # If the trial converged, lower the upper bound.
            # If the tiral did not converge, raise the lower bound.
            trial_converged = grape_sess.l <= grape_sess.conv.conv_target
            print("TRIAL_CONVERGED={}".format(trial_converged))
            if trial_converged:
                prev_converged_mid_steps = mid_steps
                max_steps = mid_steps
            else:
                min_steps = mid_steps
        # ENDWHILE
        
        # Log the best time.
        print("BEST_STEPS={}, BEST_TIME={}"
              "".format(prev_converged_mid_steps,
                        prev_converged_mid_steps * nps))


if __name__ == "__main__":
    main()
