"""
uccsd_slice_qoc.py - A module for running quantum optimal control on
                     UCCSD slices.
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

# TODO: Style - All constants should be all caps.
# We will work with g2 slices which each have 1 unique angle per slice.
SLICE_GRANULARITY = 2

# Time backoff for binary search.
BACKOFF = 1.2
# Binary search nanosecond granularity.
BNS_GRANULARITY = 10

# Grape args.
DATA_PATH = "/project/ftchong/qoc/thomas/uccsd_slice_qoc/lih_v3"

# Define hardware specific parameters.
num_qubits = 4
num_states = 2
connected_qubit_pairs = get_nearest_neighbor_coupling_list(2, 2, directed=False)
H0 = get_H0(num_qubits, num_states)
Hops, Hnames = get_Hops_and_Hnames(num_qubits, num_states,
                                   connected_qubit_pairs)
states_concerned_list = get_full_states_concerned_list(num_qubits, num_states)
maxA = get_maxA(num_qubits, num_states, connected_qubit_pairs)

# Define convergence parameters and penalties.
max_iterations = 1000
decay = max_iterations / 2
convergence = {'rate':0.01, 'conv_target': 1e-3,
               'max_iterations': max_iterations, 'learning_rate_decay':decay,
               'min_grads': 1e-5}
reg_coeffs = {}
use_gpu = False
sparse_H = False
show_plots = False
method = 'ADAM'
# Steps per nanosecond and nanoseconds per step
spn = 20.0
nps = 1 / spn

# Get slices to perform qoc on. The initial angle of each RZ
# gate does not matter.
theta = UCCSD_LIH_THETA
circuit = optimize_circuit(get_uccsd_circuit('LiH', theta),
                           connected_qubit_pairs)
uccsd_slices = get_uccsd_slices(circuit, granularity=SLICE_GRANULARITY,
                                dependence_grouping=True)

# https://ark.intel.com/products/91754/Intel-Xeon-Processor-E5-2680-v4-35M-Cache-2-40-GHz-
BROADWELL_CORE_COUNT = 14


### MAIN ###

def main():
    # Handle CLI.
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
    slices = uccsd_slices[slice_start:slice_stop + 1]

    # Get a list of the angles that each slice should be compiled for.
    angle_list_deg = list(np.arange(angle_start, angle_stop, angle_step))
    angle_list = list(np.deg2rad(angle_list_deg))
    
    # Build argument iterators to map to each process.
    job_count = slice_count
    slice_index_iter = range(slice_count)
    angles_iter = [angle_list] * job_count
    file_names_iter = list()
    for slice_index in slice_index_iter:
        file_names = list()
        for angle in angle_list_deg:
            file_names.append("s{}_{}".format(slice_index, angle))
        file_names_iter.append(file_names)
    
    # Run QOC for each slice on each angle list.
    with MPIPoolExecutor(job_count) as executor:
        executor.map(process_init, slices, slice_index_iter,
                     angles_iter, file_names_iter)


def process_init(uccsdslice, slice_index, angles,
                 file_names):
    """Do all necessary process specific tasks before running grape.
    Args: ugly
    Returns: nothing
    """
    # Redirect output to a log file.
    log_file = "s{}.log".format(slice_index)
    log_file_path = os.path.join(DATA_PATH, log_file)
    with open(log_file_path, "w") as log:
        # sys.stdout = sys.stderr = log

        # Display pid, time, slice id, and circuit.
        print("PID={}\nTIME={}\nSLICE_ID={}"
              "".format(os.getpid(), time.time(), slice_index))
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
        prev_converged_min_steps = None
        prev_converged_max_steps = None
        prev_converged_mid_steps = None
        prev_converged_sess = None
        initial_guess = None

        # We begin with no initial guess.
        grape_sess = None
        
        for i, angle in enumerate(angles):
            # Get and display necessary information, update slice angles.
            print("\nANGLE={}".format(angle))
            file_name = file_names[i]
            uccsdslice.update_angles([angle] * len(uccsdslice.angles))
            U = uccsdslice.unitary()
            search_converged = False
            # We run the first trial for the same pulse time that the
            # last angle converged to.
            if prev_converged_mid_steps is not None:
                min_steps = prev_converged_min_steps
                max_steps = prev_converged_max_steps
                mid_steps = prev_converged_mid_steps
                initial_guess = prev_converged_sess.uks
            
            # Binary search for the minimum pulse time on the current angle.
            while not search_converged:
                # Search in the search space until we have a convergence window
                # of BNS_GRANULARITY
                while min_steps + BNS_GRANULARITY < max_steps:
                    if initial_guess is not None:
                        initial_guess = resize_uks(initial_guess, mid_steps)
                    total_time = mid_steps * nps
                    print("\nMAX_STEPS={}\nMIN_STEPS={}\nMID_STEPS={}\nTIME={}"
                          "\nGRAPE_START_TIME={}"
                          "".format(max_steps, min_steps, mid_steps, total_time,
                                    time.time()))
                    grape_sess = Grape(H0, Hops, Hnames, U, total_time, mid_steps,
                                  convergence = convergence, reg_coeffs = reg_coeffs,
                                  use_gpu = use_gpu, sparse_H = sparse_H,
                                  method = method, maxA = maxA,
                                  states_concerned_list = states_concerned_list,
                                  show_plots = show_plots, file_name = file_name,
                                  data_path = DATA_PATH ,
                                  initial_guess = initial_guess)
                    print("GRAPE_END_TIME={}".format(time.time()))
                    # If the trial converged, lower the upper bound.
                    # If the tiral did not converge, raise the lower bound.
                    trial_converged = grape_sess.l <= grape_sess.conv.conv_target
                    print("TRIAL_CONVERGED={}".format(trial_converged))
                    if trial_converged:
                        search_converged = True
                        prev_converged_mid_steps = mid_steps
                        prev_converged_max_steps = max_steps
                        prev_converged_min_steps = min_steps
                        prev_converged_sess = grape_sess
                        max_steps = mid_steps
                    else:
                        min_steps = mid_steps
                    # Update mid_steps to run for the next trial.
                    mid_steps = int((max_steps + min_steps) / 2)
                # ENDWHILE
                # If binary search did not converge, then the pulse time is
                # too short and should be backed off.
                print("SEARCH_CONVERGED={}".format(search_converged))
                if not search_converged:
                    max_steps *= BACKOFF
                    mid_steps = int((max_steps + min_steps) / 2)
            # ENDWHILE
            print("CONVERGED_STEPS={}\nCONVERGED_TIME={}"
                  "".format(prev_converged_mid_steps,
                            prev_converged_mid_steps * nps))
        # ENDFOR


def resize_uks(uks, num_steps):
    """
    Truncate or extend the length of each array in a 2D numpy.ndarray.
    If the arrays are extended, fill them with zeros.
    Args:
    uks :: numpy.ndarray - the pulses to be resized
    num_steps :: int - the number of steps the pulses should be
                       extended or truncated to
    Returns:
    new_uks :: numpy.ndarray - the resized pulses
    """
    array_len = uks.shape[1]
    # If the array length is equal to the number of steps,
    # do nothing.
    if array_len == num_steps:
        return uks
    # If the array is longer than the number of steps,
    # take the first num_step elements from the array.
    elif array_len > num_steps:
        mod = lambda array: array.tolist()[:num_steps]
    # If the array is shorter than the number of steps,
    # extend the array with zeros by the difference.
    else:
        diff = num_steps - array_len
        print("diff={}".format(diff))
        mod = lambda array: array.tolist() + [0] * diff

    return np.array([mod(array) for array in uks])


if __name__ == "__main__":
    main()


if __name__ == "__main__":
    main()

