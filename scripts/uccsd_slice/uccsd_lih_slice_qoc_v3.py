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

from itertools import product
import os
import sys
import time
import argparse

from fqc.data import UCCSD_LIH_THETA, UCCSD_LIH_SLICE_TIMES
from fqc.uccsd import get_uccsd_circuit, get_uccsd_slices
from fqc.util import (optimize_circuit, get_unitary,
                      get_nearest_neighbor_coupling_list,
                      get_max_pulse_time)
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
DATA_PATH = "/project/ftchong/qoc/thomas/uccsd_slice_qoc/lih_v2"

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
reg_coeffs = {'dwdt': 0.001, 'envelope': 0.01}
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
slices = get_uccsd_slices(circuit, granularity=SLICE_GRANULARITY,
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
    slices = slices[slice_start:slice_stop + 1]

    # Get a list of the angles that each slice should be compiled for.
    angle_list_deg = np.arange(angle_start, angle_stop, angle_step)
    angle_list = list(np.deg2rad(angle_list_deg))
    
    # Build argument iterators to map to each process.
    job_count = slice_count
    slice_index_iter = range(slice_count)
    angle_iter = [angle_list] * job_count
    file_names_iter = list()
    for slice_index in slice_index_iter:
        file_names = list()
        for angle in angle_list_deg:
            file_names.append("s{}_{}".format(slice_index, angle))
        file_names_iter.append(file_names)
    
    # Run QOC for each slice on each angle list.
    with MPIPoolExecutor(job_count) as executor:
        executor.map(process_init, slices, slice_index_iter,
                     angle_iter, file_names_iter)


def process_init(uccsdslice, slice_index, angles, max_time,
                 file_names):
    """Do all necessary process specific tasks before running grape.
    Args: ugly
    Returns: nothing
    """
    # Redirect output to a log file.
    log_file = "s{}.log".format(slice_index)
    log_file_path = os.path.join(DATA_PATH, log_file)
    with open(log_file_path, "w") as log:
        sys.stdout = sys.stderr = log

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
        prev_mid_steps = mid_steps
        # We begin with no initial guess.
        grape_sess = None
        
        for i, angle in enumerate(angles):
            # Get and display necessary information, update slice angles.
            print("\nANGLE={}".format(angle))
            file_name = file_names[i]
            uccsdslice.update_angles([angle] * len(uccsdslice.angles))
            U = uccsdslice.unitary()
            # Initial guess is the previous run's pulse.
            initial_guess = None if grape_sess is None else grape_sess.uks
            search_converged = False
            
            # Binary search for the minimum pulse time on the current angle.
            while not search_converged:
                print("MAX_STEPS={}\nMIN_STEPS={}\nMID_STEPS={}"
                      "".format(max_steps, min_steps, mid_steps))
                
                while min_steps + BNS_GRANULARITY < max_steps:
                    total_time = mid_steps * nps
                    print("\nMID_STEPS={}\nTIME={}\n".format(mid_steps, total_time))
                    trial_converged = S.l <= SS.conv.conv_target
                    if trial_converged:
                        max_steps = mid_steps
                    else:
                        min_steps = mid_steps
                    prev_mid_steps = mid_steps
                    mid_steps = int((max_steps + min_steps) / 2)
                
                # If binary search did not converge, then the pulse time is
                # too short and should be backed off.
                max_steps *= BACKOFF
                mid_steps = int((max_steps + min_steps) / 2)



if __name__ == "__main__":
    main()


def binary_search_for_shortest_pulse_time(unitary, file_name, min_steps,
                                          max_steps, initial_guess=None):
    """Search between [min_steps, max_steps] (inclusive)."""

    while min_steps + 10 < max_steps:  # just estimate to +/- 0.5ns
        print("\n")
        mid_steps = int((min_steps + max_steps) / 2)
        total_time = mid_steps / 20.0
        print("MID_STEPS={}".format(mid_steps))
        print("TRIAL_TOTAL_TIME={}".format(total_time))
        print("GRAPE_START_TIME={}".format(time.time()))
        SS = Grape(H0, Hops, Hnames, U, total_time, mid_steps,
                   states_concerned_list, convergence, reg_coeffs=reg_coeffs,
                   use_gpu=False, sparse_H=False, method='ADAM', maxA=maxA,
                   show_plots=False, file_name=file_name, data_path=DATA_PATH)
        print("GRAPE_END_TIME={}".format(time.time()))
        converged = SS.l < SS.conv.conv_target
        print("CONVERGED={}".format(converged))
        if converged:
            max_steps = mid_steps
        else:
            min_steps = mid_steps
        print("MAX_STEPS={}\nMIN_STEPS={}".format(max_steps,min_steps))

    return mid_steps / 20.0


if __name__ == "__main__":
    main()
