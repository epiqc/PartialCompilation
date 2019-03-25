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
                      get_nearest_neighbor_coupling_list)
from mpi4py.futures import MPIPoolExecutor
from quantum_optimal_control.main_grape.grape import Grape
from quantum_optimal_control.core.hamiltonian import (get_H0,
                                                      get_Hops_and_Hnames,
                                                      get_full_states_concerned_list,
                                                      get_maxA)

# https://ark.intel.com/products/91754/Intel-Xeon-Processor-E5-2680-v4-35M-Cache-2-40-GHz-
BROADWELL_CORE_COUNT = 14
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
    
    # We will work with g2 slices which each have 1 unique angle per slice.
    slice_granularity = 2
    angles_per_slice = 1

    # Grape args.
    data_path = "/project/ftchong/qoc/thomas/uccsd_slice_qoc/lih_v2"

    # Define hardware specific parameters.
    num_qubits = 4
    num_states = 2
    connected_qubit_pairs = get_nearest_neighbor_coupling_list(2, 2, directed=False)
    H0 = np.zeros((num_states ** num_qubits, num_states ** num_qubits))
    Hops, Hnames = get_Hops_and_Hnames(num_qubits, num_states,
                                       connected_qubit_pairs)
    states_concerned_list = get_full_states_concerned_list(num_qubits, num_states)
    maxA = get_maxA(num_qubits, num_states, connected_qubit_pairs)
    
    # Define convergence parameters and penalties.
    convergence = {'rate': 2e-2, 'conv_target': 1e-3,
                   'max_iterations': 1e3, 'learning_rate_decay': 1e3}
    reg_coeffs = {}
    use_gpu = False
    sparse_H = False
    show_plots = False
    method = 'ADAM'

    # Get slices to perform qoc on. The initial angle of each RZ
    # gate does not matter.
    theta = UCCSD_LIH_THETA
    circuit = optimize_circuit(get_uccsd_circuit('LiH', theta),
                               connected_qubit_pairs)
    slices = get_uccsd_slices(circuit, granularity=slice_granularity,
                              dependence_grouping=True)
    # Trim slices to only include start thru stop
    slices = slices[slice_start:slice_stop + 1]
    angle_list_deg = np.arange(angle_start, angle_stop, angle_step)
    angle_list = list(np.deg2rad(angle_list_deg))
    
    # Build argument iterators to map to each process.
    job_count = slice_count
    slice_index_iter = range(slice_count)
    angle_iter = [angle_list] * job_count
    H0_iter = [H0] * job_count
    Hops_iter = [Hops] * job_count
    Hnames_iter = [Hnames] * job_count
    total_time_iter = UCCSD_LIH_SLICE_TIMES
    steps_iter = [int(pulse_time * 100) for pulse_time in UCCSD_LIH_SLICE_TIMES]
    states_concerned_list_iter = [states_concerned_list] * job_count
    convergence_iter = [convergence] * job_count
    reg_coeffs_iter = [reg_coeffs] * job_count
    method_iter = [method] * job_count
    maxA_iter = [maxA] * job_count
    use_gpu_iter = [use_gpu] * job_count
    sparse_H_iter = [sparse_H] * job_count
    show_plots_iter = [show_plots] * job_count
    file_names_iter = list()
    data_path_iter = [data_path] * job_count
    for slice_index in slice_index_iter:
        file_names = list()
        for angle in angle_list_deg:
            file_names.append("s{}_{}".format(slice_index, angle))
        file_names_iter.append(file_names)
    
    # Run QOC for each slice on each angle list.
    with MPIPoolExecutor(job_count) as executor:
        executor.map(process_init, slices, slice_index_iter,
                     angle_iter, H0_iter, Hops_iter, 
                     Hnames_iter, total_time_iter, steps_iter,
                     states_concerned_list_iter, convergence_iter,
                     reg_coeffs_iter, method_iter, maxA_iter, use_gpu_iter,
                     sparse_H_iter, show_plots_iter, file_names_iter,
                     data_path_iter)


def process_init(uccsdslice, slice_index, angles, H0, Hops, Hnames,
                 total_time, steps, states_concerned_list, convergence,
                 reg_coeffs, method, maxA, use_gpu, sparse_H, show_plots,
                 file_names, data_path):
    """Do all necessary process specific tasks before running grape.
    Args: ugly
    Returns: nothing
    """
    log_file = "s{}.log".format(slice_index)
    log_file_path = os.path.join(data_path, log_file)
    with open(log_file_path, "w") as log:
        sys.stdout = sys.stderr = log
        # Redirect everything to a log file.
        print("PID={}\nTIME={}".format(os.getpid(), time.time()))

        # Display angles, updated circuit, and get unitary.
        print("SLICE_ID={}".format(slice_index))
        print(uccsdslice.circuit)
        
        S = None
        for i, angle in enumerate(angles):
            file_name = file_names[i]
            print("ANGLE={}".format(angle))
            uccsdslice.update_angles([angle] * len(uccsdslice.angles))
            U = uccsdslice.unitary()
            # Initial guess is the previous run's pulse.
            initial_guess = None if S is None else S.uks
            
            # Run grape.
            print("GRAPE_START_TIME={}".format(time.time()))
            S = Grape(H0, Hops, Hnames, U, total_time, steps,
                      states_concerned_list, convergence = convergence,
                      reg_coeffs = reg_coeffs, method = method, maxA = maxA,
                      use_gpu = use_gpu, sparse_H = sparse_H,
                      show_plots = show_plots, file_name=file_name,
                      data_path = data_path, initial_guess = initial_guess)
            print("GRAPE_END_TIME={}".format(time.time()))

if __name__ == "__main__":
    main()
