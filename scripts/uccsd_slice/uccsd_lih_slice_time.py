"""
uccsd_slice_time.py - A script for computing the appropriate run_time's for
                      each UCCSD slice.
"""
# Set random seeds for reasonable reproducability.
import random
random.seed(0)
import numpy as np
np.random.seed(1)
import tensorflow as tf
tf.set_random_seed(2)

import os
import sys
import time

from fqc.uccsd import get_uccsd_circuit, get_uccsd_slices
from fqc.util import (optimize_circuit, get_unitary,
                      get_nearest_neighbor_coupling_list, get_max_pulse_time)
from fqc.data import UCCSD_LIH_THETA
from hyperopt import fmin, tpe, hp, STATUS_OK, Trials
from mpi4py.futures import MPIPoolExecutor
from quantum_optimal_control.main_grape.grape import Grape
from quantum_optimal_control.core.hamiltonian import (get_H0, 
        get_Hops_and_Hnames, get_full_states_concerned_list, get_maxA)

def main():
    # Get all UCCSD slices with trivial theta-dependent gates.
    slice_granularity = 2
    connected_qubit_pairs = get_nearest_neighbor_coupling_list(2, 2, directed=False)
    # The same randomly generated theta is used for computing slice and full time.
    theta = UCCSD_LIH_THETA
    circuit = optimize_circuit(get_uccsd_circuit('LiH', theta),
                               connected_qubit_pairs)
    slices = get_uccsd_slices(circuit, granularity=slice_granularity,
                              dependence_grouping=True)
    
    # Run time optimizer for each slice.
    connected_qubit_pairs_iter = [connected_qubit_pairs] * 8
    slice_index_iter = range(8)
    with MPIPoolExecutor(8) as executor:
        executor.map(process_init, slices, connected_qubit_pairs_iter,
                     slice_index_iter)


def process_init(uccsdslice, connected_qubit_pairs, slice_index):
    """Initialize a time optimization loop for a single slice.
    Args:
    uccsdslice :: fqc.UCCSDSlice  - the slice to perform time optimization on
    connected_qubit_pairs :: [(int, int)] - connected qubit list
    slice_index :: int - the index of the slice in the sequence of slices

    Returns: nothing
    """
    file_name = "uccsd_lih_s{}".format(slice_index)
    data_path = "/project/ftchong/qoc/thomas/uccsd_slice_time/lih/"
    log_file = file_name + '.log'
    log_file_path = os.path.join(data_path, log_file)
    with open(log_file_path, "w") as log:
        # Redirect everything to a log file.
        sys.stdout = log
        sys.stderr = log
        print("PID={}\nTIME={}\n".format(os.getpid(), time.time()))

        # Display slice.
        print("SLICE_INDEX={}".format(slice_index))
        print(uccsdslice.circuit)

        # Define search space.
        max_pulse_time = get_max_pulse_time(uccsdslice.circuit)
        min_steps = 0
        max_steps = max_pulse_time * 20.0
        print("MAX_PULSE_TIME={}\nMIN_STEPS={}\nMAX_STEPS={}"
              "".format(max_pulse_time, min_steps, max_steps))
        res = binary_search_for_shortest_pulse_time(uccsdslice,
                                                    connected_qubit_pairs,
                                                    file_name, data_path,
                                                    min_steps, max_steps)
        print("RES={}".format(res))

def binary_search_for_shortest_pulse_time(uccsdslice, connected_qubit_pairs,
                                          file_name, data_path, min_steps,
                                          max_steps):
    """Search between [min_steps, max_steps] (inclusive)."""
    # Get unitary.
    U = uccsdslice.unitary()

    # Define hardware specific parameters.
    num_qubits = 4
    num_states = 2
    H0 = get_H0(num_qubits, num_states)
    Hops, Hnames = get_Hops_and_Hnames(num_qubits, num_states, connected_qubit_pairs)
    states_concerned_list = get_full_states_concerned_list(num_qubits, num_states)
    maxA = get_maxA(num_qubits, num_states, connected_qubit_pairs)

    # Define convergence parameters and penalties.
    max_iterations = 1000
    decay = max_iterations / 2
    convergence = {'rate': 1e-2, 'conv_target': 1e-3,
                   'max_iterations': max_iterations,
                   'learning_rate_decay': decay,
                   'min_grads': 1e-5}
    reg_coeffs = {'dwdt': 0.001, 'envelope': 0.01}
    use_gpu = False
    sparse_H = False
    show_plots = False
    method = 'ADAM'

    while min_steps + 10 < max_steps:  # just estimate to +/- 0.5ns
        print("\n")
        mid_steps = int((min_steps + max_steps) / 2)
        total_time = mid_steps / 20.0
        print("MID_STEPS={}".format(mid_steps))
        print("TRIAL_TOTAL_TIME={}".format(total_time))
        SS = Grape(H0, Hops, Hnames, U, total_time, mid_steps,
                   states_concerned_list, convergence, reg_coeffs=reg_coeffs,
                   use_gpu=False, sparse_H=False, method='ADAM', maxA=maxA,
                   show_plots=False, file_name=file_name, data_path=data_path)
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
