"""
uccsd_slice_qoc.py - A module for running quantum optimal control on
                     UCCSD slices.
"""
from itertools import product
from multiprocessing import Pool
import os
import sys
import time

from fqc.uccsd import get_uccsd_circuit, get_uccsd_slices
from fqc.util import (optimize_circuit, get_unitary,
                      get_nearest_neighbor_coupling_list)
import numpy as np
from quantum_optimal_control.main_grape.grape import Grape
from quantum_optimal_control.core.hamiltonian import (get_H0,
                                                      get_Hops_and_Hnames,
                                                      get_full_states_concerned_list,
                                                      get_maxA)

# TODO: CLI

# https://ark.intel.com/products/91754/Intel-Xeon-Processor-E5-2680-v4-35M-Cache-2-40-GHz-
BROADWELL_CORE_COUNT = 14

def main():
    # Pair each theta dependent slice with a non-theta dependent slice.
    slice_granularity = 2
    # Run QOC for every angle in {0, 10, 20, ..., 350}
    angle_step = 10

    # Define output_path
    data_path = ("/project/ftchong/qoc/thomas/uccsd_slice_qoc_g{}"
                 "".format(slice_granularity))

    # Define hardware specific parameters.
    num_qubits = 4
    num_states = 2
    connected_qubit_pairs = get_nearest_neighbor_coupling_list(2, 2, directed=False)
    H0 = get_H0(num_qubits, num_states, connected_qubit_pairs)
    Hops, Hnames = get_Hops_and_Hnames(num_qubits, num_states)
    states_concerned_list = get_full_states_concerned_list(num_qubits, num_states)
    maxA = get_maxA(num_qubits, num_states)

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

    # Define time scale in nanoseconds.
    total_time = 50
    steps = total_time * 100

    # Get slices to perform qoc on. Initial theta does not matter.
    theta = [np.random.random() for _ in range(8)]
    circuit = optimize_circuit(get_uccsd_circuit('LiH', theta),
                               connected_qubit_pairs)
    slices = get_uccsd_slices(circuit, granularity=slice_granularity)

    # Run QOC for every slice on every angle_list.
    pool = Pool(BROADWELL_CORE_COUNT)

    for i, uccsdslice in enumerate(slices):
        # Temporary break for debugging purposes.
        if i > 0:
            break
        angle_lists = get_angle_lists(len(uccsdslice.angles), angle_step)
        for j, angle_list in enumerate(angle_lists):
            # Temporary break for debugging purposes.
            if j > 0:
                break
            print("slice {}, angle list {}".format(i, j))
            pool.apply_async(process_init,
                             (uccsdslice, angle_list, i, j, angle_step, H0, Hops,
                              Hnames, total_time, steps, states_concerned_list,
                              convergence, reg_coeffs, method, maxA, use_gpu,
                              sparse_H, show_plots, data_path))
        # END FOR
    # END FOR
    pool.close()
    pool.join()

def process_init(uccsdslice, angle_list, i, j, angle_step, H0, Hops, Hnames,
                 total_time, steps, states_concerned_list, convergence,
                 reg_coeffs, method, maxA, use_gpu, sparse_H, show_plots,
                 data_path):
    """Do all necessary process specific tasks before running grape.
    Args: ugly
    Returns: nothing
    """
    file_name = "pulse_s{}_{}".format(i, j * angle_step)
    log_file = file_name + '.log'
    log_file_path = os.path.join(data_path, log_file)
    with open(log_file_path, "w") as log:
        # Redirect everything to a log file.
        # sys.stdout = log
        # sys.stderr = log
        log.write("PID={}\nTIME={}\n".format(os.getpid(), time.time()))
        
        # Display angles, updated circuit, and get unitary.
        print(angle_list)
        uccsdslice.update_angles(angle_list)
        print(uccsdslice.circuit)
        U = uccsdslice.unitary()
        print("got unitary")
        # Run grape.
        uks, U_f = Grape(H0, Hops, Hnames, U, total_time, steps,
                         states_concerned_list, convergence = convergence,
                         reg_coeffs = reg_coeffs, method = method, maxA = maxA,
                         use_gpu = use_gpu, sparse_H = sparse_H,
                         show_plots = show_plots, file_name=file_name,
                         data_path = data_path)


def get_angle_lists(parameterized_gate_count, angle_step):
    """
    Return all possible angle combinations for the parameterized gates.
    Args:
    parameterized_gate_count :: int - the number of gates parameterized by
                                      an angle in the circuit.
    angle_step :: int - the step size between angles

    Returns:
    angle_lists :: [[float]] - a list of lists of floats with angles to
                               update angles in the slices
    """
    angle_space = list(np.deg2rad(np.arange(0, 360, angle_step)))
    angle_spaces = [angle_space for _ in range(parameterized_gate_count)]
    angle_lists = list(product(*angle_spaces))
    return angle_lists


if __name__ == "__main__":
    main()
