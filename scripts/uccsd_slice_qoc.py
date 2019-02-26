"""
uccsd_slice_qoc.py - A module for running quantum optimal control on
                     UCCSD slices.
"""
from itertools import product
import os
import sys
import time
import argparse

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
                        "(0-22)")
    parser.add_argument("--slice-stop", type=int, default=0, help="the "
                        "inclusive upper bound of slice indices to include "
                        "(0-22)")
    args = vars(parser.parse_args())
    angle_start = args["angle_start"]
    angle_stop = args["angle_stop"]
    angle_step = args["angle_step"]
    slice_start = args["slice_start"]
    slice_stop = args["slice_stop"]

    # Pair each theta dependent slice with a non-theta dependent slice.
    slice_granularity = 2
    # For granularity = 2, we assume each slice is only dependent
    # on one angle.
    angles_per_slice = 1

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
    slices = get_uccsd_slices(circuit, granularity=slice_granularity,
                              dependence_grouping=True)
    slices = slices[slice_start:slice_stop + 1]
    # Run QOC for every slice on every angle_list.
    for i, uccsdslice in enumerate(slices):
        slice_index = i + slice_start
        angle_lists = get_angle_lists(angles_per_slice, angle_start,
                                      angle_stop, angle_step)
        for j, angle_list in enumerate(angle_lists):
            angle = j * angle_step + angle_start
            file_name = "pulse_s{}_{}".format(slice_index, angle)
            process_init(uccsdslice, i, angle, file_name, H0, Hops,
                         Hnames, total_time, steps, states_concerned_list,
                         convergence, reg_coeffs, method, maxA, use_gpu,
                         sparse_H, show_plots, data_path)
        # END FOR
    # END FOR


def process_init(uccsdslice, i, angle, file_name, H0, Hops, Hnames,
                 total_time, steps, states_concerned_list, convergence,
                 reg_coeffs, method, maxA, use_gpu, sparse_H, show_plots,
                 data_path):
    """Do all necessary process specific tasks before running grape.
    Args: ugly
    Returns: nothing
    """
    log_file = file_name + '.log'
    log_file_path = os.path.join(data_path, log_file)
    with open(log_file_path, "w") as log:
        # Redirect everything to a log file.
        sys.stdout = log
        sys.stderr = log
        log.write("PID={}\nTIME={}\n".format(os.getpid(), time.time()))
        
        # Display angles, updated circuit, and get unitary.
        print("SLICE_ID={}\n".format(i))
        print("ANGLE={}\n".format(angle))
        uccsdslice.update_angles([angle] * len(uccsdslice.angles))
        print(uccsdslice.circuit)
        U = uccsdslice.unitary()

        # Run grape.
        log.write("GRAPE_START_TIME={}\n".format(time.time()))
        uks, U_f = Grape(H0, Hops, Hnames, U, total_time, steps,
                         states_concerned_list, convergence = convergence,
                         reg_coeffs = reg_coeffs, method = method, maxA = maxA,
                         use_gpu = use_gpu, sparse_H = sparse_H,
                         show_plots = show_plots, file_name=file_name,
                         data_path = data_path)
        log.write("GRAPE_END_TIME={}\n".format(time.time()))


def get_angle_lists(parameterized_gate_count, angle_start, angle_stop,
                    angle_step):
    """
    Return all possible angle combinations for the parameterized gates.
    Args:
    parameterized_gate_count :: int - the number of gates parameterized by
                                      an angle in the circuit.
    angle_start :: int - the inclusive lower bound of angles to generate
    angle_stop :: int - the exclusive upper bound of angles to generate
    angle_step :: int - the step size between angles

    Returns:
    angle_lists :: [[float]] - a list of lists of floats with angles to
                               update angles in the slices
    """
    angle_space = list(np.deg2rad(np.arange(angle_start,
                                            angle_stop,
                                            angle_step)))
    angle_spaces = [angle_space for _ in range(parameterized_gate_count)]
    angle_lists = list(product(*angle_spaces))
    return angle_lists


if __name__ == "__main__":
    main()
