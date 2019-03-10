"""
uccsd_full_time.py - A module for computing the pulse time of the full
                     uccsd circuits.
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

from fqc.data import UCCSD_LIH_THETA
from fqc.uccsd import get_uccsd_circuit, get_uccsd_slices, MOLECULE_TO_INFO
from fqc.util import (optimize_circuit, get_unitary,
                      get_nearest_neighbor_coupling_list, get_max_pulse_time,
                      merge_rotation_gates)
from hyperopt import fmin, tpe, hp, STATUS_OK, Trials
from mpi4py.futures import MPIPoolExecutor
from quantum_optimal_control.main_grape.grape import Grape
from quantum_optimal_control.core.hamiltonian import (get_H0, 
        get_Hops_and_Hnames, get_full_states_concerned_list, get_maxA)

molecule_strings = list(MOLECULE_TO_INFO.keys())
molecule_count = len(molecule_strings)

def main():
    # Get all UCCSD circuits.
    circuits = list()
    connected_qubit_pairs_list = list()
    # Connected qubit pairs currently not implemented for larger ansatze,
    # so we only run time optimization on the first molecule lih.
    for i, molecule_string in enumerate(molecule_strings):
        if i > 0:
            break
        # Some circuits are trivial without Rz gates, so we cannot
        # set them to zero. Instead we use a random theta for each circuit.
        # TODO: implement a random theta for each molecule.
        theta = UCCSD_LIH_THETA
        circuit = get_uccsd_circuit(molecule_string, theta)
        num_qubits = circuit.width()
        # TOOD: Implement connected_qubit_pairs based on number of qubits.
        connected_qubit_pairs = get_nearest_neighbor_coupling_list(2, 2, directed=False)
        optimized_circuit = optimize_circuit(circuit, connected_qubit_pairs)
        connected_qubit_pairs_list.append(connected_qubit_pairs)
        circuits.append(optimized_circuit)
    
    # Run time optimizer for each circuit.
    # with MPIPoolExecutor(molecule_count) as executor:
    with MPIPoolExecutor(1) as executor:
        executor.map(process_init, circuits, molecule_strings,
                     connected_qubit_pairs_list)


def process_init(circuit, molecule_string, connected_qubit_pairs):
    """Initialize a time optimization loop for a single circuit.
    """
    file_name = "uccsd_{}".format(molecule_string.lower())
    data_path = "/project/ftchong/qoc/thomas/uccsd_full_time/{}/"
                "".format(molecule_string.lower())
    log_file = file_name + '.log'
    log_file_path = os.path.join(data_path, log_file)
    with open(log_file_path, "w") as log:
        # Redirect everything to a log file.
        sys.stdout = log
        sys.stderr = log
        print("PID={}\nTIME={}\n".format(os.getpid(), time.time()))

        # Display slice.
        print("MOLECULE={}".format(molecule_string))
        print(circuit)

        # Define search space.
        max_pulse_time = get_max_pulse_time(circuit)
        min_steps = 0
        max_steps = max_pulse_time * 20.0
        print("MAX_PULSE_TIME={}\nMIN_STEPS={}\nMAX_STEPS={}"
              "".format(max_pulse_time, min_steps, max_steps))
        res = binary_search_for_shortest_pulse_time(circuit,
                                                    connected_qubit_pairs,
                                                    file_name, data_path,
                                                    min_steps, max_steps)
        print("RES={}".format(res))


def binary_search_for_shortest_pulse_time(circuit, connected_qubit_pairs,
                                          file_name, data_path, min_steps,
                                          max_steps):
    """Search between [min_steps, max_steps] (inclusive)."""
    # Get unitary.
    U = get_unitary(circuit)

    # Define hardware specific parameters.
    num_qubits = circuit.width()
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
        print("GRAPE_START_TIME={}".format(time.time()))
        SS = Grape(H0, Hops, Hnames, U, total_time, mid_steps,
                   states_concerned_list, convergence, reg_coeffs=reg_coeffs,
                   use_gpu=False, sparse_H=False, method='ADAM', maxA=maxA,
                   show_plots=False, file_name=file_name, data_path=data_path)
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
