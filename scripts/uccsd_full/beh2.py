"""
beh2.py - A script attempting convergence for QOC on the UCCSD BEH2 molecule.
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

def main():
    file_name = "uccsd_beh2"
    data_path = "/project/ftchong/qoc/thomas/test/"
    log_file = file_name + '.log'
    log_file_path = os.path.join(data_path, log_file)
    with open(log_file_path, "w") as log:
        # Redirect everything to a log file.
        sys.stdout = log
        sys.stderr = log
        # Get circuit.
        connected_qubit_pairs = get_nearest_neighbor_coupling_list(2, 3, directed=False)
        theta = [np.random.random() * 2 * np.pi for _ in range(26)]
        circuit = optimize_circuit(get_uccsd_circuit('BeH2', theta), connected_qubit_pairs)
        print(circuit)

        # Get unitary.
        U = get_unitary(circuit)
        print(U)

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

        # Define pulse time.
        max_pulse_time = get_max_pulse_time(circuit)
        print("MAX_PULSE_TIME={}".format(max_pulse_time))
        total_time = max_pulse_time
        steps = int(100 * total_time)

        # Run GRAPE.
        print("GRAPE_START_TIME={}".format(time.time()))
        SS = Grape(H0, Hops, Hnames, U, total_time, steps,
                   states_concerned_list, convergence, reg_coeffs=reg_coeffs,
                   use_gpu=False, sparse_H=False, method='ADAM', maxA=maxA,
                   show_plots=False, file_name=file_name, data_path=data_path)
        print("GRAPE_END_TIME={}".format(time.time()))


if __name__ == "__main__":
    main()
