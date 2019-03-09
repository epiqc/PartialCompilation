"""
uccsd_slice_hpo.py - A script for computing the appropriate hyperparameters
                     for GRAPe on a UCCSD slice.
"""

import argparse
from itertools import product
import os
import sys
import time

from fqc.uccsd import get_uccsd_circuit, get_uccsd_slices
from fqc.util import (optimize_circuit, get_unitary,
                      get_nearest_neighbor_coupling_list)
from hyperopt import fmin, tpe, hp, STATUS_OK, Trials
from mpi4py.futures import MPIPoolExecutor
import numpy as np
from quantum_optimal_control.main_grape.grape import Grape
from quantum_optimal_control.core.hamiltonian import (get_H0, 
        get_Hops_and_Hnames, get_full_states_concerned_list, get_maxA)

def main():
    # Get all UCCSD slices with trivial theta-dependent gates.
    slice_granularity = 2
    theta = [0] * 8
    circuit = optimize_circuit(get_uccsd_circuit('LiH', theta),
                               connected_qubit_pairs)
    slices = get_uccsd_slices(circuit, granularity=slice_granularity,
                              dependence_grouping=True)
    
    # Run time optimizer for each slice.
    with MPIPoolExecutor(14) as executor:
        executor.map(process_init, uccsdslice_iter, slice_index_iter,)


def process_init(uccsdslice, slice_index):
    """Initialize a time optimization loop.
    Args:
    uccsdslice :: fqc.UCCSDSlice  - the slice to perform time optimization on
    slice_index :: int - the index of the slice in the sequence of slices

    Returns: nothing
    """
    file_name = "s{}".format(slice_index)
    log_file = file_name + '.log'
    log_file_path = os.path.join(data_path, log_file)
    with open(log_file_path, "w") as log:
        # Redirect everything to a log file.
        sys.stdout = log
        sys.stderr = log
        log.write("PID={}\nTIME={}\n".format(os.getpid(), time.time()))
        
        # Display slice.
        print("SLICE_INDEX={}".format(slice_index))
        print(uccsdslice.circuit)

        # Define search space.
        lr_lb = 1e-5
        lr_ub = 1
        decay_lb = 0
        decay_ub = 100
        print("LR_LB={}, LR_UB={}, DECAY_LB={}, DECAY_UB"
              "".format(lr_lb, lr_ub, dcay_lb, decay_ub))
        space = {'lr': hp.loguniform('lr', np.log(lr_lb), np.log(lr_ub)),
                 'decay': hp.uniform('decay', decay_lb, decay_ub),}
        objective_wrapper = lambda params: objective(uccsdslice, file_name,
                                                     params)
        trials = Trials()
        best = fmin(objective_wrapper, space = space,
                    algo = tpe.suggest, max_evals = 100,
                    trials = trials)
        
        print("BEST={}".format(best))


def objective(uccsdslice, file_name, params):
    """This is the function to minimize
    """
    # Grape args.
    data_path = "/project/ftchong/qoc/thomas/uccsd_slice_time/"

    # Get unitary.
    U = uccsdslice.unitary()

    # Define hardware specific parameters.
    num_qubits = 4
    num_states = 2
    connected_qubit_pairs = get_nearest_neighbor_coupling_list(2, 2, directed=False)
    H0 = get_H0(num_qubits, num_states)
    Hops, Hnames = get_Hops_and_Hnames(num_qubits, num_states, connected_qubit_pairs)
    states_concerned_list = get_full_states_concerned_list(num_qubits, num_states)
    maxA = get_maxA(num_qubits, num_states, connected_qubit_pairs)

    # Define convergence parameters and penalties.
    max_iterations = 1000
    decay = max_iterations / 2
    convergence = {'rate': params['lr'], 'conv_target': 1e-3,
                   'max_iterations': max_iterations,
                   'learning_rate_decay': params['decay'],
                   'min_grads': 1e-5}
    reg_coeffs = {'dwdt': 0.001, 'envelope': 0.01}
    use_gpu = False
    sparse_H = False
    show_plots = False
    method = 'ADAM'
    
    # Define time.
    run_time = 50
    steps = total_time * 100
    
    print("GRAPE_START_TIME={}".format(time.time()))
    SS = Grape(H0, Hops, Hnames, U, run_time, steps,
               states_concerned_list, convergence = convergence,
               reg_coeffs = reg_coeffs, method = method, maxA = maxA,
               use_gpu = use_gpu, sparse_H = sparse_H,
               show_plots = show_plots, file_name = file_name,
               data_path = data_path)
    print("GRAPE_END_TIME={}".format(time.time()))
    
    return {
        'loss': SS.l,
        'status': STATUS_OK,
    }

if __name__ == "__main__":
    main()
