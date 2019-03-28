import sys
sys.path.append('../../..')
import config
from fqc import uccsd, util

import numpy as np
from datetime import datetime

data_path = config.DATA_PATH
file_name = datetime.today().strftime('%h%d_fixed_slice_%s' % sys.argv[1])

from quantum_optimal_control.helper_functions.grape_functions import transmon_gate
from quantum_optimal_control.main_grape.grape import Grape
from quantum_optimal_control.core import hamiltonian

d = 2  # this is the number of energy levels to consider (i.e. d-level qudits)
max_iterations = 6000
decay =  max_iterations / 2
convergence = {'rate':0.01, 'max_iterations': max_iterations,
                       'conv_target':1e-3, 'learning_rate_decay':decay, 'min_grad': 1e-12, 'update_step': 20}
reg_coeffs = {}



N = 4
connected_qubit_pairs = util.get_nearest_neighbor_coupling_list(2, 2, directed=False)
H0 = np.zeros((d ** N, d ** N))
Hops, Hnames = hamiltonian.get_Hops_and_Hnames(N, d, connected_qubit_pairs)
states_concerned_list = hamiltonian.get_full_states_concerned_list(N, d)
maxA = hamiltonian.get_maxA(N, d, connected_qubit_pairs)



circuit = uccsd.get_uccsd_circuit('LiH')


slices = uccsd.get_uccsd_slices(circuit, granularity=1)
slices = [slice for slice in slices if not slice.parameterized]


def binary_search_for_shortest_pulse_time(min_time, max_time, tolerance=1):
    """Search between [min_time, max_time] up to 1ns tolerance. Assumes 20 steps per ns."""
    min_steps, max_steps = min_time * 20, max_time * 20
    while min_steps + 20 * tolerance < max_steps:  # just estimate to +- 1ns
        mid_steps = int((min_steps + max_steps) / 2)
        total_time = mid_steps / 20.0
        print('\n\ntrying total_time: %s for unitary of size %s' % (str(total_time), str(U.shape)))
        SS = Grape(H0, Hops, Hnames, U, total_time, mid_steps, states_concerned_list, convergence,
                         reg_coeffs=reg_coeffs,
                         use_gpu=False, sparse_H=False, method='Adam', maxA=maxA,
                         show_plots=False, file_name=file_name, data_path=data_path)
        if SS.l < SS.conv.conv_target:  # if converged, search lower half
            max_steps = mid_steps
        else:
            min_steps = mid_steps

    return mid_steps / 20


slice = slices[int(sys.argv[1])]
U = slice.unitary()


shortest_time = binary_search_for_shortest_pulse_time(28.0, 80.0, tolerance=0.3)
print('\n\n^^^SHORTEST TIME was %s' % shortest_time)
