import sys
sys.path.append('../..')
import config
from fqc import uccsd, util

import numpy as np
from datetime import datetime

data_path = config.DATA_PATH
file_name = datetime.today().strftime('%h%d_block%s' % sys.argv[1])

from quantum_optimal_control.helper_functions.grape_functions import transmon_gate
from quantum_optimal_control.main_grape.grape import Grape
from quantum_optimal_control.core import hamiltonian

d = 2  # this is the number of energy levels to consider (i.e. d-level qudits)
max_iterations = 6000
decay =  max_iterations / 2
convergence = {'rate':0.01, 'max_iterations': max_iterations,
                       'conv_target':1e-3, 'learning_rate_decay':decay, 'min_grad': 1e-12, 'update_step': 20}
reg_coeffs = {}


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



from qiskit import QuantumCircuit, QuantumRegister
from copy import deepcopy

circuit = uccsd.get_uccsd_circuit('BeH2')


subcircuits = []

gates = circuit.data
qubit_block_indices = [{0, 1, 2, 3}, {2, 3, 4, 5}, {1, 2, 3, 4}]
current_block_type = 0  # start with the 0123 "L"

while len(gates) > 0:    
    uncontaminated_indices = qubit_block_indices[current_block_type].copy()
    remaining_gates = []
    affected_indices = set()
    
    subcircuit = deepcopy(circuit); subcircuit.data = []
    
    for i, gate in enumerate(gates):
        if len(uncontaminated_indices) > 0:
            if len(gate.qargs) == 1:
                if gate.qargs[0][1] in uncontaminated_indices:
                    subcircuit.data.append(gate)
                    affected_indices.add(gate.qargs[0][1])
                else:
                    remaining_gates.append(gate)
            elif len(gate.qargs) == 2:
                if gate.qargs[0][1] in uncontaminated_indices and gate.qargs[1][1] in uncontaminated_indices:
                    subcircuit.data.append(gate)
                    affected_indices.add(gate.qargs[0][1])
                    affected_indices.add(gate.qargs[1][1])
                else:
                    uncontaminated_indices.discard(gate.qargs[0][1])
                    uncontaminated_indices.discard(gate.qargs[1][1])
                    remaining_gates.append(gate)
        else:
            remaining_gates.append(gate)
                
    gates = remaining_gates
    if len(subcircuit.data) > 0:
        if current_block_type == 2:
            affected_indices = list(sorted(list(affected_indices)))
            connected_qubit_pairs = [(affected_indices.index(item[0]), affected_indices.index(item[1])) for item in [(1, 2), (1, 3), (2, 3), (2, 4), (3, 4)] \
                                     if item[0] in affected_indices and item[1] in affected_indices]
            subcircuits.append((util.circuitutil.squash_circuit(subcircuit), connected_qubit_pairs))
        else:
            subcircuit = util.circuitutil.squash_circuit(subcircuit)
            connected_qubit_pairs = [(0, 1), (1, 2), (2, 3)]  # within the "L", linear connectivity
            subcircuits.append((subcircuit,
                                connected_qubit_pairs[:subcircuit.width()-1]))

    current_block_type = (current_block_type + 1) % len(qubit_block_indices)


subcircuit, connected_qubit_pairs = subcircuits[int(sys.argv[1])]
    
 
N = subcircuit.width()
H0 = np.zeros((d ** N, d ** N))
Hops, Hnames = hamiltonian.get_Hops_and_Hnames(N, d, connected_qubit_pairs)
states_concerned_list = hamiltonian.get_full_states_concerned_list(N, d)
maxA = hamiltonian.get_maxA(N, d, connected_qubit_pairs)
U = util.circuitutil.get_unitary(subcircuit)

shortest_time = binary_search_for_shortest_pulse_time(0.0, 60.0, tolerance=0.3)
print('\n\n^^^SHORTEST TIME was %s' % shortest_time)
