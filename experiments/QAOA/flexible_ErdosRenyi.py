import random
random.seed(0)
import numpy as np
np.random.seed(1)
import tensorflow as tf
tf.set_random_seed(2)

import sys
sys.path.append('../..')
import config
from fqc import uccsd, qaoa,  util

#import numpy as np
from datetime import datetime

# Parameters for the Erdos-Renyi graph
def find_p(num_s, i):
    res = 0
    old_res = 0
    for (j, s) in enumerate(num_s):
        old_res = res
        res = res + s
        if res > i:
            return (j+1, i - old_res)
    print("Input %d has exceeded all slices." % i)
    return (-1, -1) 

graph_N = 6
num_slices = [3,8,11,14,17,20,23,26] # for p = 1,2,3,...
sbatch_id = int(sys.argv[1])
(graph_p, slice_id) = find_p(num_slices, sbatch_id)

print("Experiment FlexibleErdosRenyi N=%d p=%d slice=%d" % (graph_N, graph_p, slice_id))
data_path = config.DATA_PATH
file_name = datetime.today().strftime('%h%d_flexErdosRenyiN%dp%d_slice_%s' % (graph_N, graph_p, slice_id))


from quantum_optimal_control.helper_functions.grape_functions import transmon_gate
from quantum_optimal_control.main_grape.grape import Grape
from quantum_optimal_control.core import hamiltonian


d = 2  # this is the number of energy levels to consider (i.e. d-level qudits)
max_iterations = 6000
decay =  max_iterations / 2
convergence = {'rate':0.01, 'max_iterations': max_iterations,
               'conv_target':1e-3, 'learning_rate_decay':decay, 'min_grad': 1e-20, 'update_step': 20}
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



qaoa_circuit = qaoa.get_qaoa_circuit(graph_N, graph_p, 'ErdosRenyi')
qaoa_circuit = util.circuitutil.optimize_circuit(qaoa_circuit)
coupling_list = util.circuitutil.get_nearest_neighbor_coupling_list(2, int(graph_N/2))
qaoa_circuit = util.circuitutil.optimize_circuit(qaoa_circuit, coupling_list)


from qiskit import QuantumCircuit, QuantumRegister
from copy import deepcopy


def indices(gate):
    return [qarg[1] for qarg in gate.qargs]


def gate_block_index(gate, blocking):
    if len(indices(gate)) == 1:
        return [indices(gate)[0] in grouping for grouping in blocking].index(True)
    else:
        # check which block each qubit is in
        control_index, target_index = indices(gate)
        control_block_index = [control_index in grouping for grouping in blocking].index(True)
        target_block_index = [target_index in grouping for grouping in blocking].index(True)
        if control_block_index != target_block_index:
            return -1
        else:
            return control_block_index
        

# For N = 6
# layout is 0 1
#           2 3
#           4 5


class Blocking1(object):
    blocks = [{0, 1, 2, 3}, {4, 5}]
    connected_qubit_pairs_list = [[(0, 1), (1, 3), (2, 3), (0, 2)], [(0, 1)]]

class Blocking2(object):
    blocks = [{0, 2, 4}, {1, 3, 5}]
    connected_qubit_pairs_list = [[(0, 1), (1, 2)], [(0, 1), (1, 2)]]    

class Blocking3(object):
    blocks = [{0, 1}, {2, 3, 4, 5}]
    connected_qubit_pairs_list = [[(0, 1)], [(0, 1), (1, 3), (2, 3), (0, 2)]]
 
blockings = [Blocking1, Blocking2, Blocking3]
blockings_index = 0

gates = qaoa_circuit.data
width = qaoa_circuit.width()

slice_circuits_list = []

while len(gates) > 0:
    blocking = blockings[blockings_index]
    remaining_gates = []; contaminated_indices = set()
    
    slice_circuits = [deepcopy(qaoa_circuit) for _ in range(len(blocking.blocks))]
    for circuit in slice_circuits:
        circuit.data = []

    for i, gate in enumerate(gates):
        if len(contaminated_indices) == width:
            remaining_gates.extend(gates[i:])
            break

        block_index = gate_block_index(gate, blocking.blocks)
        if block_index == -1:
            contaminated_indices.add(indices(gate)[0]); contaminated_indices.add(indices(gate)[1]);
            remaining_gates.append(gate)
        else:
            if len(indices(gate)) == 1:
                if indices(gate)[0] in contaminated_indices:
                    remaining_gates.append(gate)
                else:
                    slice_circuits[block_index].data.append(gate)
            else:
                if indices(gate)[0] in contaminated_indices or indices(gate)[1] in contaminated_indices:
                    contaminated_indices.add(indices(gate)[0]); contaminated_indices.add(indices(gate)[1]);
                    remaining_gates.append(gate)
                else:
                    slice_circuits[block_index].data.append(gate)            
                    
    slice_circuits_list.append((slice_circuits, blocking))
    gates = remaining_gates
    blockings_index = (blockings_index + 1) % len(blockings)

#print('Num slices for N=%d, p=%d: %d' % (graph_N, graph_p, len(slice_circuits_list)))
slice_circuits, blocking = slice_circuits_list[int(slice_id)]

# HACK ALERT: to preserve all qubits, I add a dummy identity on each qubit, then squash
times = []
for slice_circuit, block, connected_qubit_pairs in zip(
    slice_circuits, blocking.blocks, blocking.connected_qubit_pairs_list):
    for index in block:
        assert len(slice_circuit.qregs) == 1
        slice_circuit.iden(slice_circuit.qregs[0][index])

    slice_circuit = util.squash_circuit(slice_circuit)
    N = slice_circuit.width()
    H0 = hamiltonian.get_H0(N, d)
    Hops, Hnames = hamiltonian.get_Hops_and_Hnames(N, d, connected_qubit_pairs)
    states_concerned_list = hamiltonian.get_full_states_concerned_list(N, d)
    maxA = hamiltonian.get_maxA(N, d, connected_qubit_pairs)
    max_time = (3*N) * 5.5 * graph_p

    time = 0.0
    for subslice in uccsd.get_uccsd_slices(slice_circuit, granularity=2, dependence_grouping=True):
        U = util.get_unitary(subslice.circuit)
        time += binary_search_for_shortest_pulse_time(0.0, max_time, tolerance=0.3)

    times.append(time)


print('\n\n\n')
print(times)
print('TIME FOR SLICE IS:')
print(max(times))
