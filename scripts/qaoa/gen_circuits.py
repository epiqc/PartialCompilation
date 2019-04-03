"""
gen_circuits.py - Get all of the circuits we care about compiling.
"""
from copy import deepcopy
import pickle
import random

from fqc.qaoa import (get_qaoa_circuit)
from fqc.uccsd import (get_uccsd_slices, get_uccsd_circuit)
from fqc.util import (get_unitary, get_max_pulse_time, CustomJSONEncoder,
                      squash_circuit, get_nearest_neighbor_coupling_list,
                      optimize_circuit)
import numpy as np
from qiskit import QuantumCircuit, QuantumRegister
from qiskit.extensions.standard import RZGate
import tensorflow as tf

### MAIN METHODS ###

def main():
    # Save a list indexed by p value for each qaoa benchmark.
    # where each item has the form ([rz_index], [(circuit, cqp)])
    n6e_list = list()
    n8e_list = list()
    n6r3_list = list()
    n8r3_list = list()
    
    # p ranges from 1 - 8.
    for p in range(1, 9):
        n6e_circuits = get_N6_erdosrenyi_qaoa_circuits_to_compile(p)
        n8e_circuits = get_N8_erdosrenyi_qaoa_circuits_to_compile(p)
        n6r3_circuits = get_N6_3reg_qaoa_circuits_to_compile(p)
        n8r3_circuits = get_N8_3reg_qaoa_circuits_to_compile(p)
        
        n6e_rz_indices = get_rz_indices(n6e_circuits)
        n8e_rz_indices = get_rz_indices(n8e_circuits)
        n6r3_rz_indices = get_rz_indices(n6r3_circuits)
        n8r3_rz_indices = get_rz_indices(n8r3_circuits)

        n6e_list.append((n6e_rz_indices, n6e_circuits))
        n8e_list.append((n8e_rz_indices, n8e_circuits))
        n6r3_list.append((n6r3_rz_indices, n6r3_circuits))
        n8r3_list.append((n8r3_rz_indices, n8r3_circuits))
    #ENDFOR
    
    with open("n6e_circuits.pickle", "wb") as f:
        pickle.dump(n6e_list, f)
    with open("n8e_circuits.pickle", "wb") as f:
        pickle.dump(n8e_list, f)
    with open("n6r3_circuits.pickle", "wb") as f:
        pickle.dump(n6r3_list, f)
    with open("n8r3_circuits.pickle", "wb") as f:
        pickle.dump(n8r3_list, f)


### HELPER METHODS ###

def get_rz_indices(circuit_list):
    rz_indices = list()
    for i, (circuit, _) in enumerate(circuit_list):
        if has_rz(circuit):
            rz_indices.append(i)
    return rz_indices

def count_rzs(circuit):
    count = 0
    for gate in circuit.data:
        if isinstance(gate, RZGate):
            count += 1
    return count

def has_rz(circuit):
    for gate in circuit.data:
        if isinstance(gate, RZGate):
            return True
    return False


### PRANAV'S CODE ###
# See fqc/experiments/CompileTimes.ipynb

def set_seeds():
    random.seed(0)
    np.random.seed(1)
    tf.set_random_seed(2)

def _get_slice_circuits_list(circuit, blockings):
    set_seeds()

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

    blockings_index = 0

    gates = circuit.data
    width = circuit.width()

    slice_circuits_list = []

    while len(gates) > 0:
        blocking = blockings[blockings_index]
        remaining_gates = []; contaminated_indices = set()

        slice_circuits = [deepcopy(circuit) for _ in range(len(blocking.blocks))]
        for slice_circuit in slice_circuits:
            slice_circuit.data = []

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


    return slice_circuits_list


def _get_circuits_to_compile(slice_circuits_list, sampling_rate=None):
    circuits_to_compile = []
    for slice_circuits, blocking in slice_circuits_list:
        if sampling_rate is not None:
            if random.random() > sampling_rate:
                continue

        for slice_circuit, block, connected_qubit_pairs in zip(
            slice_circuits, blocking.blocks, blocking.connected_qubit_pairs_list):
            for index in block:
                assert len(slice_circuit.qregs) == 1
                slice_circuit.iden(slice_circuit.qregs[0][index])
            slice_circuit = squash_circuit(slice_circuit)
            
            for subslice in get_uccsd_slices(slice_circuit, granularity=2, dependence_grouping=True):
                circuits_to_compile.append((subslice.circuit, connected_qubit_pairs))

    return circuits_to_compile


def _get_qaoa_circuits_to_compile(graph_type, graph_N, graph_p, blockings, sampling_rate=None):
    assert graph_p in [1, 2, 3, 4, 5, 6, 7, 8], 'we only did p = 1...8'
    set_seeds()
    circuit = get_qaoa_circuit(graph_N, graph_p, graph_type)
    circuit = optimize_circuit(circuit)
    coupling_list = get_nearest_neighbor_coupling_list(2, int(graph_N/2))
    circuit = optimize_circuit(circuit, coupling_list)
    slice_circuits_list = _get_slice_circuits_list(circuit, blockings)
    return _get_circuits_to_compile(slice_circuits_list, sampling_rate=sampling_rate)

def get_N6_erdosrenyi_qaoa_circuits_to_compile(graph_p, sampling_rate=None):
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
    return _get_qaoa_circuits_to_compile('ErdosRenyi', 6, graph_p, blockings, sampling_rate)


def get_N8_erdosrenyi_qaoa_circuits_to_compile(graph_p, sampling_rate=None):
    class Blocking1(object):
        blocks = [{0, 1, 2, 3}, {4, 5, 6, 7}]
        connected_qubit_pairs_list = [[(0, 1), (1, 3), (2, 3), (0, 2)],[(0, 1), (1, 3), (2, 3), (0, 2)]]

    class Blocking2(object):
        blocks = [{0, 2, 4, 6}, {1, 3, 5, 7}]
        connected_qubit_pairs_list = [[(0, 1), (1, 2), (2, 3)], [(0, 1), (1, 2), (2, 3)]]    

    class Blocking3(object):
        blocks = [{0, 1}, {2, 3, 4, 5}, {6, 7}]
        connected_qubit_pairs_list = [[(0, 1)], [(0, 1), (1, 3), (2, 3), (0, 2)], [(0, 1)]]
 
    blockings = [Blocking1, Blocking2, Blocking3]
    return _get_qaoa_circuits_to_compile('ErdosRenyi', 8, graph_p, blockings, sampling_rate)


def get_N6_3reg_qaoa_circuits_to_compile(graph_p, sampling_rate=None):
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
    return _get_qaoa_circuits_to_compile('3Reg', 6, graph_p, blockings, sampling_rate)


def get_N8_3reg_qaoa_circuits_to_compile(graph_p, sampling_rate=None):
    class Blocking1(object):
        blocks = [{0, 1, 2, 3}, {4, 5, 6, 7}]
        connected_qubit_pairs_list = [[(0, 1), (1, 3), (2, 3), (0, 2)],[(0, 1), (1, 3), (2, 3), (0, 2)]]

    class Blocking2(object):
        blocks = [{0, 2, 4, 6}, {1, 3, 5, 7}]
        connected_qubit_pairs_list = [[(0, 1), (1, 2), (2, 3)], [(0, 1), (1, 2), (2, 3)]]    

    class Blocking3(object):
        blocks = [{0, 1}, {2, 3, 4, 5}, {6, 7}]
        connected_qubit_pairs_list = [[(0, 1)], [(0, 1), (1, 3), (2, 3), (0, 2)], [(0, 1)]]
 
    blockings = [Blocking1, Blocking2, Blocking3]
    return _get_qaoa_circuits_to_compile('3Reg', 8, graph_p, blockings, sampling_rate)


if __name__ == "__main__":
    main()
