"""
uccsd_block_circuits.py - We block on all circuits larger than 4 qubits.
"""
from copy import deepcopy
import pickle
import random

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
    # beh2_circuits = get_beh2_circuits_to_compile()
    # nah_circuits = get_nah_circuits_to_compile()


    # Figure out which circuits are important to optimize over.
    # beh2_rz_indices = list()
    # for circuit_index, (circuit, _) in enumerate(beh2_circuits):
    #     if has_rz(circuit):
    #         beh2_rz_indices.append(circuit_index)

    # nah_rz_indices = list()
    # for circuit_index, (circuit, _) in enumerate(nah_circuits):
    #     if has_rz(circuit):
    #         nah_rz_indices.append(circuit_index)
            
    # Save the indices of circuits that are theta dependent.
    # with open("beh2_circuits.pickle", "wb") as f:
    #     pickle.dump((beh2_rz_indices, beh2_circuits), f)
    
    # with open("nah_circuits.pickle", "wb") as f:
    #     pickle.dump((nah_rz_indices, nah_circuits), f)

    h2o_circuits = get_h2o_circuits_to_compile()
    
    h2o_rz_indices = list()
    for circuit_index, (circuit, _) in enumerate(h2o_circuits):
        if has_rz(circuit):
            h2o_rz_indices.append(circuit_index)

    with open("h2o_circuits.pickle", "wb") as f:
        pickle.dump((h2o_rz_indices, h2o_circuits), f)


### HELPER METHODS ###

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


def _get_circuits_to_compile(slice_circuits_list):
    circuits_to_compile = []
    for slice_circuits, blocking in slice_circuits_list:
        for slice_circuit, block, connected_qubit_pairs in zip(
            slice_circuits, blocking.blocks, blocking.connected_qubit_pairs_list):
            for index in block:
                assert len(slice_circuit.qregs) == 1
                slice_circuit.iden(slice_circuit.qregs[0][index])
            slice_circuit = squash_circuit(slice_circuit)
            
            for subslice in get_uccsd_slices(slice_circuit, granularity=2, dependence_grouping=True):
                circuits_to_compile.append((subslice.circuit, connected_qubit_pairs))
    return circuits_to_compile

def get_beh2_circuits_to_compile():
    set_seeds()
    
    circuit = get_uccsd_circuit('BeH2')
    circuit = optimize_circuit(circuit)
    coupling_list = get_nearest_neighbor_coupling_list(2, 3)
    circuit = optimize_circuit(circuit, coupling_list)

    # layout is 0 2 4
    #           1 3 5
    
    class Blocking1(object):
        blocks = [{0, 1, 2, 3}, {4, 5}]
        connected_qubit_pairs_list = [[(0, 1), (1, 3), (2, 3), (0, 2)], [(0, 1)]]

    class Blocking2(object):
        blocks = [{0, 1}, {2, 3, 4, 5}]
        connected_qubit_pairs_list = [[(0, 1)], [(0, 1), (1, 3), (2, 3), (0, 2)]]

    blockings = [Blocking1, Blocking2]
    
    slice_circuits_list = _get_slice_circuits_list(circuit, blockings)
    return _get_circuits_to_compile(slice_circuits_list)

def get_nah_circuits_to_compile():
    set_seeds()
    
    circuit = get_uccsd_circuit('NaH')
    circuit = optimize_circuit(circuit)
    coupling_list = get_nearest_neighbor_coupling_list(2, 4)
    circuit = optimize_circuit(circuit, coupling_list)

    # layout is 0 2 4 6
    #           1 3 5 7
    
    class Blocking1(object):
        blocks = [{0, 1, 2, 3}, {4, 5, 6, 7}]
        connected_qubit_pairs_list = [[(0, 1), (1, 3), (2, 3), (0, 2)], [(0, 1), (1, 3), (2, 3), (0, 2)]]

    class Blocking2(object):
        blocks = [{0, 1}, {2, 3, 4, 5}, {6, 7}]
        connected_qubit_pairs_list = [[(0, 1)], [(0, 1), (1, 3), (2, 3), (0, 2)], [(0, 1)]]

    blockings = [Blocking1, Blocking2]
    
    slice_circuits_list = _get_slice_circuits_list(circuit, blockings)
    return _get_circuits_to_compile(slice_circuits_list)

def get_h2o_circuits_to_compile():
    set_seeds()
    
    circuit = get_uccsd_circuit('H2O')
    circuit = optimize_circuit(circuit)
    coupling_list = get_nearest_neighbor_coupling_list(2, 5)
    circuit = optimize_circuit(circuit, coupling_list)

    # layout is 0 2 4 6 8
    #           1 3 5 7 9
    
    class Blocking1(object):
        blocks = [{0, 1, 2, 3}, {4, 5, 6, 7}, {8, 9}]
        connected_qubit_pairs_list = [[(0, 1), (1, 3), (2, 3), (0, 2)], [(0, 1), (1, 3), (2, 3), (0, 2)], [(0, 1)]]

    class Blocking2(object):
        blocks = [{0, 1}, {2, 3, 4, 5}, {6, 7, 8, 9}]
        connected_qubit_pairs_list = [[(0, 1)], [(0, 1), (1, 3), (2, 3), (0, 2)], [(0, 1), (1, 3), (2, 3), (0, 2)]]

    blockings = [Blocking1, Blocking2]
    
    slice_circuits_list = _get_slice_circuits_list(circuit, blockings)
    return _get_circuits_to_compile(slice_circuits_list)

if __name__ == "__main__":
    main()
