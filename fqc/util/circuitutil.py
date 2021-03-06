"""
circuitutil.py - A module for extending Qiskit circuit functionality.
"""

import numpy as np
import pickle
from qiskit import Aer, BasicAer, QuantumCircuit, QuantumRegister, execute
from qiskit.extensions.standard import *
from qiskit.mapper import CouplingMap, swap_mapper
from qiskit.converters import circuit_to_dag, dag_to_circuit
from qiskit.transpiler import PassManager, transpile
from qiskit.transpiler.passes import (BasicSwap, CXCancellation, HCancellation)

### CONSTANTS ###

# NOTICE: GATE_TO_PULSE_TIME is kept here for dependency reasons, but all
# future references to this dict and any other experimental constants
# should be kept in fqc/data/data.py.
# See Gate_Times.ipynb or Realistic_Pulses.ipynb for determination of these pulse times
GATE_TO_PULSE_TIME = {'h': 1.4, 'cx': 3.8, 'rz': 0.4, 'rx': 2.5, 'x': 2.5, 'swap': 7.4, 'id': 0.0}
GATE_TO_PULSE_TIME_REALISTIC = {'h': 20, 'cx': 45, 'rz': 1, 'rx': 31, 'x': 31, 'swap': 59, 'id': 0.0}


unitary_backend = BasicAer.get_backend('unitary_simulator')
state_backend = Aer.get_backend('statevector_simulator')

### FUNCTIONS ###

def get_unitary(circuit):
    """Given a qiskit circuit, produce a unitary matrix to represent it.
    Args:
    circuit :: qiskit.QuantumCircuit - an arbitrary quantum circuit

    Returns:
    matrix :: np.matrix - the unitary representing the circuit
    """
    job = execute(circuit, unitary_backend)
    unitary = job.result().get_unitary(circuit, decimals=10)
    return np.matrix(unitary)

# TODO: deprecated, use optimize_circuit
def impose_swap_coupling(circuit, coupling_list):
    """Impose a qubit topology on the given circuit using swap gates.
    Args:
    circuit :: qiskit.QuantumCircuit - the circuit to impose a topology upon.
    coupling_list :: [(int, int)] - the list of connected qubit pairs

    Returns:
    coupled_circuit :: qiskit.QuantumCircuit - the circuit equivalent to the
    original that abides by the qubit mapping via swap gates
    """
    dag = circuit_to_dag(circuit)
    coupling_map = CouplingMap(coupling_list)
    coupled_dag = swap_mapper(dag, coupling_map)[0]
    coupled_circuit = dag_to_circuit(coupled_dag)
    return coupled_circuit

def optimize_circuit(circuit, coupling_list=None):
    """Use the qiskit transpiler module to perform a suite of optimizations on
    the given circuit, including imposing swap coupling.
    Args:
    circuit :: qiskit.QuantumCircuit - the circuit to optimize
    coupling_list :: [(int, int)] - the list of connected qubit pairs
                     if None is passed in, will not perform mapping

    Returns:
    optimized_circuit :: qiskit.QuantumCircuit - the optimized circuit
    """
    # TODO: optimize until gates count stays stagnant.
    # TODO: implement rotation merge for clifford gates as pass.
    merge_rotation_gates(circuit)

    coupling_map = None if coupling_list is None else CouplingMap(coupling_list)

    pass_manager = PassManager()
    pass_manager.append(HCancellation())
    pass_manager.append(CXCancellation())
    # Some CNOT identities are interleaved between others,
    # for this reason a second pass is required. More passes
    # may be required for other circuits.
    pass_manager.append(CXCancellation())
    if coupling_map is not None:
        pass_manager.append(BasicSwap(coupling_map))

    optimized_circuit = transpile(circuit, backend=state_backend,
                                  coupling_map=coupling_list,
                                  pass_manager=pass_manager)

    return optimized_circuit

def get_max_pulse_time(circuit, more_realistic=False):
    """Returns the maximum possible pulse duration (in ns) for this circuit.

    This value is based on the pulse times in GATE_TO_PULSE_TIME (or in
    GATE_TO_PULSE_TIME_REALISTIC if more_realistic is True).

    In principle, applying optimal control to the full circuit unitary should
    allow shorter pulses than this maximum duration.

    """
    total_time = 0.0
    gate_to_time = GATE_TO_PULSE_TIME_REALISTIC if more_realistic else GATE_TO_PULSE_TIME

    dag = circuit_to_dag(circuit)
    for layer in dag.layers():
        slice_circuit = dag_to_circuit(layer['graph'])
        gates = slice_circuit.data
        layer_time = max([gate_to_time[gate.name] for gate in gates])
        total_time += layer_time

    return total_time

def squash_circuit(circuit):
    """For a given circuit, return a new circuit that has the minimum number
    of registers possible. If a branch of the circuit does not have any gates
    that act on it, i.e. a qubit is not acted on, that branch will not appear
    in the new circuit.
    Args:
    circuit :: qiskit.QuantumCircuit - the circuit to squash
    
    Returns:
    new_circuit :: qiskit.QuantumCircuit - the squashed form of the circuit
    """
    # Hash the index of each qubit that is acted on in the circuit.
    gates = circuit.data
    qubit_indices = set()
    for gate in gates:
        for arg in gate.qargs:
            qubit_index = arg[1]
            qubit_indices.add(qubit_index)
    
    # Transform qubit_indices into a list to define an accessible ordering on 
    # the new indices.
    qubit_indices = list(qubit_indices)
    num_qubits = len(qubit_indices)

    # If all qubits are acted on, there is nothing to squash.
    if circuit.width() == num_qubits:
        return circuit

    # Otherwise, construct the new circuit.
    register = QuantumRegister(num_qubits)
    new_circuit = QuantumCircuit(register)
    
    # Append the gates from the circuit to the new circuit.
    # The index of the qubit index of the gate in qubit_indices is the gate's
    # qubit index in the new circuit. DOC: is this confusing?
    for gate in gates:
        gate_qubit_indices = [qubit_indices.index(arg[1]) for arg in gate.qargs]
        append_gate(new_circuit, register, gate, gate_qubit_indices)        

    return new_circuit


PAULI_GATE_TO_ROTATION_GATE = {XGate: RXGate,
                               YGate: RYGate,
                               ZGate: RZGate}

def _convert_pauli_gates_into_rotation_gates(circuit):
    """Mutates the circuit by transforming X into RX(PI) and similar for Y & Z."""

    new_gates = []

    for gate in circuit.data:
        rotation_gate = PAULI_GATE_TO_ROTATION_GATE.get(type(gate))
        if rotation_gate is not None:
            new_gates.append(rotation_gate(np.pi, gate.qargs[0]))
        else:
            new_gates.append(gate)

    circuit.data = new_gates


def _is_rotation_gate(gate):
    return type(gate) in PAULI_GATE_TO_ROTATION_GATE.values()


def merge_rotation_gates(circuit):
    """Mutates the circuit by merging consecutive RX (or RY or RZ) gates on the same qubit.

    NB it would be more efficient to operate directly via DAG."""

    _convert_pauli_gates_into_rotation_gates(circuit)

    new_gates = []

    for i, gate in enumerate(circuit.data):
        if type(gate) in PAULI_GATE_TO_ROTATION_GATE.values():
            if gate.params[0] == 0:  # skip 0-degree rotations
                continue

            target_qubit = gate.qargs[0]
            j = i + 1
            while j < len(circuit.data):
                if target_qubit in circuit.data[j].qargs:
                    if type(circuit.data[j]) == type(gate):
                        # add the rotation angle to the current gate and 0 the jth gate's angle
                        gate.params[0] = (gate.params[0] + circuit.data[j].params[0]) % (2 * np.pi)
                        circuit.data[j].params[0] = 0
                    else:
                        break
                j += 1

            if gate.params[0] != 0:
                new_gates.append(gate)

        else:
            new_gates.append(gate)

    circuit.data = new_gates


def append_gate(circuit, register, gate, indices=None):
    """Append a quantum gate to a new circuit.
    Args:
    circuit :: qiskit.QuantumCircuit - the circuit the gate should be 
                                       appended to
    register :: qiskit.QuantumRegister - the register associated with
                                         the circuit to be appended to
    gate :: qiskit.QuantumGate - the gate to append to the circuit
    indices :: [int] - the qubit indices the gate should act on. If no
                       indices are specified, the gate will act on
                       the qubit indices it did in the circuit that constructed
                       it.

    Returns: nothing
    """
    # num_qubits corresponds to how many qubits the gate acts on.
    num_qubits = len(gate.qargs)

    # Get the qubits the gate should act on from the register.
    qubits = list()
    # If indices were specified, snag 'em.
    if indices is not None:
        qubits = [register[index] for index in indices]
    # If indices were not specified, the gate should act on
    # the same qubit indices it previously did.
    else:
        for arg in gate.qargs:
            qubit_index = arg[1]
            qubits.append(register[qubit_index])
        
    # Switch on the type of gate and append it.
    if isinstance(gate, U1Gate):
        constructor = circuit.u1
    elif isinstance(gate, U2Gate):
        constructor = circuit.u2
    elif isinstance(gate, U3Gate):
        constructor = circuit.u3
    elif isinstance(gate, HGate):
        constructor = circuit.h
    elif isinstance(gate, XGate):
        constructor = circuit.x
    elif isinstance(gate, RXGate):
        constructor = circuit.rx
    elif isinstance(gate, RZGate):
        constructor = circuit.rz
    elif isinstance(gate, IdGate):
        constructor = circuit.iden
    elif isinstance(gate, CnotGate):
        constructor = circuit.cx
    elif isinstance(gate, SwapGate):
        constructor = circuit.swap
    # TODO: extend to all gates?
    else:
        raise ValueError("append_gate() did not recognize gate %s" % gate)

    constructor(*gate.params, *qubits)


def get_nearest_neighbor_coupling_list(width, height, directed=True):
    """Returns a coupling list for nearest neighbor (rectilinear grid) architecture.

    Qubits are numbered in row-major order with 0 at the top left and
    (width*height - 1) at the bottom right.

    If directed is True, the coupling list includes both  [a, b] and [b, a] for each edge.
    """
    coupling_list = []

    def _qubit_number(row, col):
        return row * width + col

    # horizontal edges
    for row in range(height):
        for col in range(width - 1):
            coupling_list.append((_qubit_number(row, col), _qubit_number(row, col + 1)))
            if directed:
                coupling_list.append((_qubit_number(row, col + 1), _qubit_number(row, col)))

    # vertical edges
    for col in range(width):
        for row in range(height - 1):
            coupling_list.append((_qubit_number(row, col), _qubit_number(row + 1, col)))
            if directed:
                coupling_list.append((_qubit_number(row + 1, col), _qubit_number(row, col)))

    return coupling_list


def get_two_qubit_gate_freq(circuit):
    two_qubit_gates = [gate for gate in circuit.data if len(gate.qargs) == 2]
    pairs = [frozenset([gate.qargs[0][1], gate.qargs[1][1]]) for gate in two_qubit_gates]
    freq = {pair: 0 for pair in pairs}
    for pair in pairs:
        freq[pair] += 1
    return list(sorted(freq.items(), key=lambda freq: freq[1]))


def _tests():
    """A function to run tests on the module"""
    pass

if __name__ == "__main__":
    _tests()

