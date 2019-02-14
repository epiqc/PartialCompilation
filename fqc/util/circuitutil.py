"""
circuitutil.py - A module for extending Qiskit circuit functionality.
"""

import numpy as np
import pickle
from qiskit import BasicAer, QuantumCircuit, QuantumRegister, execute
from qiskit.extensions.standard import *
from qiskit.mapper import CouplingMap, swap_mapper
from qiskit.converters import circuit_to_dag, dag_to_circuit

### CONSTANTS ###

# See Gate_Times.ipnyb for determination of these pulse times
GATE_TO_PULSE_TIME = {'h': 2.1, 'cx': 7.1, 'rz': 0.3, 'rx': 4.2, 'x': 4.2}

backend = BasicAer.get_backend('unitary_simulator')

### FUNCTIONS ###

def get_unitary(circuit):
    """Given a qiskit circuit, produce a unitary matrix to represent it.
    Args:
    circuit :: qiskit.QuantumCircuit - an arbitrary quantum circuit

    Returns:
    matrix :: np.matrix - the unitary representing the circuit
    """
    job = execute(circuit, backend)
    unitary = job.result().get_unitary(circuit, decimals=10)
    return np.matrix(unitary)

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

def get_max_pulse_time(circuit):
    """Returns the maximum possible pulse duration (in ns) for this circuit.

    This value is based on the pulse times in GATE_TO_PULSE_TIME. In principle,
    applying optimal control to the full circuit unitary should allow shorter
    pulses than this maximum duration.

    """
    total_time = 0.0

    dag = circuit_to_dag(circuit)
    for layer in dag.layers():
        slice_circuit = dag_to_circuit(layer['graph'])
        gates = slice_circuit.data
        layer_time = max([GATE_TO_PULSE_TIME[gate.name] for gate in gates])
        total_time += layer_time

    return total_time


# Note: lists are not hashable in python so I can not think of a better than O(n)
# way to compare lists for uniqueness.
# Suggestion: Tuples are hashable.
def redundant(gates, new_gate):
    """Determines if new_gate has the same parameters if as those in gates.
    Args:
    gates :: [qiskit.QuantumGate] - a list of gates
    new_gate :: qiskit.QuantumGate - the quantum gate to compare for uniqueness
    
    Returns:
    redundant :: bool - whether or not a gate with the same parameters as new_gate
                        is already contained in gates
    """
    redundant = False

    for gate in gates:
        if new_gate.params == gate.params:
            redundant = True
            break

    return redundant

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
    elif isinstance(gate, CnotGate):
        constructor = circuit.cx
    # TODO: extend to all gates?
    else:
        raise ValueError("append_gate() did not recognize gate %s" % gate)

    constructor(*gate.params, *qubits)

def _tests():
    """A function to run tests on the module"""
    pass

if __name__ == "__main__":
    _tests()