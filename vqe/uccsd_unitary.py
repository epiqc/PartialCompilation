"""
uccsd_unitary.py - A module for generating unitary matrices
                   that represent UCCSD operator circuits.
"""

from math import pi

import numpy as np

#lib from Qiskit Terra
from qiskit import BasicAer, QuantumCircuit, ClassicalRegister, QuantumRegister, execute
from qiskit.extensions.standard import *

# lib from Qiskit Aqua
from qiskit.aqua import Operator, QuantumInstance
from qiskit.aqua.algorithms import VQE, ExactEigensolver
from qiskit.aqua.components.optimizers import COBYLA

# lib from Qiskit Aqua Chemistry
from qiskit.chemistry import FermionicOperator
from qiskit.chemistry.drivers import PySCFDriver, UnitsType
from qiskit.chemistry.aqua_extensions.components.variational_forms import UCCSD
from qiskit.chemistry.aqua_extensions.components.initial_states import HartreeFock

from circuitslice import UCCSDSlice

backend = BasicAer.get_backend('unitary_simulator')

### BUILD THE UCCSD VARIATIONAL FORM ###

# This section follows:
# https://github.com/Qiskit/qiskit-terra/blob/master/qiskit/circuit/instruction.py

# TODO: This section is not wrapped in a funciton because we
# want the variational form to be precompiled in the python binary.
# When we begin to work with different variational forms we should
# then modulate it.

# using driver to get fermionic Hamiltonian
# PySCF example
driver = PySCFDriver(atom='Li .0 .0 .0; H .0 .0 1.6', unit=UnitsType.ANGSTROM,
                     charge=0, spin=0, basis='sto3g')
molecule = driver.run()

# please be aware that the idx here with respective to original idx
freeze_list = [0]
remove_list = [-3, -2] # negative number denotes the reverse order
map_type = 'parity'

h1 = molecule.one_body_integrals
h2 = molecule.two_body_integrals
nuclear_repulsion_energy = molecule.nuclear_repulsion_energy

num_particles = molecule.num_alpha + molecule.num_beta
num_spin_orbitals = molecule.num_orbitals * 2
#print("HF energy: {}".format(molecule.hf_energy - molecule.nuclear_repulsion_energy))
#print("# of electrons: {}".format(num_particles))
#print("# of spin orbitals: {}".format(num_spin_orbitals))

# prepare full idx of freeze_list and remove_list
# convert all negative idx to positive
remove_list = [x % molecule.num_orbitals for x in remove_list]
freeze_list = [x % molecule.num_orbitals for x in freeze_list]
# update the idx in remove_list of the idx after frozen, since the idx of orbitals are changed after freezing
remove_list = [x - len(freeze_list) for x in remove_list]
remove_list += [x + molecule.num_orbitals - len(freeze_list)  for x in remove_list]
freeze_list += [x + molecule.num_orbitals for x in freeze_list]

# prepare fermionic hamiltonian with orbital freezing and eliminating, and then map to qubit hamiltonian
# and if PARITY mapping is selected, reduction qubits
energy_shift = 0.0
qubit_reduction = True if map_type == 'parity' else False

ferOp = FermionicOperator(h1=h1, h2=h2)
if len(freeze_list) > 0:
    ferOp, energy_shift = ferOp.fermion_mode_freezing(freeze_list)
    num_spin_orbitals -= len(freeze_list)
    num_particles -= len(freeze_list)
if len(remove_list) > 0:
    ferOp = ferOp.fermion_mode_elimination(remove_list)
    num_spin_orbitals -= len(remove_list)

qubitOp = ferOp.mapping(map_type=map_type, threshold=0.00000001)
qubitOp = qubitOp.two_qubit_reduced_operator(num_particles) if qubit_reduction else qubitOp
qubitOp.chop(10**-10)

#print(qubitOp.print_operators())
#print(qubitOp)

# Using exact eigensolver to get the smallest eigenvalue
exact_eigensolver = ExactEigensolver(qubitOp, k=1)
ret = exact_eigensolver.run()
#print('The computed energy is: {:.12f}'.format(ret['eigvals'][0].real))
#print('The total ground state energy is: {:.12f}'.format(ret['eigvals'][0].real + energy_shift + nuclear_repulsion_energy))

# setup COBYLA optimizer
max_eval = 200
cobyla = COBYLA(maxiter=max_eval)

# setup HartreeFock state
HF_state = HartreeFock(qubitOp.num_qubits, num_spin_orbitals, num_particles, map_type, 
                       qubit_reduction)

# setup UCCSD variational form
var_form = UCCSD(qubitOp.num_qubits, depth=1, 
                   num_orbitals=num_spin_orbitals, num_particles=num_particles, 
                   active_occupied=[0], active_unoccupied=[0, 1],
                   initial_state=HF_state, qubit_mapping=map_type, 
                   two_qubit_reduction=qubit_reduction, num_time_slices=1)

### BUILD CIRCUTS AND UNITARIES ###

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

def get_uccsd_circuit(theta_vector, use_basis_gates=False):
    """Produce the full UCCSD circuit.
    Args:
    theta_vector :: array - arguments for the vqe ansatz
    use_basis_gates :: bool - Mike and Ike gates if False, Basis gates if True.
       
    Returns:
    circuit :: qiskit.QuantumCircuit - the UCCSD circuit parameterized
                                       by theta_vector
    """
    return var_form.construct_circuit(theta_vector, use_basis_gates=use_basis_gates)

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

def _is_theta_dependent(gate):
    """Return ture if a gate is dependent on the theta vector,
    false otherwise.
    Note:
    RZ is the only theta dependent gate in the UCCSD circuit and the only RZ gates
    in the UCCSD circuit are theta dependent. Therefore, if a gate is RZ then
    it is theta dependent.

    Args:
    gate :: qiskit.QuantumGate - an arbitrary quantum gate
    """
    return isinstance(gate, RZGate)

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
    elif isinstance(gate, RXGate):
        constructor = circuit.rx
    elif isinstance(gate, RZGate):
        constructor = circuit.rz
    elif isinstance(gate, CnotGate):
        constructor = circuit.cx
    # TODO: extend to all gates?
    else:
        raise ValueError("append_gate() did not recognize gate")

    constructor(*gate.params, *qubits)

    return

def get_uccsd_slices(theta_vector):
    """Return a list of the decomposition of the full UCCSD circuit
    parameterized by theta_vector into theta-dependent and non-theta-dependent
    slices
    Args:
    theta_vector :: array - arguments for the vqe ansatz
    
    Returns:
    slice list :: [Slice] - a list of slice objects that contain
                            spans of partial circuits that are either
                            theta dependent or not theta dependent
    """
    full_circuit = get_uccsd_circuit(theta_vector)
    # The circuit width is the number of registers, i.e. qubits.
    full_circuit_width = full_circuit.width()
    gates = full_circuit.data
    gate_count = len(gates)
    slices = list()

    # Walk the list of gates and make a new quantum circuit for every continuous
    # span of theta dependent or not theta dependent gates.
    gates_encountered = 0
    while gates_encountered < gate_count:
        # Construct a new circuit for the span.
        register = QuantumRegister(full_circuit_width)
        circuit = QuantumCircuit(register)

        # Traverse the gate list and construct a circuit that
        # is either a continuous span of theta-dependent gates or
        # not theta_dependent gates
        redundant = False
        gate_is_theta_dependent = False
        last_gate_was_theta_dependent = False
        first_gate = True
        for gate in gates[gates_encountered:]:
            gate_is_theta_dependent = _is_theta_dependent(gate)

            if (gate_is_theta_dependent and
                    (last_gate_was_theta_dependent or first_gate)):
                last_gate_was_theta_dependent = True
                gates_encountered += 1

            elif (not gate_is_theta_dependent and
                    (not last_gate_was_theta_dependent or first_gate)):
                last_gate_was_theta_dependent = False
                gates_encountered += 1

            else:
                break
            
            append_gate(circuit, register, gate)
            
            if first_gate:
                first_gate = False
        
        # Construct a slice from the partial circuit.
        # Check that the theta gate is not redundant.
        if last_gate_was_theta_dependent:
            params = circuit.data[0].params
            for uccsdslice in slices:
                if (uccsdslice.theta_dependent
                      and uccsdslice.circuit.data[0].params == params):
                    redundant = True
                    break
                    
        # Get the unitary of the circuit.
        unitary=np.matrix(get_unitary(circuit))
        slices.append(UCCSDSlice(circuit, unitary=unitary,
                                 theta_dependent=last_gate_was_theta_dependent,
                                 redundant=redundant))
        #ENDFOR

    #ENDWHILE

    return slices

def _tests():
    """A function to run tests on the module"""
    theta = [np.random.random() * 2 * np.pi for _ in range(8)]
    slices = get_uccsd_slices(theta)

    for uccsdslice in slices:
        squashed_circuit = squash_circuit(uccsdslice.circuit)
        squashable = False
        if squashed_circuit.width() < uccsdslice.circuit.width():
            squashable = True
        print("theta_dependent: {}, redundant: {}, squashable: {}"
              "".format(uccsdslice.theta_dependent, uccsdslice.redundant,
                        squashable))
        print(uccsdslice.circuit)
        if squashable:
            print("squashed circuit:")
            print(squashed_circuit)

if __name__ == "__main__":
    _tests()
