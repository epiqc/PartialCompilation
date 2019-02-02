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

def append_gate(circuit, register, gate):
    """Append a quantum gate to a new circuit.
    Args:
    circuit :: qiskit.QuantumCircuit - the circuit the gate should be 
                                       appended to
    register :: qiskit.QuantumRegister - the register associated with
                                         the circuit to be appended to
    gate :: qiskit.QuantumGate - the gate to append to the circuit

    Returns: nothing
    """
    # Get the qubit indices that the gate should be applied to.
    qubits = list()
    for arg in gate.qargs:
        index = arg[1]
        qubits.append(register[index])

    num_qubits = len(qubits)
    
    # Single qubit gates.
    if num_qubits == 1:
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
        else:
            raise ValueError("append_gate() did not recognize single qubit gate")

        constructor(*gate.params, *qubits)
    # Two qubit gates.
    elif num_qubits == 2:
        if isinstance(gate, CnotGate):
            circuit.cx(*qubits)
        else:
            raise ValueError("append_gate() did not recognize two qubit gate")
    else:
        raise ValueError("append_gate() did not recognize mutltiple qubit gate")

    return

def slice_uccsd(theta_vector):
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
        slices.append(UCCSDSlice(circuit, 
                                 theta_dependent=last_gate_was_theta_dependent))
        #ENDFOR

    #ENDWHILE

    return slices

def _tests():
    """A function to run tests on the module"""
    theta = [np.random.random() * 2 * np.pi for _ in range(8)]
    slices = slice_uccsd(theta)

    # Clean slices
    for uccsdslice in slices:
        # Collapse theta circuits.
        if uccsdslice.theta_dependent:
            gate = uccsdslice.circuit.data[0]
            register = QuantumRegister(1)
            circuit = QuantumCircuit(register)
            circuit.rx(*gate.params, register[0])
            uccsdslice.circuit = circuit

        # Check for redundant gates.
        if uccsdslice.theta_dependent:
            params = uccsdslice.circuit.data[0].params
            for uccsdslice2 in slices:
                if uccsdslice2.circuit.data[0].params == params:
                    uccsdslice.redundant = True
                    break
            
    for uccsdslice in slices:
        print("theta dependent: {}, redundant: {}"
              "".format(uccsdslice.theta_dependent, uccsdslice.redundant))
        print(uccsdslice.circuit)

if __name__ == "__main__":
    _tests()
