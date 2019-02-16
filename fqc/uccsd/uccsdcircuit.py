"""
uccsd_unitary.py - A module for generating unitary matrices
                   that represent UCCSD operator circuits.
"""

import numpy as np
import pickle

#lib from Qiskit Terra
from qiskit import BasicAer, QuantumCircuit, ClassicalRegister, QuantumRegister, execute
from qiskit.extensions.standard import *
from qiskit.mapper import CouplingMap, swap_mapper
from qiskit.converters import circuit_to_dag, dag_to_circuit

# lib from Qiskit Aqua
from qiskit.aqua import Operator, QuantumInstance
from qiskit.aqua.algorithms import VQE, ExactEigensolver

# lib from Qiskit Aqua Chemistry
from qiskit.chemistry import FermionicOperator
from qiskit.chemistry.drivers import PySCFDriver, UnitsType
from qiskit.chemistry.aqua_extensions.components.variational_forms import UCCSD
from qiskit.chemistry.aqua_extensions.components.initial_states import HartreeFock

from fqc.models import UCCSDSlice
from fqc.util import get_unitary, squash_circuit, append_gate, redundant

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
        slices.append(UCCSDSlice(circuit,
                                 theta_dependent=last_gate_was_theta_dependent,
                                 redundant=redundant))
        #ENDFOR

    #ENDWHILE

    return slices

def save_uccsd_slices(theta_vector, file_name):
    """Save the UCCSD slices (and their squashed versions, if they exist) to a binary file in *.npz format
    Args:
    theta_vector :: array - arguments for the vqe ansatz
    file_nmae :: string - the filepath to save to

    """
    slices = get_uccsd_slices(theta_vector)


    # In the data file map something of the form UN_M to the Nth circuit that has a circuit depth of M
    file_data = [(get_unitary(squash_circuit(uccsdslice.circuit)), uccsdslice.circuit.depth()) for uccsdslice in slices]

    # Save the matrices jto the specified filepath
    with open(file_name, 'wb') as f:
        pickle.dump(file_data, f, protocol=2, fix_imports=True)

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
