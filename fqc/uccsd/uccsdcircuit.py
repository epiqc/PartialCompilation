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

# lib from Qiskit Aqua Chemistry
from qiskit.chemistry.drivers import PySCFDriver
from qiskit.chemistry.aqua_extensions.components.variational_forms import UCCSD
from qiskit.chemistry.aqua_extensions.components.initial_states import HartreeFock
from qiskit.chemistry.core import Hamiltonian, QubitMappingType

from fqc.models import UCCSDSlice
from fqc.util import get_unitary, squash_circuit, append_gate, redundant

backend = BasicAer.get_backend('unitary_simulator')


### BUILD CIRCUTS AND UNITARIES ###


class MoleculeInfo(object):
    def __init__(self, atomic_string, orbital_reduction, active_occupied=[], active_unoccupied=[]):
        self.atomic_string = atomic_string
        self.orbital_reduction = orbital_reduction

        # TODO: what should I pass in for active (un)occupied for non LiH molecules?
        self.active_occupied = active_occupied
        self.active_unoccupied = active_unoccupied


MOLECULE_TO_INFO = {
    'LiH': MoleculeInfo('Li .0 .0 .0; H .0 .0 1.6', [-3, -2], [0], [0, 1]),

    # Minimum energy is at 1.3 Angstrom intermolecular distance. [-4, -3] reduction performs well.
    # github.com/Qiskit/qiskit-tutorials/blob/master/community/aqua/chemistry/beh2_reductions.ipynb
    'BeH2': MoleculeInfo('H .0 .0 -1.3; Be .0 .0 .0; H .0 .0 1.3', [-4, -3]),

    # Minimum energy is at 1.7/2 Angstrom intermolecular distance.
    # github.com/Qiskit/qiskit-tutorials/blob/master/community/aqua/chemistry/nah_uccsd.ipynb
    'NaH': MoleculeInfo('H .0 .0 -0.85; Na .0 .0 0.85', []),

    # Minimum energy is at 0.7/2 Angstrom intermolecular distance.
    # github.com/Qiskit/qiskit-tutorials/blob/master/community/aqua/chemistry/h2_uccsd.ipynb
    'H2': MoleculeInfo('H .0 .0 -0.35; H .0 .0 0.35', []),

    # github.com/Qiskit/qiskit-tutorials/blob/master/community/aqua/chemistry/h2o.ipynb
    'H2O': MoleculeInfo('O 0.0 0.0 0.0; H 0.757 0.586 0.0; H -0.757 0.586 0.0', []),
    }


def get_uccsd_circuit(molecule, theta_vector=None, use_basis_gates=False):
    """Produce the full UCCSD circuit.
    Args:
    molecule :: string - must be a key of MOLECULE_TO_INFO
    theta_vector :: array - arguments for the vqe ansatz. If None, will generate random angles.
    use_basis_gates :: bool - Mike and Ike gates if False, Basis gates if True.
       
    Returns:
    circuit :: qiskit.QuantumCircuit - the UCCSD circuit parameterized
                                       by theta_vector
    """
    molecule_info = MOLECULE_TO_INFO[molecule]
    driver = PySCFDriver(atom=molecule_info.atomic_string, basis='sto3g')
    qmolecule = driver.run()
    hamiltonian = Hamiltonian(qubit_mapping=QubitMappingType.PARITY, two_qubit_reduction=True,
                              freeze_core=True, orbital_reduction=molecule_info.orbital_reduction)

    energy_input = hamiltonian.run(qmolecule)
    qubit_op = energy_input.qubit_op
    num_spin_orbitals = hamiltonian.molecule_info['num_orbitals']
    num_particles = hamiltonian.molecule_info['num_particles']
    map_type = hamiltonian._qubit_mapping
    qubit_reduction = hamiltonian.molecule_info['two_qubit_reduction']

    HF_state = HartreeFock(qubit_op.num_qubits, num_spin_orbitals, num_particles, map_type,
                           qubit_reduction)
    var_form = UCCSD(qubit_op.num_qubits, depth=1,
                     num_orbitals=num_spin_orbitals, num_particles=num_particles,
                     active_occupied=molecule_info.active_occupied,
                     active_unoccupied=molecule_info.active_unoccupied,
                     initial_state=HF_state, qubit_mapping=map_type,
                     two_qubit_reduction=qubit_reduction, num_time_slices=1)

    if theta_vector is None:
        theta_vector = [np.random.random() * 2 * np.pi for _ in range(var_form._num_parameters)]

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
