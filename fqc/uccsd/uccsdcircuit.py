"""
uccsdcircuit.py -  Functions for generating circuit for UCCSD for various molecules
"""
import numpy as np
from qiskit import QuantumCircuit, QuantumRegister
from qiskit.chemistry.drivers import PySCFDriver
from qiskit.chemistry.aqua_extensions.components.variational_forms import UCCSD
from qiskit.chemistry.aqua_extensions.components.initial_states import HartreeFock
from qiskit.chemistry.core import Hamiltonian, QubitMappingType

from fqc.util import get_unitary

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


def get_uccsd_circuit(molecule, theta_vector, use_basis_gates=False):
    """Produce the full UCCSD circuit.
    Args:
    molecule :: string - must be a key of MOLECULE_TO_INFO
    theta_vector :: array - arguments for the vqe ansatz. If None, will generate random angles.
    use_basis_gates :: bool - Mike and Ike gates if False, Basis gates if True.
       
    Returns:
    circuit :: qiskit.QuantumCircuit - the UCCSD circuit parameterized
                                       by theta_vector, unoptimized
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

    return var_form.construct_circuit(theta_vector, use_basis_gates=use_basis_gates)

def _tests():
    """A function to run tests on the module"""
    theta = [np.random.random() for _ in range(8)]
    circuit = get_uccsd_circuit('LiH', theta)
    print(circuit)

if __name__ == "__main__":
    _tests()
