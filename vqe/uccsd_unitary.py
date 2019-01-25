"""
uccsd_unitary.py - A module for generating unitary matrices
                   that represent UCCSD operator circuits.
"""

from math import pi

import numpy as np

#lib from Qiskit Terra
from qiskit import BasicAer, QuantumCircuit, ClassicalRegister, QuantumRegister, execute
from qiskit.extensions.standard.u1 import U1Gate
from qiskit.extensions.standard.u2 import U2Gate
from qiskit.extensions.standard.u3 import U3Gate

# lib from Qiskit Aqua
from qiskit_aqua import Operator, QuantumInstance
from qiskit_aqua.algorithms import VQE, ExactEigensolver
from qiskit_aqua.components.optimizers import COBYLA

# lib from Qiskit Aqua Chemistry
from qiskit_chemistry import FermionicOperator
from qiskit_chemistry.drivers import PySCFDriver, UnitsType
from qiskit_chemistry.aqua_extensions.components.variational_forms import UCCSD
from qiskit_chemistry.aqua_extensions.components.initial_states import HartreeFock

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
    job = execute(circuit, backend)
    unitary = job.result().get_unitary(circuit, decimals=10)
    return unitary

def get_uccsd_circuit(theta_vector):
    """Produce the full UCCSD circuit.
       
       theta_vector :: array - arguments for the vqe ansatz
    """
    return var_form.construct_circuit(theta_vector)

# The four gates that comprise the UCCSD circuit are CX, U1, U2, and U3.
# The latter three gates are dependent on the theta vector.
def _is_theta_dependent(gate):
    """Return ture if a gate is dependent on the theta vector,
    false otherwise.
    
        gate :: qiskit.QuantumGate - an arbitrary quantum gate
    """
    return (isinstance(gate, U1Gate) 
            or isinstance(gate, U2Gate) 
            or isinstance(gate, U3Gate))

# https://github.com/Qiskit/qiskit-terra/blob/master/qiskit/circuit/instruction.py
# The syntax will change from `gate.param` to `gate.params` in a future release of
# qiskit terra
def _append_theta_dependent_gate(circuit, register, gate):
    """Take a U1, U2, or U3 gate object and append it to a circuit.
    
        circuit :: qiskit.QuantumCircuit - the circuit to apply the gate to
        register :: qiskit.QuantumRegister - the register that defines the circuit
        gate :: qiskit.QuantumGate - one of the following gates: U1, U2, or U3 
    """
    # Get the qubit index that the gate should be applied to.
    qubit = gate.qargs[0][1]

    if isinstance(gate, U1Gate):
        constructor = circuit.u1
    elif isinstance(gate, U2Gate):
        constructor = circuit.u2
    else:
        constructor = circuit.u3

    constructor(*gate.param, register[qubit])

    return

def get_uccsd_theta_circuits(theta_vector):
    """Return a list of circuits that comprise the 
       continuous spans of gates in the UCCSD circuit
       that depend on the values of the theta_vector.
       
       theta_vector :: array - arguments for the vqe ansatz
    """

    full_circuit = get_uccsd_circuit(theta_vector)
    gates = full_circuit.data
    gate_count = len(gates)

    # Walk the list of gates and make a new quantum circuit
    # for every continuous span of theta dependent gates.
    # Store the unitary of the partial circuit in theta_unitaries.
    i = 0
    theta_circuits = list()
    while(i < gate_count):
        # Construct a 4 qubit circuit.
        register = QuantumRegister(4)
        circuit = QuantumCircuit(register)

        # Add the continuous span of gates
        # to the partial circuit.
        while(i < gate_count and _is_theta_dependent(gates[i])):
            _append_theta_dependent_gate(circuit, register, gates[i])
            i += 1

        # If there were theta depedent gates found 
        # in this stretch, grab their circuit.
        # Then, turn the circuit into a unitary.
        if (len(circuit.data) > 0):
            theta_circuits.append(circuit)
        else:
            i += 1
    
    return theta_circuits


def _tests():
    """A function to run tests on the module"""
    theta = [1.0,1.1,1.2,1.3,1.4,1.5,1.6,1.7]
    circuits = get_uccsd_theta_circuits(theta)
    for circuit in circuits:
        print(circuit)

    return


if __name__ == "__main__":
    _tests()
