import numpy as np
import scipy.linalg as la
from scipy.special import factorial
import os,sys,inspect
import h5py
import random as rd
import time
from IPython import display

data_path = 'out/output_pulses/'

print("Importing Grape...", flush=True)

from quantum_optimal_control.helper_functions.grape_functions import *
from quantum_optimal_control.main_grape.grape import Grape

print("Importing Qiskit...", flush=True)

import qiskit
from qiskit import Aer

# lib from Qiskit Aqua
from qiskit.aqua import Operator, QuantumInstance

# lib from Qiskit Aqua Chemistry
from qiskit.chemistry import FermionicOperator
from qiskit.chemistry.drivers import PySCFDriver, UnitsType
from qiskit.chemistry.aqua_extensions.components.variational_forms import UCCSD
from qiskit.chemistry.aqua_extensions.components.initial_states import HartreeFock

print("Setting up parameters for GRAPE...", flush=True)

# Frequencies are in GHz:
OMEGA_DEFAULT = 2 * np.pi * 5 
ALPHA_DEFAULT = 2 * np.pi * -0.2 
G_DEFAULT = 2 * np.pi * 0.05

def get_H0(N, d, connected_qubit_pairs, omega=OMEGA_DEFAULT, alpha=ALPHA_DEFAULT, g=G_DEFAULT): 
    """Returns the drift Hamiltonian, H0."""
    return _get_single_qudit_terms(N, d, omega, alpha) + _get_coupling_terms(N, d, connected_qubit_pairs, g)

def _get_single_qudit_terms(N, d, omega=OMEGA_DEFAULT, alpha=ALPHA_DEFAULT): 
    H = np.zeros((d ** N, d ** N))
    for j in range(N):
        # qudit frequency (omega) terms:
        matrices = [np.eye(d)] * N
        matrices[j] = get_adagger(d) @ get_a(d)
        H += omega * krons(matrices)
        # anharmonicity (alpha) terms:
        matrices = [np.eye(d)] * N
        matrices[j] = get_adagger(d) @ get_a(d) @ (get_adagger(d) @ get_a(d) - np.eye(d)) 
        H += alpha / 2.0 * krons(matrices)
    return H

def _get_coupling_terms(N, d, connected_qubit_pairs, g=G_DEFAULT): 
    _validate_connectivity(N, connected_qubit_pairs)
    H = np.zeros((d ** N, d ** N))
    for (j, k) in connected_qubit_pairs:
        matrices = [np.eye(d)] * N
        matrices[j] = get_adagger(d) + get_a(d) 
        matrices[k] = get_adagger(d) + get_a(d) 
        H += g * krons(matrices)
    return H
def _validate_connectivity(N, connected_qubit_pairs): 
    """Each edge should be included only once.""" 
    for (j, k) in connected_qubit_pairs:
        assert 0 <= j < N 
        assert 0 <= k < N
        assert j < k
        assert connected_qubit_pairs.count((j, k)) == 1 
        assert connected_qubit_pairs.count((k, j)) == 0

def get_Hops_and_Hnames(N, d):
    """Returns the control Hamiltonian matrices and their labels.""" 
    hamiltonians, names = [], []
    for j in range(N):
        matrices = [np.eye(d)] * N
        matrices[j] = get_adagger(d) + get_a(d) 
        hamiltonians.append(krons(matrices)) 
        names.append("qubit %s charge drive" % j)
        matrices = [np.eye(d)] * N
        matrices[j] = get_adagger(d) @ get_a(d) 
        hamiltonians.append(krons(matrices)) 
        names.append("qubit %s flux drive" % j)
    return hamiltonians, names

def get_a(d):
    """Returns the matrix for the annihilation operator (a^{\dagger}), truncated to d-levels.""" 
    values = np.sqrt(np.arange(1, d))
    return np.diag(values, 1)

def get_adagger(d):
    """Returns the matrix for the creation operator (a^{\dagger}), truncated to d-levels.""" 
    return get_a(d).T # real matrix, so transpose is same as the dagger

def get_number_operator(d):
    """Returns the matrix for the number operator, a^\dagger * a, truncated to d-levels""" 
    return get_adagger(d) @ get_a(d)

def krons(matrices):
    """Returns the Kronecker product of the given matrices.""" 
    result = [1]
    for matrix in matrices:
        result = np.kron(result, matrix) 
    return result

def get_full_states_concerned_list(N, d): 
    states_concerned_list = []
    for i in range(2 ** N):
        bits = "{0:b}".format(i)
        states_concerned_list.append(int(bits, d)) 
    return states_concerned_list

N = 4
d = 2 # this is the number of energy levels to consider (i.e. d-level qudits)
connected_qubit_pairs = [(0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3)] 
H0 = get_H0(N, d, connected_qubit_pairs)
Hops, Hnames = get_Hops_and_Hnames(N, d) 
total_time = 40.0
steps = int(total_time * 20)
states_concerned_list = get_full_states_concerned_list(N, d)


print("Qubit concerned states:", flush=True)
print(states_concerned_list, flush=True)

print("Generating UCCSD circuit with parameters:", flush=True)

parameters = [rd.random() for _ in range(8)]

print(parameters, flush=True)

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
print("HF energy: {}".format(molecule.hf_energy - molecule.nuclear_repulsion_energy), flush=True)
print("# of electrons: {}".format(num_particles), flush=True)
print("# of spin orbitals: {}".format(num_spin_orbitals), flush=True)

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
print(qubitOp, flush=True)

# setup HartreeFock state
HF_state = HartreeFock(qubitOp.num_qubits, num_spin_orbitals, num_particles, map_type, 
                       qubit_reduction)

# setup UCCSD variational form
var_form = UCCSD(qubitOp.num_qubits, depth=1, num_orbitals=num_spin_orbitals, num_particles=num_particles, active_occupied=[0], active_unoccupied=[0, 1], initial_state=HF_state, qubit_mapping=map_type, two_qubit_reduction=qubit_reduction, num_time_slices=1)

circuit = var_form.construct_circuit(parameters, use_basis_gates=False)

print("Target circuit initialized.", flush=True)
#circuit.draw(output='text', line_length=350)

backend_sim = Aer.get_backend('unitary_simulator')

result = qiskit.execute(circuit, backend_sim).result()

U = result.get_unitary()
print(U.shape, flush=True)
print(U, flush=True)

U = transmon_gate(U, d)
print("Target U initialized.", flush=True)

max_iterations = 8  # proof of concept for now. Change this if want to see convergence.
decay = max_iterations/2
convergence = {'rate':0.01, 'update_step':1, 'max_iterations':max_iterations, 'conv_target':1e-4,'learning_rate_decay':decay} 
reg_coeffs = {'speed_up': 0.001}

uks, U_f = Grape(H0, Hops, Hnames, U, total_time, steps, states_concerned_list, convergence, reg_coeffs=reg_coeffs, use_gpu=False, sparse_H=False, method='L-BFGS-B', maxA=[2*np.pi*0.3] * len(Hops), show_plots = False, file_name='uccsd4', data_path = data_path)

