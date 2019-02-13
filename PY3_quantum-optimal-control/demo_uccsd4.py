import numpy as np
import scipy.linalg as la
from scipy.special import factorial
import os,sys,inspect
import h5py
import random as rd
import time
from IPython import display

data_path = '../out/output_pulses/'

print("Importing Grape", flush=True)

from quantum_optimal_control.helper_functions.grape_functions import *
from quantum_optimal_control.main_grape.grape import Grape


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
        matrices[j] = get_adagger(d).dot(get_a(d)) 
        H += omega * krons(matrices)
        # anharmonicity (alpha) terms:
        matrices = [np.eye(d)] * N
        matrices[j] = get_adagger(d).dot(get_a(d).dot(get_adagger(d).dot(get_a(d)) - np.eye(d))) 
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
        matrices[j] = get_adagger(d).dot(get_a(d)) 
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
    return get_adagger(d).dot(get_a(d))

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

N=4
d = 2 # this is the number of energy levels to consider (i.e. d-level qudits)
connected_qubit_pairs = [(0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3)] 
H0 = get_H0(N, d, connected_qubit_pairs)
Hops, Hnames = get_Hops_and_Hnames(N, d) 
total_time = 40.0
steps = 4000
states_concerned_list = get_full_states_concerned_list(N, d)


print("Qubit concerned states:", flush=True)
print(states_concerned_list, flush=True)

U = np.array([[ 0.11647636+6.36313165e-02j, -0.18140114-9.90998820e-02j,
   0.        +0.00000000e+00j,  0.13841994+7.56191499e-02j,
  -0.18140114-9.90998820e-02j,  0.28251547+1.54338888e-01j,
   0.        +0.00000000e+00j, -0.21557624-1.17769822e-01j,
   0.        +0.00000000e+00j,  0.        +0.00000000e+00j,
   0.        +0.00000000e+00j,  0.        +0.00000000e+00j,
  -0.33574003-1.83415591e-01j,  0.522884  +2.85652795e-01j,
   0.        +0.00000000e+00j, -0.39899183-2.17970204e-01j],
 [ 0.        +0.00000000e+00j,  0.        +0.00000000e+00j,
   0.25618976+1.39957089e-01j,  0.        +0.00000000e+00j,
   0.        +0.00000000e+00j,  0.        +0.00000000e+00j,
  -0.39899183-2.17970204e-01j,  0.        +0.00000000e+00j,
   0.        +0.00000000e+00j,  0.        +0.00000000e+00j,
   0.        +0.00000000e+00j,  0.        +0.00000000e+00j,
   0.        +0.00000000e+00j,  0.        +0.00000000e+00j,
  -0.73846024-4.03422617e-01j,  0.        +0.00000000e+00j],
 [ 0.21557624+1.17769822e-01j,  0.13841994+7.56191499e-02j,
   0.        +0.00000000e+00j,  0.        +0.00000000e+00j,
  -0.33574003-1.83415591e-01j, -0.21557624-1.17769822e-01j,
   0.        +0.00000000e+00j,  0.        +0.00000000e+00j,
   0.        +0.00000000e+00j,  0.        +0.00000000e+00j,
   0.        +0.00000000e+00j,  0.        +0.00000000e+00j,
  -0.62139282-3.39468405e-01j, -0.39899183-2.17970204e-01j,
   0.        +0.00000000e+00j,  0.        +0.00000000e+00j],
 [-0.4638616 -2.53408716e-01j,  0.3596192 +1.96460842e-01j,
   0.        +0.00000000e+00j, -0.38117394-2.08236248e-01j,
   0.20528031+1.12145133e-01j, -0.15366596-8.39480864e-02j,
   0.        +0.00000000e+00j,  0.19465128+1.06338465e-01j,
   0.        +0.00000000e+00j,  0.        +0.00000000e+00j,
   0.        +0.00000000e+00j,  0.        +0.00000000e+00j,
  -0.29778427-1.62680269e-01j,  0.24819518+1.35589629e-01j,
   0.        +0.00000000e+00j, -0.16262041-8.88399235e-02j],
 [-0.27568431-1.50607010e-01j, -0.6164152 -3.36749121e-01j,
   0.        +0.00000000e+00j,  0.16262041+8.88399235e-02j,
   0.30546062+1.66873878e-01j,  0.00287674+1.57157135e-03j,
   0.        +0.00000000e+00j,  0.22089374+1.20674788e-01j,
   0.        +0.00000000e+00j,  0.        +0.00000000e+00j,
   0.        +0.00000000e+00j,  0.        +0.00000000e+00j,
  -0.3354713 -1.83268787e-01j, -0.09892734-5.40442440e-02j,
   0.        +0.00000000e+00j,  0.15264379+8.33896700e-02j],
 [ 0.        +0.00000000e+00j,  0.        +0.00000000e+00j,
   0.73846024+4.03422617e-01j,  0.        +0.00000000e+00j,
   0.        +0.00000000e+00j,  0.        +0.00000000e+00j,
   0.47415997+2.59034741e-01j,  0.        +0.00000000e+00j,
   0.        +0.00000000e+00j,  0.        +0.00000000e+00j,
   0.        +0.00000000e+00j,  0.        +0.00000000e+00j,
   0.        +0.00000000e+00j,  0.        +0.00000000e+00j,
   0.        +0.00000000e+00j,  0.        +0.00000000e+00j],
 [ 0.39867248+2.17795743e-01j,  0.11756477+6.42259192e-02j,
   0.        +0.00000000e+00j, -0.18140114-9.90998820e-02j,
   0.11756477+6.42259192e-02j,  0.29106373+1.59008820e-01j,
   0.        +0.00000000e+00j,  0.28251547+1.54338888e-01j,
   0.        +0.00000000e+00j,  0.        +0.00000000e+00j,
   0.        +0.00000000e+00j,  0.        +0.00000000e+00j,
  -0.18140114-9.90998820e-02j,  0.28251547+1.54338888e-01j,
   0.        +0.00000000e+00j,  0.522884  +2.85652795e-01j],
 [ 0.21557624+1.17769822e-01j, -0.33574003-1.83415591e-01j,
   0.        +0.00000000e+00j, -0.62139282-3.39468405e-01j,
   0.13841994+7.56191499e-02j, -0.21557624-1.17769822e-01j,
   0.        +0.00000000e+00j, -0.39899183-2.17970204e-01j,
   0.        +0.00000000e+00j,  0.        +0.00000000e+00j,
   0.        +0.00000000e+00j,  0.        +0.00000000e+00j,
   0.        +0.00000000e+00j,  0.        +0.00000000e+00j,
   0.        +0.00000000e+00j,  0.        +0.00000000e+00j],
 [ 0.        +0.00000000e+00j,  0.        +0.00000000e+00j,
   0.        +0.00000000e+00j,  0.        +0.00000000e+00j,
   0.        +0.00000000e+00j,  0.        +0.00000000e+00j,
   0.        +0.00000000e+00j,  0.        +0.00000000e+00j,
   0.39899183+2.17970204e-01j, -0.62139282-3.39468405e-01j,
   0.        +1.08246745e-15j,  0.47415997+2.59034741e-01j,
   0.        +0.00000000e+00j,  0.        +0.00000000e+00j,
   0.        +0.00000000e+00j,  0.        +0.00000000e+00j],
 [ 0.        +0.00000000e+00j,  0.        +0.00000000e+00j,
   0.        +0.00000000e+00j,  0.        +0.00000000e+00j,
   0.        +0.00000000e+00j,  0.        +0.00000000e+00j,
   0.        +0.00000000e+00j,  0.        +0.00000000e+00j,
   0.        +0.00000000e+00j,  0.        +0.00000000e+00j,
   0.87758259+4.79425495e-01j,  0.        +0.00000000e+00j,
   0.        +0.00000000e+00j,  0.        +0.00000000e+00j,
   0.        +0.00000000e+00j,  0.        +0.00000000e+00j],
 [ 0.        +0.00000000e+00j,  0.        +0.00000000e+00j,
   0.        +0.00000000e+00j,  0.        +0.00000000e+00j,
   0.        +0.00000000e+00j,  0.        +0.00000000e+00j,
   0.        +0.00000000e+00j,  0.        +0.00000000e+00j,
   0.73846024+4.03422617e-01j,  0.47415997+2.59034741e-01j,
   0.        +0.00000000e+00j,  0.        +0.00000000e+00j,
   0.        +0.00000000e+00j,  0.        +0.00000000e+00j,
   0.        +0.00000000e+00j,  0.        +0.00000000e+00j],
 [ 0.        +0.00000000e+00j,  0.        +0.00000000e+00j,
   0.        +0.00000000e+00j,  0.        +0.00000000e+00j,
   0.        +0.00000000e+00j,  0.        +0.00000000e+00j,
   0.        +0.00000000e+00j,  0.        +0.00000000e+00j,
   0.25618976+1.39957089e-01j, -0.39899183-2.17970204e-01j,
   0.        +0.00000000e+00j, -0.73846024-4.03422617e-01j,
   0.        +0.00000000e+00j,  0.        +0.00000000e+00j,
   0.        +0.00000000e+00j,  0.        +0.00000000e+00j],
 [-0.38668145-2.11245015e-01j,  0.0371896 +2.03167699e-02j,
   0.        +0.00000000e+00j, -0.19465128-1.06338465e-01j,
  -0.20317892-1.10997137e-01j,  0.5750229 +3.14136402e-01j,
   0.        +0.00000000e+00j, -0.31824149-1.73856095e-01j,
   0.        +0.00000000e+00j,  0.        +0.00000000e+00j,
   0.        +0.00000000e+00j,  0.        +0.00000000e+00j,
  -0.06477959-3.53892470e-02j, -0.23485182-1.28300116e-01j,
   0.        +0.00000000e+00j,  0.22089374+1.20674788e-01j],
 [ 0.        +0.00000000e+00j,  0.        +0.00000000e+00j,
   0.39899183+2.17970204e-01j,  0.        +0.00000000e+00j,
   0.        +0.00000000e+00j,  0.        +0.00000000e+00j,
  -0.62139282-3.39468405e-01j,  0.        +0.00000000e+00j,
   0.        +0.00000000e+00j,  0.        +0.00000000e+00j,
   0.        +0.00000000e+00j,  0.        +0.00000000e+00j,
   0.        +0.00000000e+00j,  0.        +0.00000000e+00j,
   0.47415997+2.59034741e-01j,  0.        +0.00000000e+00j],
 [-0.22458799-1.22692963e-01j, -0.21899495-1.19637470e-01j,
   0.        +0.00000000e+00j, -0.09801147-5.35439031e-02j,
  -0.61798678-3.37607674e-01j, -0.28032848-1.53144126e-01j,
   0.        +0.00000000e+00j,  0.15264379+8.33896700e-02j,
   0.        +1.08073273e-15j,  0.        +0.00000000e+00j,
   0.        +0.00000000e+00j,  0.        +0.00000000e+00j,
   0.11756477+6.42259192e-02j,  0.29106373+1.59008820e-01j,
   0.        +0.00000000e+00j,  0.28251547+1.54338888e-01j],
 [ 0.11647636+6.36313165e-02j, -0.18140114-9.90998820e-02j,
   0.        +0.00000000e+00j, -0.33574003-1.83415591e-01j,
  -0.18140114-9.90998820e-02j,  0.28251547+1.54338888e-01j,
   0.        +0.00000000e+00j,  0.522884  +2.85652795e-01j,
   0.        +0.00000000e+00j,  0.        +0.00000000e+00j,
   0.        +0.00000000e+00j,  0.        +0.00000000e+00j,
   0.13841994+7.56191499e-02j, -0.21557624-1.17769822e-01j,
   0.        +0.00000000e+00j, -0.39899183-2.17970204e-01j]])

U = transmon_gate(U, d)
print("Target U initialized.", flush=True)

max_iterations = 200
decay = max_iterations/2
convergence = {'rate':0.01, 'update_step':10, 'max_iterations':max_iterations, 'conv_target':1e-4,'learning_rate_decay':decay} 
reg_coeffs = {'speed_up': 0.001}

uks, U_f = Grape(H0, Hops, Hnames, U, total_time, steps, states_concerned_list, convergence, reg_coeffs=reg_coeffs, use_gpu=False, sparse_H=False, method='L-BFGS-B', maxA=[2*np.pi*0.3] * len(Hops), show_plots = False, file_name='uccsd4', data_path = data_path)
