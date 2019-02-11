import numpy as np
import scipy.linalg as la
from scipy.special import factorial
import os,sys,inspect
import h5py
import random as rd
import time
from IPython import display

data_path = '../pulses/output_pulses/'

from quantum_optimal_control.helper_functions.grape_functions import *
from quantum_optimal_control.main_grape.grape import Grape

print("Setting up parameters for GRAPE...", flush=True)
 
#Defining time scales
total_time = 10.0
steps = 1000
qubit_state_num = 5

#Defining H0
alpha = 0.225
freq_ge = 3.9#GHz
ens_1 = np.array([ 2*np.pi*ii*(freq_ge - 0.5*(ii-1)*alpha) for ii in np.arange(qubit_state_num)])
H0_1 = np.diag(ens_1)

alpha = 0.225
freq_ge = 3.5#GHz
ens_2 = np.array([ 2*np.pi*ii*(freq_ge - 0.5*(ii-1)*alpha) for ii in np.arange(qubit_state_num)])
H0_2 = np.diag(ens_2)

g = 2*np.pi*0.1

Q_x   = np.diag(np.sqrt(np.arange(1,qubit_state_num)),1)+np.diag(np.sqrt(np.arange(1,qubit_state_num)),-1)
Q_y   = (0+1j) *(np.diag(np.sqrt(np.arange(1,qubit_state_num)),1)-np.diag(np.sqrt(np.arange(1,qubit_state_num)),-1))
Q_z   = np.diag(np.arange(0,qubit_state_num))
Q_I   = np.identity(qubit_state_num)

H0 = np.kron(H0_1,Q_I) + np.kron(Q_I,H0_2) + g * np.kron(Q_x,Q_x)

# Defining dressed info
is_dressed = False
#w_c, v_c, dressed_id = get_dressed_info(H0)
#dressed_info = {'dressed_id':dressed_id, 'eigenvectors':v_c, 'eigenvalues':w_c,'is_dressed':is_dressed}
dressed_info = None

#Defining Concerned states (starting states)
psi0 = [0,1,qubit_state_num,qubit_state_num+1] #[gg,ge,eg,ee]

#Defining states to include in the drawing of occupation
states_draw_list = [0,1,qubit_state_num,qubit_state_num+1]
states_draw_names = ['00','01','10','11']


#Defining U (Target)


if is_dressed:
    p2 = get_state_index(qubit_state_num+1,dressed_id) #[ee]
    p3 = get_state_index(qubit_state_num,dressed_id) #[eg]
    U = np.identity(qubit_state_num**2)
    U = U - np.outer(v_c[:,p2],v_c[:,p2])
    U = U - np.outer(v_c[:,p3],v_c[:,p3])
    U = U + np.outer(v_c[:,p3],v_c[:,p2])
    U = U + np.outer(v_c[:,p2],v_c[:,p3])
else:   
    U= np.identity(4)
    U[2,2] = 0
    U[3,3] = 0
    U[2,3] = 1
    U[3,2] = 1
    U = transmon_gate(U,qubit_state_num)


# Defining controls
Q_x1 = np.kron(Q_x,Q_I)
Q_y1 = np.kron(Q_y,Q_I)
Q_z1 = np.kron(Q_z,Q_I)
Q_x2 = np.kron(Q_I,Q_x)
Q_y2 = np.kron(Q_I,Q_y)
Q_z2 = np.kron(Q_I,Q_z)

Hops = [Q_x1, Q_x2, Q_z2]
ops_max_amp = [2*np.pi*0.5,2*np.pi*0.5,2*np.pi*0.5]
Hnames =['x1', 'x2','z2']

#Defining convergence parameters
max_iterations = 1000
decay = max_iterations/2
convergence = {'rate':0.01, 'update_step':10, 'max_iterations':max_iterations,\
               'conv_target':1e-3,'learning_rate_decay':decay}

# Defining reg coeffs

states_forbidden_list = []

for ii in range(qubit_state_num):
    forbid_state = (qubit_state_num-1)*qubit_state_num+ii
    if not forbid_state in states_forbidden_list:
        states_forbidden_list.append(forbid_state)
        
    forbid_state = (qubit_state_num-2)*qubit_state_num+ii
    if not forbid_state in states_forbidden_list:
        states_forbidden_list.append(forbid_state)
        
        
for ii in range(qubit_state_num):
    forbid_state = ii*qubit_state_num + (qubit_state_num-1)
    if not forbid_state in states_forbidden_list:
        states_forbidden_list.append(forbid_state)
        
    forbid_state = ii*qubit_state_num + (qubit_state_num-2)
    if not forbid_state in states_forbidden_list:
        states_forbidden_list.append(forbid_state)
        
#print states_forbidden_list

## nothing
#reg_coeffs = {'envelope' : 0.0, 'dwdt':0.0,'d2wdt2':0.0,'forbidden':0.0,
#             'states_forbidden_list': states_forbidden_list,'forbid_dressed':False}

## forbid
#reg_coeffs = {'envelope' : 0.0, 'dwdt':0.0,'d2wdt2':0.0, 'forbidden':50.0,
#              'states_forbidden_list': states_forbidden_list,'forbid_dressed':False}

## forbid + pulse reg
reg_coeffs = {'amplitude':0.01,'dwdt':0.00007,'d2wdt2':0.0, 'forbidden_coeff_list':[10] * len(states_forbidden_list),
             'states_forbidden_list': states_forbidden_list,'forbid_dressed':False}

uks,U_f = Grape(H0,Hops,Hnames,U,total_time,steps,psi0,convergence=convergence, method = 'L-BFGS-B', draw = [states_draw_list,states_draw_names] ,maxA = ops_max_amp, use_gpu=False, sparse_H = False,reg_coeffs=reg_coeffs, unitary_error = 1e-08, show_plots = False, dressed_info = dressed_info, file_name='transmon_transmon_CNOT',Taylor_terms = [20,0], data_path = data_path)
