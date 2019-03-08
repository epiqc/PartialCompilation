"""
pulseutil.py - A module for pulse functionality.
"""


import numpy as np
import scipy.linalg as la
from scipy.special import factorial
import os,sys,inspect
import h5py
import random as rd
import time
from IPython import display
import matplotlib.pyplot as plt
import re, math
from datetime import datetime

# Either define a file named config and append the directory
# in which it sits to your python path, or define an environment
# variable that points to an output directory.
try:
    import config
    data_path = config.DATA_PATH
except ImportError:
    data_path = os.environ['FQC_DATA_PATH']

file_name = datetime.today().strftime('%h%d')

from quantum_optimal_control.helper_functions.grape_functions import *
from quantum_optimal_control.main_grape.grape import Grape
from quantum_optimal_control.core import hamiltonian

from fqc.models import Pulse

def evol_pulse(pulse, U=None, save=True, out_file=file_name, out_path=data_path):
    """
    """
    if (U is None):
        U = pulse.U
    
    SS = Grape(pulse.H0, pulse.Hops, pulse.Hnames, U, 
                pulse.total_time, pulse.steps, pulse.states_concerned_list,
                {}, initial_guess=pulse.uks, reg_coeffs={}, 
                use_gpu=False, sparse_H=False, method='EVOLVE', 
                maxA=pulse.maxA, show_plots=False, 
                save=save, file_name=out_file, data_path=out_path)
    return SS


def evol_pulse_from_file(filename, U=None, save=True, out_file=file_name, out_path=data_path):
    N = None
    d = None
    qubits = None
    pulse = Pulse(N, d, qubits, fname=filename)
    if (U == None):
        U = pulse.U
    res = Grape(pulse.H0, pulse.Hops, pulse.Hnames, U, 
                pulse.total_time, pulse.steps, pulse.states_concerned_list,
                {}, initial_guess=pulse.uks, reg_coeffs={}, 
                use_gpu=False, sparse_H=False, method='EVOLVE', 
                maxA=pulse.maxA, show_plots=False, 
                save=save, file_name=out_file, data_path=out_path)
    return res

def plot_pulse(pulse, save_plot=True, file_name=file_name, data_path=data_path):
    """
    """
    print("Error: %e" % pulse.error)
    dt = pulse.total_time / pulse.steps
    uks = pulse.uks
    for jj in range(len(uks)):

        plt.plot(np.array([dt * ii for ii in range(pulse.steps)]), np.array(uks[jj, :]), label='u'+pulse.Hnames[jj])

        # Control Fields
        plt.title('Optimized pulse')

        plt.ylabel('Amplitude')
        plt.xlabel('Time (ns)')
        plt.legend()
    fig = plt.gcf()
    fig.set_size_inches(12,9)
    if (save_plot):
        fig.savefig(data_path+file_name)
    else:
        fig.show()

def plot_pulse_from_file(filename, save_plot=True, file_name=file_name, data_path=data_path):
    f = h5py.File(filename, 'r')
    print("Error: %e" % list(f['error'])[-1])
    uks = list(f['uks'])[-1]
    Hnames = [b.decode('utf-8') for b in list(f['Hnames'])]
    t = np.array(f['total_time'])
    steps = np.array(f['steps'])
    dt = t / steps
    for jj in range(len(uks)):

        plt.plot(np.array([dt * ii for ii in range(steps)]), np.array(uks[jj, :]), label='u'+Hnames[jj])

        # Control Fields
        plt.title('Optimized pulse')

        plt.ylabel('Amplitude')
        plt.xlabel('Time (ns)')
        plt.legend()
    fig = plt.gcf()
    fig.set_size_inches(12,9)
    if (save_plot):
        out_name = data_path+'/'+file_name+'.pdf'
        fig.savefig(out_name)
        print("Figure saved at: "+out_name)
    else:
        fig.show()


