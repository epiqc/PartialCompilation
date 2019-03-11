"""
grapeseed.py - Functions for seeding the grape optimzations
adjust_slices, get_initial_pulses, get_opt_pulses

"""

import numpy as np
#import pickle
import scipy.linalg as la
from scipy.special import factorial
import os,sys,inspect
import h5py
import random as rd
import time
from IPython import display
import matplotlib.pyplot as plt
import re, math

from fqc.uccsd import UCCSDSlice
from fqc.models import Pulse
from fqc.util import get_unitary, squash_circuit, append_gate
from fqc.util import evol_pulse, evol_pulse_from_file, plot_pulse, plot_pulse_from_file 

from datetime import datetime

import config

data_path = config.DATA_PATH
file_name = datetime.today().strftime('%h%d')

from quantum_optimal_control.helper_functions.grape_functions import *
from quantum_optimal_control.main_grape.grape import Grape
from quantum_optimal_control.core import hamiltonian

### BUILD CIRCUTS AND UNITARIES ###


def get_initial_pulses(slices):
    """ Produce the initial guesses of optimal control pulses
    Args:
    slices :: list of uccsdslice - slices of circuit
    option :: string - Default is 'gaussian' (random guess). 
                       

    """
    seeds = []
    #for s in slices:
    #TODO: integrate loop-up table method here
    
    return seeds

def get_opt_pulses(seeds, convergence, reg_coeffs={}, method='ADAM'):
    """
    TODO: multi-threading for slices of pulses
    Args: 
    seeds :: list of initial pulses
    """
    opt_pulses = []
    for s in seeds:
        #TODO: parallelize for loop
        res = Grape(s.H0, s.Hops, s.Hnames, s.U, s.total_time, s.steps, 
              s.states_concerned_list, convergence, initial_guess = s.uks, 
              reg_coeffs=reg_coeffs, use_gpu=False, sparse_H=False, 
              method=method, maxA=s.maxA, show_plots=False, save=False)
        opt_pulse = Pulse(s.N, s.d, s.qubits, 
                 uks=res.uks, total_time=s.total_time, steps=s.steps, 
                 H0=s.H0, Hops=s.Hops, Hnames=s.Hnames, 
                 U=s.U, error=s.error, 
                 states_concerned_list=s.states_concerned_list, maxA=s.maxA)
        opt_pulses.append(opt_pulse)

    return opt_pulses

def _extend_uks(pulse, N, d):
    """
    TODO: extend pulses on subset of qubits to all qubits
    """
    if (pulse.N < N):
        #TODO: extend to full qubits (by idle gates?)
        print("Pulse acts on subset of qubits. _extend_uks has not been implemented.");
        print("N: %d, pulse.N: %d" % (N, pulse.N))
        sys.exit()
        
    elif (pulse.N > N):
        print("Pulse segment exceed specified qubits number");
        print("N: %d, pulse.N: %d" % (N, pulse.N))
        sys.exit()
    return pulse

def _ops_all_equal(Hops1, Hops2):
    if (len(Hops1) != len(Hops2)):
        return False
    for i in range(len(Hops1)):
        if (not np.array_equal(Hops1[i],Hops2[i])):
            return False
    return True

def concat_and_evol(N, d, pulses, U, file_name=file_name, data_path=data_path):
    """
    Assume each pulse has the same dt interval, same complete Hops!
    """
    total_time = 0.0
    steps = 0
    uks = []
    maxA = []
    #Hops, Hnames = hamiltonian.get_Hops_and_Hnames(N, d)
    for (i, p) in enumerate(pulses):
        p = _extend_uks(p, N, d)
        if (i==0):
            H0 = p.H0
            Hops = p.Hops
            Hnames = p.Hnames
            states_concerned_list = p.states_concerned_list
            uks = p.uks
            maxA = p.maxA
        else:
            assert(len(uks) == len(p.uks))
            assert(len(maxA) == len(p.maxA))
            assert(_ops_all_equal(Hops, p.Hops)) 
            #assert(Hnames == p.Hnames)
            uks = [np.concatenate((uks[ll],p.uks[ll]), axis=0) for ll in range(len(p.uks))]
            maxA = [max(maxA[ll], p.maxA[ll]) for ll in range(len(p.maxA))]
        total_time += p.total_time
        steps += p.steps

        

    res = Grape(H0, Hops, Hnames, U, total_time, steps, states_concerned_list, 
          {},initial_guess=uks, reg_coeffs={},use_gpu=False, 
          sparse_H=False, method='EVOLVE', maxA=maxA,show_plots=False, 
          file_name=file_name, data_path = data_path)
    
    return res


def adjust_slices(slices):
    """ Procedure that perform optimizations on the slices
    Args:
    slices :: list of uccsdslice
    """
    return slices


def _tests():
    """A function to run tests on the module"""
    theta = [np.random.random() * 2 * np.pi for _ in range(8)]
    slices = get_uccsd_slices(theta)

    
    #for uccsdslice in slices:
    #    squashed_circuit = squash_circuit(uccsdslice.circuit)
    #    squashable = False
    #    if squashed_circuit.width() < uccsdslice.circuit.width():
    #        squashable = True
    #    print("theta_dependent: {}, redundant: {}, squashable: {}"
    #          "".format(uccsdslice.theta_dependent, uccsdslice.redundant,
    #                    squashable))
    #    print(uccsdslice.circuit)
    #    if squashable:
    #        print("squashed circuit:")
    #        print(squashed_circuit)

if __name__ == "__main__":
    _tests()
