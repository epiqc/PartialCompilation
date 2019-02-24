"""
pulse.py - A module for defining objects that represent pulse
                  of quantum circuits.
"""

#from fqc.util import get_unitary
import numpy as np
import scipy.linalg as la
from scipy.special import factorial
import os,sys,inspect
import h5py
import matplotlib.pyplot as plt
import random as rd
import time
import re, math
from datetime import datetime

class Pulse(object):
    """
    The Pulse object is the main object that will contain a reference to a
    pulse information required by Grape,
    and any attributes that should be noted about the partial circuit.

    Fields:
    uks :: 2D np.array - the amplitudes of all control fields
    U :: np.matrix - target unitary
    error :: float - error rate of uks from target U
    total_time :: float - duration of pulse
    steps :: int - number of discrete steps of the pulse
    N :: int - number of qubits
    d :: int - number of states per qubit
    H0 :: np.matrix - drift hamiltonian
    qubits :: int list - index of target qubits
    Hops :: np.array of np.matrix - control field hamiltonians
    Hnames :: np.array of strings - names of control fields
    states_concerned_list :: int list - list of concerned states
    maxA :: int list - max amplitude of each control field
     
    """
    
    def __init__(self, N, d, qubits, fname='', 
                 uks=[], total_time=0.0, steps=0, 
                 H0=[], Hops=[], Hnames=[], 
                 U=[], error=0.0, 
                 states_concerned_list=[], maxA=[]):
        """
        Args:
        circuit :: qiskit.QuantumCircuit - the partial circuit the slice
                                           represents
        """
        super().__init__()
        if (fname != ''):
            f = h5py.File(fname, 'r')
            self.H0 = np.array(f['H0'])
            self.Hops = np.array(f['Hops'])
            self.Hnames = [b.decode('utf-8') for b in list(f['Hnames'])]
            self.maxA = np.array(f['maxA'])
            self.uks = list(f['uks'])[-1]
            self.total_time = np.array(f['total_time'])
            self.steps = np.array(f['steps'])
            self.U = np.array(f['U'])
            self.error = np.array(f['error'])[-1]
            self.states_concerned_list = np.array(f['states_concerned_list'])
            self.N = N
            self.d = d
            self.qubits = qubits

        else:
            self.uks = uks
            self.total_time = total_time
            self.steps = steps
            self.H0 = H0
            self.Hops = Hops
            self.Hnames = Hnames
            self.U = U
            self.error = error
            self.N =N
            self.d = d
            self.qubits = qubits
            self.states_concerned_list = states_concerned_list
            self.maxA = maxA


    def set_params(self, uks=[], total_time=0.0, steps=0, 
                 H0=[], Hops=[], Hnames=[],  
                 U=[], error=0.0, N=0, d=0, qubits=[], 
                 states_concerned_list=[], maxA=[]):
        if (uks != []):
            self.uks = uks
        if (total_time != 0.0):
            self.total_time = total_time
        if (steps != 0):
            self.steps = steps
        if (H0 != []):
            self.H0 = H0
        if (Hops != []):
            self.Hops = Hops
        if (Hnames != []):
            self.Hnames = Hanems
        if (U != []):
            self.U = U
        if (error != 0.0):
            self.error = error
        if (N != 0):
            self.N = N
        if (d != 0):
            self.d = d
        if (qubits != []):
            self.qubits = qubits
        if (states_concerned_list != []):
            self.states_concerned_list = states_concerned_list
        if (maxA != []):
            self.maxA = maxA

    #def save_pulse():
    #    return

