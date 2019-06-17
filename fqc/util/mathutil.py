"""
mathutil.py - A module for math and linear algebra utilites.
"""

import numpy as np

# constants
PAULI_X = np.matrix([[0, 1],
                     [1, 0]])
PAULI_Y = np.matrix([[0,    0-1j],
                     [0+1j,    0]])
PAULI_Z = np.matrix([[1,  0],
                     [0, -1]])
