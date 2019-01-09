"""A module for quantum / linear algebra utilities."""

from functools import reduce

import numpy as np

# Math

def kron_many(*matrices):
    """Compute the kronecker product of multiple matrices."""
    return reduce(np.kron, matrices)

def matprod_many(*matrices):
    """Compute the matrix product of multiple matrices."""
    return reduce(np.matmul, matrices)

def extend_gate(gate, level):
    """Extend the gate to the computational basis of dimension 'levels'.
    
    """
    return

# Gates

# 2-qubit CNOT
CNOT = np.matrix([[1,0,0,0], 
                  [0,1,0,0], 
                  [0,0,0,1], 
                  [0,0,1,0]])

# Hadamard
H = np.matrix([[1,  1], 
               [1, -1]])/np.sqrt(2)

# Pauli Y
Y = np.matrix([[0, 0-1j],
               [0+1j,  0]])

def ID(n):
    """Return the identity matrix.
    
    n :: int -- the dimension of the matrix 
    """
    return np.matrix(np.identity(n))

def RZ(theta):
    """Return the unitary matrix that rotates blocsphere by \'theta\' in the 
    z-direction.

    theta :: number -- rotation argument in radians
    """
    return np.matrix([[np.exp(-0.5j*theta),0],[0,np.exp(0.5j*theta)]])
    
if __name__ == "__main__":
    # Run tests
    # print("Tests passing.")
    print("No tests currently implemented.")

