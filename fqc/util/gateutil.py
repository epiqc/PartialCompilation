"""A module for quantum / linear algebra utilities.
Some of these utilities are implemented in schuster lab's 
quantum_optimal_control.helper_functions.grape_functions
but it is hard to tell that their code is doing what we want it to do
so we implement some helper functions here for clarity.
"""

from functools import reduce

import numpy as np

# Math

def krons(*matrices):
    """Compute the kronecker product of multiple matrices.
    matrices :: [np.matrix] - list of matrices to compute the kroenecker product of
    """
    return reduce(np.kron, matrices)

def matprods(*matrices):
    """Compute the matrix product of multiple matrices.
    matrices :: [np.matrix] - list of matrices to compute the matrix product of
    """
    return reduce(np.matmul, matrices)

#TODO: implement me
def extend_gate(gate, n):
    """Extend the gate to the computational basis space with n basis vectors.
    gate :: np.matrix - a unitary matrix to be extended
    n :: int >= 2 - the number of rows (and columns) to extend the gate to
    """
    return

# Gates

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

