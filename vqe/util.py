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

# Gates

# 2-qubit CNOT
CNOT = np.array([[1,0,0,0], 
                 [0,1,0,0], 
                 [0,0,0,1], 
                 [0,0,1,0]])
# Hadamard
H = np.array([[1, 1], 
              [1,-1]])/np.sqrt(2)

def ID(n):
    """Return the identity matrix.
    
    n :: int -- the dimension of the matrix 
    """
    return np.identity(n)

def RZ(theta):
    """Return the unitary matrix that rotates blocsphere by \'theta\' in the 
    z-direction.

    theta :: number -- rotation argument in radians
    """
    return np.array([[np.exp(-0.5j*theta),0],[0,np.exp(0.5j*theta)]])

# Display

def print_matrix(matrix):
    """Display a matrix nicely."""

    print_str = "["
    last_row_index = len(matrix) - 1
    for i, row in enumerate(matrix):
        print_str += "["
        last_element_index = len(row) - 1
        for j, element in enumerate(row):
            if j == last_element_index:
                print_str += str(element)
            else:
                print_str += str(element) + ", "
        if i == last_row_index:
            print_str += "]"
        else:
            print_str += "],\n"

    print_str += "]"

    print(print_str)

    return
    
if __name__ == "__main__":
    # Run tests
    # print("Tests passing.")
    print("No tests currently implemented.")

