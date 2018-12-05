import numpy as np
from quantum_optimal_control.core.util import kron_many, matprod_many, CNOT, H, ID, RZ

I = ID(2)

U1 = kron_many(H,H)
U2 = CNOT
U4 = CNOT
U5 = U1

def unitary(theta):
    U3 = kron_many(I, RZ(theta))
    U = matprod_many(U5, U4, U3, U2, U1)
    return U

if __name__ == "__main__":
    THETA = 2.0
    U = unitary(THETA)
    print(U)
    print(U.shape)
