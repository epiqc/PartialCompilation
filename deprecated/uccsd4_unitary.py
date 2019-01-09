import numpy as np
from quantum_optimal_control.core.util import kron_many, matprod_many, CNOT, H, ID, RZ

I = ID(2)

U1 = kron_many(H,H,H,H)
U2 = kron_many(CNOT, I, I)
U3 = kron_many(I, CNOT, I)
U4 = kron_many(I, I, CNOT)
U6 = kron_many(I, I, CNOT)
U7 = kron_many(I, CNOT, I)
U8 = kron_many(CNOT, I, I)
U9 = kron_many(H, H, H, H)

def unitary(theta):
    U5 = kron_many(I, I, I, RZ(theta))
    U  = matprod_many(U9, U8, U7, U6, U5, U4, U3, U2, U1)

    return U

if __name__ == "__main__":
    THETA = 0
    U = unitary(THETA)
    print(U)
    print(U.shape)

