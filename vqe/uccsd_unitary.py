""" A module for computing the UCCSD ansatz unitary matrix."""
from functools import reduce
import numpy as np
from quantum_optimal_control.core.util import kron_many, matprod_many, CNOT, H, Y, ID, RZ

# TODO: Implement num_qubit_states?

def uccsd_unitary(num_qubits, sqops_str, theta):
    """Build the UCCSD ansatz unitary matrix for the specified inputs.
    
    num_qubits (int) -- the number of qubits to build the ansatz for, domain?. 
    sqops_str (str) -- a string specifying the single qubit operations (U1, U2, U3, U4)
                      in the ansatz--see figure 1b: https://arxiv.org/pdf/1805.04340.pdf
                      e.g. \"HHYH\"
    theta (float) -- the angle in radians of the Z-rotation gate.    
    """

    # Constants
    # The number of qubits that are acted on by a single qubit operator.
    num_sqop_qubits = 4
    
    # Parse input.
    if num_qubits < 4 or not isinstance(num_qubits, int):
        raise ValueError("At least 4 qubits are required for the uccsd ansatz.")
    if not len(sqops_str) == 4:
        raise ValueError("Only 4 single qubit operations may comprise the uccsd ansatz.")
    
    # Parse the sqops_str into a set of gates.
    sqops = []
    sqops_str = sqops_str.lower()
    for char in sqops_str:
        if char == "h":
            sqops.append(H)
        elif char == "y":
            sqops.append(Y)
        else:
            raise ValueError("Only haddamard \'h\' and pauli-y \'y\'"
                             "gates may comprise the sinqle qubit "
                             "operations of the uccsd ansatz.")
    #print(sqops)

    # Do initialization.
    I = ID(2) # Should this be dependent on number of qubit states?
    # The number of qubits that will be not be acted on by a single qubit operator.
    num_nsqop_qubits = num_qubits - num_sqop_qubits 
    # The number of qubits between the first and second qubits acted on by a
    # single qubit operator. num_sqop_qubits is distributed about evenly between
    # top and bottom.
    num_top_qubits = int((num_nsqop_qubits - (num_nsqop_qubits % 2)) / 2 + (num_nsqop_qubits % 2))
    # The number of qubits between the third and fourth qubits acted on by a
    # single qubit operator.
    num_bot_qubits = num_nsqop_qubits - num_top_qubits
    #print("num_top_qubits", num_top_qubits, "num_bot_qubits", num_bot_qubits)

    # Construct the matrices at each step in the ansatz and collect them in 'intermediaries'.
    intermediaries = []

    # Construct the initial matrix.
    intermediaries.append(kron_many(sqops[0],
                                    *[I for qubit in range(num_top_qubits)],
                                    sqops[1], sqops[2],
                                    *[I for qubit in range(num_bot_qubits)],
                                    sqops[3]))

    # Construct the top-to-bottom CNOT cascade. Each step in the cascade will be
    # one of the (num_qubits - 1) permutations of the kroencker product of 
    # (num_qubits - 2) identity matrices and one CNOT matrix.
    base = [I for qubit in range(num_qubits - 2)]
    permutations = []
    for i in range(num_qubits - 1):
        gates = base[:]
        gates.insert(i, CNOT)
        permutations.append(gates)
        intermediaries.append(kron_many(*gates))

    # Construct the middle matrix.
    intermediaries.append(kron_many(*[I for qubit in range(num_qubits - 1)],
                                     RZ(theta)))
    
    # Construct the bottom-to-top CNOT cascade. Simply reverse the order of the
    # permutations.
    for permutation in permutations:
        permutation.reverse()
    intermediaries += [kron_many(*permutation) for permutation in permutations]


    # Construct the final matrix.
    intermediaries.append(kron_many(sqops[0].H,
                                    *[I for qubit in range(num_top_qubits)],
                                    sqops[1].H, sqops[2].H,
                                    *[I for qubit in range(num_bot_qubits)],
                                    sqops[3].H))

    return matprod_many(*intermediaries)

if __name__ == "__main__":
    U = uccsd_unitary(4, "HHHH", 0)
    print(U)
