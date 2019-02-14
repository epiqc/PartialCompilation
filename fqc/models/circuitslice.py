"""
circuitslice.py - A  module for defining objects that represent slices
                  of quantum circuits.
"""

from fqc.util import get_unitary

class CircuitSlice(object):
    """
    The circuit slice object is the main object that will contain a reference to a
    circuit (presumably the circuit it references is a slice of a larger circuit)
    and any attributes that should be noted about the partial circuit.

    Fields:
    circuit :: qiskit.QuantumCircuit - the partial circuit the slice represents
    unitary :: np.matrix - the unitary for the partial circuit
    """
    
    def __init__(self, circuit):
        """
        Args:
        circuit :: qiskit.QuantumCircuit - the partial circuit the slice
                                           represents
        """
        super().__init__()
        self.circuit = circuit
        self.unitary = get_unitary(circuit)
