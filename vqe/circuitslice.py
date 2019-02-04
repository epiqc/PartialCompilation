"""
circuitslice.py - A  module for defining objects that represent slices
                  of quantum circuits.
"""

class Slice(object):
    """
    The slice object is the main object that will contain a reference to a
    circuit (presumably the circuit it references is a slice of a larger circuit)
    and any attributes that should be noted about the circuit.
    """
    
    def __init__(self, circuit, unitary=None):
        """
        Args:
        circuit :: qiskit.QuantumCircuit - the partial circuit the slice
                                           represents
        unitary :: numpy.matrix - the unitary for the partial circuit
        """
        super().__init__()
        self.circuit = circuit
        self.unitary = unitary

class UCCSDSlice(Slice):
    """
    The UCCSD slice is used to split the UCCSD circuit into partial circuits
    that depend on its parameteraization vector (theta vector) from those that
    do not.
    """
    
    def __init__(self, circuit, unitary=None, theta_dependent=False,
                 redundant=False):
        """
        Args:
        circuit :: qiskit.QuantumCircuit - the partial circuit the slice
                                           represents
        unitary :: numpy.matrix = the unitary for the partial circuit
        theta_dependent :: bool - whether or not the partical circuit
                                  is parameterized by the theta vector
        redundant :: bool - whether this theta dependent circuit has
                            appeared earlier in a group of slices

        Note: The theta dependent slices of the UCCSD circuit are typically
        circuits comprised of a single Rz gate. These Rz gates often have
        the same parametarized value and hence their slice is identical to
        some set of other theta dependent gates. We should note this 
        redundancy for brevity's sake.
        """
        super().__init__(circuit, unitary)
        self.theta_dependent = theta_dependent
        self.redundant = redundant
        
        
        
