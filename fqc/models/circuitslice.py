"""
circuitslice.py - A  module for defining objects and mehods for slicing
                  quantum circuits.
"""
from qiskit import QuantumRegister, QuantumCircuit

from fqc.util import get_unitary, append_gate

class CircuitSlice(object):
    """
    The circuit slice object is the main object that will contain a reference to a
    circuit (presumably the circuit it references is a slice of a larger circuit)
    and any attributes that should be noted about the partial circuit.

    Fields:
    circuit :: qiskit.QuantumCircuit - the partial circuit the slice represents
    register :: qiskit.QuantumRegister - the register for the circuit
    unitary :: np.matrix - the unitary for the partial circuit
    """
    
    def __init__(self, circuit, register):
        """
        Args:
        circuit :: qiskit.QuantumCircuit - see class fields
        """
        super().__init__()
        self.circuit = circuit
        self.register = register
        self.unitary = get_unitary(circuit)

def get_slices(circuit):
    """Greedily slice a circuit into continuous runs of gates with
    attribute and runs without attribute and pass information to a constructor.
    Args:
    circuit :: qiskit.QuantumCircuit - the circuit to slice
    has_attribute :: qiskit.QuantumGate -> bool - a function that determines
                                                  if a given gate has the 
                                                  attribute in question
    slices :: [fqc.models.CircuitSlice] - the slices of the circuit
    """
    slices = list()
    # The circuit width is the number of registers, i.e. qubits.
    circuit_width = circuit.width()
    gates = circuit.data
    gate_count = len(gates)

    # Walk the list of gates and make a new quantum circuit for every continuous
    # span of gates that have attribute or do not have attribute.
    gates_encountered = 0
    while gates_encountered < gate_count:
        # Construct a new circuit for the span.
        register = QuantumRegister(circuit_width)
        circuit = QuantumCircuit(register)

        # Traverse the gate list and construct a circuit that is either
        # a continuous span of attribute gates or non-attribute gates.
        redundant = False
        gate_has_attribute = False
        last_gate_had_attribute = False
        first_gate = True
        for gate in gates[gates_encountered:]:
            gate_has_attribute = has_attribute(gate) 

            if (gate_has_attribute and
                    (last_gate_had_attribute or first_gate)):
                last_gate_had_attribute = True
                gates_encountered += 1

            elif (not gate_has_attribute and
                    (not last_gate_had_attribute or first_gate)):
                last_gate_had_attribute = False
                gates_encountered += 1

            else:
                break
            
            append_gate(circuit, register, gate)
            
            if first_gate:
                first_gate = False

            slices.append(CircuitSlice(circuit, register))

        #ENDFOR

    #ENDWHILE

    return slices
