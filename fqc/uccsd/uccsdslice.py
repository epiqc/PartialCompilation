"""
uccsdslice.py - A  module for defining uccsd circuit slice classes and methods.
"""
import numpy as np
from qiskit import QuantumRegister, QuantumCircuit
from qiskit.extensions.standard import RZGate

from fqc.models import CircuitSlice
from fqc.uccsd import get_uccsd_circuit
from fqc.util import (append_gate, optimize_circuit,
                      get_nearest_neighbor_coupling_list)

### CLASS DEFINITIONS ###

class UCCSDSlice(CircuitSlice):
    """
    The UCCSD slice is used to split the UCCSD circuit into partial circuits
    that depend on its parameteraization vector (theta vector) from those that
    do not.
    Fields:
    circuit :: qiskit.QuantumCircuit - the partial circuit the slice
                                       represents
    register :: qiskit.QuantumRegister - the register of the circuit
    unitary :: np.matrix - the unitary matrix that represents the circuit
    parameterized :: bool - whether or not the partical circuit
                              is parameterized by the theta vector
    angles :: [float] - the parameterized values in the slice
                        that correspond to the angle of the RZGates
                        sequentially contained in the slice
    _parameterized_gates :: [qiskit.QuantumGate] - list of gates that depend 
                                                   on circuit parameteriation
    """
    
    def __init__(self, circuit, register, parameterized):
        """
        Args:
        circuit :: qiskit.QuantumCircuit - see class fields
        register :: qiskit.QuantumRegister - see class fields
        parameterized :: bool - see class fields
        """
        super().__init__(circuit, register)
        self.parameterized = parameterized

        # Get each of the the parameterized gates in the circuit,
        # as well as their rotation angle value.
        self._parameterized_gates = list()
        self.angles = list()
        for gate in circuit.data:
            if _is_theta_dependent(gate):
                self._parameterized_gates.append(gate)
                self.angles.append(gate.params[0])
                
    
    def __add__(self, right):
        """
        Concatenate two slices without modifying the original slices.
        Args:
        right :: fqc.models.UCCSDSlice - the slice to concatenate to self
        
        Returns:
        new_slice :: fqc.models.UCCSDSlice - the slice that is the concatenation
                                             of each of the slices
        """
        if not self.circuit.width() == right.circuit.width():
            raise ValueError("Incompatible qubit circuits with different"
                             " qubit counts")
        # Concatenate right to the right side of this circuit.
        register = QuantumRegister(self.circuit.width())
        circuit = QuantumCircuit(register)
        for gate in self.circuit.data:
            append_gate(circuit, register, gate)
        for gate in right.circuit.data:
            append_gate(circuit, register, gate)
        parameterized = self.parameterized or right.parameterized

        return UCCSDSlice(circuit, register, parameterized)


    def update_angles(self, angles):
        """
        Update the value of each of the parameterized gates and the
        corresponding class field.
        Args:
        angles :: [float] - a list of new values to store in each of the
                            parameterized gates
        
        Returns: nothing
        """
        self.angles = angles
        for i, gate in enumerate(self._parameterized_gates):
            gate.params = [angles[i]]


### HELPER METHODS ###

def _is_theta_dependent(gate):
    """Return ture if a gate is dependent on the theta vector,
    false otherwise.
    Note:
    RZ is the only theta dependent gate in the UCCSD circuit and the only RZ gates
    in the UCCSD circuit are theta dependent. Therefore, if a gate is RZ then
    it is theta dependent.

    Args:
    gate :: qiskit.QuantumGate - an arbitrary quantum gate
    """
    return isinstance(gate, RZGate)

### PUBLIC METHODS ###

def get_uccsd_slices(circuit):
    """Greedily slice a UCCSD circuit into continuous runs of theta dependent
    gates and non-theta-dependent gates.
    Args:
    circuit :: qiskit.QuantumCircuit - the UCCSD circuit to slice

    Returns:
    slices :: [fqc.models.UCCSDSlice] - the slices of the circuit
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
            gate_has_attribute = _is_theta_dependent(gate)

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
            
        #ENDFOR

        slices.append(UCCSDSlice(circuit, register, last_gate_had_attribute))

    #ENDWHILE

    return slices

def _tests():
    """Run tests on the module.
    """
    coupling_list = get_nearest_neighbor_coupling_list(2, 2)
    theta = [np.random.random() for _ in range(8)]
    circuit = optimize_circuit(get_uccsd_circuit('LiH', theta), coupling_list)
    slices = get_uccsd_slices(circuit)
    for uccsdslice in slices:
        print(uccsdslice.angles)
        print(uccsdslice.circuit)

if __name__ == "__main__":
    _tests()
