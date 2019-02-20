"""
uccsdslice.py - A  module for defining uccsd circuit slice classes and methods.
"""
from qiskit import QuantumRegister, QuantumCircuit
from qiskit.extensions.standard import RZGate

from fqc.models import CircuitSlice
from fqc.util import append_gate

class UCCSDSlice(CircuitSlice):
    """
    The UCCSD slice is used to split the UCCSD circuit into partial circuits
    that depend on its parameteraization vector (theta vector) from those that
    do not.
    Fields:
    circuit :: qiskit.QuantumCircuit - the partial circuit the slice
                                       represents
    unitary :: np.matrix - the unitary matrix that represents the circuit
    theta_dependent :: bool - whether or not the partical circuit
                              is parameterized by the theta vector
    thetas :: [float] - the parameterized theta values in the slice
                        that correspond to the RZGates sequentially
                        contained in the slice
    """
    
    def __init__(self, circuit, theta_dependent, thetas=None,
                 redundant=False):
        """
        Args:
        circuit :: qiskit.QuantumCircuit - see class fields
        theta_dependent :: bool - see class fields
        thetas :: [float] - see class fields
        """
        super().__init__(circuit)
        self.theta_dependent = theta_dependent
        self.thetas = thetas

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
        # Collect theta values if the slice is theta dependent.
        thetas = None
        if last_gate_had_attribute:
            thetas = list()
            for gate in circuit.data:
                if _is_theta_dependent(gate):
                    thetas.append(gate.params)

        slices.append(UCCSDSlice(circuit, last_gate_had_attribute, thetas))

    #ENDWHILE

    return slices

def _tests():
    """Run tests on the module.
    """
    import numpy as np
    from fqc.uccsd import get_uccsd_circuit
    theta = [np.random.random() for _ in range(8)]
    circuit = get_uccsd_circuit('LiH', theta)
    slices = get_uccsd_slices(circuit)
    for uccsdslice in slices:
        print(uccsdslice.circuit)

if __name__ == "__main__":
    _tests()
