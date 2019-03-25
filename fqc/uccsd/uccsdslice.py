"""
uccsdslice.py - A  module for defining uccsd circuit slice classes and methods.
"""
import numpy as np
from qiskit import QuantumRegister, QuantumCircuit
from qiskit.extensions.standard import RZGate

from fqc.models import CircuitSlice
from fqc.uccsd import get_uccsd_circuit
from fqc.util import (append_gate, optimize_circuit,
                      get_nearest_neighbor_coupling_list,
                      GATE_TO_PULSE_TIME)

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
        register = self.circuit.qregs[0]
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
            if _is_theta_dependent(gate):
                gate.params = [angles[i]]


# Each index of the list corresponds to the slice index to which it corresponds.
# Times are given in nanosceonds. Times were computed with Rz(0) in
# /projects/ftchong/qoc/thomas/uccsd_slice_time/
# The times computed there were then added to by the number of Rz gates times
# the maximum time required to execute one Rz gate. These times are for the g2_s8
# class of circuits.
rz = GATE_TO_PULSE_TIME['rz']
UCCSD_LIH_SLICE_TIMES = [1.05 + rz * 2, 3.35 + rz * 2, 3.55 + rz * 2, 3.1 + rz * 2,
                         8.95 + rz * 8, 6.25 + rz * 8, 8.9 + rz * 8, 5.25 + rz * 8]


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

def get_uccsd_slices(circuit, granularity=1, dependence_grouping=False):
    """Greedily slice a UCCSD circuit into continuous runs of theta dependent
    gates and non-theta-dependent gates.
    Args:
    circuit :: qiskit.QuantumCircuit - the UCCSD circuit to slice
    granularity :: int > 0 - the base set of slices (granularity = 1) are
    the sequential theta-dependent and non-theta-dependent slice. Specifying
    a granularity greater than 1 concatenates these base slices. For instance,
    granularity = 2 will concatenate every two slices in the base slice list,
    resulting in a theta-dependent and non theta-dependent slice being
    concatenated together. 
    dependence_grouping :: bool - whether or not consecutive theta dependent
                                  gates who share the same theta dependence will
                                  be grouped into the same slice

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
        register = circuit.qregs[0]
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

    if granularity > 1:
        # Walk the list of slices and concatenate granularity number of
        # gates together.
        i = 0
        slice_count = len(slices)
        new_slices = list()
        while i < slice_count:
            new_slice = slices[i]
            # Concatenate granularity - 1 slices to the current slice.
            for j in range(i + 1, i + granularity):
                # If the end of the slices list has been reached, pull out.
                if j > slice_count - 1:
                    break
                new_slice += slices[j]
            # ENDFOR
            # Append the last unparameterized base slice to the final new slice
            # if it is the lone last slice.
            if slice_count % granularity == 1 and i + granularity == slice_count - 1:
                new_slice += slices[-1]
                i += 1
            new_slices.append(new_slice)
            i += granularity
        #ENDWHILE
        slices = new_slices
    elif granularity != 1:
        raise ValueError("granularity must be greater than 0 but got {}"
                         "".format(granularity))

    # # Concatenate neighboring slices that have the same theta dependence.
    if dependence_grouping:
        i = 0
        slice_count = len(slices)
        new_slices = list()
        # Consider each slice.
        while i < slice_count:
            new_slice = slices[i]
            cur_angles = np.array(new_slice.angles)
            num_concatenated = 0
            # Consider all slices after the current slice.
            for j in range(i + 1, slice_count):
                # Do not consider the next slice if it is out of bounds.
                if j > slice_count - 1:
                    break
                next_slice = slices[j]
                next_angles = np.array(next_slice.angles)
                # Concatenate the next slice to the current slice, if it has
                # the same theta dependence.
                if cur_angles.all() == next_angles.all():
                    new_slice += next_slice
                    num_concatenated += 1
                else:
                    break
            # END FOR
            # Add the new slice to the list.
            new_slices.append(new_slice)
            # Next, in the outer loop, consider the slice after
            # the last slice concatenated to `new_slice`.
            i += 1 + num_concatenated
        # END WHILE
        slices = new_slices
    # END IF
    return slices

def _tests():
    """Run tests on the module.
    """
    coupling_list = get_nearest_neighbor_coupling_list(2, 2)
    theta = [np.random.random() for _ in range(8)]
    circuit = optimize_circuit(get_uccsd_circuit('LiH', theta), coupling_list)
    slices = get_uccsd_slices(circuit, granularity=2)
    grouped_slices = get_uccsd_slices(circuit, granularity=2,
                                      dependence_grouping=True)
    angle_count = 0
    for uccsdslice in grouped_slices:
        print(uccsdslice.angles)
        print(uccsdslice.circuit)
        for angle in uccsdslice.angles:
            assert angle == slices[angle_count].angles[0]
            angle_count += 1

    print("grouped_slices_count: {}".format(len(grouped_slices)))
    assert angle_count == 40

if __name__ == "__main__":
    _tests()
