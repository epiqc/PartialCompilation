# -*- coding: utf-8 -*-

# Copyright 2018 IBM.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# =============================================================================

import numpy as np
import logging

from qiskit import QuantumRegister, QuantumCircuit, transpiler
from qiskit import execute as q_execute
from qiskit.transpiler.passes import Unroller
from qiskit.transpiler import PassManager

from qiskit.aqua.components.initial_states import InitialState
from qiskit.aqua.utils.backend_utils import get_aer_backend

logger = logging.getLogger(__name__)


class Custom(InitialState):
    """A custom initial state."""

    CONFIGURATION = {
        'name': 'CUSTOM',
        'description': 'Custom initial state',
        'input_schema': {
            '$schema': 'http://json-schema.org/schema#',
            'id': 'custom_state_schema',
            'type': 'object',
            'properties': {
                'state': {
                    'type': 'string',
                    'default': 'zero',
                    'oneOf': [
                        {'enum': ['zero', 'uniform', 'random']}
                    ]
                },
                'state_vector': {
                    'type': ['array', 'null'],
                    "items": {
                        "type": "number"
                    },
                    'default': None
                }
            },
            'additionalProperties': False
        }
    }

    def __init__(self, num_qubits, state="zero", state_vector=None, circuit=None):
        """Constructor.

        Args:
            num_qubits (int): number of qubits
            state (str): `zero`, `uniform` or `random`
            state_vector: customized vector
            circuit (QuantumCircuit): the actual custom circuit for the desired initial state
        """
        loc = locals().copy()
        # since state_vector is a numpy array of complex numbers which aren't json valid,
        # remove it from validation
        del loc['state_vector']
        self.validate(loc)
        super().__init__()
        self._num_qubits = num_qubits
        self._state = state
        size = np.power(2, self._num_qubits)
        self._circuit = None
        if circuit is not None:
            if circuit.width() != num_qubits:
                logger.warning('The specified num_qubits and the provided custom circuit do not match.')
            self._circuit = Custom._convert_to_basis_gates(circuit)
            if state_vector is not None:
                self._state = None
                self._state_vector = None
                logger.warning('The provided state_vector is ignored in favor of the provided custom circuit.')
        else:
            if state_vector is None:
                if self._state == 'zero':
                    self._state_vector = np.array([1.0] + [0.0] * (size - 1))
                elif self._state == 'uniform':
                    self._state_vector = np.array([1.0 / np.sqrt(size)] * size)
                elif self._state == 'random':
                    self._state_vector = Custom._normalize(np.random.rand(size))
                else:
                    raise ValueError('Unknown state {}'.format(self._state))
            else:
                if len(state_vector) != np.power(2, self._num_qubits):
                    raise ValueError('State vector length {} incompatible with num qubits {}'
                                     .format(len(state_vector), self._num_qubits))
                self._state_vector = Custom._normalize(state_vector)
                self._state = None

    @staticmethod
    def _normalize(vector):
        return vector / np.linalg.norm(vector)

    @staticmethod
    def _convert_to_basis_gates(circuit):
        # get the circuits from compiled circuit
        unroller = Unroller(basis=['u1', 'u2', 'u3', 'cx', 'id'])
        pm = PassManager(passes=[unroller])
        qc = transpiler.transpile(circuit, get_aer_backend('qasm_simulator'),
                                  pass_manager=pm)
        return qc

    def construct_circuit(self, mode, register=None):
        """
        Construct the statevector of desired initial state.

        Args:
            mode (string): `vector` or `circuit`. The `vector` mode produces the vector.
                            While the `circuit` constructs the quantum circuit corresponding that
                            vector.
            register (QuantumRegister): register for circuit construction.

        Returns:
            QuantumCircuit or numpy.ndarray: statevector.

        Raises:
            ValueError: when mode is not 'vector' or 'circuit'.
        """
        if mode == 'vector':
            if self._state_vector is None:
                if self._circuit is not None:
                    self._state_vector = np.asarray(q_execute(self._circuit, get_aer_backend(
                        'statevector_simulator')).result().get_statevector(self._circuit))
            return self._state_vector
        elif mode == 'circuit':
            if self._circuit is None:
                if register is None:
                    register = QuantumRegister(self._num_qubits, name='q')
                circuit = QuantumCircuit(register)
                if self._state is None or self._state == 'random':
                    circuit.initialize(self._state_vector, [
                                       register[i] for i in range(self._num_qubits)])
                    circuit = Custom._convert_to_basis_gates(circuit)
                elif self._state == 'zero':
                    pass
                elif self._state == 'uniform':
                    for i in range(self._num_qubits):
                        circuit.u2(0.0, np.pi, register[i])
                else:
                    pass
                self._circuit = circuit
            return self._circuit
        else:
            raise ValueError('Mode should be either "vector" or "circuit"')
