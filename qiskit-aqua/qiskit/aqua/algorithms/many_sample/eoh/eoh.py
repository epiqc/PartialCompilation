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
"""
The Quantum Dynamics algorithm.
"""

import logging

from qiskit import QuantumRegister
from qiskit.aqua.algorithms import QuantumAlgorithm
from qiskit.aqua import AquaError, PluggableType, get_pluggable_class

logger = logging.getLogger(__name__)


class EOH(QuantumAlgorithm):
    """
    The Quantum EOH (Evolution of Hamiltonian) algorithm.
    """

    PROP_OPERATOR_MODE = 'operator_mode'
    PROP_EVO_TIME = 'evo_time'
    PROP_NUM_TIME_SLICES = 'num_time_slices'
    PROP_EXPANSION_MODE = 'expansion_mode'
    PROP_EXPANSION_ORDER = 'expansion_order'

    CONFIGURATION = {
        'name': 'EOH',
        'description': 'Evolution of Hamiltonian for Quantum Systems',
        'input_schema': {
            '$schema': 'http://json-schema.org/schema#',
            'id': 'EOH_schema',
            'type': 'object',
            'properties': {
                PROP_OPERATOR_MODE: {
                    'type': 'string',
                    'default': 'paulis',
                    'oneOf': [
                        {'enum': [
                            'paulis',
                            'grouped_paulis',
                            'matrix'
                        ]}
                    ]
                },
                PROP_EVO_TIME: {
                    'type': 'number',
                    'default': 1,
                    'minimum': 0
                },
                PROP_NUM_TIME_SLICES: {
                    'type': 'integer',
                    'default': 1,
                    'minimum': 0
                },
                PROP_EXPANSION_MODE: {
                    'type': 'string',
                    'default': 'trotter',
                    'oneOf': [
                        {'enum': [
                            'trotter',
                            'suzuki'
                        ]}
                    ]
                },
                PROP_EXPANSION_ORDER: {
                    'type': 'integer',
                    'default': 1,
                    'minimum': 1
                }
            },
            'additionalProperties': False
        },
        'problems': ['eoh'],
        'depends': ['initial_state'],
        'defaults': {
            'initial_state': {
                'name': 'ZERO'
            }
        }
    }

    def __init__(self, operator, initial_state, evo_operator, operator_mode='paulis', evo_time=1, num_time_slices=1,
                 expansion_mode='trotter', expansion_order=1):
        self.validate(locals())
        super().__init__()
        self._operator = operator
        self._operator_mode = operator_mode
        self._initial_state = initial_state
        self._evo_operator = evo_operator
        self._evo_time = evo_time
        self._num_time_slices = num_time_slices
        self._expansion_mode = expansion_mode
        self._expansion_order = expansion_order
        self._ret = {}

    @classmethod
    def init_params(cls, params, algo_input):
        """
        Initialize via parameters dictionary and algorithm input instance
        Args:
            params: parameters dictionary
            algo_input: EnergyInput instance
        """
        if algo_input is None:
            raise AquaError("EnergyInput instance is required.")

        # For getting the extra operator, caller has to do something like: algo_input.add_aux_op(evo_op)
        operator = algo_input.qubit_op
        aux_ops = algo_input.aux_ops
        if aux_ops is None or len(aux_ops) != 1:
            raise AquaError("EnergyInput, a single aux op is required for evaluation.")
        evo_operator = aux_ops[0]
        if evo_operator is None:
            raise AquaError("EnergyInput, invalid aux op.")

        dynamics_params = params.get(QuantumAlgorithm.SECTION_KEY_ALGORITHM)
        operator_mode = dynamics_params.get(EOH.PROP_OPERATOR_MODE)
        evo_time = dynamics_params.get(EOH.PROP_EVO_TIME)
        num_time_slices = dynamics_params.get(EOH.PROP_NUM_TIME_SLICES)
        expansion_mode = dynamics_params.get(EOH.PROP_EXPANSION_MODE)
        expansion_order = dynamics_params.get(EOH.PROP_EXPANSION_ORDER)

        # Set up initial state, we need to add computed num qubits to params
        initial_state_params = params.get(QuantumAlgorithm.SECTION_KEY_INITIAL_STATE)
        initial_state_params['num_qubits'] = operator.num_qubits
        initial_state = get_pluggable_class(PluggableType.INITIAL_STATE,
                                            initial_state_params['name']).init_params(initial_state_params)

        return cls(operator, initial_state, evo_operator, operator_mode, evo_time, num_time_slices,
                   expansion_mode=expansion_mode,
                   expansion_order=expansion_order)

    def construct_circuit(self):
        """
        Construct the circuit.

        Returns:
            QuantumCircuit: the circuit.
        """
        quantum_registers = QuantumRegister(self._operator.num_qubits, name='q')
        qc = self._initial_state.construct_circuit('circuit', quantum_registers)

        qc += self._evo_operator.evolve(
            None,
            self._evo_time,
            'circuit',
            self._num_time_slices,
            quantum_registers=quantum_registers,
            expansion_mode=self._expansion_mode,
            expansion_order=self._expansion_order,
        )

        return qc

    def _run(self):
        qc = self.construct_circuit()
        qc_with_op = self._operator.construct_evaluation_circuit(self._operator_mode,
                                                                 qc, self._quantum_instance.backend)
        result = self._quantum_instance.execute(qc_with_op)
        self._ret['avg'], self._ret['std_dev'] = self._operator.evaluate_with_result(self._operator_mode,
                                                                                     qc_with_op, self._quantum_instance.backend, result)
        return self._ret
