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


import logging

from qiskit.aqua.algorithms import QuantumAlgorithm
from qiskit.aqua import AquaError, PluggableType, get_pluggable_class
from qiskit.aqua.algorithms.adaptive import VQE
from .varform import QAOAVarForm

logger = logging.getLogger(__name__)


class QAOA(VQE):
    """
    The Quantum Approximate Optimization Algorithm.

    See https://arxiv.org/abs/1411.4028
    """

    CONFIGURATION = {
        'name': 'QAOA.Variational',
        'description': 'Quantum Approximate Optimization Algorithm',
        'input_schema': {
            '$schema': 'http://json-schema.org/schema#',
            'id': 'qaoa_schema',
            'type': 'object',
            'properties': {
                'operator_mode': {
                    'type': 'string',
                    'default': 'matrix',
                    'oneOf': [
                        {'enum': ['matrix', 'paulis', 'grouped_paulis']}
                    ]
                },
                'p': {
                    'type': 'integer',
                    'default': 1,
                    'minimum': 1
                },
                'initial_point': {
                    'type': ['array', 'null'],
                    "items": {
                        "type": "number"
                    },
                    'default': None
                },
                'batch_mode': {
                    'type': 'boolean',
                    'default': False
                }
            },
            'additionalProperties': False
        },
        'problems': ['ising'],
        'depends': ['optimizer', 'initial_state'],
        'defaults': {
            'optimizer': {
                'name': 'COBYLA'
            },
            'initial_state': {
                'name': 'ZERO'
            }
        }
    }

    def __init__(self, operator, optimizer, p=1, initial_state=None, operator_mode='matrix', initial_point=None,
                 batch_mode=False, aux_operators=None):
        """
        Args:
            operator (Operator): Qubit operator
            operator_mode (str): operator mode, used for eval of operator
            p (int) : the integer parameter p as specified in https://arxiv.org/abs/1411.4028
            initial_state (InitialState): the initial state to prepend the QAOA circuit with
            optimizer (Optimizer) : the classical optimization algorithm.
            initial_point (numpy.ndarray) : optimizer initial point.
        """
        self.validate(locals())
        var_form = QAOAVarForm(operator, p, initial_state=initial_state)
        super().__init__(operator, var_form, optimizer,
                         operator_mode=operator_mode, initial_point=initial_point,
                         batch_mode=batch_mode, aux_operators=aux_operators)

    @classmethod
    def init_params(cls, params, algo_input):
        """
        Initialize via parameters dictionary and algorithm input instance

        Args:
            params (dict): parameters dictionary
            algo_input (EnergyInput): EnergyInput instance
        """
        if algo_input is None:
            raise AquaError("EnergyInput instance is required.")

        operator = algo_input.qubit_op

        qaoa_params = params.get(QuantumAlgorithm.SECTION_KEY_ALGORITHM)
        operator_mode = qaoa_params.get('operator_mode')
        p = qaoa_params.get('p')
        initial_point = qaoa_params.get('initial_point')
        batch_mode = qaoa_params.get('batch_mode')

        init_state_params = params.get(QuantumAlgorithm.SECTION_KEY_INITIAL_STATE)
        init_state_params['num_qubits'] = operator.num_qubits
        init_state = get_pluggable_class(PluggableType.INITIAL_STATE,
                                         init_state_params['name']).init_params(init_state_params)

        # Set up optimizer
        opt_params = params.get(QuantumAlgorithm.SECTION_KEY_OPTIMIZER)
        optimizer = get_pluggable_class(PluggableType.OPTIMIZER,
                                        opt_params['name']).init_params(opt_params)

        return cls(operator, optimizer, p=p, initial_state=init_state, operator_mode=operator_mode,
                   initial_point=initial_point, batch_mode=batch_mode,
                   aux_operators=algo_input.aux_ops)
