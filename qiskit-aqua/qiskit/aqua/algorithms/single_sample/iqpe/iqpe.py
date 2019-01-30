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
The Iterative Quantum Phase Estimation Algorithm.
See https://arxiv.org/abs/quant-ph/0610214
"""

import logging
import numpy as np

from qiskit import QuantumRegister, ClassicalRegister, QuantumCircuit
from qiskit.quantum_info import Pauli

from qiskit.aqua import Operator, AquaError
from qiskit.aqua import PluggableType, get_pluggable_class
from qiskit.aqua.utils import get_subsystem_density_matrix
from qiskit.aqua.algorithms import QuantumAlgorithm


logger = logging.getLogger(__name__)


class IQPE(QuantumAlgorithm):
    """
    The Iterative Quantum Phase Estimation algorithm.

    See https://arxiv.org/abs/quant-ph/0610214
    """

    PROP_NUM_TIME_SLICES = 'num_time_slices'
    PROP_EXPANSION_MODE = 'expansion_mode'
    PROP_EXPANSION_ORDER = 'expansion_order'
    PROP_NUM_ITERATIONS = 'num_iterations'

    CONFIGURATION = {
        'name': 'IQPE',
        'description': 'Iterative Quantum Phase Estimation for Quantum Systems',
        'input_schema': {
            '$schema': 'http://json-schema.org/schema#',
            'id': 'IQPE_schema',
            'type': 'object',
            'properties': {
                PROP_NUM_TIME_SLICES: {
                    'type': 'integer',
                    'default': 1,
                    'minimum': 1
                },
                PROP_EXPANSION_MODE: {
                    'type': 'string',
                    'default': 'suzuki',
                    'oneOf': [
                        {'enum': [
                            'suzuki',
                            'trotter'
                        ]}
                    ]
                },
                PROP_EXPANSION_ORDER: {
                    'type': 'integer',
                    'default': 2,
                    'minimum': 1
                },
                PROP_NUM_ITERATIONS: {
                    'type': 'integer',
                    'default': 1,
                    'minimum': 1
                }
            },
            'additionalProperties': False
        },
        'problems': ['energy'],
        'depends': ['initial_state'],
        'defaults': {
            'initial_state': {
                'name': 'ZERO'
            },
        }
    }

    def __init__(self, operator, state_in, num_time_slices=1, num_iterations=1,
                 expansion_mode='suzuki', expansion_order=2,
                 shallow_circuit_concat=False):
        """
        Constructor.

        Args:
            operator (Operator): the hamiltonian Operator object
            state_in (InitialState): the InitialState pluggable component representing the initial quantum state
            num_time_slices (int): the number of time slices
            num_iterations (int): the number of iterations
            expansion_mode (str): the expansion mode (trotter|suzuki)
            expansion_order (int): the suzuki expansion order
            shallow_circuit_concat (bool): indicate whether to use shallow (cheap) mode for circuit concatenation
        """
        self.validate(locals())
        super().__init__()
        self._operator = operator
        self._state_in = state_in
        self._num_time_slices = num_time_slices
        self._num_iterations = num_iterations
        self._expansion_mode = expansion_mode
        self._expansion_order = expansion_order
        self._shallow_circuit_concat = shallow_circuit_concat
        self._state_register = None
        self._ancillary_register = None
        self._pauli_list = None
        self._ret = {}
        self._setup()

    @classmethod
    def init_params(cls, params, algo_input):
        """
        Initialize via parameters dictionary and algorithm input instance.

        Args:
            params: parameters dictionary
            algo_input: EnergyInput instance
        """
        if algo_input is None:
            raise AquaError("EnergyInput instance is required.")

        operator = algo_input.qubit_op

        iqpe_params = params.get(QuantumAlgorithm.SECTION_KEY_ALGORITHM)
        num_time_slices = iqpe_params.get(IQPE.PROP_NUM_TIME_SLICES)
        expansion_mode = iqpe_params.get(IQPE.PROP_EXPANSION_MODE)
        expansion_order = iqpe_params.get(IQPE.PROP_EXPANSION_ORDER)
        num_iterations = iqpe_params.get(IQPE.PROP_NUM_ITERATIONS)

        # Set up initial state, we need to add computed num qubits to params
        init_state_params = params.get(QuantumAlgorithm.SECTION_KEY_INITIAL_STATE)
        init_state_params['num_qubits'] = operator.num_qubits
        init_state = get_pluggable_class(PluggableType.INITIAL_STATE,
                                         init_state_params['name']).init_params(init_state_params)

        return cls(operator, init_state, num_time_slices=num_time_slices, num_iterations=num_iterations,
                   expansion_mode=expansion_mode,
                   expansion_order=expansion_order)

    def _setup(self):
        self._pauli_list = self._operator.get_flat_pauli_list()
        self._ret['translation'] = sum([abs(p[0]) for p in self._pauli_list])
        self._ret['stretch'] = 0.5 / self._ret['translation']

        # translate the operator
        self._operator._simplify_paulis()
        translation_op = Operator([
            [
                self._ret['translation'],
                Pauli(
                    np.zeros(self._operator.num_qubits),
                    np.zeros(self._operator.num_qubits)
                )
            ]
        ])
        translation_op._simplify_paulis()
        self._operator += translation_op

        # stretch the operator
        for p in self._pauli_list:
            p[0] = p[0] * self._ret['stretch']

        if len(self._pauli_list) == 1:
            slice_pauli_list = self._pauli_list
        else:
            if self._expansion_mode == 'trotter':
                slice_pauli_list = self._pauli_list
            else:
                slice_pauli_list = Operator._suzuki_expansion_slice_pauli_list(self._pauli_list, 1, self._expansion_order)
        self._slice_pauli_list = slice_pauli_list

    def construct_circuit(self, k=None, omega=0):
        """Construct the kth iteration Quantum Phase Estimation circuit.

        For details of parameters, please see Fig. 2 in https://arxiv.org/pdf/quant-ph/0610214.pdf.

        Args:
            k (int): the iteration idx.
            omega (float): the feedback angle.
        Returns:
            QuantumCircuit: the quantum circuit per iteration
        """
        k = self._num_iterations if k is None else k
        a = QuantumRegister(1, name='a')
        q = QuantumRegister(self._operator.num_qubits, name='q')
        self._ancillary_register = a
        self._state_register = q
        qc = QuantumCircuit(q)
        qc += self._state_in.construct_circuit('circuit', q)
        # hadamard on a[0]
        qc.add_register(a)
        qc.u2(0, np.pi, a[0])
        # controlled-U
        qc_evolutions = Operator.construct_evolution_circuit(
            self._slice_pauli_list, -2 * np.pi, self._num_time_slices, q, a, unitary_power=2 ** (k - 1),
            shallow_slicing=self._shallow_circuit_concat
        )
        if self._shallow_circuit_concat:
            qc.data += qc_evolutions.data
        else:
            qc += qc_evolutions
        # global phase due to identity pauli
        qc.u1(2 * np.pi * self._ancilla_phase_coef * (2 ** (k - 1)), a[0])
        # rz on a[0]
        qc.u1(omega, a[0])
        # hadamard on a[0]
        qc.u2(0, np.pi, a[0])
        return qc

    def _estimate_phase_iteratively(self):
        """Iteratively construct the different order of controlled evolution circuit to carry out phase estimation."""
        self._ret['top_measurement_label'] = ''

        omega_coef = 0
        # k runs from the number of iterations back to 1
        for k in range(self._num_iterations, 0, -1):
            omega_coef /= 2
            qc = self.construct_circuit(k, -2 * np.pi * omega_coef)
            if self._quantum_instance.is_statevector:
                result = self._quantum_instance.execute(qc)
                complete_state_vec = result.get_statevector(qc)
                ancilla_density_mat = get_subsystem_density_matrix(
                    complete_state_vec,
                    range(self._operator.num_qubits)
                )
                ancilla_density_mat_diag = np.diag(ancilla_density_mat)
                max_amplitude = max(ancilla_density_mat_diag.min(), ancilla_density_mat_diag.max(), key=abs)
                x = np.where(ancilla_density_mat_diag == max_amplitude)[0][0]
            else:
                c = ClassicalRegister(1, name='c')
                qc.add_register(c)
                qc.barrier(self._ancillary_register)
                qc.measure(self._ancillary_register, c)
                measurements = self._quantum_instance.execute(qc).get_counts(qc)

                if '0' not in measurements:
                    if '1' in measurements:
                        x = 1
                    else:
                        raise RuntimeError('Unexpected measurement {}.'.format(measurements))
                else:
                    if '1' not in measurements:
                        x = 0
                    else:
                        x = 1 if measurements['1'] > measurements['0'] else 0
            self._ret['top_measurement_label'] = '{}{}'.format(x, self._ret['top_measurement_label'])
            omega_coef = omega_coef + x / 2
            logger.info('Reverse iteration {} of {} with measured bit {}'.format(k, self._num_iterations, x))
        return omega_coef

    def _compute_energy(self):
        # check for identify paulis to get its coef for applying global phase shift on ancilla later
        num_identities = 0
        self._pauli_list = self._operator.get_flat_pauli_list()
        for p in self._pauli_list:
            if np.all(np.logical_not(p[1].z)) and np.all(np.logical_not(p[1].x)):
                num_identities += 1
                if num_identities > 1:
                    raise RuntimeError('Multiple identity pauli terms are present.')
                self._ancilla_phase_coef = p[0].real if isinstance(p[0], complex) else p[0]

        self._ret['phase'] = self._estimate_phase_iteratively()
        self._ret['top_measurement_decimal'] = sum([t[0] * t[1] for t in zip(
            [1 / 2 ** p for p in range(1, self._num_iterations + 1)],
            [int(n) for n in self._ret['top_measurement_label']]
        )])
        self._ret['energy'] = self._ret['phase'] / self._ret['stretch'] - self._ret['translation']

    def _run(self):
        self._compute_energy()
        return self._ret
