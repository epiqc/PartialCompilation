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
The Amplitude Estimation Algorithm.
"""

import logging
from collections import OrderedDict
import numpy as np

from qiskit import ClassicalRegister
from qiskit.aqua import AquaError
from qiskit.aqua import PluggableType, get_pluggable_class
from qiskit.aqua.algorithms import QuantumAlgorithm
from qiskit.aqua.algorithms.single_sample import PhaseEstimation
from qiskit.aqua.components.iqfts import Standard
from .q_factory import QFactory

logger = logging.getLogger(__name__)


class AmplitudeEstimation(QuantumAlgorithm):
    """
    The Amplitude Estimation algorithm.
    """

    CONFIGURATION = {
        'name': 'AmplitudeEstimation',
        'description': 'Amplitude Estimation Algorithm',
        'input_schema': {
            '$schema': 'http://json-schema.org/schema#',
            'id': 'AmplitudeEstimation_schema',
            'type': 'object',
            'properties': {
                'num_eval_qubits': {
                    'type': 'integer',
                    'default': 5,
                    'minimum': 1
                }
            },
            'additionalProperties': False
        },
        'problems': ['uncertainty'],
        'depends': ['uncertainty_problem', 'uncertainty_model', 'iqft'],
        'defaults': {
            'uncertainty_model': {
                'name': 'NormalDistribution'
            },
            'iqft': {
                'name': 'STANDARD'
            }
        }
    }

    def __init__(self, num_eval_qubits, a_factory, q_factory=None, iqft=None):
        """
        Constructor.

        Args:
            num_eval_qubits (int): number of evaluation qubits
            a_factory (CircuitFactory): the CircuitFactory subclass object representing the problem unitary
            q_factory (CircuitFactory): the CircuitFactory subclass object representing an amplitude estimation sample (based on a_factory)
            iqft (IQFT): the Inverse Quantum Fourier Transform pluggable component, defaults to using a standard iqft when None
        """
        self.validate(locals())
        super().__init__()

        # get/construct A/Q operator
        self.a_factory = a_factory
        if q_factory is None:
            self.q_factory = QFactory(a_factory)
        else:
            self.q_factory = q_factory

        # get parameters
        self._m = num_eval_qubits
        self._M = 2 ** num_eval_qubits

        # determine number of ancillas
        self._num_ancillas = self.q_factory.required_ancillas_controlled()
        self._num_qubits = self.a_factory.num_target_qubits + self._m + self._num_ancillas

        if iqft is None:
            iqft = Standard(self._m)

        self._iqft = iqft
        self._circuit = None
        self._ret = {}

    @classmethod
    def init_params(cls, params, algo_input):
        """
        Initialize via parameters dictionary and algorithm input instance
        Args:
            params: parameters dictionary
            algo_input: Input instance
        """
        if algo_input is not None:
            raise AquaError("Input instance not supported.")

        ae_params = params.get(QuantumAlgorithm.SECTION_KEY_ALGORITHM)
        num_eval_qubits = ae_params.get('num_eval_qubits')

        # Set up uncertainty model and problem
        uncertainty_model_params = params.get(QuantumAlgorithm.SECTION_KEY_UNCERTAINTY_MODEL)
        uncertainty_model_params['num_target_qubits'] = num_eval_qubits
        uncertainty_model = get_pluggable_class(
            PluggableType.UNCERTAINTY_MODEL,
            uncertainty_model_params['name']).init_params(uncertainty_model_params)

        uncertainty_problem_params = params.get(QuantumAlgorithm.SECTION_KEY_UNCERTAINTY_PROBLEM)
        uncertainty_problem_params['uncertainty_model'] = uncertainty_model
        uncertainty_problem = get_pluggable_class(
            PluggableType.UNCERTAINTY_PROBLEM,
            uncertainty_problem_params['name']).init_params(uncertainty_problem_params)

        # Set up iqft, we need to add num qubits to params which is our num_ancillae bits here
        iqft_params = params.get(QuantumAlgorithm.SECTION_KEY_IQFT)
        iqft_params['num_qubits'] = num_eval_qubits
        iqft = get_pluggable_class(PluggableType.IQFT, iqft_params['name']).init_params(iqft_params)

        return cls(num_eval_qubits, uncertainty_problem, q_factory=None, iqft=iqft)

    def construct_circuit(self):
        """
        Construct the Amplitude Estimation quantum circuit.

        Returns:
            the QuantumCircuit object for the constructed circuit
        """
        pe = PhaseEstimation(None, None, self._iqft, num_ancillae=self._m,
                             state_in_circuit_factory=self.a_factory,
                             unitary_circuit_factory=self.q_factory)

        self._circuit = pe.construct_circuit()
        return self._circuit

    def _evaluate_statevector_results(self, probabilities):
        # map measured results to estimates
        y_probabilities = OrderedDict()
        for i, probability in enumerate(probabilities):
            b = "{0:b}".format(i).rjust(self._num_qubits, '0')[::-1]
            y = int(b[:self._m], 2)
            y_probabilities[y] = y_probabilities.get(y, 0) + probability

        a_probabilities = OrderedDict()
        for y, probability in y_probabilities.items():
            if y >= int(self._M / 2):
                y = self._M - y
            a = np.power(np.sin(y * np.pi / 2 ** self._m), 2)
            a_probabilities[a] = a_probabilities.get(a, 0) + probability

        return a_probabilities, y_probabilities

    def _run(self):

        # construct circuit
        self.construct_circuit()

        if self._quantum_instance.is_statevector:
            # run circuit on statevector simlator
            ret = self._quantum_instance.execute(self._circuit)
            state_vector = np.asarray([ret.get_statevector(self._circuit)])
            self._ret['statevector'] = state_vector

            # get state probabilities
            state_probabilities = np.real(state_vector.conj() * state_vector)[0]

            # evaluate results
            a_probabilities, y_probabilities = self._evaluate_statevector_results(state_probabilities)
        else:
            # run circuit on QASM simulator
            qc = self._circuit
            cr = ClassicalRegister(self._m)
            qc.add_register(cr)
            qc.measure([q for q in qc.qregs if q.name == 'a'][0], cr)
            ret = self._quantum_instance.execute(self._circuit)

            # get counts
            self._ret['counts'] = ret.get_counts()

            # construct probabilities
            y_probabilities = {}
            a_probabilities = {}
            shots = sum(ret.get_counts().values())
            for state, counts in ret.get_counts().items():
                y = int(state.replace(' ', '')[:self._m][::-1], 2)
                p = counts / shots
                y_probabilities[y] = p
                a = np.power(np.sin(y * np.pi / 2 ** self._m), 2)
                a_probabilities[a] = a_probabilities.get(a, 0.0) + p

        # construct a_items and y_items
        a_items = [(a, p) for (a, p) in a_probabilities.items() if p > 1e-6]
        y_items = [(y, p) for (y, p) in y_probabilities.items() if p > 1e-6]
        a_items = sorted(a_items)
        y_items = sorted(y_items)
        self._ret['a_items'] = a_items
        self._ret['y_items'] = y_items

        # map estimated values to original range and extract probabilities
        self._ret['mapped_values'] = [self.a_factory.value_to_estimation(a_item[0]) for a_item in self._ret['a_items']]
        self._ret['values'] = [a_item[0] for a_item in self._ret['a_items']]
        self._ret['y_values'] = [y_item[0] for y_item in y_items]
        self._ret['probabilities'] = [a_item[1] for a_item in self._ret['a_items']]
        self._ret['mapped_items'] = [(self._ret['mapped_values'][i], self._ret['probabilities'][i]) for i in range(len(self._ret['mapped_values']))]

        # determine most likely estimator
        self._ret['estimation'] = None
        self._ret['max_probability'] = 0
        for val, prob in self._ret['mapped_items']:
            if prob > self._ret['max_probability']:
                self._ret['max_probability'] = prob
                self._ret['estimation'] = val

        return self._ret
