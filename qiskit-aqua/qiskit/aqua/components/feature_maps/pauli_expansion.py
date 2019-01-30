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
This module contains the definition of a base class for
feature map. Several types of commonly used approaches.
"""

from collections import OrderedDict
import copy
import itertools
import logging

import numpy as np
from qiskit import QuantumCircuit, QuantumRegister
from qiskit.quantum_info import Pauli
from qiskit.qasm import pi
from sympy.core.numbers import NaN, Float

from qiskit.aqua import Operator
from qiskit.aqua.components.feature_maps import FeatureMap, self_product

logger = logging.getLogger(__name__)


class PauliExpansion(FeatureMap):
    """
    Mapping data with the second order expansion followed by entangling gates.
    Refer to https://arxiv.org/pdf/1804.11326.pdf for details.
    """

    CONFIGURATION = {
        'name': 'PauliExpansion',
        'description': 'Pauli expansion for feature map (any order)',
        'input_schema': {
            '$schema': 'http://json-schema.org/schema#',
            'id': 'Pauli_Expansion_schema',
            'type': 'object',
            'properties': {
                'depth': {
                    'type': 'integer',
                    'default': 2,
                    'minimum': 1
                },
                'entangler_map': {
                    'type': ['object', 'null'],
                    'default': None
                },
                'entanglement': {
                    'type': 'string',
                    'default': 'full',
                    'oneOf': [
                        {'enum': ['full', 'linear']}
                    ]
                },
                'paulis': {
                    'type': ['array'],
                    "items": {
                        "type": "string"
                    },
                    'default': ['Z', 'ZZ']
                }
            },
            'additionalProperties': False
        }
    }

    def __init__(self, num_qubits, depth=2, entangler_map=None,
                 entanglement='full', paulis=['Z', 'ZZ'], data_map_func=self_product):
        """Constructor.

        Args:
            num_qubits (int): number of qubits
            depth (int): the number of repeated circuits
            entangler_map (dict): describe the connectivity of qubits
            entanglement (str): ['full', 'linear'], generate the qubit connectivitiy by predefined
                                topology
            paulis (str): a comma-seperated string for to-be-used paulis
            data_map_func (Callable): a mapping function for data x
        """
        self.validate(locals())
        super().__init__()
        self._num_qubits = num_qubits
        self._depth = depth
        if entangler_map is None:
            self._entangler_map = self.get_entangler_map(entanglement, num_qubits)
        else:
            self._entangler_map = self.validate_entangler_map(entangler_map, num_qubits)

        self._pauli_strings = self._build_subset_paulis_string(paulis)
        self._data_map_func = data_map_func

        self._magic_num = np.nan
        self._param_pos = OrderedDict()
        self._circuit_template = self._build_circuit_template()

    def _build_subset_paulis_string(self, paulis):
        # fill out the paulis to the number of qubits
        temp_paulis = []
        for pauli in paulis:
            len_pauli = len(pauli)
            for possible_pauli_idx in itertools.combinations(range(self._num_qubits), len_pauli):
                string_temp = ['I'] * self._num_qubits
                for idx in range(len(possible_pauli_idx)):
                    string_temp[-possible_pauli_idx[idx] - 1] = pauli[-idx - 1]
                temp_paulis.append(''.join(string_temp))
        # clean up string that can not be entangled.
        final_paulis = []
        for pauli in temp_paulis:
            where_z = np.where(np.asarray(list(pauli[::-1])) != 'I')[0]
            if len(where_z) == 1:
                final_paulis.append(pauli)
            else:
                is_valid = True
                for src, targ in itertools.combinations(where_z, 2):
                    if src not in self._entangler_map:
                        is_valid = False
                        break
                    else:
                        if targ not in self._entangler_map[src]:
                            is_valid = False
                            break
                if is_valid:
                    final_paulis.append(pauli)
                else:
                    logger.warning("Due to the limited entangler_map,"
                                   " {} is skipped.".format(pauli))
        logger.info("Pauli terms include: {}".format(final_paulis))
        return final_paulis

    def _build_circuit_template(self):
        x = np.asarray([self._magic_num] * self._num_qubits)
        qr = QuantumRegister(self._num_qubits, name='q')
        qc = self.construct_circuit(x, qr)

        for index in range(len(qc.data)):
            gate_param = qc.data[index].params
            param_sub_pos = []
            for x in range(len(gate_param)):
                if isinstance(gate_param[x], NaN):
                    param_sub_pos.append(x)
            if param_sub_pos != []:
                self._param_pos[index] = param_sub_pos
        return qc

    def _extract_data_for_rotation(self, pauli, x):
        where_non_i = np.where(np.asarray(list(pauli[::-1])) != 'I')[0]
        return x[where_non_i]

    def _construct_circuit_with_template(self, x):
        coeffs = [self._data_map_func(self._extract_data_for_rotation(pauli, x))
                  for pauli in self._pauli_strings] * self._depth
        qc = copy.deepcopy(self._circuit_template)
        data_idx = 0
        for key, value in self._param_pos.items():
            new_param = coeffs[data_idx]
            for pos in value:
                qc.data[key].params[pos] = Float(2. * new_param)  # rotation angle is 2x
            data_idx += 1
        return qc

    def construct_circuit(self, x, qr=None, inverse=False):
        """
        Construct the second order expansion based on given data.

        Args:
            x (numpy.ndarray): 1-D to-be-transformed data.
            qr (QauntumRegister): the QuantumRegister object for the circuit, if None,
                                  generate new registers with name q.
            inverse (bool): whether or not inverse the circuit

        Returns:
            QuantumCircuit: a quantum circuit transform data x.
        """
        if not isinstance(x, np.ndarray):
            raise TypeError("x must be numpy array.")
        if x.ndim != 1:
            raise ValueError("x must be 1-D array.")
        if x.shape[0] != self._num_qubits:
            raise ValueError("number of qubits and data dimension must be the same.")

        if qr is None:
            qc = self._construct_circuit_with_template(x)
        else:
            qc = QuantumCircuit(qr)
            for _ in range(self._depth):
                for i in range(self._num_qubits):
                    qc.u2(0, pi, qr[i])
                for pauli in self._pauli_strings:
                    coeff = self._data_map_func(self._extract_data_for_rotation(pauli, x))
                    p = Pauli.from_label(pauli)
                    qc += Operator.construct_evolution_circuit([[coeff, p]], 1, 1, qr)

        if inverse:
            qc.data = [gate.inverse() for gate in reversed(qc.data)]

        return qc
