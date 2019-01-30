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
variational forms. Several types of commonly used ansatz.
"""

from qiskit.aqua import Pluggable
from abc import abstractmethod
from qiskit.aqua.utils import get_entangler_map, validate_entangler_map


class VariationalForm(Pluggable):

    """Base class for VariationalForms.

        This method should initialize the module and its configuration, and
        use an exception if a component of the module is
        available.

        Args:
            configuration (dict): configuration dictionary
    """

    @abstractmethod
    def __init__(self):
        super().__init__()
        self._num_parameters = 0
        self._bounds = list()
        pass

    @classmethod
    def init_params(cls, params):
        args = {k: v for k, v in params.items() if k != 'name'}
        return cls(**args)

    @abstractmethod
    def construct_circuit(self, parameters, q=None):
        """Construct the variational form, given its parameters.

        Args:
            parameters (numpy.ndarray[float]): circuit parameters.
            q (QuantumRegister): Quantum Register for the circuit.

        Returns:
            A quantum circuit.
        """
        raise NotImplementedError()

    @property
    def num_parameters(self):
        """Number of parameters of the variational form.

        Returns:
            An integer indicating the number of parameters.
        """
        return self._num_parameters

    @property
    def parameter_bounds(self):
        """Parameter bounds.

        Returns:
            A list of pairs indicating the bounds, as (lower,
            upper). None indicates an unbounded parameter in the
            corresponding direction. If None is returned, problem is
            fully unbounded.
        """
        return self._bounds

    @property
    def setting(self):
        ret = "Variational Form: {}\n".format(self._configuration['name'])
        params = ""
        for key, value in self.__dict__.items():
            if key != "_configuration" and key[0] == "_":
                params += "-- {}: {}\n".format(key[1:], value)
        ret += "{}".format(params)
        return ret

    @property
    def preferred_init_points(self):
        return None

    @staticmethod
    def get_entangler_map(map_type, num_qubits):
        return get_entangler_map(map_type, num_qubits)

    @staticmethod
    def validate_entangler_map(entangler_map, num_qubits):
        return validate_entangler_map(entangler_map, num_qubits)
