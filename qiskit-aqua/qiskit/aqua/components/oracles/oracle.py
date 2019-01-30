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
This module contains the definition of a base class for Oracle.
"""
from qiskit.aqua import Pluggable
from abc import abstractmethod


class Oracle(Pluggable):

    """
        Base class for oracles.

        This method should initialize the module and its configuration, and
        use an exception if a component of the module is
        available.

        Args:
            configuration (dict): configuration dictionary
    """

    @abstractmethod
    def __init__(self):
        super().__init__()

    @classmethod
    def init_params(cls, params):
        args = {k: v for k, v in params.items() if k != 'name'}
        return cls(**args)

    @property
    @abstractmethod
    def variable_register(self):
        raise NotImplementedError()

    @property
    @abstractmethod
    def ancillary_register(self):
        raise NotImplementedError()

    @property
    @abstractmethod
    def outcome_register(self):
        raise NotImplementedError()

    @abstractmethod
    def construct_circuit(self):
        """Construct the oracle circuit.

        Returns:
            A quantum circuit for the oracle.
        """
        raise NotImplementedError()

    @abstractmethod
    def evaluate_classically(self, assignment):
        """For an assignment, check the oracle result (True|False) classically
        Args:
            assignment (list of bools) : a particular variable assignment

        Returns:
            The True|False value of the oracle under the given assignment
        """
        raise NotImplementedError

    @abstractmethod
    def interpret_measurement(self, *args, **kwargs):
        raise NotImplementedError
