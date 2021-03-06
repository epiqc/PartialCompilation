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
The Univariate Log-Normal Distribution.
"""

from scipy.stats.distributions import lognorm
from qiskit.aqua.components.random_distributions.univariate_distribution import UnivariateDistribution
import numpy as np


class LogNormalDistribution(UnivariateDistribution):
    """
    The Univariate Log-Normal Distribution.
    """

    CONFIGURATION = {
        'name': 'LogNormalDistribution',
        'description': 'Log-Normal Distribution',
        'input_schema': {
            '$schema': 'http://json-schema.org/schema#',
            'id': 'LogNormalDistribution_schema',
            'type': 'object',
            'properties': {
                'num_target_qubits': {
                    'type': 'integer',
                    'default': 2,
                },
                'mu': {
                    'type': 'number',
                    'default': 0,
                },
                'sigma': {
                    'type': 'number',
                    'default': 1,
                },
                'low': {
                    'type': 'number',
                    'default': 0,
                },
                'high': {
                    'type': 'number',
                    'default': 3,
                },
            },
            'additionalProperties': False
        }
    }

    def __init__(self, num_target_qubits, mu=0, sigma=1, low=0, high=1):
        self.validate(locals())
        probabilities, _ = UnivariateDistribution.\
        pdf_to_probabilities(lambda x: lognorm.pdf(x, s=sigma, scale=np.exp(mu)), low, high, 2 ** num_target_qubits)
        super().__init__(num_target_qubits, probabilities, low, high)
