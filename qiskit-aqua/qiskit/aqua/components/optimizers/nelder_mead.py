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

from scipy.optimize import minimize

from qiskit.aqua.components.optimizers import Optimizer

logger = logging.getLogger(__name__)


class NELDER_MEAD(Optimizer):
    """Nelder-Mead algorithm.

    Uses scipy.optimize.minimize Nelder-Mead
    See https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.minimize.html
    """

    CONFIGURATION = {
        'name': 'NELDER_MEAD',
        'description': 'NELDER_MEAD Optimizer',
        'input_schema': {
            '$schema': 'http://json-schema.org/schema#',
            'id': 'nelder_mead_schema',
            'type': 'object',
            'properties': {
                'maxiter': {
                    'type': ['integer', 'null'],
                    'default': None
                },
                'maxfev': {
                    'type': ['integer', 'null'],
                    'default': 1000
                },
                'disp': {
                    'type': 'boolean',
                    'default': False
                },
                'xatol': {
                    'type': 'number',
                    'default': 0.0001
                },
                'tol': {
                    'type': ['number', 'null'],
                    'default': None
                },
                'adaptive': {
                    'type': 'boolean',
                    'default': False
                }
            },
            'additionalProperties': False
        },
        'support_level': {
            'gradient': Optimizer.SupportLevel.ignored,
            'bounds': Optimizer.SupportLevel.ignored,
            'initial_point': Optimizer.SupportLevel.required
        },
        'options': ['maxiter', 'maxfev', 'disp', 'xatol', 'adaptive'],
        'optimizer': ['local']
    }

    def __init__(self, maxiter=None, maxfev=1000, disp=False, xatol=0.0001, tol=None, adaptive=False):
        """
        Constructor.

        For details, please refer to
        https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.minimize.html.

        Args:
            maxiter (int): Maximum allowed number of iterations. If both maxiter and maxfev are set,
                           minimization will stop at the first reached.
            maxfev (int): Maximum allowed number of function evaluations. If both maxiter and
                          maxfev are set, minimization will stop at the first reached.
            disp (bool): Set to True to print convergence messages.
            xatol (float): Absolute error in xopt between iterations that is acceptable for convergence.
            tol (float or None): Tolerance for termination.
            adaptive (bool): Adapt algorithm parameters to dimensionality of problem.
        """
        self.validate(locals())
        super().__init__()
        for k, v in locals().items():
            if k in self._configuration['options']:
                self._options[k] = v
        self._tol = tol

    def optimize(self, num_vars, objective_function, gradient_function=None, variable_bounds=None, initial_point=None):
        super().optimize(num_vars, objective_function, gradient_function, variable_bounds, initial_point)

        res = minimize(objective_function, initial_point, tol=self._tol, method="Nelder-Mead", options=self._options)
        return res.x, res.fun, res.nfev
