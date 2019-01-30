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
The Fixed Income Expected Value.
"""

from qiskit.aqua.components.uncertainty_problems import UncertaintyProblem
from qiskit.aqua.utils.circuit_utils import cry
import numpy as np


class FixedIncomeExpectedValue(UncertaintyProblem):
    """
    The Fixed Income Expected Value.

    Evaluates a fixed income asset with uncertain interest rates.
    """

    def __init__(self, uncertainty_model, A, b, cash_flow, c_approx, i_state=None, i_objective=None):
        """
        Constructor.

        Args:
            uncertainty_model:  multivariate distribution
            A: PCA matrix for delta_r (changes in interest rates)
            b: offset for interest rates (= initial interest rates)
            cash_flow: cash flow time series
            c_approx: approximation scaling factor
        """

        if i_state is None:
            i_state = list(range(uncertainty_model.num_target_qubits))
        if i_objective is None:
            i_objective = uncertainty_model.num_target_qubits

        self._params = {
            'i_state': i_state,
            'i_objective': i_objective
        }

        # get number of time steps
        self.T = len(cash_flow)

        # get dimension of uncertain model
        self.K = uncertainty_model.dimension

        # get total number of target qubits
        num_target_qubits = 1 + uncertainty_model.num_target_qubits

        # initialize parent class
        super().__init__(num_target_qubits)

        self.uncertainty_model = uncertainty_model
        self.cash_flow = cash_flow
        self.c_approx = c_approx
        self.A = A
        self.b = b

        # construct PCA-based cost function (1st order approximation):
        # c_t / (1 + A_t x + b_t)^{t+1} ~ c_t / (1 + b_t)^{t+1} - (t+1) c_t A_t / (1 + b_t)^{t+2} x = h + np.dot(g, x)
        self.h = 0
        self.g = np.zeros(self.K)
        for t in range(self.T):
            self.h += cash_flow[t] / pow(1 + b[t], (t+1))
            self.g += -1.0 * (t+1) * cash_flow[t] * A[t, :] / pow(1 + b[t], (t+2))

        # compute overall offset using lower bound for x (corresponding to x = min)
        self.offset = np.dot(uncertainty_model.low, self.g) + self.h

        # compute overall slope
        self.slope = np.zeros(uncertainty_model.num_target_qubits)
        index = 0
        for k in range(self.K):
            nk = uncertainty_model.num_qubits[k]
            for i in range(nk):
                self.slope[index] = pow(2.0, i) / (pow(2.0, nk) - 1) * (uncertainty_model.high[k] - uncertainty_model.low[k]) * self.g[k]
                index += 1

        # evaluate min and max values
        # for scaling to [0, 1] is then given by (V - min) / (max - min)
        self.min_value = self.offset + sum(self.slope)
        self.max_value = self.offset

        # reset offset / slope accordingly
        self.offset -= self.min_value
        self.offset /= (self.max_value - self.min_value)
        self.slope /= (self.max_value - self.min_value)

        # apply approximation scaling
        self.offset_angle = (self.offset - 1/2) * np.pi/2 * self.c_approx + np.pi/4
        self.slope_angle = self.slope * np.pi/2 * self.c_approx

    def value_to_estimation(self, value):
        estimator = value - 1/2
        estimator *= 2 / np.pi / self.c_approx
        estimator += 1/2
        estimator *= (self.max_value - self.min_value)
        estimator += self.min_value
        return estimator

    def required_ancillas(self):
        return 0

    def required_ancillas_controlled(self):
        return self.uncertainty_model.required_ancillas_controlled()

    def build(self, qc, q, q_ancillas=None, params=None):

        if params is None:
            params = self._params

        # get qubits
        q_objective = q[params['i_objective']]

        # apply uncertainty model
        self.uncertainty_model.build(qc, q, q_ancillas, params)

        # apply approximate payoff function
        qc.ry(2 * self.offset_angle, q_objective)
        for i in params['i_state']:
            cry(2 * self.slope_angle[i], q[i], q_objective, qc)
