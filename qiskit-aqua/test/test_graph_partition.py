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

import numpy as np

from test.common import QiskitAquaTestCase
from qiskit.aqua import get_aer_backend

from qiskit.aqua import run_algorithm
from qiskit.aqua.input import EnergyInput
from qiskit.aqua.translators.ising import graphpartition
from qiskit.aqua.algorithms import ExactEigensolver


class TestGraphPartition(QiskitAquaTestCase):
    """Cplex Ising tests."""

    def setUp(self):
        super().setUp()
        np.random.seed(100)
        self.num_nodes = 4
        self.w = graphpartition.random_graph(self.num_nodes, edge_prob=0.8, weight_range=10)
        self.qubit_op, self.offset = graphpartition.get_graphpartition_qubitops(self.w)
        self.algo_input = EnergyInput(self.qubit_op)

    def brute_force(self):
        # use the brute-force way to generate the oracle
        def bitfield(n, L):
            result = np.binary_repr(n, L)
            return [int(digit) for digit in result]  # [2:] to chop off the "0b" part

        L = self.num_nodes
        max = 2**L
        minimal_v = np.inf
        for i in range(max):
            cur = bitfield(i, L)

            how_many_nonzero = np.count_nonzero(cur)
            if how_many_nonzero * 2 != L:  # not balanced
                continue

            cur_v = graphpartition.objective_value(np.array(cur), self.w)
            if cur_v < minimal_v:
                minimal_v = cur_v
        return minimal_v

    def test_graph_partition(self):
        params = {
            'problem': {'name': 'ising'},
            'algorithm': {'name': 'ExactEigensolver'}
        }
        result = run_algorithm(params, self.algo_input)
        x = graphpartition.sample_most_likely(result['eigvecs'][0])
        # check against the oracle
        ising_sol = graphpartition.get_graph_solution(x)
        np.testing.assert_array_equal(ising_sol, [0, 1, 0, 1])
        oracle = self.brute_force()
        self.assertEqual(graphpartition.objective_value(x, self.w), oracle)

    def test_graph_partition_direct(self):
        algo = ExactEigensolver(self.algo_input.qubit_op, k=1, aux_operators=[])
        result = algo.run()
        x = graphpartition.sample_most_likely(result['eigvecs'][0])
        # check against the oracle
        ising_sol = graphpartition.get_graph_solution(x)
        np.testing.assert_array_equal(ising_sol, [0, 1, 0, 1])
        oracle = self.brute_force()
        self.assertEqual(graphpartition.objective_value(x, self.w), oracle)

    def test_graph_partition_vqe(self):
        algorithm_cfg = {
            'name': 'VQE',
            'operator_mode': 'matrix',
            'batch_mode': True
        }

        optimizer_cfg = {
            'name': 'SPSA',
            'max_trials': 300
        }

        var_form_cfg = {
            'name': 'RY',
            'depth': 5,
            'entanglement': 'linear'
        }

        params = {
            'problem': {'name': 'ising', 'random_seed': 10598},
            'algorithm': algorithm_cfg,
            'optimizer': optimizer_cfg,
            'variational_form': var_form_cfg
        }
        backend = get_aer_backend('statevector_simulator')
        result = run_algorithm(params, self.algo_input, backend=backend)
        x = graphpartition.sample_most_likely(result['eigvecs'][0])
        # check against the oracle
        ising_sol = graphpartition.get_graph_solution(x)
        np.testing.assert_array_equal(ising_sol, [1, 0, 0, 1])
        oracle = self.brute_force()
        self.assertEqual(graphpartition.objective_value(x, self.w), oracle)
