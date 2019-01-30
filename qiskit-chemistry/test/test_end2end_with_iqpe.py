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

import unittest

from parameterized import parameterized
import numpy as np
import qiskit
from qiskit.transpiler import PassManager
from qiskit.aqua.utils import decimal_to_binary
from qiskit.aqua import QuantumInstance
from qiskit.aqua.algorithms.single_sample import IQPE
from qiskit.aqua.algorithms.classical import ExactEigensolver
from qiskit.qobj import RunConfig
from test.common import QiskitAquaChemistryTestCase
from qiskit.chemistry.drivers import PySCFDriver, UnitsType
from qiskit.chemistry import FermionicOperator, QiskitChemistryError
from qiskit.chemistry.aqua_extensions.components.initial_states import HartreeFock


class TestIQPE(QiskitAquaChemistryTestCase):
    """IQPE tests."""

    @parameterized.expand([
        [0.5],
        [0.735],
        [1],
    ])
    def test_iqpe(self, distance):
        self.algorithm = 'IQPE'
        self.log.debug('Testing End-to-End with IQPE on H2 with '
                       'inter-atomic distance {}.'.format(distance))
        try:
            driver = PySCFDriver(atom='H .0 .0 .0; H .0 .0 {}'.format(distance),
                                 unit=UnitsType.ANGSTROM,
                                 charge=0,
                                 spin=0,
                                 basis='sto3g')
        except QiskitChemistryError:
            self.skipTest('PYSCF driver does not appear to be installed')
        self.molecule = driver.run()
        qubit_mapping = 'parity'
        fer_op = FermionicOperator(h1=self.molecule.one_body_integrals, h2=self.molecule.two_body_integrals)
        self.qubit_op = fer_op.mapping(map_type=qubit_mapping, threshold=1e-10).two_qubit_reduced_operator(2)

        exact_eigensolver = ExactEigensolver(self.qubit_op, k=1)
        results = exact_eigensolver.run()
        self.reference_energy = results['energy']
        self.log.debug('The exact ground state energy is: {}'.format(results['energy']))

        num_particles = self.molecule.num_alpha + self.molecule.num_beta
        two_qubit_reduction = True
        num_orbitals = self.qubit_op.num_qubits + (2 if two_qubit_reduction else 0)

        num_time_slices = 50
        num_iterations = 12
        state_in = HartreeFock(self.qubit_op.num_qubits, num_orbitals,
                               num_particles, qubit_mapping, two_qubit_reduction)
        iqpe = IQPE(self.qubit_op, state_in, num_time_slices, num_iterations,
                    expansion_mode='suzuki', expansion_order=2,
                    shallow_circuit_concat=True)
        backend = qiskit.Aer.get_backend('qasm_simulator')
        run_config = RunConfig(shots=100, max_credits=10, memory=False)
        quantum_instance = QuantumInstance(backend, run_config, pass_manager=PassManager())

        result = iqpe.run(quantum_instance)

        self.log.debug('top result str label:     {}'.format(result['top_measurement_label']))
        self.log.debug('top result in decimal:    {}'.format(result['top_measurement_decimal']))
        self.log.debug('stretch:                  {}'.format(result['stretch']))
        self.log.debug('translation:              {}'.format(result['translation']))
        self.log.debug('final energy from QPE:    {}'.format(result['energy']))
        self.log.debug('reference energy:         {}'.format(self.reference_energy))
        self.log.debug('ref energy (transformed): {}'.format(
            (self.reference_energy + result['translation']) * result['stretch'])
        )
        self.log.debug('ref binary str label:     {}'.format(decimal_to_binary(
            (self.reference_energy + result['translation']) * result['stretch'],
            max_num_digits=num_iterations + 3,
            fractional_part_only=True
        )))

        np.testing.assert_approx_equal(result['energy'], self.reference_energy, significant=2)


if __name__ == '__main__':
    unittest.main()
