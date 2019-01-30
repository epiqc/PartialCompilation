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
from test.common import QiskitAquaChemistryTestCase
from qiskit.chemistry.drivers import PySCFDriver, UnitsType
from qiskit.chemistry.core import Hamiltonian, TransformationType, QubitMappingType
from qiskit.chemistry import QiskitChemistryError


class TestCoreHamiltonianOrbReduce(QiskitAquaChemistryTestCase):
    """core/hamiltonian Driver tests."""

    def setUp(self):
        try:
            driver = PySCFDriver(atom='Li .0 .0 -0.8; H .0 .0 0.8',
                                 unit=UnitsType.ANGSTROM,
                                 charge=0,
                                 spin=0,
                                 basis='sto3g')
        except QiskitChemistryError:
            self.skipTest('PYSCF driver does not appear to be installed')
        self.qmolecule = driver.run()

    def _validate_vars(self, core, energy_shift=0.0, ph_energy_shift=0.0):
        self.assertAlmostEqual(core._hf_energy, -7.862, places=3)
        self.assertAlmostEqual(core._energy_shift, energy_shift)
        self.assertAlmostEqual(core._ph_energy_shift, ph_energy_shift)

    def _validate_info(self, core, num_particles=4, num_orbitals=12, actual_two_qubit_reduction=False):
        self.assertEqual(core.molecule_info, {'num_particles': num_particles,
                                              'num_orbitals': num_orbitals,
                                              'two_qubit_reduction': actual_two_qubit_reduction})

    def _validate_input_object(self, input_object, num_qubits=12, num_paulis=631):
        self.assertEqual(type(input_object).__name__, 'EnergyInput')
        self.assertIsNotNone(input_object.qubit_op)
        self.assertEqual(input_object.qubit_op.num_qubits, num_qubits)
        self.assertEqual(len(input_object.qubit_op.save_to_dict()['paulis']), num_paulis)

    def test_output(self):
        core = Hamiltonian(transformation=TransformationType.FULL,
                           qubit_mapping=QubitMappingType.JORDAN_WIGNER,
                           two_qubit_reduction=False,
                           freeze_core=False,
                           orbital_reduction=[])
        input_object = core.run(self.qmolecule)
        self._validate_vars(core)
        self._validate_info(core)
        self._validate_input_object(input_object)

    def test_parity(self):
        core = Hamiltonian(transformation=TransformationType.FULL,
                           qubit_mapping=QubitMappingType.PARITY,
                           two_qubit_reduction=True,
                           freeze_core=False,
                           orbital_reduction=[])
        input_object = core.run(self.qmolecule)
        self._validate_vars(core)
        self._validate_info(core, actual_two_qubit_reduction=True)
        self._validate_input_object(input_object, num_qubits=10)

    def test_freeze_core(self):
        core = Hamiltonian(transformation=TransformationType.FULL,
                           qubit_mapping=QubitMappingType.PARITY,
                           two_qubit_reduction=False,
                           freeze_core=True,
                           orbital_reduction=[])
        input_object = core.run(self.qmolecule)
        self._validate_vars(core, energy_shift=-7.7962196)
        self._validate_info(core, num_particles=2, num_orbitals=10)
        self._validate_input_object(input_object, num_qubits=10, num_paulis=276)

    def test_freeze_core_orb_reduction(self):
        core = Hamiltonian(transformation=TransformationType.FULL,
                           qubit_mapping=QubitMappingType.PARITY,
                           two_qubit_reduction=False,
                           freeze_core=True,
                           orbital_reduction=[-3, -2])
        input_object = core.run(self.qmolecule)
        self._validate_vars(core, energy_shift=-7.7962196)
        self._validate_info(core, num_particles=2, num_orbitals=6)
        self._validate_input_object(input_object, num_qubits=6, num_paulis=118)

    def test_freeze_core_all_reduction(self):
        core = Hamiltonian(transformation=TransformationType.FULL,
                           qubit_mapping=QubitMappingType.PARITY,
                           two_qubit_reduction=True,
                           freeze_core=True,
                           orbital_reduction=[-3, -2])
        input_object = core.run(self.qmolecule)
        self._validate_vars(core, energy_shift=-7.7962196)
        self._validate_info(core, num_particles=2, num_orbitals=6, actual_two_qubit_reduction=True)
        self._validate_input_object(input_object, num_qubits=4, num_paulis=100)

    def test_freeze_core_all_reduction_ph(self):
        core = Hamiltonian(transformation=TransformationType.PH,
                           qubit_mapping=QubitMappingType.PARITY,
                           two_qubit_reduction=True,
                           freeze_core=True,
                           orbital_reduction=[-2, -1])
        input_object = core.run(self.qmolecule)
        self._validate_vars(core, energy_shift=-7.7962196, ph_energy_shift=-1.05785247)
        self._validate_info(core, num_particles=2, num_orbitals=6, actual_two_qubit_reduction=True)
        self._validate_input_object(input_object, num_qubits=4, num_paulis=52)


if __name__ == '__main__':
    unittest.main()
