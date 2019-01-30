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

from qiskit.chemistry.drivers import BaseDriver
import logging
from qiskit.chemistry import QMolecule, QiskitChemistryError
import os

logger = logging.getLogger(__name__)


class HDF5Driver(BaseDriver):
    """Python implementation of a hdf5 driver."""

    KEY_HDF5_INPUT = 'hdf5_input'

    CONFIGURATION = {
        "name": "HDF5",
        "description": "HDF5 Driver",
        "input_schema": {
            "$schema": "http://json-schema.org/schema#",
            "id": "hdf5_schema",
            "type": "object",
            "properties": {
                "hdf5_input": {
                    "type": "string",
                    "default": "molecule.hdf5"
                }
            },
            "additionalProperties": False
        }
    }

    def __init__(self, hdf5_input):
        """
        Initializer
        Args:
            hdf5_input (str): path to hdf5 file
        """
        self.validate(locals())
        super().__init__()
        self._hdf5_input = hdf5_input

    @classmethod
    def init_from_input(cls, section):
        """
        Initialize via section dictionary.

        Args:
            params (dict): section dictionary

        Returns:
            Driver: Driver object
        """
        if section is None or not isinstance(section, dict):
            raise QiskitChemistryError('Invalid or missing section {}'.format(section))

        kwargs = section
        logger.debug('init_from_input: {}'.format(kwargs))
        return cls(**kwargs)

    def run(self):
        hdf5_file = self._hdf5_input
        if self.work_path is not None and not os.path.isabs(hdf5_file):
            hdf5_file = os.path.abspath(os.path.join(self.work_path, hdf5_file))

        if not os.path.isfile(hdf5_file):
            raise LookupError('HDF5 file not found: {}'.format(hdf5_file))

        molecule = QMolecule(hdf5_file)
        molecule.load()
        return molecule
