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
This trial wavefunction is a Unitary Coupled-Cluster Single and Double excitations
variational form.
For more information, see https://arxiv.org/abs/1805.04340
"""

import logging
import numpy as np
from qiskit import QuantumRegister, QuantumCircuit

from qiskit.aqua import Operator
from qiskit.aqua.components.variational_forms import VariationalForm
from qiskit.chemistry.fermionic_operator import FermionicOperator

logger = logging.getLogger(__name__)


class UCCSD(VariationalForm):
    """
        This trial wavefunction is a Unitary Coupled-Cluster Single and Double excitations
        variational form.
        For more information, see https://arxiv.org/abs/1805.04340
    """

    CONFIGURATION = {
        'name': 'UCCSD',
        'description': 'UCCSD Variational Form',
        'input_schema': {
            '$schema': 'http://json-schema.org/schema#',
            'id': 'uccsd_schema',
            'type': 'object',
            'properties': {
                'depth': {
                    'type': 'integer',
                    'default': 1,
                    'minimum': 1
                },
                'num_orbitals': {
                    'type': 'integer',
                    'default': 4,
                    'minimum': 1
                },
                'num_particles': {
                    'type': 'integer',
                    'default': 2,
                    'minimum': 1
                },
                'active_occupied': {
                    'type': ['array', 'null'],
                    'default': None
                },
                'active_unoccupied': {
                    'type': ['array', 'null'],
                    'default': None
                },
                'qubit_mapping': {
                    'type': 'string',
                    'default': 'parity',
                    'oneOf': [
                        {'enum': ['jordan_wigner', 'parity', 'bravyi_kitaev']}
                    ]
                },
                'two_qubit_reduction': {
                    'type': 'boolean',
                    'default': True
                },
                'num_time_slices': {
                    'type': 'integer',
                    'default': 1,
                    'minimum': 1
                },
            },
            'additionalProperties': False
        }
    }

    def __init__(self, num_qubits, depth, num_orbitals, num_particles,
                 active_occupied=None, active_unoccupied=None, initial_state=None,
                 qubit_mapping='parity', two_qubit_reduction=False, num_time_slices=1,
                 cliffords=None, sq_list=None, tapering_values=None, symmetries=None):
        """Constructor.

        Args:
            num_orbitals (int): number of spin orbitals
            depth (int): number of replica of basic module
            num_particles (int): number of particles
            active_occupied (list): list of occupied orbitals to consider as active space
            active_unoccupied (list): list of unoccupied orbitals to consider as active space
            initial_state (InitialState): An initial state object.
            qubit_mapping (str): qubit mapping type.
            two_qubit_reduction (bool): two qubit reduction is applied or not.
            num_time_slices (int): parameters for dynamics.
            cliffords ([Operator]): list of unitary Clifford transformation
            sq_list ([int]): position of the single-qubit operators that anticommute
                            with the cliffords
            tapering_values ([int]): array of +/- 1 used to select the subspace. Length
                                    has to be equal to the length of cliffords and sq_list
            symmetries ([Pauli]): represent the Z2 symmetries
        """
        self.validate(locals())
        super().__init__()
        self._cliffords = cliffords
        self._sq_list = sq_list
        self._tapering_values = tapering_values
        self._symmetries = symmetries

        if self._cliffords is not None and self._sq_list is not None and \
                self._tapering_values is not None and self._symmetries is not None:
            self._qubit_tapering = True
        else:
            self._qubit_tapering = False

        self._num_qubits = num_orbitals if not two_qubit_reduction else num_orbitals - 2
        self._num_qubits = self._num_qubits if not self._qubit_tapering else self._num_qubits - len(sq_list)
        if self._num_qubits != num_qubits:
            raise ValueError('Computed num qubits {} does not match actual {}'
                             .format(self._num_qubits, num_qubits))
        self._depth = depth
        self._num_orbitals = num_orbitals
        self._num_particles = num_particles

        if self._num_particles > self._num_orbitals:
            raise ValueError('# of particles must be less than or equal to # of orbitals.')

        self._initial_state = initial_state
        self._qubit_mapping = qubit_mapping
        self._two_qubit_reduction = two_qubit_reduction
        self._num_time_slices = num_time_slices

        self._single_excitations, self._double_excitations = \
            UCCSD.compute_excitation_lists(num_particles, num_orbitals,
                                           active_occupied, active_unoccupied)

        self._hopping_ops, self._num_parameters = self._build_hopping_operators()
        self._bounds = [(-np.pi, np.pi) for _ in range(self._num_parameters)]

    def _build_hopping_operators(self):

        hopping_ops = {}
        for s_e_qubits in self._single_excitations:
            qubit_op = self._build_hopping_operator(s_e_qubits)
            hopping_ops['_'.join([str(x) for x in s_e_qubits])] = qubit_op

        for d_e_qubits in self._double_excitations:
            qubit_op = self._build_hopping_operator(d_e_qubits)
            hopping_ops['_'.join([str(x) for x in d_e_qubits])] = qubit_op

        # count the number of parameters
        num_parmaeters = len([1 for k, v in hopping_ops.items() if v is not None]) * self._depth
        return hopping_ops, num_parmaeters

    def _build_hopping_operator(self, index):

        def check_commutativity(op_1, op_2):
            com = op_1 * op_2 - op_2 * op_1
            com.zeros_coeff_elimination()
            return True if com.is_empty() else False

        two_d_zeros = np.zeros((self._num_orbitals, self._num_orbitals))
        four_d_zeros = np.zeros((self._num_orbitals, self._num_orbitals,
                                 self._num_orbitals, self._num_orbitals))

        dummpy_fer_op = FermionicOperator(h1=two_d_zeros, h2=four_d_zeros)
        h1 = two_d_zeros.copy()
        h2 = four_d_zeros.copy()
        if len(index) == 2:
            i, j = index
            h1[i, j] = 1.0
            h1[j, i] = -1.0
        elif len(index) == 4:
            i, j, k, m = index
            h2[i, j, k, m] = 1.0
            h2[m, k, j, i] = -1.0
        dummpy_fer_op.h1 = h1
        dummpy_fer_op.h2 = h2

        qubit_op = dummpy_fer_op.mapping(self._qubit_mapping)
        qubit_op = qubit_op.two_qubit_reduced_operator(
            self._num_particles) if self._two_qubit_reduction else qubit_op

        if self._qubit_tapering:
            for symmetry in self._symmetries:
                symmetry_op = Operator(paulis=[[1.0, symmetry]])
                symm_commuting = check_commutativity(symmetry_op, qubit_op)
                if not symm_commuting:
                    break

        if self._qubit_tapering:
            if symm_commuting:
                qubit_op = Operator.qubit_tapering(qubit_op, self._cliffords,
                                                   self._sq_list, self._tapering_values)
            else:
                qubit_op = None

        if qubit_op is None:
            logger.debug('excitation ({}) is skipped since it is not commuted '
                         'with symmetries'.format(','.join([str(x) for x in index])))
        return qubit_op

    def construct_circuit(self, parameters, q=None):
        """
        Construct the variational form, given its parameters.

        Args:
            parameters (numpy.ndarray): circuit parameters
            q (QuantumRegister): Quantum Register for the circuit.

        Returns:
            QuantumCircuit: a quantum circuit with given `parameters`

        Raises:
            ValueError: the number of parameters is incorrect.
        """
        if len(parameters) != self._num_parameters:
            raise ValueError('The number of parameters has to be {}'.format(self._num_parameters))

        if q is None:
            q = QuantumRegister(self._num_qubits, name='q')
        if self._initial_state is not None:
            circuit = self._initial_state.construct_circuit('circuit', q)
        else:
            circuit = QuantumCircuit(q)

        param_idx = 0

        for d in range(self._depth):
            for s_e_qubits in self._single_excitations:
                qubit_op = self._hopping_ops['_'.join([str(x) for x in s_e_qubits])]
                if qubit_op is not None:
                    circuit.extend(qubit_op.evolve(None, parameters[param_idx] * -1j,
                                                   'circuit', self._num_time_slices, q))
                    param_idx += 1

            for d_e_qubits in self._double_excitations:
                qubit_op = self._hopping_ops['_'.join([str(x) for x in d_e_qubits])]
                if qubit_op is not None:
                    circuit.extend(qubit_op.evolve(None, parameters[param_idx] * -1j,
                                                   'circuit', self._num_time_slices, q))
                    param_idx += 1

        return circuit

    @property
    def preferred_init_points(self):
        """Getter of preferred initial points based on the given initial state."""
        if self._initial_state is None:
            return None
        else:
            bitstr = self._initial_state.bitstr
            if bitstr is not None:
                return np.zeros(self._num_parameters, dtype=np.float)
            else:
                return None

    @staticmethod
    def compute_excitation_lists(num_particles, num_orbitals, active_occ_list=None,
                                 active_unocc_list=None, same_spin_doubles=True):
        """
        Computes single and double excitation lists

        Args:
            num_particles: Total number of particles
            num_orbitals:  Total number of spin orbitals
            active_occ_list: List of occupied orbitals to include, indices are
                             0 to n where n is num particles // 2
            active_unocc_list: List of unoccupied orbitals to include, indices are
                               0 to m where m is (num_orbitals - num particles) // 2
            same_spin_doubles: True to include alpha,alpha and beta,beta double excitations
                               as well as alpha,beta pairings. False includes only alpha,beta

        Returns:
            Single and double excitation lists
        """
        if num_particles < 2 or num_particles % 2 != 0:
            raise ValueError('Invalid number of particles {}'.format(num_particles))
        if num_orbitals < 4 or num_orbitals % 2 != 0:
            raise ValueError('Invalid number of orbitals {}'.format(num_orbitals))
        if num_orbitals <= num_particles:
            raise ValueError('No unoccupied orbitals')
        if active_occ_list is not None:
            active_occ_list = [i if i >= 0 else i + num_particles // 2 for i in active_occ_list]
            for i in active_occ_list:
                if i >= num_particles // 2:
                    raise ValueError('Invalid index {} in active active_occ_list {}'
                                     .format(i, active_occ_list))
        if active_unocc_list is not None:
            active_unocc_list = [i + num_particles // 2 if i >=
                                 0 else i + num_orbitals // 2 for i in active_unocc_list]
            for i in active_unocc_list:
                if i < 0 or i >= num_orbitals // 2:
                    raise ValueError('Invalid index {} in active active_unocc_list {}'
                                     .format(i, active_unocc_list))

        if active_occ_list is None or len(active_occ_list) <= 0:
            active_occ_list = [i for i in range(0, num_particles // 2)]

        if active_unocc_list is None or len(active_unocc_list) <= 0:
            active_unocc_list = [i for i in range(num_particles // 2, num_orbitals // 2)]

        single_excitations = []
        double_excitations = []

        logger.debug('active_occ_list {}'.format(active_occ_list))
        logger.debug('active_unocc_list {}'.format(active_unocc_list))

        beta_idx = num_orbitals // 2
        for occ_alpha in active_occ_list:
            for unocc_alpha in active_unocc_list:
                single_excitations.append([occ_alpha, unocc_alpha])

        for occ_beta in [i + beta_idx for i in active_occ_list]:
            for unocc_beta in [i + beta_idx for i in active_unocc_list]:
                single_excitations.append([occ_beta, unocc_beta])

        for occ_alpha in active_occ_list:
            for unocc_alpha in active_unocc_list:
                for occ_beta in [i + beta_idx for i in active_occ_list]:
                    for unocc_beta in [i + beta_idx for i in active_unocc_list]:
                        double_excitations.append([occ_alpha, unocc_alpha, occ_beta, unocc_beta])

        if same_spin_doubles and len(active_occ_list) > 1 and len(active_unocc_list) > 1:
            for i, occ_alpha in enumerate(active_occ_list[:-1]):
                for j, unocc_alpha in enumerate(active_unocc_list[:-1]):
                    for occ_alpha_1 in active_occ_list[i + 1:]:
                        for unocc_alpha_1 in active_unocc_list[j + 1:]:
                            double_excitations.append([occ_alpha, unocc_alpha,
                                                       occ_alpha_1, unocc_alpha_1])

            up_active_occ_list = [i + beta_idx for i in active_occ_list]
            up_active_unocc_list = [i + beta_idx for i in active_unocc_list]
            for i, occ_beta in enumerate(up_active_occ_list[:-1]):
                for j, unocc_beta in enumerate(up_active_unocc_list[:-1]):
                    for occ_beta_1 in up_active_occ_list[i + 1:]:
                        for unocc_beta_1 in up_active_unocc_list[j + 1:]:
                            double_excitations.append([occ_beta, unocc_beta,
                                                       occ_beta_1, unocc_beta_1])

        logger.debug('single_excitations {}'.format(single_excitations))
        logger.debug('double_excitations {}'.format(double_excitations))

        return single_excitations, double_excitations
