# -*- coding: utf-8 -*-

# Copyright 2018 IBM RESEARCH. All Rights Reserved.
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
from qiskit.converters import circuit_to_dag


def summarize_circuits(circuits):
    """Summarize circuits based on QuantumCircuit, and four metrics are summarized.

    Number of qubits and classical bits, and number of operations and depth of circuits.
    The average statistic is provided if multiple circuits are inputed.

    Args:
        circuits (QuantumCircuit or [QuantumCircuit]): the to-be-summarized circuits

    """
    if not isinstance(circuits, list):
        circuits = [circuits]
    ret = ""
    ret += "Submitting {} circuits.\n".format(len(circuits))
    ret += "============================================================================\n"
    stats = np.zeros(4)
    for i, circuit in enumerate(circuits):
        dag = circuit_to_dag(circuit)
        depth = dag.depth()
        width = dag.width()
        size = dag.size()
        classical_bits = dag.num_cbits()
        op_counts = dag.count_ops()
        stats[0] += width
        stats[1] += classical_bits
        stats[2] += size
        stats[3] += depth
        ret = ''.join([ret, "{}-th circuit: {} qubits, {} classical bits and {} operations with depth {}\n op_counts: {}\n".format(
            i, width, classical_bits, size, depth, op_counts)])
    if len(circuits) > 1:
        stats /= len(circuits)
        ret = ''.join([ret, "Average: {:.2f} qubits, {:.2f} classical bits and {:.2f} operations with depth {:.2f}\n".format(
            stats[0], stats[1], stats[2], stats[3])])
    ret += "============================================================================\n"
    return ret
