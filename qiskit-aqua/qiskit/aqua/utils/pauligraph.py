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
For coloring Pauli Graph for transforming paulis into grouped Paulis
"""

import copy

import numpy as np


class PauliGraph(object):
    """Pauli Graph."""

    def __init__(self, paulis, mode="largest-degree"):
        self.nodes, self.weights = self._create_nodes(paulis)  # must be pauli list
        self._nqbits = self._get_nqbits()
        self.edges = self._create_edges()
        self._grouped_paulis = self._coloring(mode)

    def _create_nodes(self, paulis):
        """
        Check the validity of the pauli list and return immutable list as list of nodes.

        Args:
            paulis: list of [weight, Pauli object]

        Returns:
            Pauli object as immutable list
        """
        pauliOperators = [x[1] for x in paulis]
        pauliWeights = [x[0] for x in paulis]
        return tuple(pauliOperators), tuple(pauliWeights)  # fix their ordering

    def _get_nqbits(self):
        nqbits = self.nodes[0].numberofqubits
        for i in range(1, len(self.nodes)):
            assert nqbits == self.nodes[i].numberofqubits, "different number of qubits"
        return nqbits

    def _create_edges(self):
        """
        Create edges (i,j) if i and j is not commutable under Paulis.

        Returns:
            dictionary of graph connectivity with node index as key and list of neighbor as values
        """
        conv = {
            'I': 0,
            'X': 1,
            'Y': 2,
            'Z': 3
        }
        a = np.array([[conv[e] for e in reversed(n.to_label())] for n in self.nodes], dtype=np.int8)
        b = a[:, None]
        c = (((a * b) * (a - b)) == 0).all(axis=2)  # i and j are commutable with TPB if c[i, j] is True
        edges = {i: np.where(c[i] == False)[0] for i in range(len(self.nodes))}
        return edges

    def _coloring(self, mode="largest-degree"):
        if mode == "largest-degree":
            nodes = sorted(self.edges.keys(), key=lambda x: len(self.edges[x]), reverse=True)
            # -1 means not colored; 0 ... len(self.nodes)-1 is valid colored
            max_node = max(nodes)
            color = np.array([-1] * (max_node + 1))
            all_colors = np.arange(len(nodes))
            for i in nodes:
                neighbors = self.edges[i]
                color_neighbors = color[neighbors]
                color_neighbors = color_neighbors[color_neighbors >= 0]
                mask = np.ones(len(nodes), dtype=bool)
                mask[color_neighbors] = False
                color[i] = np.min(all_colors[mask])
            assert np.min(color[nodes]) >= 0, "Uncolored node exists!"

            # post-processing to grouped_paulis
            maxColor = np.max(color[nodes])  # the color used is 0, 1, 2, ..., maxColor
            temp_gp = []  # list of indices of grouped paulis
            for c in range(maxColor+1):  # maxColor is included
                temp_gp.append([i for i, icolor in enumerate(color) if icolor == c])

            # create _grouped_paulis as dictated in the operator.py
            gp = []
            for c in range(maxColor+1):  # maxColor is included
                # add all paulis
                gp.append([[self.weights[i], self.nodes[i]] for i in temp_gp[c]])

            # create header (measurement basis)
            for i, groupi in enumerate(gp):
                header = [0.0, copy.deepcopy(groupi[0][1])]
                for _, p in groupi:
                    for k in range(self._nqbits):
                        if p.z[k] or p.x[k]:
                            header[1].update_z(p.z[k], k)
                            header[1].update_x(p.x[k], k)
                gp[i].insert(0, header)
            return gp
        else:
            return self._coloring("largest-degree")  # this is the default implementation

    @property
    def grouped_paulis(self):
        """Getter of grouped Pauli list."""
        return self._grouped_paulis
