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

from .tensorproduct import tensorproduct
from .pauligraph import PauliGraph
from .jsonutils import convert_dict_to_json, convert_json_to_dict
from .random_matrix_generator import (random_unitary, random_h2_body,
                                      random_h1_body, random_hermitian,
                                      random_non_hermitian)
from .decimal_to_binary import decimal_to_binary
from .summarize_circuits import summarize_circuits
from .mct import mct
from .mcu1 import mcu1
from .mcu3 import mcu3
from .subsystem import get_subsystem_density_matrix, get_subsystems_counts
from .entangler_map import get_entangler_map, validate_entangler_map
from .dataset_helper import (get_feature_dimension, get_num_classes,
                             split_dataset_to_data_and_labels, map_label_to_class_name,
                             reduce_dim_to_via_pca)
from .qpsolver import optimize_svm
from .circuit_factory import CircuitFactory
from .run_circuits import compile_and_run_circuits, find_regs_by_name
from .circuit_cache import CircuitCache
from .boolean_logic import CNF, DNF


__all__ = ['tensorproduct',
           'PauliGraph',
           'convert_dict_to_json',
           'convert_json_to_dict',
           'random_unitary',
           'random_h2_body',
           'random_h1_body',
           'random_hermitian',
           'random_non_hermitian',
           'decimal_to_binary',
           'summarize_circuits',
           'mct',
           'mcu1',
           'mcu3',
           'get_subsystem_density_matrix',
           'get_subsystems_counts',
           'get_entangler_map',
           'validate_entangler_map',
           'get_feature_dimension',
           'get_num_classes',
           'split_dataset_to_data_and_labels',
           'map_label_to_class_name',
           'reduce_dim_to_via_pca',
           'optimize_svm',
           'CircuitFactory',
           'compile_and_run_circuits',
           'find_regs_by_name',
           'CircuitCache',
           'CNF',
           'DNF',
           ]
