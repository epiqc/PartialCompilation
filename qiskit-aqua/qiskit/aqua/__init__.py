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

"""Algorithm discovery methods, Error and Base classes"""

from .aqua_error import AquaError
from ._discover import (PLUGGABLES_ENTRY_POINT,
                        PluggableType,
                        refresh_pluggables,
                        local_pluggables_types,
                        local_pluggables,
                        get_pluggable_class,
                        get_pluggable_configuration,
                        register_pluggable,
                        deregister_pluggable)
from .utils.backend_utils import (get_aer_backend,
                                  get_backends_from_provider,
                                  get_backend_from_provider,
                                  get_local_providers,
                                  register_ibmq_and_get_known_providers,
                                  get_provider_from_backend,
                                  enable_ibmq_account,
                                  disable_ibmq_account)
from .pluggable import Pluggable
from .utils.mct import mct
from .utils.mcu1 import mcu1
from .utils.mcu3 import mcu3
from .quantum_instance import QuantumInstance
from .operator import Operator
from .algorithms import QuantumAlgorithm
from ._aqua import run_algorithm, run_algorithm_to_json, build_algorithm_from_dict
from ._logging import (get_logging_level,
                       build_logging_config,
                       set_logging_config,
                       get_aqua_logging,
                       set_aqua_logging)

__version__ = '0.4.2'

__all__ = ['AquaError',
           'Pluggable',
           'Operator',
           'QuantumAlgorithm',
           'PLUGGABLES_ENTRY_POINT',
           'PluggableType',
           'refresh_pluggables',
           'QuantumInstance',
           'get_aer_backend',
           'get_backends_from_provider',
           'get_backend_from_provider',
           'get_local_providers',
           'register_ibmq_and_get_known_providers',
           'get_provider_from_backend',
           'enable_ibmq_account',
           'disable_ibmq_account',
           'mct',
           'mcu1',
           'mcu3',
           'local_pluggables_types',
           'local_pluggables',
           'get_pluggable_class',
           'get_pluggable_configuration',
           'register_pluggable',
           'deregister_pluggable',
           'run_algorithm',
           'run_algorithm_to_json',
           'build_algorithm_from_dict',
           'get_logging_level',
           'build_logging_config',
           'set_logging_config',
           'get_aqua_logging',
           'set_aqua_logging']
