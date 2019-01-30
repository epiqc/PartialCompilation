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

import sys
import os

algorithms_directory = os.path.dirname(os.path.realpath(__file__))
algorithms_directory = os.path.join(algorithms_directory, '../../..')
sys.path.insert(0, 'qiskit_aqua_ui')
sys.path.insert(0, 'qiskit_aqua')
sys.path.insert(0, algorithms_directory)

from qiskit_aqua_ui.browser.command_line import main

main()
