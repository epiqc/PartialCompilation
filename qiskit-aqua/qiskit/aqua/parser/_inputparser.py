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

from .base_parser import BaseParser
import json
import logging
import os
import copy
from qiskit.aqua import (local_pluggables_types,
                         PluggableType,
                         get_pluggable_configuration,
                         local_pluggables,
                         get_backends_from_provider)
from qiskit.aqua.aqua_error import AquaError
from .jsonschema import JSONSchema

logger = logging.getLogger(__name__)


class InputParser(BaseParser):
    """Aqua input Parser."""

    def __init__(self, input=None):
        """Create Parser object."""
        super().__init__(JSONSchema(os.path.join(os.path.dirname(__file__), 'input_schema.json')))
        if input is not None:
            if isinstance(input, dict):
                self._sections = input
            elif isinstance(input, str):
                self._filename = input
            else:
                raise AquaError("Invalid parser input type.")

        self._section_order = [JSONSchema.PROBLEM,
                               PluggableType.INPUT.value,
                               PluggableType.ALGORITHM.value]
        for pluggable_type in local_pluggables_types():
            if pluggable_type not in [PluggableType.INPUT, PluggableType.ALGORITHM]:
                self._section_order.append(pluggable_type.value)

        self._section_order.extend([JSONSchema.BACKEND, InputParser._UNKNOWN])

    def parse(self):
        """Parse the data."""
        if self._sections is None:
            if self._filename is None:
                raise AquaError("Missing input file")

            with open(self._filename) as json_file:
                self._sections = json.load(json_file)

        self._json_schema.update_backend_schema()
        self._json_schema.update_pluggable_input_schemas(self)
        self._update_algorithm_input_schema()
        self._sections = self._order_sections(self._sections)
        self._original_sections = copy.deepcopy(self._sections)

    def get_default_sections(self):
        return self._json_schema.get_default_sections()

    def _merge_default_values(self):
        section_names = self.get_section_names()
        if PluggableType.ALGORITHM.value in section_names:
            if JSONSchema.PROBLEM not in section_names:
                self.set_section(JSONSchema.PROBLEM)

        self._json_schema.update_backend_schema()
        self._json_schema.update_pluggable_input_schemas(self)
        self._update_algorithm_input_schema()
        self._merge_dependencies()

        # do not merge any pluggable that doesn't have name default in schema
        default_section_names = []
        pluggable_type_names = [pluggable_type.value for pluggable_type in local_pluggables_types()]
        for section_name in self.get_default_section_names():
            if section_name in pluggable_type_names:
                if self.get_property_default_value(section_name, JSONSchema.NAME) is not None:
                    default_section_names.append(section_name)
            else:
                default_section_names.append(section_name)

        section_names = set(self.get_section_names()) | set(default_section_names)
        for section_name in section_names:
            if section_name not in self._sections:
                self.set_section(section_name)

            new_properties = self.get_section_default_properties(section_name)
            if new_properties is not None:
                if self.section_is_text(section_name):
                    text = self.get_section_text(section_name)
                    if (text is None or len(text) == 0) and \
                            isinstance(new_properties, str) and \
                            len(new_properties) > 0 and \
                            text != new_properties:
                        self.set_section_data(section_name, new_properties)
                else:
                    properties = self.get_section_properties(section_name)
                    new_properties.update(properties)
                    self.set_section_properties(section_name, new_properties)

        self._sections = self._order_sections(self._sections)

    def validate_merge_defaults(self):
        super().validate_merge_defaults()
        self._validate_input_problem()

    def save_to_file(self, file_name):
        if file_name is None:
            raise AquaError('Missing file path')

        file_name = file_name.strip()
        if len(file_name) == 0:
            raise AquaError('Missing file path')

        with open(file_name, 'w') as f:
            print(json.dumps(self.get_sections(), sort_keys=True, indent=4), file=f)

    def delete_section(self, section_name):
        """
        Args:
            section_name (str): the name of the section, case insensitive
        """
        super().delete_section(section_name)
        self._update_algorithm_input_schema()

    def set_section_property(self, section_name, property_name, value):
        section_name = JSONSchema.format_section_name(section_name).lower()
        property_name = JSONSchema.format_property_name(property_name)
        value = self._json_schema.check_property_value(section_name, property_name, value)
        types = self.get_property_types(section_name, property_name)

        sections_temp = copy.deepcopy(self._sections)
        InputParser._set_section_property(sections_temp, section_name, property_name, value, types)
        msg = self._json_schema.validate_property(sections_temp, section_name, property_name)
        if msg is not None:
            raise AquaError("{}.{}: Value '{}': '{}'".format(section_name, property_name, value, msg))

        # check if this provider is loadable and valid
        if JSONSchema.BACKEND == section_name and property_name == JSONSchema.PROVIDER:
            get_backends_from_provider(value)

        InputParser._set_section_property(self._sections, section_name, property_name, value, types)
        if property_name == JSONSchema.NAME:
            if PluggableType.INPUT.value == section_name:
                self._update_algorithm_input_schema()
                # remove properties that are not valid for this section
                default_properties = self.get_section_default_properties(section_name)
                if isinstance(default_properties, dict):
                    properties = self.get_section_properties(section_name)
                    for property_name in list(properties.keys()):
                        if property_name != JSONSchema.NAME and property_name not in default_properties:
                            self.delete_section_property(section_name, property_name)
            elif JSONSchema.PROBLEM == section_name:
                self._update_algorithm_problem()
                self._update_input_problem()
            elif JSONSchema.BACKEND == section_name:
                self._json_schema.update_backend_schema()
            elif InputParser.is_pluggable_section(section_name):
                self._json_schema.update_pluggable_input_schemas(self)
                # remove properties that are not valid for this section
                default_properties = self.get_section_default_properties(section_name)
                if isinstance(default_properties, dict):
                    properties = self.get_section_properties(section_name)
                    for property_name in list(properties.keys()):
                        if property_name != JSONSchema.NAME and property_name not in default_properties:
                            self.delete_section_property(section_name, property_name)

                if section_name == PluggableType.ALGORITHM.value:
                    self._update_dependency_sections()

        self._sections = self._order_sections(self._sections)

    @staticmethod
    def get_input_problems(input_name):
        config = get_pluggable_configuration(PluggableType.INPUT, input_name)
        if 'problems' in config:
            return config['problems']

        return []

    def _update_algorithm_input_schema(self):
        # find algorithm input
        default_name = self.get_property_default_value(PluggableType.INPUT.value, JSONSchema.NAME)
        input_name = self.get_section_property(PluggableType.INPUT.value, JSONSchema.NAME, default_name)
        if input_name is None:
            # find the first valid input for the problem
            problem_name = self.get_section_property(JSONSchema.PROBLEM, JSONSchema.NAME)
            if problem_name is None:
                problem_name = self.get_property_default_value(JSONSchema.PROBLEM, JSONSchema.NAME)

            if problem_name is None:
                raise AquaError("No algorithm 'problem' section found on input.")

            for name in local_pluggables(PluggableType.INPUT):
                if problem_name in self.get_input_problems(name):
                    # set to the first input to solve the problem
                    input_name = name
                    break

        if input_name is None:
            # just remove fromm schema if none solves the problem
            if PluggableType.INPUT.value in self._json_schema.schema['properties']:
                del self._json_schema.schema['properties'][PluggableType.INPUT.value]
            return

        if default_name is None:
            default_name = input_name

        config = {}
        try:
            config = get_pluggable_configuration(PluggableType.INPUT, input_name)
        except:
            pass

        input_schema = config['input_schema'] if 'input_schema' in config else {}
        properties = input_schema['properties'] if 'properties' in input_schema else {}
        properties[JSONSchema.NAME] = {'type': 'string'}
        required = input_schema['required'] if 'required' in input_schema else []
        additionalProperties = input_schema['additionalProperties'] if 'additionalProperties' in input_schema else True
        if default_name is not None:
            properties[JSONSchema.NAME]['default'] = default_name
            required.append(JSONSchema.NAME)

        if PluggableType.INPUT.value not in self._json_schema.schema['properties']:
            self._json_schema.schema['properties'][PluggableType.INPUT.value] = {'type': 'object'}

        self._json_schema.schema['properties'][PluggableType.INPUT.value]['properties'] = properties
        self._json_schema.schema['properties'][PluggableType.INPUT.value]['required'] = required
        self._json_schema.schema['properties'][PluggableType.INPUT.value]['additionalProperties'] = additionalProperties

    def _validate_input_problem(self):
        input_name = self.get_section_property(PluggableType.INPUT.value, JSONSchema.NAME)
        if input_name is None:
            return

        problem_name = self.get_section_property(JSONSchema.PROBLEM, JSONSchema.NAME)
        if problem_name is None:
            problem_name = self.get_property_default_value(JSONSchema.PROBLEM, JSONSchema.NAME)

        if problem_name is None:
            raise AquaError("No algorithm 'problem' section found on input.")

        problems = InputParser.get_input_problems(input_name)
        if problem_name not in problems:
            raise AquaError("Problem: {} not in the list of problems: {} for input: {}.".format(problem_name, problems, input_name))

    def _update_input_problem(self):
        problem_name = self.get_section_property(JSONSchema.PROBLEM, JSONSchema.NAME)
        if problem_name is None:
            problem_name = self.get_property_default_value(JSONSchema.PROBLEM, JSONSchema.NAME)

        if problem_name is None:
            raise AquaError("No algorithm 'problem' section found on input.")

        input_name = self.get_section_property(PluggableType.INPUT.value, JSONSchema.NAME)
        if input_name is not None and problem_name in InputParser.get_input_problems(input_name):
            return

        for input_name in local_pluggables(PluggableType.INPUT):
            if problem_name in self.get_input_problems(input_name):
                # set to the first input to solve the problem
                self.set_section_property(PluggableType.INPUT.value, JSONSchema.NAME, input_name)
                return

        # no input solve this problem, remove section
        self.delete_section(PluggableType.INPUT.value)
    