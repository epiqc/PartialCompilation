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
"""Optimizer interface
"""

from qiskit.aqua import Pluggable
from abc import abstractmethod
from enum import IntEnum
import logging
import numpy as np

logger = logging.getLogger(__name__)


class Optimizer(Pluggable):
    """Base class for optimization algorithm."""

    class SupportLevel(IntEnum):
        not_supported = 0  # Does not support the corresponding parameter in optimize()
        ignored = 1        # Feature can be passed as non None but will be ignored
        supported = 2      # Feature is supported
        required = 3       # Feature is required and must be given, None is invalid

    """
    Base class for Optimizers.

        This method should initialize the module and its configuration, and
        use an exception if a component of the module is
        available.

        Args:
            configuration (dict): configuration dictionary
    """
    DEFAULT_CONFIGURATION = {
        'support_level': {
            'gradient': SupportLevel.not_supported,
            'bounds': SupportLevel.not_supported,
            'initial_point': SupportLevel.not_supported
        },
        'options': []
    }

    @abstractmethod
    def __init__(self):
        """Constructor.

        Initialize the optimization algorithm, setting the support
        level for _gradient_support_level, _bound_support_level,
        _initial_point_support_level, and empty options.

        """
        super().__init__()
        if 'support_level' not in self._configuration:
            self._configuration['support_level'] = self.DEFAULT_CONFIGURATION['support_level']
        if 'options' not in self._configuration:
            self._configuration['options'] = self.DEFAULT_CONFIGURATION['options']
        self._gradient_support_level = self._configuration['support_level']['gradient']
        self._bounds_support_level = self._configuration['support_level']['bounds']
        self._initial_point_support_level = self._configuration['support_level']['initial_point']
        self._options = {}
        self._batch_mode = False

    @classmethod
    def init_params(cls, params):
        """Initialize with a params dictionary.

        A dictionary of config params as per the configuration object. Some of these params get
        passed to scipy optimizers in an options dictionary. We can specify an options array of
        names in config dictionary to have the options dictionary automatically populated. All
        other config items, excluding name, will be passed to init_args

        Args:
            params (dict): configuration dict
        """
        logger.debug('init_params: {}'.format(params))
        args = {k: v for k, v in params.items() if k != 'name'}
        optimizer = cls(**args)
        return optimizer

    def set_options(self, **kwargs):
        """
        Sets or updates values in the options dictionary.

        The options dictionary may be used internally by a given optimizer to
        pass additional optional values for the underlying optimizer/optimization
        function used. The options dictionary may be initially populated with
        a set of key/values when the given optimizer is constructed.

        Args:
            kwargs (dict): options, given as name=value.
        """
        for name, value in kwargs.items():
            self._options[name] = value
        logger.debug('options: {}'.format(self._options))

    @staticmethod
    def gradient_num_diff(x_center, f, epsilon):
        """
        We compute the gradient with the numeric differentiation in the parallel way, around the point x_center.
        Args:
            x_center (ndarray): point around which we compute the gradient
            f (func): the function of which the gradient is to be computed.
            epsilon (float): the epsilon used in the numeric differentiation.
        Returns:
            grad: the gradient computed

        """
        forig = f(*((x_center,)))
        grad = np.zeros((len(x_center),), float)
        ei = np.zeros((len(x_center),), float)
        todos = []
        for k in range(len(x_center)):
            ei[k] = 1.0
            d = epsilon * ei
            todos.append(x_center + d)
            ei[k] = 0.0
        parallel_parameters = np.concatenate(todos)
        todos_results = f(parallel_parameters)
        for k in range(len(x_center)):
            grad[k] = (todos_results[k] - forig) / epsilon
        return grad

    @staticmethod
    def wrap_function(function, args):
        """
        Wrap the function to implicitly inject the args at the call of the function.
        Args:
            function (func): the target function
            args (tuple): the args to be injected

        """
        def function_wrapper(*wrapper_args):
            return function(*(wrapper_args + args))
        return function_wrapper

    @property
    def setting(self):
        ret = "Optimizer: {}\n".format(self._configuration['name'])
        params = ""
        for key, value in self.__dict__.items():
            if key != "_configuration" and key[0] == "_":
                params += "-- {}: {}\n".format(key[1:], value)
        ret += "{}".format(params)
        return ret

    @abstractmethod
    def optimize(self, num_vars, objective_function, gradient_function=None, variable_bounds=None, initial_point=None):
        """Perform optimization.

        Args:
            num_vars (int) : number of parameters to be optimized.
            objective_function (callable) : handle to a function that
                computes the objective function.
            gradient_function (callable) : handle to a function that
                computes the gradient of the objective function, or
                None if not available.
            variable_bounds (list[(float, float)]) : list of variable
                bounds, given as pairs (lower, upper). None means
                unbounded.
            initial_point (numpy.ndarray[float]) : initial point.

        Returns:
            point, value, nfev
               point: is a 1D numpy.ndarray[float] containing the solution
               value: is a float with the objective function value
               nfev: number of objective function calls made if available or None
        """

        if initial_point is not None and len(initial_point) != num_vars:
            raise ValueError('Initial point does not match dimension')
        if variable_bounds is not None and len(variable_bounds) != num_vars:
            raise ValueError('Variable bounds not match dimension')

        has_bounds = False
        if variable_bounds is not None:
            # If *any* value is *equal* in bounds array to None then the does *not* have bounds
            has_bounds = not np.any(np.equal(variable_bounds, None))

        if gradient_function is None and self.is_gradient_required:
            raise ValueError('Gradient is required but None given')
        if not has_bounds and self.is_bounds_required:
            raise ValueError('Variable bounds is required but None given')
        if initial_point is None and self.is_initial_point_required:
            raise ValueError('Initial point is required but None given')

        if gradient_function is not None and self.is_gradient_ignored:
            logger.debug('WARNING: {} does not support gradient function. It will be ignored.'.format(self.configuration['name']))
        if has_bounds and self.is_bounds_ignored:
            logger.debug('WARNING: {} does not support bounds. It will be ignored.'.format(self.configuration['name']))
        if initial_point is not None and self.is_initial_point_ignored:
            logger.debug('WARNING: {} does not support initial point. It will be ignored.'.format(self.configuration['name']))
        pass

    @property
    def gradient_support_level(self):
        return self._gradient_support_level

    @property
    def is_gradient_ignored(self):
        return self._gradient_support_level == self.SupportLevel.ignored

    @property
    def is_gradient_supported(self):
        return self._gradient_support_level != self.SupportLevel.not_supported

    @property
    def is_gradient_required(self):
        return self._gradient_support_level == self.SupportLevel.required

    @property
    def bounds_support_level(self):
        return self._bounds_support_level

    @property
    def is_bounds_ignored(self):
        return self._bounds_support_level == self.SupportLevel.ignored

    @property
    def is_bounds_supported(self):
        return self._bounds_support_level != self.SupportLevel.not_supported

    @property
    def is_bounds_required(self):
        return self._bounds_support_level == self.SupportLevel.required

    @property
    def initial_point_support_level(self):
        return self._initial_point_support_level

    @property
    def is_initial_point_ignored(self):
        return self._initial_point_support_level == self.SupportLevel.ignored

    @property
    def is_initial_point_supported(self):
        return self._initial_point_support_level != self.SupportLevel.not_supported

    @property
    def is_initial_point_required(self):
        return self._initial_point_support_level == self.SupportLevel.required

    def print_options(self):
        """Print algorithm-specific options."""
        for name in sorted(self._options):
            logger.debug('{:s} = {:s}'.format(name, str(self._options[name])))

    def set_batch_mode(self, mode):
        self._batch_mode = mode
