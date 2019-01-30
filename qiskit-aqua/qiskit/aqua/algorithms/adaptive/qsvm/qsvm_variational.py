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

import logging

import numpy as np
from sklearn.utils import shuffle

from qiskit import ClassicalRegister, QuantumCircuit, QuantumRegister
from qiskit.aqua.algorithms import QuantumAlgorithm
from qiskit.aqua import AquaError, PluggableType, get_pluggable_class
from qiskit.aqua.algorithms.adaptive.qsvm import (cost_estimate, return_probabilities)
from qiskit.aqua.utils import (get_feature_dimension, map_label_to_class_name,
                               split_dataset_to_data_and_labels)

logger = logging.getLogger(__name__)


class QSVMVariational(QuantumAlgorithm):

    CONFIGURATION = {
        'name': 'QSVM.Variational',
        'description': 'QSVM_Variational Algorithm',
        'input_schema': {
            '$schema': 'http://json-schema.org/schema#',
            'id': 'SVM_Variational_schema',
            'type': 'object',
            'properties': {
                'override_SPSA_params': {
                    'type': 'boolean',
                    'default': True
                },
                'batch_mode': {
                    'type': 'boolean',
                    'default': False
                },
                'minibatch_size': {
                    'type': 'integer',
                    'default': -1
                }
            },
            'additionalProperties': False
        },
        'depends': ['optimizer', 'feature_map', 'variational_form'],
        'problems': ['svm_classification'],
        'defaults': {
            'optimizer': {
                'name': 'SPSA'
            },
            'feature_map': {
                'name': 'SecondOrderExpansion',
                'depth': 2
            },
            'variational_form': {
                'name': 'RYRZ',
                'depth': 3
            }
        }
    }

    def __init__(self, optimizer, feature_map, var_form, training_dataset,
                 test_dataset=None, datapoints=None, batch_mode=False,
                 minibatch_size=-1):
        """Initialize the object
        Args:
            training_dataset (dict): {'A': numpy.ndarray, 'B': numpy.ndarray, ...}
            test_dataset (dict): the same format as `training_dataset`
            datapoints (numpy.ndarray): NxD array, N is the number of data and D is data dimension
            optimizer (Optimizer): Optimizer instance
            feature_map (FeatureMap): FeatureMap instance
            var_form (VariationalForm): VariationalForm instance
            batch_mode (boolean): Batch mode for circuit compilation and execution
        Notes:
            We used `label` denotes numeric results and `class` means the name of that class (str).
        """
        self.validate(locals())
        super().__init__()
        if training_dataset is None:
            raise AquaError('Training dataset must be provided')

        self._training_dataset, self._class_to_label = split_dataset_to_data_and_labels(
            training_dataset)
        self._label_to_class = {label: class_name for class_name, label
                                in self._class_to_label.items()}
        self._num_classes = len(list(self._class_to_label.keys()))

        if test_dataset is not None:
            self._test_dataset = split_dataset_to_data_and_labels(test_dataset,
                                                                  self._class_to_label)
        else:
            self._test_dataset = test_dataset

        if datapoints is not None and not isinstance(datapoints, np.ndarray):
            datapoints = np.asarray(datapoints)
        self._datapoints = datapoints
        self._optimizer = optimizer
        self._feature_map = feature_map
        self._var_form = var_form
        self._num_qubits = self._feature_map.num_qubits
        self._optimizer.set_batch_mode(batch_mode)
        self._minibatch_size = minibatch_size
        self._ret = {}

    @classmethod
    def init_params(cls, params, algo_input):
        algo_params = params.get(QuantumAlgorithm.SECTION_KEY_ALGORITHM)
        override_spsa_params = algo_params.get('override_SPSA_params')
        batch_mode = algo_params.get('batch_mode')
        minibatch_size = algo_params.get('minibatch_size')

        # Set up optimizer
        opt_params = params.get(QuantumAlgorithm.SECTION_KEY_OPTIMIZER)
        # If SPSA then override SPSA params as reqd to our predetermined values
        if opt_params['name'] == 'SPSA' and override_spsa_params:
            opt_params['c0'] = 4.0
            opt_params['c1'] = 0.1
            opt_params['c2'] = 0.602
            opt_params['c3'] = 0.101
            opt_params['c4'] = 0.0
            opt_params['skip_calibration'] = True
        optimizer = get_pluggable_class(PluggableType.OPTIMIZER,
                                        opt_params['name']).init_params(opt_params)

        # Set up feature map
        fea_map_params = params.get(QuantumAlgorithm.SECTION_KEY_FEATURE_MAP)
        num_qubits = get_feature_dimension(algo_input.training_dataset)
        fea_map_params['num_qubits'] = num_qubits
        feature_map = get_pluggable_class(PluggableType.FEATURE_MAP,
                                          fea_map_params['name']).init_params(fea_map_params)

        # Set up variational form
        var_form_params = params.get(QuantumAlgorithm.SECTION_KEY_VAR_FORM)
        var_form_params['num_qubits'] = num_qubits
        var_form = get_pluggable_class(PluggableType.VARIATIONAL_FORM,
                                       var_form_params['name']).init_params(var_form_params)

        return cls(optimizer, feature_map, var_form, algo_input.training_dataset,
                   algo_input.test_dataset, algo_input.datapoints, batch_mode,
                   minibatch_size)

    def construct_circuit(self, x, theta, measurement=False):
        """
        Construct circuit based on data and parameters in variational form.

        Args:
            x (numpy.ndarray): 1-D array with D dimension
            theta ([numpy.ndarray]): list of 1-D array, parameters sets for variational form
            measurement (bool): flag to add measurement
        Returns:
            QuantumCircuit: the circuit
        """
        qr = QuantumRegister(self._num_qubits, name='q')
        cr = ClassicalRegister(self._num_qubits, name='c')
        qc = QuantumCircuit(qr, cr)
        qc += self._feature_map.construct_circuit(x, qr)
        qc += self._var_form.construct_circuit(theta, qr)

        if measurement:
            qc.barrier(qr)
            qc.measure(qr, cr)
        return qc

    def _cost_function(self, predicted_probs, labels):
        """
        Calculate cost of predicted probability of ground truth label based on
        cross entropy function.

        Args:
            predicted_probs (numpy.ndarray): NxK array
            labels (numpy.ndarray): Nx1 array
        Returns:
            float: cost
        """
        total_loss = cost_estimate(predicted_probs, labels)
        return total_loss

    def _get_prediction(self, data, theta):
        """
        Make prediction on data based on each theta.

        Args:
            data (numpy.ndarray): 2-D array, NxD, N data points, each with D dimension
            theta ([numpy.ndarray]): list of 1-D array, parameters sets for variational form
        Returns:
            numpy.ndarray or [numpy.ndarray]: list of NxK array
            numpy.ndarray or [numpy.ndarray]: list of Nx1 array
        """
        if self._quantum_instance.is_statevector:
            raise ValueError('Selected backend "{}" is not supported.'.format(
                self._quantum_instance.backend_name))

        predicted_probs = []
        predicted_labels = []
        circuits = {}
        circuit_id = 0

        num_theta_sets = len(theta) // self._var_form.num_parameters
        theta_sets = np.split(theta, num_theta_sets)

        for theta in theta_sets:
            for datum in data:
                circuit = self.construct_circuit(datum, theta, measurement=True)
                circuits[circuit_id] = circuit
                circuit_id += 1

        results = self._quantum_instance.execute(list(circuits.values()))

        circuit_id = 0
        predicted_probs = []
        predicted_labels = []
        for theta in theta_sets:
            counts = []
            for datum in data:
                counts.append(results.get_counts(circuits[circuit_id]))
                circuit_id += 1
            probs = return_probabilities(counts, self._num_classes)
            predicted_probs.append(probs)
            predicted_labels.append(np.argmax(probs, axis=1))

        if len(predicted_probs) == 1:
            predicted_probs = predicted_probs[0]
        if len(predicted_labels) == 1:
            predicted_labels = predicted_labels[0]

        return predicted_probs, predicted_labels

    # Breaks data into minibatches. Labels are optional, but will be broken into batches if included.
    def batch_data(self, data, labels=None, minibatch_size=-1):
        label_batches = None

        if minibatch_size > 0 and minibatch_size < len(data):
            batch_size = min(minibatch_size, len(data))
            if labels is not None:
                shuffled_samples, shuffled_labels = shuffle(data, labels, random_state=self.random)
                label_batches = np.array_split(shuffled_labels, batch_size)
            else:
                shuffled_samples = shuffle(data, random_state=self.random)
            batches = np.array_split(shuffled_samples, batch_size)
        else:
            batches = np.asarray([data])
            label_batches = np.asarray([labels])
        return batches, label_batches

    def train(self, data, labels, quantum_instance=None):
        """Train the models, and save results.

        Args:
            data (numpy.ndarray): NxD array, N is number of data and D is dimension
            labels (numpy.ndarray): Nx1 array, N is number of data
            quantum_instance (QuantumInstance): quantum backend with all setting
        """
        self._quantum_instance = self._quantum_instance if quantum_instance is None else quantum_instance
        batches, label_batches = self.batch_data(data, labels, self._minibatch_size)
        self.batch_num = 0

        def _cost_function_wrapper(theta):
            batch_num = self.batch_num % len(batches)
            predicted_probs, predicted_labels = self._get_prediction(batches[batch_num], theta)
            self.batch_num += 1
            total_cost = []
            if isinstance(predicted_probs, list):
                for predicted_prob in predicted_probs:
                    total_cost.append(self._cost_function(predicted_prob, label_batches[batch_num]))
            else:
                total_cost.append(self._cost_function(predicted_probs, label_batches[batch_num]))
            logger.debug('Intermediate batch cost: {:.2f}%'.format(sum(total_cost) * 100.0))
            return total_cost if len(total_cost) > 1 else total_cost[0]

        initial_theta = self.random.randn(self._var_form.num_parameters)

        theta_best, cost_final, _ = self._optimizer.optimize(initial_theta.shape[0],
                                                             _cost_function_wrapper,
                                                             initial_point=initial_theta)

        self._ret['opt_params'] = theta_best
        self._ret['training_loss'] = cost_final

    def test(self, data, labels, quantum_instance=None, minibatch_size=-1):
        """Predict the labels for the data, and test against with ground truth labels.

        Args:
            data (numpy.ndarray): NxD array, N is number of data and D is data dimension
            labels (numpy.ndarray): Nx1 array, N is number of data
            quantum_instance (QuantumInstance): quantum backend with all setting
        Returns:
            float: classification accuracy
        """
        # minibatch size defaults to setting in instance variable if not set
        minibatch_size = minibatch_size if minibatch_size > 0 else self._minibatch_size

        batches, label_batches = self.batch_data(data, labels, minibatch_size)
        self.batch_num = 0
        total_cost = 0
        total_correct = 0
        total_samples = 0

        self._quantum_instance = self._quantum_instance if quantum_instance is None else quantum_instance
        for batch, label_batch in zip(batches, label_batches):
            predicted_probs, predicted_labels = self._get_prediction(batch, self._ret['opt_params'])
            total_cost += self._cost_function(predicted_probs, label_batch)
            total_correct += np.sum((np.argmax(predicted_probs, axis=1) == label_batch))
            total_samples += label_batch.shape[0]
            int_accuracy = np.sum((np.argmax(predicted_probs, axis=1) == label_batch)) / label_batch.shape[0]
            logger.debug('Intermediate batch accuracy: {:.2f}%'.format(int_accuracy * 100.0))
        total_accuracy = total_correct / total_samples
        logger.info('Accuracy is {:.2f}%'.format(total_accuracy * 100.0))
        self._ret['testing_accuracy'] = total_accuracy
        self._ret['test_success_ratio'] = total_accuracy
        self._ret['testing_loss'] = total_cost / len(batches)
        return total_accuracy

    def predict(self, data, quantum_instance=None, minibatch_size=-1):
        """Predict the labels for the data.

        Args:
            data (numpy.ndarray): NxD array, N is number of data, D is data dimension
            quantum_instance (QuantumInstance): quantum backend with all setting
        Returns:
            list: for each data point, generates the predicted probability for each class
            list: for each data point, generates the predicted label (that with the highest prob)
        """

        # minibatch size defaults to setting in instance variable if not set
        minibatch_size = minibatch_size if minibatch_size > 0 else self._minibatch_size
        batches, _ = self.batch_data(data, None, minibatch_size)
        predicted_probs = None
        predicted_labels = None

        self._quantum_instance = self._quantum_instance if quantum_instance is None else quantum_instance
        for i, batch in enumerate(batches):
            if len(batches) > 0:
                logger.debug('Predicting batch {}'.format(i))
            batch_probs, batch_labels = self._get_prediction(batch, self._ret['opt_params'])
            if not predicted_probs and not predicted_labels:
                predicted_probs = batch_probs
                predicted_labels = batch_labels
            else:
                np.concatenate((predicted_probs, batch_probs))
                np.concatenate((predicted_labels, batch_labels))
        self._ret['predicted_probs'] = predicted_probs
        self._ret['predicted_labels'] = predicted_labels
        return predicted_probs, predicted_labels

    def _run(self):
        self.train(self._training_dataset[0], self._training_dataset[1])

        if self._test_dataset is not None:
            self.test(self._test_dataset[0], self._test_dataset[1])

        if self._datapoints is not None:
            predicted_probs, predicted_labels = self.predict(self._datapoints)
            self._ret['predicted_classes'] = map_label_to_class_name(predicted_labels,
                                                                     self._label_to_class)

        return self._ret

    @property
    def ret(self):
        return self._ret

    @ret.setter
    def ret(self, new_value):
        self._ret = new_value

    @property
    def label_to_class(self):
        return self._label_to_class

    @property
    def class_to_label(self):
        return self._class_to_label

    def load_model(self, file_path):
        model_npz = np.load(file_path)
        self._ret['opt_params'] = model_npz['opt_params']

    def save_model(self, file_path):
        model = {'opt_params': self._ret['opt_params']}
        np.savez(file_path, **model)

    @property
    def test_dataset(self):
        return self._test_dataset

    @property
    def train_dataset(self):
        return self._train_dataset

    @property
    def datapoints(self):
        return self._datapoints
