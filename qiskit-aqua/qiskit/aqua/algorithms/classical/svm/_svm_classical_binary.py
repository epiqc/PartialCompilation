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
from sklearn.metrics.pairwise import rbf_kernel

from qiskit.aqua.algorithms.classical.svm import _SVM_Classical_ABC
from qiskit.aqua.utils import map_label_to_class_name, optimize_svm

logger = logging.getLogger(__name__)


class _SVM_Classical_Binary(_SVM_Classical_ABC):
    """
    the binary classifier
    """

    def construct_kernel_matrix(self, points_array, points_array2, gamma=None):
        return rbf_kernel(points_array, points_array2, gamma)

    def train(self, data, labels):
        """
        train the svm
        Args:
            data (dict): dictionary which maps each class to the points in the class
            class_labels (list): list of classes. For example: ['A', 'B']
        """
        labels = labels.astype(np.float)
        labels = labels * 2. - 1.
        kernel_matrix = self.construct_kernel_matrix(data, data, self.gamma)
        self._ret['kernel_matrix_training'] = kernel_matrix
        [alpha, b, support] = optimize_svm(kernel_matrix, labels)
        alphas = np.array([])
        svms = np.array([])
        yin = np.array([])
        for alphindex in range(len(support)):
            if support[alphindex]:
                alphas = np.vstack([alphas, alpha[alphindex]]) if alphas.size else alpha[alphindex]
                svms = np.vstack([svms, data[alphindex]]) if svms.size else data[alphindex]
                yin = np.vstack([yin, labels[alphindex]]
                                ) if yin.size else labels[alphindex]

        self._ret['svm'] = {}
        self._ret['svm']['alphas'] = alphas
        self._ret['svm']['bias'] = b
        self._ret['svm']['support_vectors'] = svms
        self._ret['svm']['yin'] = yin

    def test(self, data, labels):
        """
        test the svm
        Args:
            data (dict): dictionary which maps each class to the points in the class
            labels (list): list of classes. For example: ['A', 'B']
        """
        alphas = self._ret['svm']['alphas']
        bias = self._ret['svm']['bias']
        svms = self._ret['svm']['support_vectors']
        yin = self._ret['svm']['yin']

        kernel_matrix = self.construct_kernel_matrix(data, svms, self.gamma)
        self._ret['kernel_matrix_testing'] = kernel_matrix

        success_ratio = 0
        l = 0
        total_num_points = len(data)
        lsign = np.zeros(total_num_points)
        for tin in range(total_num_points):
            ltot = 0
            for sin in range(len(svms)):
                l = yin[sin] * alphas[sin] * kernel_matrix[tin][sin]
                ltot += l
            lsign[tin] = (np.sign(ltot + bias) + 1.) / 2.

            logger.debug("\n=============================================")
            logger.debug('classifying {}.'.format(data[tin]))
            logger.debug('Label should be {}.'.format(self.label_to_class[np.int(labels[tin])]))
            logger.debug('Predicted label is {}.'.format(self.label_to_class[np.int(lsign[tin])]))
            if np.int(labels[tin]) == np.int(lsign[tin]):
                logger.debug('CORRECT')
            else:
                logger.debug('INCORRECT')
            if lsign[tin] == labels[tin]:
                success_ratio += 1
        final_success_ratio = success_ratio / total_num_points
        logger.debug('Classification success is {} %% \n'.format(100 * final_success_ratio))
        self._ret['test_success_ratio'] = final_success_ratio
        self._ret['testing_accuracy'] = final_success_ratio

        return final_success_ratio

    def predict(self, data):
        """
        predict using the svm
        Args:
            data (numpy.ndarray): the points
        """
        alphas = self._ret['svm']['alphas']
        bias = self._ret['svm']['bias']
        svms = self._ret['svm']['support_vectors']
        yin = self._ret['svm']['yin']
        kernel_matrix = self.construct_kernel_matrix(data, svms, self.gamma)
        self._ret['kernel_matrix_prediction'] = kernel_matrix

        total_num_points = len(data)
        lsign = np.zeros(total_num_points)
        for tin in range(total_num_points):
            ltot = 0
            for sin in range(len(svms)):
                l = yin[sin] * alphas[sin] * kernel_matrix[tin][sin]
                ltot += l
            lsign[tin] = np.int((np.sign(ltot + bias) + 1.) / 2.)
        self._ret['predicted_labels'] = lsign
        return lsign

    def run(self):
        """
        put the train, test, predict together
        """

        self.train(self.training_dataset[0], self.training_dataset[1])
        if self.test_dataset is not None:
            self.test(self.test_dataset[0], self.test_dataset[1])

        if self.datapoints is not None:
            predicted_labels = self.predict(self.datapoints)
            predicted_classes = map_label_to_class_name(predicted_labels, self.label_to_class)
            self._ret['predicted_classes'] = predicted_classes
        return self._ret
