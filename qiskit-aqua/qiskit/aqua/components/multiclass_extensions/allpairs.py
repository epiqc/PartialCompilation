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
from sklearn.utils.multiclass import _ovr_decision_function

from qiskit.aqua.components.multiclass_extensions import MulticlassExtension

logger = logging.getLogger(__name__)


class AllPairs(MulticlassExtension):
    """
    The multiclass extension based on the all-pairs algorithm.
    """

    CONFIGURATION = {
        'name': 'AllPairs',
        'description': 'AllPairs extension',
        'input_schema': {
            '$schema': 'http://json-schema.org/schema#',
            'id': 'allpairs_schema',
            'type': 'object',
            'properties': {
            },
            'additionalProperties': False
        }
    }

    def __init__(self, estimator_cls, params=None):
        super().__init__()
        self.estimator_cls = estimator_cls
        self.params = params or []

    def train(self, x, y):
        """
        training multiple estimators each for distinguishing a pair of classes.
        Args:
            x (numpy.ndarray): input points
            y (numpy.ndarray): input labels
        """
        self.classes_ = np.unique(y)
        if len(self.classes_) == 1:
            raise ValueError(" can not be fit when only one class is present.")
        n_classes = self.classes_.shape[0]
        self.estimators = {}
        logger.info("Require {} estimators.".format(n_classes * (n_classes - 1) / 2))
        for i in range(n_classes):
            estimators_from_i = {}
            for j in range(i + 1, n_classes):
                estimator = self.estimator_cls(*self.params)
                cond = np.logical_or(y == i, y == j)
                indcond = np.arange(x.shape[0])[cond]
                x_filtered = x[indcond]
                y_filtered = y[indcond]
                y_filtered[y_filtered == i] = 0
                y_filtered[y_filtered == j] = 1
                estimator.fit(x_filtered, y_filtered)
                estimators_from_i[j] = estimator
            self.estimators[i] = estimators_from_i

    def test(self, x, y):
        """
        testing multiple estimators each for distinguishing a pair of classes.
        Args:
            X (numpy.ndarray): input points
            y (numpy.ndarray): input labels

        Returns:
            float: accuracy
        """
        A = self.predict(x)
        B = y
        l = len(A)
        diff = np.sum(A != B)
        logger.debug("%d out of %d are wrong" % (diff, l))
        return 1. - (diff * 1.0 / l)

    def predict(self, x):
        """
        applying multiple estimators for prediction
        Args:
            x (numpy.ndarray): NxD array
        Returns:
            numpy.ndarray: predicted labels, Nx1 array
        """
        predictions = []
        confidences = []
        for i in self.estimators:
            estimators_from_i = self.estimators[i]
            for j in estimators_from_i:
                estimator = estimators_from_i[j]
                confidence = np.ravel(estimator.decision_function(x))

                indices = (confidence > 0).astype(np.int)
                prediction = self.classes_[indices]

                predictions.append(prediction.reshape(-1, 1))
                confidences.append(confidence.reshape(-1, 1))

        predictions = np.hstack(predictions)
        confidences = np.hstack(confidences)
        y = _ovr_decision_function(predictions,
                                   confidences, len(self.classes_))
        return self.classes_[y.argmax(axis=1)]
