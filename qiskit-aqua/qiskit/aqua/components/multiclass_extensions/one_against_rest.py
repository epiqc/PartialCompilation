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
from sklearn.utils.validation import _num_samples
from sklearn.preprocessing import LabelBinarizer

from qiskit.aqua.components.multiclass_extensions import MulticlassExtension

logger = logging.getLogger(__name__)


class OneAgainstRest(MulticlassExtension):
    """
      the multiclass extension based on the one-against-rest algorithm.
    """
    CONFIGURATION = {
        'name': 'OneAgainstRest',
        'description': 'OneAgainstRest extension',
        'input_schema': {
            '$schema': 'http://json-schema.org/schema#',
            'id': 'one_against_rest_schema',
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

    def train(self, X, y):
        """
        training multiple estimators each for distinguishing a pair of classes.
        Args:
            X (numpy.ndarray): input points
            y (numpy.ndarray): input labels
        """
        self.label_binarizer_ = LabelBinarizer(neg_label=0)
        Y = self.label_binarizer_.fit_transform(y)
        self.classes = self.label_binarizer_.classes_
        columns = (np.ravel(col) for col in Y.T)
        self.estimators = []
        for i, column in enumerate(columns):
            unique_y = np.unique(column)
            if len(unique_y) == 1:
                raise Exception("given all data points are assigned to the same class, the prediction would be boring.")
            estimator = self.estimator_cls(*self.params)
            estimator.fit(X, column)
            self.estimators.append(estimator)

    def test(self, x, y):
        """
        testing multiple estimators each for distinguishing a pair of classes.
        Args:
            x (numpy.ndarray): input points
            y (numpy.ndarray): input labels
        Returns:
            float: accuracy
        """
        A = self.predict(x)
        B = y
        l = len(A)
        diff = np.sum(A != B)
        logger.debug("%d out of %d are wrong" % (diff, l))
        return 1 - (diff * 1.0 / l)

    def predict(self, x):
        """
        applying multiple estimators for prediction
        Args:
            x (numpy.ndarray): NxD array
        Returns:
            numpy.ndarray: predicted labels, Nx1 array
        """
        n_samples = _num_samples(x)
        maxima = np.empty(n_samples, dtype=float)
        maxima.fill(-np.inf)
        argmaxima = np.zeros(n_samples, dtype=int)
        for i, e in enumerate(self.estimators):
            pred = np.ravel(e.decision_function(x))
            np.maximum(maxima, pred, out=maxima)
            argmaxima[maxima == pred] = i
        return self.classes[np.array(argmaxima.T)]
