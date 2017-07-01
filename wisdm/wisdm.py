# Copyright 2016 The TensorFlow Authors. All Rights Reserved.
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
# ==============================================================================

"""Functions for downloading and reading MNIST data."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections
import numpy as np
from six.moves import xrange  # pylint: disable=redefined-builtin
import csv
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import Imputer
from sklearn.preprocessing import StandardScaler
import scipy

Datasplit = collections.namedtuple('Datasplit', ['train', 'validation', 'test'])
Datasets = collections.namedtuple('Datasets',
                                  ['data', 'labels',
                                   'n_features', 'n_classes',
                                   'label_mapping'])

def read_data_sets(csv,
                   validation_size=500,
                   seed=None):

    wisdm = pd.read_csv(csv, comment='@', skip_blank_lines=True, header=None, na_values='?')

    # 1. extract labels of each sample, encode from strings to numbers
    wisdm_label_col = wisdm.shape[1] - 1
    labels_str = wisdm[wisdm_label_col].copy()
    le = LabelEncoder()
    labels = le.fit_transform(labels_str)

    # 2. delete the first two columns, and the label column (last)
    data = wisdm.drop([wisdm_label_col,0,1], axis=1)

    # 3. Impute empty values with mean
    im = Imputer(strategy="median")
    im.fit(data)
    data = im.transform(data)
    data = data.astype(np.float32)
    nf = data.shape[1]
    nc = len(le.classes_)

    print("Data file: ", csv, len(data), " samples,",
          nf, " features,", nc, " labels")

    validation_size = int(len(data) * 0.1)

    if not 0 <= validation_size <= len(data):
        raise ValueError(
            'Validation size should be between 0 and {}. Received: {}.'
            .format(len(data), validation_size))

    validation_data = data[:validation_size]
    validation_labels = labels[:validation_size]
    train_data = data[validation_size:]
    train_labels = labels[validation_size:]

    # Split using train_test_split
    X_train, X_test, y_train, y_test = train_test_split(
        train_data, train_labels, test_size=0.2)

    validation_data = X_train[:validation_size]
    validation_labels = y_train[:validation_size]
    X_train = X_train[validation_size:]
    y_train = y_train[validation_size:]

    print("training set: ", X_train.shape[0], "test set: ",  X_test.shape[0],
          "validation set: ", validation_data.shape[0])

    corpus_data = Datasplit(train=X_train,
                            validation=validation_data,
                            test=X_test)

    corpus_labels = Datasplit(train=y_train,
                              validation=validation_labels,
                              test=y_test)

    return Datasets(data = corpus_data,
                    labels = corpus_labels,label_mapping = le.classes_,
                    n_classes = nc, n_features = nf)
