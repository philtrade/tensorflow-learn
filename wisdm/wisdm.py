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

import gzip

import numpy as np
from six.moves import xrange  # pylint: disable=redefined-builtin

from tensorflow.contrib.learn.python.learn.datasets import base
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import random_seed
import csv
import pandas as pd
from tensorflow.python.platform import gfile
from sklearn.model_selection import train_test_split
import scipy

class DataSet(object):

  def __init__(self,
               data,
               labels,
               dtype=dtypes.float32,
               seed=None):
    """Construct a DataSet.
     Seed arg provides for convenient deterministic testing.
    """
    seed1, seed2 = random_seed.get_seed(seed)
    # If op level seed is not set, use whatever graph level seed is returned
    np.random.seed(seed1 if seed is None else seed2)

    assert data.shape[0] == labels.shape[0], (
          'data.shape: %s labels.shape: %s' % (data.shape, labels.shape))

    self._num_examples = data.shape[0]
    self._data = data
    self._labels = labels
    self._epochs_completed = 0
    self._index_in_epoch = 0

  @property
  def data(self):
    return self._data

  @property
  def labels(self):
    return self._labels

  @property
  def num_examples(self):
    return self._num_examples

  @property
  def epochs_completed(self):
    return self._epochs_completed

  def next_batch(self, batch_size, fake_data=False, shuffle=True):
    """Return the next `batch_size` examples from this data set."""
    start = self._index_in_epoch
    # Shuffle for the first epoch
    if self._epochs_completed == 0 and start == 0 and shuffle:
      perm0 = np.arange(self._num_examples)
      np.random.shuffle(perm0)
      self._data = self.data[perm0]
      self._labels = self.labels[perm0]
    # Go to the next epoch
    if start + batch_size > self._num_examples:
      # Finished epoch
      self._epochs_completed += 1
      # Get the rest examples in this epoch
      rest_num_examples = self._num_examples - start
      data_rest_part = self._data[start:self._num_examples]
      labels_rest_part = self._labels[start:self._num_examples]
      # Shuffle the data
      if shuffle:
        perm = np.arange(self._num_examples)
        np.random.shuffle(perm)
        self._data = self.data[perm]
        self._labels = self.labels[perm]
      # Start next epoch
      start = 0
      self._index_in_epoch = batch_size - rest_num_examples
      end = self._index_in_epoch
      data_new_part = self._data[start:end]
      labels_new_part = self._labels[start:end]
      return np.concatenate((data_rest_part, data_new_part), axis=0) , np.concatenate((labels_rest_part, labels_new_part), axis=0)
    else:
      self._index_in_epoch += batch_size
      end = self._index_in_epoch
      return self._data[start:end], self._labels[start:end]


def read_data_sets(train_dir,
                   dtype=dtypes.float32,
                   validation_size=500,
                   seed=None):

  csvfilename = train_dir + '/' + 'wisdm1k.csv'

  wisdm = base.load_csv_without_header(csvfilename,
                                      target_dtype=np.str,
                                      features_dtype=np.float32)
  data = wisdm.data.astype(np.float32)
  data = scipy.delete(data, [0, 1], 1) # First 2 columns are useless, remove.

  # Enumerate target classifications from text to 0..n 
  labels = wisdm.target.astype(np.str)
  one_hot = np.asarray(pd.get_dummies(labels), np.uint32)
  labels = np.where(one_hot != 0)[1]

  print("data shape ", data.shape)
  print("target shape ", labels.shape)

  validation_size = int(len(data) * 0.1)
  print("validation size", validation_size)

  if not 0 <= validation_size <= len(data):
    raise ValueError(
        'Validation size should be between 0 and {}. Received: {}.'
        .format(len(data), validation_size))

  validation_data = data[:validation_size]
  validation_labels = labels[:validation_size]
  train_data = data[validation_size:]
  train_labels = labels[validation_size:]

  options = dict(dtype=dtype, seed=seed)

  # Split using train_test_split
  # RANDOM_SEED=42
  X_train, X_test, y_train, y_test = train_test_split(
        train_data, train_labels, test_size=0.2)

  validation_data = X_train[:validation_size]
  validation_labels = y_train[:validation_size]
  X_train = X_train[validation_size:]
  y_train = y_train[validation_size:]

  print("training set shape ", X_train.shape)
  print("test set shape ", X_test.shape)
  print("validation set shape ", validation_data.shape)

  train = DataSet(X_train, y_train, **options)
  test = DataSet(X_test, y_test, **options)
  validation = DataSet(validation_data, validation_labels, **options)

  return base.Datasets(train=train, validation=validation, test=test)


def load_wisdm(train_dir='/tmp/data'):
  return read_data_sets(train_dir)
