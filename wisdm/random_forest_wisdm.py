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
"""A stand-alone example for tf.learn's random forest model on WISDM."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import sys
import tempfile
import numpy as np

from tensorflow.contrib.learn.python.learn import metric_spec
from tensorflow.contrib.learn.python.learn.estimators import estimator
from tensorflow.contrib.tensor_forest.client import eval_metrics
from tensorflow.contrib.tensor_forest.client import random_forest
from tensorflow.contrib.tensor_forest.python import tensor_forest
from wisdm import read_data_sets
from tensorflow.python.platform import app

FLAGS = None

def build_estimator(model_dir, nclasses, nfeatures):
  """Build an estimator."""

  print("Num trees: ", FLAGS.num_trees)

  params = tensor_forest.ForestHParams(
      num_classes=nclasses, num_features=nfeatures,
      num_trees=FLAGS.num_trees, max_nodes=FLAGS.max_nodes)

  if FLAGS.use_training_loss:
    graph_builder_class = tensor_forest.TrainingLossForest
  else:
    graph_builder_class = tensor_forest.RandomForestGraphs

  # Use the SKCompat wrapper, which gives us a convenient way to split
  # in-memory data like WISDM into batches.
  return estimator.SKCompat(random_forest.TensorForestEstimator(
      params, graph_builder_class=graph_builder_class,
      model_dir=model_dir))


def train_and_eval(wisdmFilename='../data/wisdm.txt'):

  wisdm = read_data_sets(csv = wisdmFilename)
  nclasses = len(np.unique(wisdm.train.labels))
  nfeatures = wisdm.train.data.shape[1]

  """Train and evaluate the model."""
  model_dir = FLAGS.model_dir or tempfile.mkdtemp()
  print('model directory = %s' % model_dir)

  print(nclasses,' classes from ', nfeatures, 'features')
  est = build_estimator(model_dir, nclasses, nfeatures)
  est.fit(x=wisdm.train.data, y=wisdm.train.labels,
          batch_size=FLAGS.batch_size)
  print('Done Fitting\n')

  metric_name = 'accuracy'
  mspec = metric_spec.MetricSpec(
    eval_metrics.get_metric(metric_name),
    prediction_key=eval_metrics.get_prediction_key(metric_name))

  metric = {metric_name: mspec}

  results = est.score(x=wisdm.test.data, y=wisdm.test.labels,
                      # batch_size=FLAGS.batch_size,
                      metrics=metric)
  for key in sorted(results):
    print('%s: %s' % (key, results[key]))

def main(_):
  train_and_eval(FLAGS.data)


if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument(
      '--model_dir',
      type=str,
      default='',
      help='Base directory for output models.'
  )
  parser.add_argument(
      '--data_dir',
      type=str,
      default='/tmp/data/',
      help='Directory for storing data'
  )

  parser.add_argument(
        '--data',
        type=str,
        help='path to data file'
  )

  parser.add_argument(
      '--train_steps',
      type=int,
      default=1000,
      help='Number of training steps.'
  )
  parser.add_argument(
      '--batch_size',
      type=int,
      default=1000,
      help='Number of examples in a training batch.'
  )
  parser.add_argument(
      '--num_trees',
      type=int,
      default=40,
      help='Number of trees in the forest.'
  )
  parser.add_argument(
      '--max_nodes',
      type=int,
      default=1000,
      help='Max total nodes in a single tree.'
  )
  parser.add_argument(
      '--use_training_loss',
      type=bool,
      default=False,
      help='If true, use training loss as termination criteria.'
  )
  FLAGS, unparsed = parser.parse_known_args()
  app.run(main=main, argv=[sys.argv[0]] + unparsed)
