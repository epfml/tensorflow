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
"""Tests for SdcaModel."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from threading import Thread

import tensorflow as tf

from tensorflow.contrib.linear_optimizer.python.ops.sdca_ops import _ShardedMutableDenseHashTable
from tensorflow.contrib.linear_optimizer.python.ops.sdca_ops import SdcaModel
from tensorflow.contrib.linear_optimizer.python.ops.sdca_ops import SparseFeatureColumn
from tensorflow.python.framework.test_util import TensorFlowTestCase
from tensorflow.python.ops import gen_sdca_ops
from tensorflow.python.platform import googletest

_MAX_ITERATIONS = 100
_SHARD_NUMBERS = [None, 1, 3, 10]
_NUM_LOSS_PARTITIONS = [2, 4]

def make_example_proto(feature_dict, target, value=1.0):
  e = tf.train.Example()
  features = e.features

  features.feature['target'].float_list.value.append(target)

  for key, values in feature_dict.items():
    features.feature[key + '_indices'].int64_list.value.extend(values)
    features.feature[key + '_values'].float_list.value.extend([value] *
                                                              len(values))

  return e


def make_example_dict(example_protos, example_weights):

  def parse_examples(example_protos):
    features = {
        'target': tf.FixedLenFeature(shape=[1],
                                     dtype=tf.float32,
                                     default_value=0),
        'age_indices': tf.VarLenFeature(dtype=tf.int64),
        'age_values': tf.VarLenFeature(dtype=tf.float32),
        'gender_indices': tf.VarLenFeature(dtype=tf.int64),
        'gender_values': tf.VarLenFeature(dtype=tf.float32)
    }
    return tf.parse_example(
        [e.SerializeToString() for e in example_protos], features)

  parsed = parse_examples(example_protos)
  sparse_features = [
      SparseFeatureColumn(
          tf.reshape(
              tf.split(1, 2, parsed['age_indices'].indices)[0], [-1]),
          tf.reshape(parsed['age_indices'].values, [-1]),
          tf.reshape(parsed['age_values'].values, [-1])), SparseFeatureColumn(
              tf.reshape(
                  tf.split(1, 2, parsed['gender_indices'].indices)[0], [-1]),
              tf.reshape(parsed['gender_indices'].values, [-1]),
              tf.reshape(parsed['gender_values'].values, [-1]))
  ]
  return dict(sparse_features=sparse_features,
              dense_features=[],
              example_weights=example_weights,
              example_labels=tf.reshape(parsed['target'], [-1]),
              example_ids=['%d' % i for i in range(0, len(example_protos))])


def make_variable_dict(max_age, max_gender):
  # TODO(sibyl-toe9oF2e):  Figure out how to derive max_age & max_gender from
  # examples_dict.
  age_weights = tf.Variable(tf.zeros([max_age + 1], dtype=tf.float32))
  gender_weights = tf.Variable(tf.zeros([max_gender + 1], dtype=tf.float32))
  return dict(sparse_features_weights=[age_weights, gender_weights],
              dense_features_weights=[])


def make_dense_examples_and_variables_dicts(dense_features_values, weights,
                                            labels):
  """Creates examples and variables dictionaries for dense features.

  Variables shapes are inferred from the list of dense feature values passed as
  argument.

  Args:
    dense_features_values: The values of the dense features
    weights: The example weights.
    labels: The example labels.
  Returns:
    One dictionary for the examples and one for the variables.
  """
  dense_tensors = []
  dense_weights = []
  for dense_feature in dense_features_values:
    dense_tensor = tf.convert_to_tensor(dense_feature, dtype=tf.float32)
    check_shape_op = tf.Assert(
        tf.less_equal(tf.rank(dense_tensor), 2),
        ['dense_tensor shape must be [batch_size, dimension] or [batch_size]'])
    # Reshape to [batch_size, dense_column_dimension].
    with tf.control_dependencies([check_shape_op]):
      dense_tensor = tf.reshape(dense_tensor,
                                [dense_tensor.get_shape().as_list()[0], -1])
    dense_tensors.append(dense_tensor)
    # Add variables of shape [feature_column_dimension].
    dense_weights.append(
        tf.Variable(
            tf.zeros(
                [dense_tensor.get_shape().as_list()[1]], dtype=tf.float32)))

  examples_dict = dict(
      sparse_features=[],
      dense_features=dense_tensors,
      example_weights=weights,
      example_labels=labels,
      example_ids=['%d' % i for i in range(0, len(labels))])
  variables_dict = dict(
      sparse_features_weights=[], dense_features_weights=dense_weights)

  return examples_dict, variables_dict


def get_binary_predictions_for_logistic(predictions, cutoff=0.5):
  return tf.cast(
      tf.greater_equal(predictions, tf.ones_like(predictions) * cutoff),
      dtype=tf.int32)


def get_binary_predictions_for_hinge(predictions):
  return tf.cast(
      tf.greater_equal(predictions, tf.zeros_like(predictions)),
      dtype=tf.int32)


# TODO(sibyl-Mooth6ku): Add tests that exercise L1 and Shrinking.
# TODO(sibyl-vie3Poto): Refactor tests to avoid repetition of boilerplate code.
class SdcaModelTest(TensorFlowTestCase):
  """Base SDCA optimizer test class for any loss type."""

  def _single_threaded_test_session(self):
    config = tf.ConfigProto(inter_op_parallelism_threads=1,
                            intra_op_parallelism_threads=1)
    return self.test_session(use_gpu=False, config=config)

class SdcaWithHingeLossTest(SdcaModelTest):
  """SDCA optimizer test class for hinge loss."""

  def testSimple(self):
    # Setup test data
    example_protos = [
        make_example_proto(
            {'age': [0],
             'gender': [0]}, 0),
        make_example_proto(
            {'age': [1],
             'gender': [1]}, 1),
    ]
    example_weights = [1.0, 1.0]
    with self._single_threaded_test_session():
      examples = make_example_dict(example_protos, example_weights)
      variables = make_variable_dict(1, 1)
      options = dict(symmetric_l2_regularization=1.0,
                     symmetric_l1_regularization=0,
                     loss_type='hinge_loss')
      model = SdcaModel(examples, variables, options)
      tf.initialize_all_variables().run()

      # Before minimization, the weights default to zero. There is no loss due
      # to regularization, only unregularized loss which is 0.5 * (1+1) = 1.0.
      predictions = model.predictions(examples)
      self.assertAllClose([0.0, 0.0], predictions.eval())
      unregularized_loss = model.unregularized_loss(examples)
      regularized_loss = model.regularized_loss(examples)
      self.assertAllClose(1.0, unregularized_loss.eval())
      self.assertAllClose(1.0, regularized_loss.eval())

      # After minimization, the model separates perfectly the data points. There
      # are 4 sparse weights: 2 for age (say w1, w2) and 2 for gender (say w3
      # and w4). Solving the system w1 + w3 = 1.0, w2 + w4 = -1.0 and minimizing
      # wrt to \|\vec{w}\|_2, gives w1=w3=1/2 and w2=w4=-1/2. This gives 0.0
      # unregularized loss and 0.25 L2 loss.
      train_op = model.minimize()
      for _ in range(_MAX_ITERATIONS):
        train_op.run()
      model.update_weights(train_op).run()

      binary_predictions = get_binary_predictions_for_hinge(predictions)
      self.assertAllEqual([-1.0, 1.0], predictions.eval())
      self.assertAllEqual([0, 1], binary_predictions.eval())
      self.assertAllClose(0.0, unregularized_loss.eval())
      self.assertAllClose(0.25, regularized_loss.eval(), atol=0.05)

if __name__ == '__main__':
  googletest.main()
