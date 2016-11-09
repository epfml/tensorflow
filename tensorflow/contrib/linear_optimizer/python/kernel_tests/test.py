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

from my_sdca_ops_test import *

_SHARD_NUMBERS = [None, 1, 3, 10]

def display_proto():
    e = make_example_proto(
            {'age': [0, 1],
             'gender': [0, 2]}, 0)
    print(e)

def disp_make_example_dict():
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
    examples = make_example_dict(example_protos, example_weights)
    print(examples.items())

class MySdcaWithLogisticLossTest(SdcaModelTest):
  """SDCA optimizer test class for logistic loss."""

  def testSimple(self):
    print("testSimple")
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
    for num_shards in _SHARD_NUMBERS:
      with self._single_threaded_test_session():
        examples = make_example_dict(example_protos, example_weights)
        variables = make_variable_dict(1, 1)
        options = dict(symmetric_l2_regularization=1,
                       symmetric_l1_regularization=0,
                       num_table_shards=num_shards,
                       loss_type='logistic_loss')

        lr = SdcaModel(examples, variables, options)
        tf.global_variables_initializer().run()
        unregularized_loss = lr.unregularized_loss(examples)
        loss = lr.regularized_loss(examples)
        predictions = lr.predictions(examples)
        self.assertAllClose(0.693147, unregularized_loss.eval())
        self.assertAllClose(0.693147, loss.eval())
        train_op = lr.minimize()
        for _ in range(_MAX_ITERATIONS):
          train_op.run()
        lr.update_weights(train_op).run()
        # The high tolerance in unregularized_loss comparisons is due to the
        # fact that it's possible to trade off unregularized_loss vs.
        # regularization and still have a sum that is quite close to the
        # optimal regularized_loss value.  SDCA's duality gap only ensures that
        # the regularized_loss is within 0.01 of optimal.
        # 0.525457 is the optimal regularized_loss.
        # 0.411608 is the unregularized_loss at that optimum.
        self.assertAllClose(0.411608, unregularized_loss.eval(), atol=0.05)
        self.assertAllClose(0.525457, loss.eval(), atol=0.01)
        predicted_labels = get_binary_predictions_for_logistic(predictions)
        self.assertAllEqual([0, 1], predicted_labels.eval())
        self.assertAllClose(0.01,
                            lr.approximate_duality_gap().eval(),
                            rtol=1e-2,
                            atol=1e-2)


if __name__ == "__main__":
    #display_proto()
    #disp_make_example_dict()
    googletest.main()
