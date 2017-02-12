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
"""Proximal stochastic dual coordinate ascent optimizer for linear models."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections

from six.moves import range

from tensorflow.contrib.linear_optimizer.python.ops.sharded_mutable_dense_hashtable import ShardedMutableDenseHashTable
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.framework.ops import internal_convert_to_tensor
from tensorflow.python.framework.ops import name_scope
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import control_flow_ops
from tensorflow.python.ops import gen_sdca_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import nn_ops
from tensorflow.python.ops import state_ops
from tensorflow.python.ops import variables as var_ops
from tensorflow.python.ops.nn import sigmoid_cross_entropy_with_logits
from tensorflow.python.summary import summary

from tensorflow.python.platform import tf_logging as logging
from tensorflow.python.ops import data_flow_ops
from tensorflow.core.framework import types_pb2
from tensorflow.python.training import queue_runner


__all__ = ['SdcaModel']

# TODO: example_state_data returned by gen_sdca_ops contains v in its first 
# column. This v will be collected to parameter server and wait for it to send
# back the aggregated v. 
# 
# This step is slimilar to `SyncReplicaOptimizer`. When constructing graphs, 
# each replica will register an operation in accumulator in parameter server.
# This is done by using the device of variable.
# 
# We also register an update operator on the device where global variable 
# is located. When the tokens are all dequeued, the v from each replicas will be 
# aggregated. Before next batch, v will be updated.
# 
# At the same time, there is a queue runner in chief worker which constantly 
# ask for aggregated v. 


# TODO(sibyl-Aix6ihai): add name_scope to appropriate methods.
class SdcaModel(object):
  """Stochastic dual coordinate ascent solver for linear models.

    This class currently only supports a single machine (multi-threaded)
    implementation. We expect the weights and duals to fit in a single machine.

    Loss functions supported:

     * Binary logistic loss
     * Squared loss
     * Hinge loss
     * Smooth hinge loss

    This class defines an optimizer API to train a linear model.

    ### Usage

    ```python
    # Create a solver with the desired parameters.
    lr = tf.contrib.linear_optimizer.SdcaModel(examples, variables, options)
    min_op = lr.minimize()
    opt_op = lr.update_weights(min_op)

    predictions = lr.predictions(examples)
    # Primal loss + L1 loss + L2 loss.
    regularized_loss = lr.regularized_loss(examples)
    # Primal loss only
    unregularized_loss = lr.unregularized_loss(examples)

    examples: {
      sparse_features: list of SparseFeatureColumn.
      dense_features: list of dense tensors of type float32.
      example_labels: a tensor of type float32 and shape [Num examples]
      example_weights: a tensor of type float32 and shape [Num examples]
      example_ids: a tensor of type string and shape [Num examples]
    }
    variables: {
      sparse_features_weights: list of tensors of shape [vocab size]
      dense_features_weights: list of tensors of shape [dense_feature_dimension]
    }
    options: {
      symmetric_l1_regularization: 0.0
      symmetric_l2_regularization: 1.0
      loss_type: "logistic_loss"
      num_loss_partitions: 1 (Optional, with default value of 1. Number of
      partitions of the global loss function, 1 means single machine solver,
      and >1 when we have more than one optimizer working concurrently.)
      num_table_shards: 1 (Optional, with default value of 1. Number of shards
      of the internal state table, typically set to match the number of
      parameter servers for large data sets.
    }
    ```

    In the training program you will just have to run the returned Op from
    minimize().

    ```python
    # Execute opt_op and train for num_steps.
    for _ in range(num_steps):
      opt_op.run()

    # You can also check for convergence by calling
    lr.approximate_duality_gap()
    ```
  """

  def __init__(self, examples, variables, options):
    """Create a new sdca optimizer."""

    if not examples or not variables or not options:
      raise ValueError('examples, variables and options must all be specified.')

    supported_losses = ('logistic_loss', 'squared_loss', 'hinge_loss',
                        'smooth_hinge_loss')
    if options['loss_type'] not in supported_losses:
      raise ValueError('Unsupported loss_type: ', options['loss_type'])

    self._assertSpecified([
        'example_labels', 'example_weights', 'example_ids', 'sparse_features',
        'dense_features'
    ], examples)
    self._assertList(['sparse_features', 'dense_features'], examples)

    self._assertSpecified(['sparse_features_weights', 'dense_features_weights'],
                          variables)
    self._assertList(['sparse_features_weights', 'dense_features_weights'],
                     variables)

    self._assertSpecified([
        'loss_type', 'symmetric_l2_regularization',
        'symmetric_l1_regularization'
    ], options)

    for name in ['symmetric_l1_regularization', 'symmetric_l2_regularization']:
      value = options[name]
      if value < 0.0:
        raise ValueError('%s should be non-negative. Found (%f)' %
                         (name, value))

    self._examples = examples
    self._variables = variables
    self._options = options
    self._create_slots()
    self._hashtable = ShardedMutableDenseHashTable(
        key_dtype=dtypes.int64,
        value_dtype=dtypes.float32,
        num_shards=self._num_table_shards(),
        # Default values of elements of `example_state_data` in `minimize`
        default_value=[0.0, 0.0, 0.0, 0.0],
        # SdcaFprint never returns 0 or 1 for the low64 bits, so this a safe
        # empty_key (that will never collide with actual payloads).
        empty_key=[0, 0])

    summary.scalar('approximate_duality_gap', self.approximate_duality_gap())
    summary.scalar('examples_seen', self._hashtable.size())

  def _symmetric_l1_regularization(self):
    return self._options['symmetric_l1_regularization']

  def _symmetric_l2_regularization(self):
    # Algorithmic requirement (for now) is to have minimal l2 of 1.0.
    return max(self._options['symmetric_l2_regularization'], 1.0)

  def _num_loss_partitions(self):
    # Number of partitions of the global objective.
    # TODO(andreasst): set num_loss_partitions automatically based on the number
    # of workers
    return self._options.get('num_loss_partitions', 1)

  def _num_table_shards(self):
    # Number of hash table shards.
    # Return 1 if not specified or if the value is 'None'
    # TODO(andreasst): set num_table_shards automatically based on the number
    # of parameter servers
    num_shards = self._options.get('num_table_shards')
    return 1 if num_shards is None else num_shards

  # TODO(sibyl-Aix6ihai): Use optimizer interface to make use of slot creation logic.
  def _create_slots(self):
    # Make internal variables which have the updates before applying L1
    # regularization.
    self._slots = collections.defaultdict(list)
    for name in ['sparse_features_weights', 'dense_features_weights']:
      for var in self._variables[name]:
        with ops.device(var.device):
          # TODO(andreasst): remove SDCAOptimizer suffix once bug 30843109 is
          # fixed
          self._slots['unshrinked_' + name].append(
              var_ops.Variable(
                  array_ops.zeros_like(var.initialized_value(), dtypes.float32),
                  name=var.op.name + '_unshrinked/SDCAOptimizer'))

  def _assertSpecified(self, items, check_in):
    for x in items:
      if check_in[x] is None:
        raise ValueError(check_in[x] + ' must be specified.')

  def _assertList(self, items, check_in):
    for x in items:
      if not isinstance(check_in[x], list):
        raise ValueError(x + ' must be a list.')

  def _l1_loss(self):
    """Computes the (un-normalized) l1 loss of the model."""
    with name_scope('sdca/l1_loss'):
      sums = []
      for name in ['sparse_features_weights', 'dense_features_weights']:
        for weights in self._convert_n_to_tensor(self._variables[name]):
          with ops.device(weights.device):
            sums.append(
                math_ops.reduce_sum(
                    math_ops.abs(math_ops.cast(weights, dtypes.float64))))
      sum = math_ops.add_n(sums)
      # SDCA L1 regularization cost is: l1 * sum(|weights|)
      return self._options['symmetric_l1_regularization'] * sum

  def _l2_loss(self, l2):
    """Computes the (un-normalized) l2 loss of the model."""
    with name_scope('sdca/l2_loss'):
      sums = []
      for name in ['sparse_features_weights', 'dense_features_weights']:
        for weights in self._convert_n_to_tensor(self._variables[name]):
          with ops.device(weights.device):
            sums.append(
                math_ops.reduce_sum(
                    math_ops.square(math_ops.cast(weights, dtypes.float64))))
      sum = math_ops.add_n(sums)
      # SDCA L2 regularization cost is: l2 * sum(weights^2) / 2
      return l2 * sum / 2.0

  def _convert_n_to_tensor(self, input_list, as_ref=False):
    """Converts input list to a set of tensors."""
    return [internal_convert_to_tensor(x, as_ref=as_ref) for x in input_list]

  def _linear_predictions(self, examples):
    """Returns predictions of the form w*x."""
    with name_scope('sdca/prediction'):
      sparse_variables = self._convert_n_to_tensor(self._variables[
          'sparse_features_weights'])
      result = 0.0
      for sfc, sv in zip(examples['sparse_features'], sparse_variables):
        # TODO(sibyl-Aix6ihai): following does not take care of missing features.
        result += math_ops.segment_sum(
            math_ops.multiply(
                array_ops.gather(sv, sfc.feature_indices), sfc.feature_values),
            sfc.example_indices)
      dense_features = self._convert_n_to_tensor(examples['dense_features'])
      dense_variables = self._convert_n_to_tensor(self._variables[
          'dense_features_weights'])

      for i in range(len(dense_variables)):
        result += math_ops.matmul(dense_features[i],
                                  array_ops.expand_dims(dense_variables[i], -1))

    # Reshaping to allow shape inference at graph construction time.
    return array_ops.reshape(result, [-1])

  def predictions(self, examples):
    """Add operations to compute predictions by the model.

    If logistic_loss is being used, predicted probabilities are returned.
    Otherwise, (raw) linear predictions (w*x) are returned.

    Args:
      examples: Examples to compute predictions on.

    Returns:
      An Operation that computes the predictions for examples.

    Raises:
      ValueError: if examples are not well defined.
    """
    self._assertSpecified(
        ['example_weights', 'sparse_features', 'dense_features'], examples)
    self._assertList(['sparse_features', 'dense_features'], examples)

    result = self._linear_predictions(examples)
    if self._options['loss_type'] == 'logistic_loss':
      # Convert logits to probability for logistic loss predictions.
      with name_scope('sdca/logistic_prediction'):
        result = math_ops.sigmoid(result)
    return result

  def minimize(self, global_step=None, name=None):
    """Add operations to train a linear model by minimizing the loss function.

    Args:
      global_step: Optional `Variable` to increment by one after the
        variables have been updated.
      name: Optional name for the returned operation.

    Returns:
      An Operation that updates the variables passed in the constructor.
    """
    # Technically, the op depends on a lot more than the variables,
    # but we'll keep the list short.
    with name_scope(name, 'sdca/minimize'):
      sparse_example_indices = []
      sparse_feature_indices = []
      sparse_features_values = []
      for sf in self._examples['sparse_features']:
        sparse_example_indices.append(sf.example_indices)
        sparse_feature_indices.append(sf.feature_indices)
        # If feature values are missing, sdca assumes a value of 1.0f.
        if sf.feature_values is not None:
          sparse_features_values.append(sf.feature_values)

      # pylint: disable=protected-access
      example_ids_hashed = gen_sdca_ops._sdca_fprint(
          internal_convert_to_tensor(self._examples['example_ids']))
      # pylint: enable=protected-access
      example_state_data = self._hashtable.lookup(example_ids_hashed)

      # Convert internal weight variables to tensor.
      weights_tensor = self._convert_n_to_tensor(self._slots[
          'unshrinked_sparse_features_weights'])
      sparse_weights = []
      sparse_indices = []
      for w, i in zip(weights_tensor, sparse_feature_indices):
        # Find the feature ids to lookup in the variables.
        with ops.device(w.device):
          sparse_indices.append(
              math_ops.cast(
                  # array_ops.unique returns a tuple (unique_value, value_index)
                  # here only the value of `math_ops.cast(i, dtypes.int32)` is 
                  # our concern.
                  array_ops.unique(math_ops.cast(i, dtypes.int32))[0],
                  dtypes.int64))
          sparse_weights.append(array_ops.gather(w, sparse_indices[-1]))

      # pylint: disable=protected-access
      # Solver returns example_state_update, new delta sparse_feature_weights
      # and delta dense_feature_weights.
      esu, sfw, dfw = gen_sdca_ops._sdca_optimizer(
          sparse_example_indices,
          sparse_feature_indices,
          sparse_features_values,
          self._convert_n_to_tensor(self._examples['dense_features']),
          internal_convert_to_tensor(self._examples['example_weights']),
          internal_convert_to_tensor(self._examples['example_labels']),
          # Sparse weights indices
          sparse_indices,
          # Sparse weights
          sparse_weights,
          # Dense weights
          self._convert_n_to_tensor(self._slots[
              'unshrinked_dense_features_weights']),
          example_state_data,
          loss_type=self._options['loss_type'],
          l1=self._options['symmetric_l1_regularization'],
          l2=self._symmetric_l2_regularization(),
          num_loss_partitions=self._num_loss_partitions(),
          num_inner_iterations=1,
          dual_method=self._options['dual_method'])
      # pylint: enable=protected-access

      # update nominal weights with delta weights obtained from sfw/dfw.
      # In the original settings, the output of _sdca_optimizer will be stored 
      # in slots. Then in update_weight(), we will update the weight from slots.
      # 
      # For synchronous training, the update slots will be delayed until it is 
      # allowed to perform next local step.
      # 
      # Note that the local weight (self._variables will not be synchronoused)
      self._example_ids_hashed = example_ids_hashed

     # update nominal weights with delta weights obtained from sfw/dfw.
      with ops.control_dependencies([esu]):
        update_ops = [self._hashtable.insert(example_ids_hashed, esu)]
        # Update the weights before the proximal step.
        for w, i, u in zip(self._slots['unshrinked_sparse_features_weights'],
                           sparse_indices, sfw):
          update_ops.append(state_ops.scatter_add(w, i, u))
        for w, u in zip(self._slots['unshrinked_dense_features_weights'], dfw):
          update_ops.append(w.assign_add(u))

      if not global_step:
        return control_flow_ops.group(*update_ops)
      with ops.control_dependencies(update_ops):
        return state_ops.assign_add(global_step, 1, name=name).op

  def update_weights(self, train_op):
    """Updates the model weights.

    This function must be called on at least one worker after `minimize`.
    In distributed training this call can be omitted on non-chief workers to
    speed up training.

    Args:
      train_op: The operation returned by the `minimize` call.

    Returns:
      An Operation that updates the model weights.
    """
    with ops.control_dependencies([train_op]):
      update_ops = []
      # Copy over unshrinked weights to user provided variables.
      for name in ['sparse_features_weights', 'dense_features_weights']:
        for var, slot_var in zip(self._variables[name],
                                 self._slots['unshrinked_' + name]):
          update_ops.append(var.assign(slot_var))

    # We don't need to apply shrinkage to weight for lasso solver.
    if self._options['dual_method']:
      return control_flow_ops.group(*update_ops)

    # Apply proximal step.
    with ops.control_dependencies(update_ops):
      update_ops = []
      for name in ['sparse_features_weights', 'dense_features_weights']:
        for var in self._variables[name]:
          with ops.device(var.device):
            # pylint: disable=protected-access
            update_ops.append(
                gen_sdca_ops._sdca_shrink_l1(
                    self._convert_n_to_tensor(
                        [var], as_ref=True),
                    l1=self._symmetric_l1_regularization(),
                    l2=self._symmetric_l2_regularization()))
      return control_flow_ops.group(*update_ops)

  def approximate_duality_gap(self):
    """Add operations to compute the approximate duality gap.

    Returns:
      An Operation that computes the approximate duality gap over all
      examples.
    """
    with name_scope('sdca/approximate_duality_gap'):
      _, values_list = self._hashtable.export_sharded()
      shard_sums = []
      for values in values_list:
        with ops.device(values.device):
          # For large tables to_double() below allocates a large temporary
          # tensor that is freed once the sum operation completes. To reduce
          # peak memory usage in cases where we have multiple large tables on a
          # single device, we serialize these operations.
          # Note that we need double precision to get accurate results.
          with ops.control_dependencies(shard_sums):
            shard_sums.append(
                math_ops.reduce_sum(math_ops.to_double(values), 0))
      summed_values = math_ops.add_n(shard_sums)

      primal_loss = summed_values[1]
      dual_loss = summed_values[2]
      example_weights = summed_values[3]
      # Note: we return NaN if there are no weights or all weights are 0, e.g.
      # if no examples have been processed
      return (primal_loss + dual_loss + self._l1_loss() +
              (2.0 * self._l2_loss(self._symmetric_l2_regularization()))
             ) / example_weights

  def unregularized_loss(self, examples):
    """Add operations to compute the loss (without the regularization loss).

    Args:
      examples: Examples to compute unregularized loss on.

    Returns:
      An Operation that computes mean (unregularized) loss for given set of
      examples.

    Raises:
      ValueError: if examples are not well defined.
    """
    self._assertSpecified([
        'example_labels', 'example_weights', 'sparse_features', 'dense_features'
    ], examples)
    self._assertList(['sparse_features', 'dense_features'], examples)
    with name_scope('sdca/unregularized_loss'):
      predictions = math_ops.cast(
          self._linear_predictions(examples), dtypes.float64)
      labels = math_ops.cast(
          internal_convert_to_tensor(examples['example_labels']),
          dtypes.float64)
      weights = math_ops.cast(
          internal_convert_to_tensor(examples['example_weights']),
          dtypes.float64)

      if self._options['loss_type'] == 'logistic_loss':
        return math_ops.reduce_sum(math_ops.multiply(
            sigmoid_cross_entropy_with_logits(labels=labels,
                                              logits=predictions),
            weights)) / math_ops.reduce_sum(weights)

      if self._options['loss_type'] in ['hinge_loss', 'smooth_hinge_loss']:
        # hinge_loss = max{0, 1 - y_i w*x} where y_i \in {-1, 1}. So, we need to
        # first convert 0/1 labels into -1/1 labels.
        all_ones = array_ops.ones_like(predictions)
        adjusted_labels = math_ops.subtract(2 * labels, all_ones)
        # Tensor that contains (unweighted) error (hinge loss) per
        # example.
        error = nn_ops.relu(
            math_ops.subtract(all_ones,
                              math_ops.multiply(adjusted_labels, predictions)))
        weighted_error = math_ops.multiply(error, weights)
        return math_ops.reduce_sum(weighted_error) / math_ops.reduce_sum(
            weights)

      # squared loss
      err = math_ops.subtract(labels, predictions)

      weighted_squared_err = math_ops.multiply(math_ops.square(err), weights)
      # SDCA squared loss function is sum(err^2) / (2*sum(weights))
      return (math_ops.reduce_sum(weighted_squared_err) /
              (2.0 * math_ops.reduce_sum(weights)))

  def regularized_loss(self, examples):
    """Add operations to compute the loss with regularization loss included.

    Args:
      examples: Examples to compute loss on.

    Returns:
      An Operation that computes mean (regularized) loss for given set of
      examples.
    Raises:
      ValueError: if examples are not well defined.
    """
    self._assertSpecified([
        'example_labels', 'example_weights', 'sparse_features', 'dense_features'
    ], examples)
    self._assertList(['sparse_features', 'dense_features'], examples)
    with name_scope('sdca/regularized_loss'):
      weights = internal_convert_to_tensor(examples['example_weights'])
      return ((
          self._l1_loss() +
          # Note that here we are using the raw regularization
          # (as specified by the user) and *not*
          # self._symmetric_l2_regularization().
          self._l2_loss(self._options['symmetric_l2_regularization'])) /
              math_ops.reduce_sum(math_ops.cast(weights, dtypes.float64)) +
              self.unregularized_loss(examples))

# The `SyncSdcaModel` is adapt from `SyncReplicaOptimizer`. The body of 
# `apply_gradient` is kept but named with `minimize` so that `SDCAOptimizer` can
# use this class like `SdcaModel`. 
# 
# The differences here are, instead of providing variables and associated 
# gradients, the variable here `global_v` is defined inside the function and the
# gradient comes from (`example_state_data` - `global_v`). The `example_state_data`
# is computed in `optimize` of `SdcaModel` which is local `v`.
# 
# This part of data come from `SyncReplicaOptimizer`, an example of it is 
# `mnist_replica.py`
class SyncSdcaModel(SdcaModel):
  def __init__(self,
               opt,
               replicas_to_aggregate,
               total_num_replicas=None,
               variable_averages=None,
               variables_to_average=None,
               use_locking=False,
               name="sync_replicas"):
    """Construct a sync_replicas optimizer.

    Args:
      opt: The actual optimizer that will be used to compute and apply the
        gradients. Must be one of the Optimizer classes.
      replicas_to_aggregate: number of replicas to aggregate for each variable
        update.
      total_num_replicas: Total number of tasks/workers/replicas, could be
        different from replicas_to_aggregate.
        If total_num_replicas > replicas_to_aggregate: it is backup_replicas +
        replicas_to_aggregate.
        If total_num_replicas < replicas_to_aggregate: Replicas compute
        multiple batches per update to variables.
      variable_averages: Optional `ExponentialMovingAverage` object, used to
        maintain moving averages for the variables passed in
        `variables_to_average`.
      variables_to_average: a list of variables that need to be averaged. Only
        needed if variable_averages is passed in.
      use_locking: If True use locks for update operation.
      name: string. Optional name of the returned operation.
    """
    if total_num_replicas is None:
      total_num_replicas = replicas_to_aggregate

    # super(SyncReplicasOptimizer, self).__init__(use_locking, name)
    logging.info(
        "SyncReplicasV2: replicas_to_aggregate=%s; total_num_replicas=%s",
        replicas_to_aggregate, total_num_replicas)

    self._name = name

    self._opt = opt
    self._replicas_to_aggregate = replicas_to_aggregate
    self._gradients_applied = False
    self._variable_averages = variable_averages
    self._variables_to_average = variables_to_average
    self._total_num_replicas   = total_num_replicas
    self._tokens_per_step  = max(total_num_replicas, replicas_to_aggregate) 
    self._global_step      = None
    self._sync_token_queue = None

    # The synchronization op will be executed in a queue runner which should
    # only be executed by one of the replicas (usually the chief).
    self._chief_queue_runner = None

    # Remember which accumulator is on which device to set the initial step in
    # the accumulator to be global step. This list contains list of the
    # following format: (accumulator, device).
    self._accumulator_list = []

  def minimize(self, global_step=None, name=None):
    """Apply gradients to variables.

    This contains most of the synchronization implementation and also wraps the
    apply_gradients() from the real optimizer.

    Args:
      grads_and_vars: List of (gradient, variable) pairs as returned by
        compute_gradients().
      global_step: Optional Variable to increment by one after the
        variables have been updated.
      name: Optional name for the returned operation.  Default to the
        name passed to the Optimizer constructor.

    Returns:
      train_op: The op to dequeue a token so the replicas can exit this batch
      and start the next one. This is executed by each replica.

    Raises:
      ValueError: If the grads_and_vars is empty.
      ValueError: If global step is not provided, the staleness cannot be
        checked.
    """

    if global_step is None:
      raise ValueError("Global step is required to check staleness")

    self._local_step = var_ops.Variable(
        initial_value=0,
        trainable=False,
        collections=[ops.GraphKeys.LOCAL_VARIABLES],
        dtype=global_step.dtype.base_dtype,
        name="sync_rep_local_step")

    # This step is used for `local_init_op` for supervisor to call.
    self.local_step_init_op = state_ops.assign(self._local_step, global_step)
    chief_init_ops = [self.local_step_init_op]
    self.ready_for_local_init_op = var_ops.report_uninitialized_variables(
        var_ops.global_variables())

    local_train_op = self._opt.minimize(self._local_step, name, sync=True)

    self._global_v = var_ops.Variable(
        initial_value=array_ops.zeros(
          [self._opt._example_ids_hashed.get_shape()[0], 4], dtypes.float32),
        name='global_v')

    with ops.control_dependencies([local_train_op]):
      delta_example_state_data = self._opt._hashtable.lookup(self._opt._example_ids_hashed) - self._global_v

    grads_and_vars = [(delta_example_state_data, self._global_v)]

    self._global_step = global_step
    train_ops = []
    aggregated_grad = []
    var_list = []

    with ops.name_scope(None, self._name):
      for grad, var in grads_and_vars:
        var_list.append(var)
        with ops.device(var.device):
          if grad is None:
            aggregated_grad.append(None)  # pass-through.
            continue
          elif isinstance(grad, ops.Tensor):
            # get shared grad accumulator in the ps server
            grad_accum = data_flow_ops.ConditionalAccumulator(
                grad.dtype,
                shape=var.get_shape(),
                shared_name=var.name + "/grad_accum")

            # Create an operation that push local `grad` to corresponding 
            # `grad_accum` in ps server. 
            train_ops.append(grad_accum.apply_grad(
                grad, local_step=self._local_step))

            # grad_accum.take_grad create a number of `_replicas_to_aggregate`
            # **blocking** operations in parameter server. The operation will 
            # return a grad tensor for each varaiable.  
            aggregated_grad.append(grad_accum.take_grad(
                  self._replicas_to_aggregate))
          else:
            if not isinstance(grad, ops.IndexedSlices):
              raise ValueError("Unknown grad type!")
            grad_accum = data_flow_ops.SparseConditionalAccumulator(
                grad.dtype, shape=(), shared_name=var.name + "/grad_accum")
            train_ops.append(grad_accum.apply_indexed_slices_grad(
                grad, local_step=self._local_step))
            aggregated_grad.append(grad_accum.take_indexed_slices_grad(
                self._replicas_to_aggregate))

          self._accumulator_list.append((grad_accum, var.device))

      # updated grads from ps server and the corresponding var.
      aggregated_grads_and_vars = zip(aggregated_grad, var_list)

      # sync_op will be assigned to the same device as the global step.
      with ops.device(global_step.device), ops.name_scope(""), ops.colocate_with(self._global_v):
        # The minimize() function is still method in the base class where 
        # apply_gradients is followed by compute_gradient. In sync_replicas,
        # we add above code to push our local grads_and_vars and get
        # `aggregated_grads_and_vars`. Then we apply `aggregated_grads_and_vars`
        # to local gradient.
        # This operation is only performed by chief worker as the chief qr will
        # call sync_op and sync_op relie on update_op.
        with ops.control_dependencies([aggregated_grad[0]]):
          update_op = state_ops.assign_add(self._global_v, aggregated_grad[0])

      # Create token queue.
      with ops.device(global_step.device), ops.name_scope(""):
        # get global_step in parameter server.
        sync_token_queue = (
            data_flow_ops.FIFOQueue(-1,
                                    global_step.dtype.base_dtype,
                                    shapes=(),
                                    name="sync_token_q",
                                    shared_name="sync_token_q"))
        self._sync_token_queue = sync_token_queue

        # dummy_queue is passed to the queue runner. Don't use the real queues
        # because the queue runner doesn't automatically reopen it once it
        # closed queues in PS devices.
        dummy_queue = (
            data_flow_ops.FIFOQueue(1,
                                    types_pb2.DT_INT32,
                                    shapes=(),
                                    name="dummy_queue",
                                    shared_name="dummy_queue"))

      with ops.device(global_step.device), ops.name_scope(""):
        # Replicas have to wait until they can get a token from the token queue.
        # when all the gradients have been computed, ask for the next 
        with ops.control_dependencies(train_ops):
          # in `local_train_op`, the v is updated locally. We pull `global_v` to it so
          # that when global_v is updated (by only chief worker), each worker will get
          # the aggregated v.
          update_v_op = self._opt._hashtable.insert(self._opt._example_ids_hashed, internal_convert_to_tensor(self._global_v))
          with ops.control_dependencies([update_v_op]):
            token = sync_token_queue.dequeue()

        train_op = state_ops.assign(self._local_step, token)

        with ops.control_dependencies([update_op]):
          # Sync_op needs to insert tokens to the token queue at the end of the
          # step so the replicas can fetch them to start the next step.
          update_global_step_op = state_ops.assign_add(global_step, 1)
          with ops.control_dependencies([update_global_step_op]):
            tokens = array_ops.fill([self._tokens_per_step], global_step)
          sync_op = sync_token_queue.enqueue_many((tokens,))

        if self._variable_averages is not None:
          with ops.control_dependencies([sync_op]), ops.name_scope(""):
            sync_op = self._variable_averages.apply(
                self._variables_to_average)

        # The enqueue operation for dummy_queue is actually not for dummy queue
        # but for sync_token_queue. So no matter how may times we have enqueued,
        # the dummy_queue is always empty. We are trying to enqueue tokens for
        # sync_token_queue.
        self._chief_queue_runner = queue_runner.QueueRunner(dummy_queue,
                                                            [sync_op])

      # Set all the accumulators to same global_step.
      for accum, dev in self._accumulator_list:
        with ops.device(dev):
          chief_init_ops.append(
              accum.set_global_step(
                  global_step, name="SetGlobalStep"))
      self.chief_init_op = control_flow_ops.group(*(chief_init_ops))
      self._gradients_applied = True

      return train_op
  
  def update_weights(self, train_op):
    return self._opt.update_weights(train_op)

  def get_init_tokens_op(self, num_tokens=-1):
    """Returns the op to fill the sync_token_queue with the tokens.

    This is supposed to be executed in the beginning of the chief/sync thread
    so that even if the total_num_replicas is less than replicas_to_aggregate,
    the model can still proceed as the replicas can compute multiple steps per
    variable update. Make sure:
    `num_tokens >= replicas_to_aggregate - total_num_replicas`.

    Args:
      num_tokens: Number of tokens to add to the queue.

    Returns:
      An op for the chief/sync replica to fill the token queue.

    Raises:
      ValueError: If this is called before apply_gradients().
      ValueError: If num_tokens are smaller than replicas_to_aggregate -
        total_num_replicas.
    """
    if self._gradients_applied is False:
      raise ValueError(
          "get_init_tokens_op() should be called after apply_gradients().")

    tokens_needed = self._replicas_to_aggregate - self._total_num_replicas
    if num_tokens == -1:
      num_tokens = self._replicas_to_aggregate
    elif num_tokens < tokens_needed:
      raise ValueError(
          "Too few tokens to finish the first step: %d (given) vs %d (needed)" %
          (num_tokens, tokens_needed))

    if num_tokens > 0:
      with ops.device(self._global_step.device), ops.name_scope(""):
        tokens = array_ops.fill([num_tokens], self._global_step)
        init_tokens = self._sync_token_queue.enqueue_many((tokens,))
    else:
      init_tokens = control_flow_ops.no_op(name="no_init_tokens")

    return init_tokens