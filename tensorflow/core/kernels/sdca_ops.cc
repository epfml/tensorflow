/* Copyright 2016 The TensorFlow Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

// See docs in ../ops/sdca_ops.cc.

#define EIGEN_USE_THREADS

#include <stdint.h>
#include <atomic>
#include <limits>
#include <memory>
#include <new>
#include <string>
#include <vector>
#include <chrono>

#include "third_party/eigen3/unsupported/Eigen/CXX11/Tensor"
#include "tensorflow/core/framework/device_base.h"
#include "tensorflow/core/framework/kernel_def_builder.h"
#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/op_def_builder.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/framework/tensor_shape.h"
#include "tensorflow/core/framework/tensor_types.h"
#include "tensorflow/core/framework/types.h"
#include "tensorflow/core/kernels/hinge-loss.h"
#include "tensorflow/core/kernels/logistic-loss.h"
#include "tensorflow/core/kernels/loss.h"
#include "tensorflow/core/kernels/sdca_internal.h"
#include "tensorflow/core/kernels/smooth-hinge-loss.h"
#include "tensorflow/core/kernels/squared-loss.h"
#include "tensorflow/core/lib/core/coding.h"
#include "tensorflow/core/lib/core/errors.h"
#include "tensorflow/core/lib/core/status.h"
#include "tensorflow/core/lib/core/stringpiece.h"
#include "tensorflow/core/lib/gtl/inlined_vector.h"
#include "tensorflow/core/lib/strings/stringprintf.h"
#include "tensorflow/core/platform/fingerprint.h"
#include "tensorflow/core/platform/macros.h"
#include "tensorflow/core/platform/mutex.h"
#include "tensorflow/core/platform/types.h"
#include "tensorflow/core/util/work_sharder.h"

namespace tensorflow {

namespace {

using sdca::Regularizations;
using sdca::Example;
using sdca::Examples;
using sdca::ExampleStatistics;
using sdca::ModelWeights;

struct ComputeOptions {
  ComputeOptions(OpKernelConstruction* const context) {
    string loss_type;
    OP_REQUIRES_OK(context, context->GetAttr("loss_type", &loss_type));
    if (loss_type == "logistic_loss") {
      loss_updater.reset(new LogisticLossUpdater);
    } else if (loss_type == "squared_loss") {
      loss_updater.reset(new SquaredLossUpdater);
    } else if (loss_type == "hinge_loss") {
      loss_updater.reset(new HingeLossUpdater);
    } else if (loss_type == "smooth_hinge_loss") {
      loss_updater.reset(new SmoothHingeLossUpdater);
    } else {
      OP_REQUIRES(context, false, errors::InvalidArgument(
                                      "Unsupported loss type: ", loss_type));
    }
    OP_REQUIRES_OK(context, context->GetAttr("adaptative", &adaptative));
    OP_REQUIRES_OK(context, context->GetAttr("dual_method", &dual_method));

    OP_REQUIRES_OK(
        context, context->GetAttr("num_sparse_features", &num_sparse_features));
    OP_REQUIRES_OK(context, context->GetAttr("num_sparse_features_with_values",
                                             &num_sparse_features_with_values));
    OP_REQUIRES_OK(context,
                   context->GetAttr("num_dense_features", &num_dense_features));
    OP_REQUIRES(
        context, num_sparse_features + num_dense_features > 0,
        errors::InvalidArgument("Requires at least one feature to train."));

    OP_REQUIRES(context, static_cast<int64>(num_sparse_features) +
                                 static_cast<int64>(num_dense_features) <=
                             std::numeric_limits<int>::max(),
                errors::InvalidArgument(
                    strings::Printf("Too many feature groups: %lld > %d",
                                    static_cast<int64>(num_sparse_features) +
                                        static_cast<int64>(num_dense_features),
                                    std::numeric_limits<int>::max())));
    OP_REQUIRES_OK(
        context, context->GetAttr("num_loss_partitions", &num_loss_partitions));
    OP_REQUIRES_OK(context, context->GetAttr("num_inner_iterations",
                                             &num_inner_iterations));
    OP_REQUIRES_OK(context, regularizations.Initialize(context));
  }

  std::unique_ptr<DualLossUpdater> loss_updater;
  int num_sparse_features = 0;
  int num_sparse_features_with_values = 0;
  int num_dense_features = 0;
  int num_inner_iterations = 0;
  int num_loss_partitions = 0;
  bool adaptative = false;
  bool dual_method  = false;
  Regularizations regularizations;
};

// TODO(shengx): The helper classes/methods are changed to support multiclass
// SDCA, which lead to changes within this function. Need to revisit the
// convergence once the multiclass SDCA is in.
void DoCompute(const ComputeOptions& options, OpKernelContext* const context) {
  ModelWeights model_weights;
  OP_REQUIRES_OK(context, model_weights.Initialize(context));

  Examples examples;
  OP_REQUIRES_OK(
      context,
      examples.Initialize(context, model_weights, options.num_sparse_features,
                          options.num_sparse_features_with_values,
                          options.num_dense_features));

  const Tensor* example_state_data_t;
  OP_REQUIRES_OK(context,
                 context->input("example_state_data", &example_state_data_t));
  TensorShape expected_example_state_shape({examples.num_examples(), 4});
  OP_REQUIRES(context,
              example_state_data_t->shape() == expected_example_state_shape,
              errors::InvalidArgument(
                  "Expected shape ", expected_example_state_shape.DebugString(),
                  " for example_state_data, got ",
                  example_state_data_t->shape().DebugString()));

  Tensor mutable_example_state_data_t(*example_state_data_t);
  auto example_state_data = mutable_example_state_data_t.matrix<float>();
  context->set_output("out_example_state_data", mutable_example_state_data_t);

  if (options.adaptative) {
    OP_REQUIRES_OK(context,
                   examples.SampleAdaptativeProbabilities(
                       options.num_loss_partitions, options.regularizations,
                       model_weights, example_state_data, options.loss_updater,
                       /*num_weight_vectors =*/1));
  }

  mutex mu;
  Status train_step_status GUARDED_BY(mu);
  std::atomic<std::int64_t> atomic_index(-1);
  auto train_step = [&](const int64 begin, const int64 end) {
    // The static_cast here is safe since begin and end can be at most
    // num_examples which is an int.
    for (int id = static_cast<int>(begin); id < end; ++id) {
      const int64 example_index =
          examples.sampled_index(++atomic_index, options.adaptative);
      const Example& example = examples.example(example_index);
      const float dual = example_state_data(example_index, 0);
      const float example_weight = example.example_weight();
      float example_label = example.example_label();
      const Status conversion_status =
          options.loss_updater->ConvertLabel(&example_label);
      if (!conversion_status.ok()) {
        mutex_lock l(mu);
        train_step_status = conversion_status;
        // Return from this worker thread - the calling thread is
        // responsible for checking context status and returning on error.
        return;
      }

      // Compute wx, example norm weighted by regularization, dual loss,
      // primal loss.
      // For binary SDCA, num_weight_vectors should be one.
      const ExampleStatistics example_statistics =
          example.ComputeWxAndWeightedExampleNorm(
              options.num_loss_partitions, model_weights,
              options.regularizations, 1 /* num_weight_vectors */);

      const double new_dual = options.loss_updater->ComputeUpdatedDual(
          options.num_loss_partitions, example_label, example_weight, dual,
          example_statistics.wx[0], example_statistics.normalized_squared_norm);

      // Compute new weights.
      const double normalized_bounded_dual_delta =
          (new_dual - dual) * example_weight /
          options.regularizations.symmetric_l2();
      model_weights.UpdateDeltaWeights(
          context->eigen_cpu_device(), example,
          std::vector<double>{normalized_bounded_dual_delta});

      // Update example data.
      example_state_data(example_index, 0) = new_dual;
      example_state_data(example_index, 1) =
          options.loss_updater->ComputePrimalLoss(
              example_statistics.prev_wx[0], example_label, example_weight);
      example_state_data(example_index, 2) =
          options.loss_updater->ComputeDualLoss(dual, example_label,
                                                example_weight);
      example_state_data(example_index, 3) = example_weight;
    }
  };
  // TODO(sibyl-Aix6ihai): Tune this properly based on sparsity of the data,
  // number of cpus, and cost per example.
  const int64 kCostPerUnit = examples.num_features();
  const DeviceBase::CpuWorkerThreads& worker_threads =
      *context->device()->tensorflow_cpu_worker_threads();

  Shard(worker_threads.num_threads, worker_threads.workers,
        examples.num_examples(), kCostPerUnit, train_step);
  OP_REQUIRES_OK(context, train_step_status);
}

//////////////////////////////////////////////////////////////////////////////
// Compute the soft threshold for this function
double SoftThreshold(const double alpha, const double gamma){
  double shrink = std::max(std::abs(alpha) - gamma, 0.0);
  return std::copysign(shrink, alpha);
}

// TODO: Add Primal-Dual Certificates.
// See Primal-Dual Rates and Certificates (2016) formula (18)
//    G(\alpha) = <w, A\alpha> + B[||A^Tw||_\infty-\lambda]_+ + \lambda||\alpha||_1
// Compute the duality gap for L1_regularized Problem.
// float ComputeL1DualityGap(const ModelWeights& model_weights, const Examples& examples,
//   const ComputeOptions& options){

//   // Part: \lambda||\alpha||_1
//   float l1 = options.regularizations.symmetric_l1();
//   float l1_regularized = l1 * model_weights.l1_norm();

//   // B = 1/\lambda f(0) = 1/(2\lambda) ||b||_2^2
//   float B = 0;
//   for (int i = 0; i < examples.num_examples(); ++i){
//     B += std::pow(examples.example(i).example_label(), 2);
//   }
//   B = B/(2*l1); 

//   // <w, A\alpha>
//   float w_A_alpha = 0;
//   for (size_t i = 0; i < examples.num_examples(); ++i){
//     const ExampleStatistics example_statistics =
//     examples.example(i).ComputeWxAndWeightedExampleNorm(
//         options.num_loss_partitions, model_weights,
//         options.regularizations, 1 /* num_weight_vectors */);

//     w_A_alpha += example_statistics.wx[0] * examples.weight(i);
//   }

//   float max_At_w = std::numeric_limits<float>::min();
//   for (int i = 0; i < examples.num_features(); ++i){
//     auto Ai = examples.Ai(i);
//     auto w  = examples.WAsCol();
//     // float sn =0;
//     // sn() = (Ai * w).sum();
//     Eigen::Tensor<float, 0, Eigen::RowMajor> sn = (Ai * w).sum();
//     max_At_w = std::max(std::abs(sn()), max_At_w);
//   }

//   return w_A_alpha + B * std::max(max_At_w - l1, static_cast<float>(0)) + l1_regularized;
// }

// TODO(shengx): The helper classes/methods are changed to support multiclass
// SDCA, which lead to changes within this function. Need to revisit the
// convergence once the multiclass SDCA is in.
// TODO: Add `example_weight` back.
// TODO: For the moment, the loss function is fixed to be squared norm.
void DoComputeDual(const ComputeOptions& options, OpKernelContext* const context){
  // Implement lasso solver: 
  //  1. dual variable `\alpha` is stored in ModelWeights
  //  2. primal weight `w = A\alpha - b` is not explictly stored. We store
  //      `v = A\alpha`. In distributed computing, we will push local `v` and 
  //      aggregate it in parameter server. When they are accumulated, we fetch
  //      updates from `v` and continue to the next step.
  //  3. Note that our code is based on the framework proxSDCA. The `features`
  //     below actually means the columns of `A` which in fact is the example 
  //     associated with `\alpha`.
  ModelWeights model_weights;
  OP_REQUIRES_OK(context, model_weights.Initialize(context));

  Examples examples;
  OP_REQUIRES_OK(
      context,
      examples.Initialize(context, model_weights, options.num_sparse_features,
                          options.num_sparse_features_with_values,
                          options.num_dense_features));

  const Tensor* example_state_data_t;
  OP_REQUIRES_OK(context,
                 context->input("example_state_data", &example_state_data_t));
  TensorShape expected_example_state_shape({examples.num_examples(), 4});
  OP_REQUIRES(context,
              example_state_data_t->shape() == expected_example_state_shape,
              errors::InvalidArgument(
                  "Expected shape ", expected_example_state_shape.DebugString(),
                  " for example_state_data, got ",
                  example_state_data_t->shape().DebugString()));

  Tensor mutable_example_state_data_t(*example_state_data_t);
  // In this function only the first column of `example_state_data` will be 
  // used. It is used for storing the intermediate variable `v =  A\alpha`.
  auto example_state_data = mutable_example_state_data_t.matrix<float>();
  context->set_output("out_example_state_data", mutable_example_state_data_t);

  if (options.adaptative) {
    OP_REQUIRES_OK(context,
                   examples.SampleAdaptativeProbabilities(
                       options.num_loss_partitions, options.regularizations,
                       model_weights, example_state_data, options.loss_updater,
                       /*num_weight_vectors =*/1));
  }

  int num_examples = examples.num_examples();

  // The slice `v` is used for dense features operation.
  auto v = example_state_data.slice(
    Eigen::array<int, 2>({0,0}), Eigen::array<int, 2>({num_examples, 1}));

  // Compute dot production of a dense feature and a residual.
  //              <A_k, r>
  // where r = b - A\alpha = b - v is the residual.
  auto dense_feature_dot_residual = [&](int k){
    const Eigen::Tensor<float, 0, Eigen::RowMajor> sn = 
        (examples.DenseFeatureAsMatrix(k) * (examples.labels() - v)).sum();
    return sn();
  };

  mutex mu;
  Status train_step_status GUARDED_BY(mu);
  std::atomic<std::int64_t> atomic_index(-1);
  // The minimization objective is:
  //    min D(\alpha) = 1/2 || b - A\alpha||_2^2 + \lambda_1 ||\alpha||_1
  // Apply coordinate descent to \alpha_i, the updated \alpha_i is given by 
  // proximal operator (soft-thresholding for squared norm):
  //    \alpha_i = S_{\lambda/||A_i||^2}(\frac{A_i^Tr}{||A_i||^2}+\alpha_i^{\text{old}})
  // where soft-thresholding is defined as 
  //    S_{\gamma}(g)=\text{sgn}(g) (|g|-\gamma)_+
  auto train_step_dense = [&](const int64 begin, const int64 end) {
    // The static_cast here is safe since begin and end can be at most
    // num_examples which is an int.
    for (int id = static_cast<int>(begin); id < end; ++id) {
      float ai_squared = examples.DenseFeatureSquaredNorm(id);

      // In case A_i is a vector of 0s, then the corresponding \alpha_i is 0 and 
      // don't update.
      if (std::abs(ai_squared) < 10e-10){
        continue;
      }

      // Compute: A_i^T*r/||A_i||^2+\alpha_i^{old}
      float ai_dot_residual = dense_feature_dot_residual(id);
      float alpha = model_weights.dense_weights()[id].weight();
      float candidate = ai_dot_residual/ai_squared + alpha;

      // Use soft-thresholding to perform coordinate descent.
      float new_alpha = SoftThreshold(candidate, 
        options.regularizations.symmetric_l1()/ai_squared);

      // Update delta dense.
      model_weights.UpdateDenseDeltaWeights(
          context->eigen_cpu_device(), new_alpha - alpha, id);

      // Primal weight is 
      //    W = \nabla f(A\alpha) = A\alpha - b
      // Thus we update it with
      //    \Delta W = A_k \Delta\alpha_k
      // We use V = W - b, so that use example_state_data to store v, initially 
      //    \alpha = 0, V = 0
      v += examples.DenseFeatureAsMatrix(id) * v.constant(new_alpha - alpha);
    }
  };

  // TODO(sibyl-Aix6ihai): Tune this properly based on sparsity of the data,
  // number of cpus, and cost per example.
  const int64 kCostPerUnit = examples.num_examples();
  const DeviceBase::CpuWorkerThreads& worker_threads =
      *context->device()->tensorflow_cpu_worker_threads();

  // Updating dense features
  Shard(worker_threads.num_threads, worker_threads.workers,
        options.num_dense_features, kCostPerUnit, train_step_dense);

  // Apply changes to `example_state_data`
  example_state_data.slice(
    Eigen::array<int, 2>({0,0}), Eigen::array<int, 2>({num_examples, 1})) = v;

  // Apply LASSO solver to sparse features. 
  auto train_step_sparse = [&](const int64 begin, const int64 end) {
    // The static_cast here is safe since begin and end can be at most
    // num_examples which is an int.
    for (int sfw_idx = static_cast<int>(begin); sfw_idx < end; ++sfw_idx) {
      const sdca::FeatureWeightsSparseStorage& sparse_weights =
          model_weights.sparse_weights()[sfw_idx];

      int num_features = sparse_weights.num_weights();

      // This 2D object is used to store (example_index, feature_value) pair.
      // The first dimension is the index of the feature in this feature group.
      std::vector<std::vector<std::pair<int, float> > > feature_values(num_features);

      std::vector<float> ai_squared(num_features);
      std::vector<float> ai_dot_residual(num_features);

      // Compute squared norm of a column and construct `feature_values` at the 
      // same time.
      for (int example_id = 0; example_id < num_examples; ++example_id){
        const Example& example = examples.example(example_id);
        const Example::SparseFeatures& sf = example.sparse_feature(sfw_idx);
        for (int i = 0; i < sf.indices->size(); ++i){
          int id = sparse_weights.indices_to_id((*sf.indices)(i));
          float feature_value = sf.values == nullptr ? 1.0 : (*sf.values)(i);
          ai_squared[id] += feature_value * feature_value;
          feature_values[id].push_back(std::make_pair(example_id, feature_value));
        }
      } 

      for (int id = 0; id < num_features; ++id){
        // If this feature is almost 0, then we don't do the proximal step.
        if (std::abs(ai_squared[id]) < 10e-10) {
          continue;
        }

        int64 indices = sparse_weights.id_to_indices(id);

        float alpha = sparse_weights.nominals_by_id(0, id) 
                    + sparse_weights.deltas_by_id(0, id);

        // compute inner production of a sparse feature with feature value
        float ai_dot_residual = 0;
        for (int i = 0; i < feature_values[id].size(); ++i){
          int example_id = feature_values[id][i].first;
          ai_dot_residual += feature_values[id][i].second * 
          (examples.labels()(example_id, 0) - example_state_data(example_id, 0));
        }

        float candidate = ai_dot_residual/ai_squared[id] + alpha;

        // Apply softthresholding in this coordinate
        float new_alpha = SoftThreshold(candidate, 
          options.regularizations.symmetric_l1()/ai_squared[id]);

        float delta_alpha = new_alpha - alpha;

        // Update sparse delta weights
        model_weights.UpdateSparseDeltaWeights(context->eigen_cpu_device(), 
          delta_alpha, sfw_idx, indices);

        // Update `v`
        for (int i = 0; i < feature_values[id].size(); ++i){
          int example_id = feature_values[id][i].first;
          example_state_data(example_id, 0) += feature_values[id][i].second * delta_alpha;
        }
      }
    }
  };

  //  Number of features in each group cane be unbalanced. Tune this parameter 
  //  for better performance.
  Shard(worker_threads.num_threads, worker_threads.workers,
        options.num_sparse_features, kCostPerUnit, train_step_sparse);

  OP_REQUIRES_OK(context, train_step_status);
}

}  // namespace

class SdcaOptimizer : public OpKernel {
 public:
  explicit SdcaOptimizer(OpKernelConstruction* const context)
      : OpKernel(context), options_(context) {}

  void Compute(OpKernelContext* context) override {
    // The input attribute 'dual_method' specifies the method to be used. 
    // If it is false, we use primal solver. Otherwise, we use dual method.
    
    auto t1 = std::chrono::high_resolution_clock::now();
    if (!options_.dual_method){
      DoCompute(options_, context);
    } else {
      DoComputeDual(options_, context);
    }
    auto t2 = std::chrono::high_resolution_clock::now();
    std::cout << "Compute took "
              << std::chrono::duration_cast<std::chrono::milliseconds>(t2-t1).count()
              << " milliseconds\n";    
  }

 private:
  // TODO(sibyl-Aix6ihai): We could use the type-constraint on loss_type, and
  // template the entire class to avoid the virtual table lookup penalty in
  // the inner loop.
  ComputeOptions options_;
};
REGISTER_KERNEL_BUILDER(Name("SdcaOptimizer").Device(DEVICE_CPU),
                        SdcaOptimizer);

class SdcaShrinkL1 : public OpKernel {
 public:
  explicit SdcaShrinkL1(OpKernelConstruction* const context)
      : OpKernel(context) {
    OP_REQUIRES_OK(context, regularizations_.Initialize(context));
  }

  void Compute(OpKernelContext* context) override {
    OpMutableInputList weights_inputs;
    OP_REQUIRES_OK(context,
                   context->mutable_input_list("weights", &weights_inputs));

    auto do_work = [&](const int64 begin, const int64 end) {
      for (int i = begin; i < end; ++i) {
        auto prox_w = weights_inputs.at(i, /*lock_held=*/true).flat<float>();
        prox_w.device(context->eigen_cpu_device()) =
            regularizations_.EigenShrinkVector(prox_w);
      }
    };

    if (weights_inputs.size() > 0) {
      int64 num_weights = 0;
      for (int i = 0; i < weights_inputs.size(); ++i) {
        num_weights += weights_inputs.at(i, /*lock_held=*/true).NumElements();
      }
      // TODO(sibyl-Aix6ihai): Tune this value.
      const int64 kCostPerUnit = (num_weights * 50) / weights_inputs.size();
      const DeviceBase::CpuWorkerThreads& worker_threads =
          *context->device()->tensorflow_cpu_worker_threads();
      Shard(worker_threads.num_threads, worker_threads.workers,
            weights_inputs.size(), kCostPerUnit, do_work);
    }
  }

 private:
  Regularizations regularizations_;
};
REGISTER_KERNEL_BUILDER(Name("SdcaShrinkL1").Device(DEVICE_CPU), SdcaShrinkL1);

// Computes platform independent, compact and unique (with very high
// probability) representation of an example id. It shouldn't be put in
// persistent storage, as its implementation may change in the future.
//
// The current probability of at least one collision for 1B example_ids is
// approximately 10^-21 (ie 2^60 / 2^129).
class SdcaFprint : public OpKernel {
 public:
  explicit SdcaFprint(OpKernelConstruction* const context)
      : OpKernel(context) {}

  void Compute(OpKernelContext* context) override {
    const Tensor& input = context->input(0);
    OP_REQUIRES(context, TensorShapeUtils::IsVector(input.shape()),
                errors::InvalidArgument("Input must be a vector, got shape ",
                                        input.shape().DebugString()));
    Tensor* out;
    const int64 num_elements = input.NumElements();
    OP_REQUIRES_OK(context, context->allocate_output(
                                0, TensorShape({num_elements, 2}), &out));

    const auto in_values = input.flat<string>();
    auto out_values = out->matrix<int64>();

    for (int64 i = 0; i < num_elements; ++i) {
      const Fprint128 fprint = Fingerprint128(in_values(i));
      // Never return 0 or 1 as the first value of the hash to allow these to
      // safely be used as sentinel values (e.g. dense hash table empty key).
      out_values(i, 0) = TF_PREDICT_TRUE(fprint.low64 >= 2)
                             ? fprint.low64
                             : fprint.low64 + ~static_cast<uint64>(1);
      out_values(i, 1) = fprint.high64;
    }
  }
};
REGISTER_KERNEL_BUILDER(Name("SdcaFprint").Device(DEVICE_CPU), SdcaFprint);

}  // namespace tensorflow
