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

#include <stddef.h>
#include <atomic>
#include <cmath>
#include <functional>
#include <limits>
#include <string>
#include <unordered_set>

#include <iostream>

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
#include "tensorflow/core/kernels/smooth-hinge-loss.h"
#include "tensorflow/core/kernels/squared-loss.h"
#include "tensorflow/core/lib/core/coding.h"
#include "tensorflow/core/lib/core/errors.h"
#include "tensorflow/core/lib/core/status.h"
#include "tensorflow/core/lib/core/stringpiece.h"
#include "tensorflow/core/lib/gtl/inlined_vector.h"
#include "tensorflow/core/lib/hash/hash.h"
#include "tensorflow/core/lib/random/distribution_sampler.h"
#include "tensorflow/core/lib/strings/stringprintf.h"
#include "tensorflow/core/platform/fingerprint.h"
#include "tensorflow/core/platform/mutex.h"
#include "tensorflow/core/platform/types.h"
#include "tensorflow/core/util/guarded_philox_random.h"
#include "tensorflow/core/util/sparse/group_iterator.h"
#include "tensorflow/core/util/sparse/sparse_tensor.h"
#include "tensorflow/core/util/work_sharder.h"


namespace tensorflow {

namespace {

using UnalignedFloatVector = TTypes<const float>::UnalignedConstVec;
using UnalignedInt64Vector = TTypes<const int64>::UnalignedConstVec;

// Statistics computed with input (ModelWeights, Feature)
struct FeatureStatistics{
  // Ai is a column of Y*X^T, r = b - A*\alpha is residual.
  // Using InlinedVector is to avoid heap allocation for small number of
  // classes, and 3 is chosen to minimize memory usage for the multiclass case.
  gtl::InlinedVector<double, 3> Air;
  double normalized_squared_norm_inv = 0;

  FeatureStatistics(const int num_weight_vectors)
      : Air(num_weight_vectors, 0.0){}
};

class Regularizations {
 public:
  Regularizations(){};

  // Initialize() must be called immediately after construction.
  Status Initialize(OpKernelConstruction* const context) {
    TF_RETURN_IF_ERROR(context->GetAttr("l1", &symmetric_l1_));
    TF_RETURN_IF_ERROR(context->GetAttr("l2", &symmetric_l2_));
    shrinkage_ = symmetric_l1_ / symmetric_l2_;
    return Status::OK();
  }

    // Vectorized float variant of the above.
  Eigen::Tensor<float, 1, Eigen::RowMajor> EigenShrinkVector(
      const Eigen::Tensor<float, 1, Eigen::RowMajor> weights) const {
    // Proximal step on the weights which is sign(w)*|w - shrinkage|+.
    return weights.sign() * ((weights.abs() - weights.constant(shrinkage_))
                                 .cwiseMax(weights.constant(0.0)));
  }

  // Matrix float variant of the above.
  Eigen::Tensor<float, 2, Eigen::RowMajor> EigenShrinkMatrix(
      const Eigen::Tensor<float, 2, Eigen::RowMajor> weights) const {
    // Proximal step on the weights which is sign(w)*|w - shrinkage|+.
    return weights.sign() * ((weights.abs() - weights.constant(shrinkage_))
                                 .cwiseMax(weights.constant(0.0)));
  }


  float symmetric_l2() const { return symmetric_l2_; }
  float symmetric_l1() const { return symmetric_l1_; }

 private:
  float symmetric_l1_ = 0;
  float symmetric_l2_ = 0;

  // L1 divided by L2, pre-computed for use during weight shrinking.
  double shrinkage_ = 0;

  TF_DISALLOW_COPY_AND_ASSIGN(Regularizations);
};

class ModelWeights;
class Residual;

// Represent one feature of the A=YX^T.
class Feature{
public:
  const FeatureStatistics get_feature_statistics(
      const Regularizations& regularization, const Residual& residual, 
      const int num_weight_vectors) const;

  double squared_norm() const {return squared_norm_;}

  // A dense vector which is a col-slice of the underlying matrix.
  struct DenseVector {
    // Returns a col slice from the matrix.
    Eigen::TensorMap<Eigen::Tensor<const float, 1, Eigen::RowMajor>> Col()
        const {
      return Eigen::TensorMap<Eigen::Tensor<const float, 1, Eigen::RowMajor>>(
          data_matrix.data(), data_matrix.dimension(0));
    }

    // Returns a col slice as a N * 1 matrix, where N is the number of examples.
    Eigen::TensorMap<Eigen::Tensor<const float, 2, Eigen::RowMajor>>
    ColAsMatrix() const {
      return Eigen::TensorMap<Eigen::Tensor<const float, 2, Eigen::RowMajor>>(
          data_matrix.data(), data_matrix.dimension(0), 1);
    }

    const TTypes<float>::ConstMatrix data_matrix; // Matrix A = Y*X^T
    // TODO: don't need this.
    const int64 col_index;
    };
  private:
    // std::vector<std::unique_ptr<DenseVector>> dense_vectors_;
    std::unique_ptr<DenseVector> feature_;

    // float  feature_label_  = 0; 
    // float  feature_weight_ = 0; 
    double squared_norm_   = 0;  // sum squared norm of the features.

    // Examples fills Example in a multi-threaded way.
    friend class Features;

    // ModelWeights use each example for model update w += \alpha * x_{i};
    friend class ModelWeights;
    friend class Residual;
};

class FeatureWeightsDenseStorage {
 public:
  FeatureWeightsDenseStorage(const TTypes<const float>::Matrix nominals,
                             TTypes<float>::Matrix deltas)
      : nominals_(nominals), deltas_(deltas) {}

  // Check if a feature index is with-in the bounds.
  bool IndexValid(const int64 index) const {
    return index >= 0 && index < deltas_.dimension(0);
  }

  // Nominals here are the original weight matrix.
  TTypes<const float>::Matrix nominals() const { return nominals_; }

  // Delta weights durining mini-batch updates.
  TTypes<float>::Matrix deltas() const { return deltas_; }

  // Updates delta weights based on active dense features in the example and
  // the corresponding dual residual.
  void UpdateDenseDeltaWeights(const Eigen::ThreadPoolDevice& device,
      const std::vector<double>& delta_alpha) {
      deltas_.device(device) +=  deltas_.constant(delta_alpha[0]);
  }

  float get_weight() const{
    return nominals_(0, 0) + deltas_(0, 0);
  }

 private:
  // The nominal value of the weight for a feature (indexed by its id).
  const TTypes<const float>::Matrix nominals_;
  // The accumulated delta weight for a feature (indexed by its id).
  TTypes<float>::Matrix deltas_;
};

// Weights in the model, wraps both current weights, and the delta weights
// for both sparse and dense features.
class ModelWeights {
 public:
  ModelWeights() {}

  bool DenseIndexValid(const int col, const int64 index) const {
    return dense_weights_[col].IndexValid(index);
  }

  // Go through all the features present in the example, and update the
  // weights based on the dual delta.
  void UpdateDeltaWeights(
      const Eigen::ThreadPoolDevice& device, const int feature_index,
      const std::vector<double>& delta_alpha) {
      dense_weights_[feature_index].UpdateDenseDeltaWeights(device, delta_alpha);
  }

  Status Initialize(OpKernelContext* const context) {
    OpInputList dense_weights_inputs;
    TF_RETURN_IF_ERROR(
        context->input_list("dense_weights", &dense_weights_inputs));

    OpOutputList sparse_weights_outputs;
    TF_RETURN_IF_ERROR(context->output_list("out_delta_sparse_weights",
                                            &sparse_weights_outputs));

    OpOutputList dense_weights_outputs;
    TF_RETURN_IF_ERROR(context->output_list("out_delta_dense_weights",
                                            &dense_weights_outputs));

    // Reads in the weights, and allocates and initializes the delta weights.
    const auto intialize_weights = [&](
        const OpInputList& weight_inputs, OpOutputList* const weight_outputs,
        std::vector<FeatureWeightsDenseStorage>* const feature_weights) {
      for (int i = 0; i < weight_inputs.size(); ++i) {
        Tensor* delta_t;
        weight_outputs->allocate(i, weight_inputs[i].shape(), &delta_t);
        // Convert the input vector to a row matrix in internal representation.
        auto deltas = delta_t->shaped<float, 2>({delta_t->NumElements(), 1});
        deltas.setZero();

        // push_back a is column vector
        feature_weights->emplace_back(
            FeatureWeightsDenseStorage{weight_inputs[i].shaped<float, 2>(
                                           {weight_inputs[i].NumElements(), 1}),
                                       deltas});
        std::cout << "Initiate ModelWeight: " << i << "th weight = "
            << (*feature_weights)[i].nominals()(0, 0)
            << ", shape0 = " << ((*feature_weights)[i].nominals()).dimension(0)
            << ", shape1 = " << ((*feature_weights)[i].nominals()).dimension(1)
            << std::endl;
      }
    };

    // TODO: use shard
    intialize_weights(dense_weights_inputs, &dense_weights_outputs,
                      &dense_weights_);

    return Status::OK();
  }

  const std::vector<FeatureWeightsDenseStorage>& dense_weights() const {
    return dense_weights_;
  }

 private:
  std::vector<FeatureWeightsDenseStorage> dense_weights_;

  TF_DISALLOW_COPY_AND_ASSIGN(ModelWeights);
};



// Features contains all the training examples that SDCA uses for a mini-batch.
class Features {
 public:
  Features() {}

  // Returns the Feature at |example_index|.
  const Feature& feature(const int feature_index) const {
    return features_.at(feature_index);
  }

  // Feature* get_feature(const int feature_index){
  //   return &(features_[feature_index]);
  // }

  int sampled_index(const int id, const bool adaptative) const {
    if (adaptative) return sampled_index_[id];
    return id;
  }

  int num_examples() const { return num_examples_; }

  int num_features() const { return num_features_; }

  // Initialize() must be called immediately after construction.
  // TODO(sibyl-Aix6ihai): Refactor/shorten this function.
  Status Initialize(OpKernelContext* const context, 
                    const ModelWeights& weights, 
                    int num_dense_features);

 private:
  // Reads the input tensors, and builds the internal representation for dense
  // features per example. This function modifies the |examples| passed in
  // to build the sparse representations.
  // static Status CreateDenseFeatureRepresentation(
  //     const DeviceBase::CpuWorkerThreads& worker_threads, 
  //     int num_examples,
  //     int num_dense_features, 
  //     const ModelWeights& weights,
  //     const OpInputList& dense_features_inputs,
  //     std::vector<Feature>* const features);

  static Status CreateDenseFeatureRepresentation(
    const DeviceBase::CpuWorkerThreads& worker_threads, 
    const int num_examples,
    const int num_dense_features, 
    const ModelWeights& weights,
    const OpInputList& dense_features_inputs,
    std::vector<Feature>* const features);

  // Computes squared example norm per example i.e |x|^2. This function modifies
  // the |examples| passed in and adds the squared norm per example.
  static void ComputeSquaredNormPerFeature(
      const DeviceBase::CpuWorkerThreads& worker_threads, int num_examples,
      int num_dense_features, std::vector<Feature>* const features);

  // All examples in the batch.
  std::vector<Feature> features_;

  // Adaptative sampling variables
  std::vector<int> sampled_index_;
  std::vector<int> sampled_count_;

  int num_features_ = 0;
  int num_examples_ = 0;

  TF_DISALLOW_COPY_AND_ASSIGN(Features);
};

Status Features::Initialize(OpKernelContext* const context,
                            const ModelWeights& weights,
                            const int num_dense_features) {
  num_features_ = num_dense_features;

  std::cout << "Features::Initialize 1" << std::endl;

  const Tensor* example_weights_t;
  TF_RETURN_IF_ERROR(context->input("example_weights", &example_weights_t));
  auto example_weights = example_weights_t->flat<float>();

  if (example_weights.size() >= std::numeric_limits<int>::max()) {
    return errors::InvalidArgument(strings::Printf(
        "Too many examples in a mini-batch: %zu > %d", example_weights.size(),
        std::numeric_limits<int>::max()));
  }

  std::cout << "Features::Initialize 2" << std::endl;

  // The static_cast here is safe since num_examples can be at max an int.
  const int num_examples = static_cast<int>(example_weights.size());
  // const Tensor* example_labels_t;
  // TF_RETURN_IF_ERROR(context->input("example_labels", &example_labels_t));
  // auto example_labels = example_labels_t->flat<float>();

  num_examples_ = num_examples;

  OpInputList dense_features_inputs;
  TF_RETURN_IF_ERROR(
      context->input_list("dense_features", &dense_features_inputs));

  features_.clear();
  features_.resize(num_features_);
  sampled_index_.resize(num_features_);
  sampled_count_.resize(num_features_);

  std::cout << "Features::Initialize 3" << std::endl;

  // for (int feature_id = 0; feature_id < num_features; ++feature_id) {
  //   Feature* const feature = &features_[feature_id];
  //   feature->dense_vectors_.resize(num_dense_features);
  // }

  const DeviceBase::CpuWorkerThreads& worker_threads =
      *context->device()->tensorflow_cpu_worker_threads();

  std::cout << "Features::Initialize 4" << std::endl;

  TF_RETURN_IF_ERROR(CreateDenseFeatureRepresentation(
      worker_threads, num_examples, num_dense_features, weights,
      dense_features_inputs, &features_));

    std::cout << "Features::Initialize 5" << std::endl;


  ComputeSquaredNormPerFeature(worker_threads, num_examples,
                               num_dense_features, &features_);

  std::cout << "Features::Initialize 6" << std::endl;
  return Status::OK();
}

Status Features::CreateDenseFeatureRepresentation(
    const DeviceBase::CpuWorkerThreads& worker_threads, 
    const int num_examples,
    const int num_dense_features, 
    const ModelWeights& weights,
    const OpInputList& dense_features_inputs,
    std::vector<Feature>* const features) {
  mutex mu;
  Status result GUARDED_BY(mu);
  auto parse_partition = [&](const int64 begin, const int64 end) {
    // The static_cast here is safe since begin and end can be at most
    // num_features which is an int.
    for (int i = static_cast<int>(begin); i < end; ++i) {
      auto dense_features = dense_features_inputs[i].template matrix<float>();

      (*features)[i].feature_.reset(
        new Feature::DenseVector{dense_features, i});
      std::cout << "Creating Dense Features Representation...\n"
            << "shape of " << i << "th is: " << 
            (*features)[i].feature_->data_matrix.dimension(0) 
            << ", " << (*features)[i].feature_->data_matrix.dimension(1)
            << std::endl;

      if (!weights.DenseIndexValid(i, dense_features.dimension(1) - 1)) {
        mutex_lock l(mu);
        result = errors::InvalidArgument(
            "More dense features than we have parameters for: ",
            dense_features.dimension(1));
        return;
      }
    }
  };
  // TODO(sibyl-Aix6ihai): Tune this as a function of dataset size.
  const int64 kCostPerUnit = num_examples;
  Shard(worker_threads.num_threads, worker_threads.workers, num_dense_features,
        kCostPerUnit, parse_partition);
  return result;
}

void Features::ComputeSquaredNormPerFeature(
    const DeviceBase::CpuWorkerThreads& worker_threads, const int num_examples,
    const int num_dense_features, std::vector<Feature>* const features) {
  // Compute norm of examples.
  auto compute_feature_norm = [&](const int64 begin, const int64 end) {
    // The static_cast here is safe since begin and end can be at most
    // num_examples which is an int.
    for (int feature_id = static_cast<int>(begin); feature_id < end;
         ++feature_id) {
      // double squared_norm = 0;
      Feature* const feature = &(*features)[feature_id];
      const Eigen::Tensor<float, 0, Eigen::RowMajor> sn = 
        feature->feature_->Col().square().sum();
      feature->squared_norm_ = sn();
      std::cout << "feature_id = " << feature_id 
          << ", squared_norm_ = " << sn() << std::endl;
      // std::cout << "column "<<feature_id << "******************************************"<<std::endl;
      // std::cout << feature->feature_->Col() << std::cout;
      // for (int j = 0; j < num_dense_features; ++j) {
      //   const Eigen::Tensor<float, 0, Eigen::RowMajor> sn =
      //       feature->dense_vectors_[j]->Col().square().sum();
      //   squared_norm += sn();
      // }
      // feature->squared_norm_ = squared_norm;
    }
  };
  // TODO(sibyl-Aix6ihai): Compute the cost optimally.
  const int64 kCostPerUnit = num_examples;
  Shard(worker_threads.num_threads, worker_threads.workers, num_dense_features,
        kCostPerUnit, compute_feature_norm);
}

class Residual{
public:

  Residual(const ModelWeights& model_weights, const Features& features, 
    TTypes<float>::Matrix init_r)
    :r_(init_r) {
    // int num_examples = features.num_examples();
    int num_features = features.num_features();
    // Tensor<float, 1, RowMajor> b(num_examples);
    // r_ = TTypes<float>(b.constant(1).data(), num_examples,1);

    for (int i = 0; i < num_features; i++){
      Eigen::Tensor<float, 2, Eigen::RowMajor> weight = model_weights.dense_weights()[i].nominals() 
                   + model_weights.dense_weights()[i].deltas();
      std::cout << "Residual Loop " << i << "th turn. Dims = " << weight.dimension(0)
                << ", " << weight.dimension(1) << ". Weight = " << weight << std::endl;
      const Feature& feature = features.feature(i);
      r_ = r_ - feature.feature_->ColAsMatrix() * 
          (feature.feature_->ColAsMatrix()).constant(weight(0, 0));
    }

    // std::cout << "End of Initializing residual, residual = \n" << r_ << std::endl;
  }

  void update_residual(const Feature& feature, const double delta_alpha){
    r_ -= feature.feature_->Col() * (feature.feature_->Col().constant(delta_alpha));
  }

  double compute_squared_norm(){
    const Eigen::Tensor<float, 0, Eigen::RowMajor> squared_norm = r_.square().sum();
    return squared_norm();
  }

  // double compute_Air(Features& features){
  //   for (int i = 0; i < features.num_features(); i++){
  //     auto feature = features[i];
  //     auto tmp = (feature.feature_.ColAsMatrix()*r_).sum();
  //     feature->Air_ = tmp();
  //   }
  // }

  const TTypes<float>::Matrix& residual() const{
    return r_;
  }

private:
  TTypes<float>::Matrix r_;
};

// Computes the example statistics for given example, and model. Defined here
// as we need definition of ModelWeights and Regularizations.
const FeatureStatistics Feature::get_feature_statistics(
      const Regularizations& regularization, const Residual& residual,
      const int num_weight_vectors) const {
  FeatureStatistics result(num_weight_vectors);

  result.normalized_squared_norm_inv =
      regularization.symmetric_l1() / squared_norm_;

  const Eigen::Tensor<float, 0, Eigen::RowMajor> tmp = (feature_->ColAsMatrix()*residual.residual()).sum();
  result.Air[0] = tmp();

  return result;
}

struct ComputeOptions {
  ComputeOptions(OpKernelConstruction* const context) {
    string loss_type;
    OP_REQUIRES_OK(context, context->GetAttr("loss_type", &loss_type));
    if (loss_type == "hinge_loss") {
      loss_updater.reset(new HingeLossUpdater);
    } else {
      OP_REQUIRES(context, false, errors::InvalidArgument(
                                      "Unsupported loss type: ", loss_type));
    }

    OP_REQUIRES_OK(context, context->GetAttr("adaptative", &adaptative));
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
  Regularizations regularizations;
};

double SoftThreshold(const double alpha, const double gamma){
  double shrink = std::max(std::abs(alpha) - gamma, 0.0);
  return std::copysign(shrink, alpha);
}

double ComputeUpdatedDual(const int num_loss_partitions, 
                        const double example_weight,
                        const double current_alpha, 
                        const double Ai_r_divided_by_l1,
                        const double weighted_feature_norm) {

  // Ai : i-th column of matrix A=Y*tX, (num_examples, num_features)
  // weighted_feature_norm : \lambda_{l1}/||A_i||^2

  // #### Lasso problem
  // $$ f(x) = \frac{1}{2}||b-Ax||^2 + \lambda ||x||_1$$
  // Closed form of **coordinate descent** for each coordinate (feature):
  // $$x_i = S_{\lambda/||A_i||^2}\left(\frac{A_i^Tr}{||A_i||^2}+x_i^{\text{old}}\right)$$
  // where **soft-thresholding** operator is defined as:
  // $$S_{\gamma}(g)=\text{sgn}(g) (|g|-\gamma)_+$$

  const double candidate_optimal_alpha = current_alpha + Ai_r_divided_by_l1 * weighted_feature_norm;
  return SoftThreshold(candidate_optimal_alpha, weighted_feature_norm);
}


// TODO(shengx): The helper classes/methods are changed to support multiclass
// SDCA, which lead to changes within this function. Need to revisit the
// convergence once the multiclass SDCA is in.
void DoCompute(const ComputeOptions& options, OpKernelContext* const context) {
  std::cout << "Begin doing Computing" << std::endl;
  ModelWeights model_weights;
  OP_REQUIRES_OK(context, model_weights.Initialize(context));

  std::cout << "After Initializing Model weight" << std::endl;
  Features features;
  OP_REQUIRES_OK(
      context,
      features.Initialize(context, model_weights, options.num_dense_features));

  std::cout << "After Initializing Features" << std::endl;
  Eigen::Tensor<float, 2, Eigen::RowMajor> b = 
      (Eigen::Tensor<float, 2, Eigen::RowMajor>
        (features.num_examples(), 1)).constant(1.0);
  // std::cout << "b = \n" << b << std::endl;
  Residual residual(model_weights, features, b);

  std::cout << "After Creating Residuals" << std::endl;

  const Tensor* example_state_data_t;
  OP_REQUIRES_OK(context,
                 context->input("example_state_data", &example_state_data_t));

  // // Allocate a 
  // Tensor* out_primal_loss;
  // const int64 num_elements = input.NumElements();
  // OP_REQUIRES_OK(context, context->allocate_output(
  //                             0, TensorShape({num_elements, 2}), &out_primal_loss));

  
  //  TODO: features.num_examples() to features.num_features()
  TensorShape expected_example_state_shape({features.num_examples(), 4});
  OP_REQUIRES(context,
              example_state_data_t->shape() == expected_example_state_shape,
              errors::InvalidArgument(
                  "Expected shape ", expected_example_state_shape.DebugString(),
                  " for example_state_data, got ",
                  example_state_data_t->shape().DebugString()));

  std::cout << "After get example state_data" << std::endl;

  Tensor mutable_example_state_data_t(*example_state_data_t);
  auto example_state_data = mutable_example_state_data_t.matrix<float>();
  context->set_output("out_example_state_data", mutable_example_state_data_t);

  std::cout << "After setting output data." << std::endl;

  // Check Input
  OP_REQUIRES(context, options.regularizations.symmetric_l1() != 0.0,
    errors::InvalidArgument("l1 should not be zero"));
  

  mutex mu;
  Status train_step_status GUARDED_BY(mu);
  std::atomic<std::int64_t> atomic_index(-1);
  auto train_step = [&](const int64 begin, const int64 end) {
    // The static_cast here is safe since begin and end can be at most
    // num_examples which is an int.
      std::cout << "Beginning of traning step." << std::endl; 
    for (int id = static_cast<int>(begin); id < end; ++id) {
      const int64 feature_index =
          features.sampled_index(++atomic_index, options.adaptative);
      std::cout << "After selecting feature" << std::endl;
      const Feature& feature = features.feature(feature_index);
      std::cout << "After Getting feature" << std::endl;
      // const float alpha = example_state_data(feature_index, 0);
      const float alpha = 
        model_weights.dense_weights()[feature_index].get_weight();

      std::cout << "After getting alpha feature" << std::endl;
      const float example_weight = 1.0;
      // float example_label = feature.example_label();
      // const Status conversion_status =
      //     options.loss_updater->ConvertLabel(&example_label);
      // if (!conversion_status.ok()) {
      //   mutex_lock l(mu);
      //   train_step_status = conversion_status;
      //   // Return from this worker thread - the calling thread is
      //   // responsible for checking context status and returning on error.
      //   return;
      // }

      const FeatureStatistics feature_statistics = 
            feature.get_feature_statistics(options.regularizations, residual, 1);
      std::cout << "After getting feature_statistics" << std::endl;


      const double new_alpha = ComputeUpdatedDual(
          options.num_loss_partitions, 
          example_weight, 
          alpha,
          feature_statistics.Air[0]/options.regularizations.symmetric_l1(), 
          feature_statistics.normalized_squared_norm_inv);

      std::cout << "--- Air = " << feature_statistics.Air[0] << std::endl;
      std::cout << "--- l1  = " << options.regularizations.symmetric_l1() << std::endl;
      std::cout << "--- inv = " << feature_statistics.normalized_squared_norm_inv/options.regularizations.symmetric_l1() << std::endl;
      std::cout << "After ComputeUpdatedDual" << std::endl;
      // Compute new weights.
      // const double normalized_bounded_dual_delta =
      //     (new_dual - dual) * example_weight /
      //     options.regularizations.symmetric_l2();

      const double delta_alpha = new_alpha - alpha;
      model_weights.UpdateDeltaWeights(
          context->eigen_cpu_device(), feature_index,
          std::vector<double>{delta_alpha});
      std::cout << "After UpdateDeltaWeights" << std::endl;

      residual.update_residual(feature, delta_alpha);
      std::cout << "After update residual" << std::endl;

      // feature.update_Air(delta_alpha);

      // Update example data.
      example_state_data(feature_index, 0) = new_alpha;
      example_state_data(feature_index, 1) = residual.compute_squared_norm()/2;
      example_state_data(feature_index, 2) = 0;
      example_state_data(feature_index, 3) = example_weight;
      std::cout << "After update residual" << std::endl;
      std::cout << "round " << id << ", alpha = " << alpha
      << ", new_alpha = " << new_alpha
      << "loss = " << example_state_data(feature_index, 1) << std::endl;
      std::cout << "========================================================" << std::endl;

    }
  };
  // TODO(sibyl-Aix6ihai): Tune this properly based on sparsity of the data,
  // number of cpus, and cost per example.
  const int64 kCostPerUnit = features.num_examples();
  const DeviceBase::CpuWorkerThreads& worker_threads =
      *context->device()->tensorflow_cpu_worker_threads();

  Shard(worker_threads.num_threads, worker_threads.workers,
        features.num_features(), kCostPerUnit, train_step);
  OP_REQUIRES_OK(context, train_step_status);

  std::cout << "\n+++++++++++++++++++++++ weights +++++++++++++" << std::endl;
  for (int i = 0; i < features.num_features(); ++i){
    std::cout << "weight " << i << " = " << model_weights.dense_weights()[i].get_weight() << std::endl;
  }
  std::cout << "++++++++++++++++++++++++ weights +++++++++++++\n" << std::endl;
}

} //namespace

class SdcaOptimizer : public OpKernel {
 public:
  explicit SdcaOptimizer(OpKernelConstruction* const context)
      : OpKernel(context), options_(context) {}

  void Compute(OpKernelContext* const context) override {
    DoCompute(options_, context);
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

  void Compute(OpKernelContext* const context) override {
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

  void Compute(OpKernelContext* const context) override {
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


}