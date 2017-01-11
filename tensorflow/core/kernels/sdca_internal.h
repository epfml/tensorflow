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

#ifndef TENSORFLOW_KERNELS_SDCA_INTERNAL_H_
#define TENSORFLOW_KERNELS_SDCA_INTERNAL_H_

#define EIGEN_USE_THREADS

#include <stddef.h>
#include <algorithm>
#include <cmath>
#include <memory>
#include <new>
#include <unordered_map>
#include <utility>
#include <vector>

#include "third_party/eigen3/unsupported/Eigen/CXX11/Tensor"
#include "tensorflow/core/framework/device_base.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/framework/tensor_shape.h"
#include "tensorflow/core/framework/tensor_types.h"
#include "tensorflow/core/framework/types.h"
#include "tensorflow/core/kernels/loss.h"
#include "tensorflow/core/lib/core/coding.h"
#include "tensorflow/core/lib/core/errors.h"
#include "tensorflow/core/lib/core/status.h"
#include "tensorflow/core/lib/core/stringpiece.h"
#include "tensorflow/core/lib/gtl/inlined_vector.h"
#include "tensorflow/core/lib/random/distribution_sampler.h"
#include "tensorflow/core/lib/strings/stringprintf.h"
#include "tensorflow/core/util/guarded_philox_random.h"
#include "tensorflow/core/util/sparse/group_iterator.h"
#include "tensorflow/core/util/sparse/sparse_tensor.h"
#include "tensorflow/core/util/work_sharder.h"

namespace tensorflow {

namespace sdca {

// Statistics computed with input (ModelWeights, Example).
struct ExampleStatistics {
  // Logits for each class.
  // For binary case, this should be a vector of length 1; while for multiclass
  // case, this vector has the same length as the number of classes, where each
  // value corresponds to one class.
  // Use InlinedVector to avoid heap allocation for small number of classes.
  gtl::InlinedVector<double, 1> wx;

  // Logits for each class, using the previous weights.
  gtl::InlinedVector<double, 1> prev_wx;

  // Sum of squared feature values occurring in the example divided by
  // L2 * sum(example_weights).
  double normalized_squared_norm = 0;

  // Num_weight_vectors equals to the number of classification classes in the
  // multiclass case; while for binary case, it is 1.
  ExampleStatistics(const int num_weight_vectors)
      : wx(num_weight_vectors, 0.0), prev_wx(num_weight_vectors, 0.0) {}
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

  // Proximal SDCA shrinking for L1 regularization.
  double Shrink(const double weight) const {
    const double shrinked = std::max(std::abs(weight) - shrinkage_, 0.0);
    if (shrinked > 0.0) {
      return std::copysign(shrinked, weight);
    }
    return 0.0;
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

// Struct describing a single example.
class Example {
 public:
  // Compute matrix vector product between weights (a matrix) and features
  // (a vector). This method also computes the normalized example norm used
  // in SDCA update.
  // For multiclass case, num_weight_vectors equals to the number of classes;
  // while for binary case, it is 1.
  const ExampleStatistics ComputeWxAndWeightedExampleNorm(
      const int num_loss_partitions, const ModelWeights& model_weights,
      const Regularizations& regularization,
      const int num_weight_vectors) const;

  float example_label() const { return example_label_; }

  float example_weight() const { return example_weight_; }

  double squared_norm() const { return squared_norm_; }

  // Sparse features associated with the example.
  // Indices and Values are the associated feature index, and values. Values
  // can be optionally absent, in which we case we implicitly assume a value of
  // 1.0f.
  struct SparseFeatures {
    std::unique_ptr<TTypes<const int64>::UnalignedConstVec> indices;
    std::unique_ptr<TTypes<const float>::UnalignedConstVec> values;  // nullptr encodes optional.

    // float query_value(const int64 index) const{
    //   for (int i = 0; i < indices->size(); i++){
    //     if (index == (*indices)(i)){
    //       return values == nullptr ? 1.0 : (*values)(i);
    //     }
    //   }
    //   return 0.0;
    // }
  };

  // A dense vector which is a row-slice of the underlying matrix.
  struct DenseVector {
    // Returns a row slice from the matrix.
    Eigen::TensorMap<Eigen::Tensor<const float, 1, Eigen::RowMajor>> Row()
        const {
      return Eigen::TensorMap<Eigen::Tensor<const float, 1, Eigen::RowMajor>>(
          data_matrix.data() + row_index * data_matrix.dimension(1),
          data_matrix.dimension(1));
    }

    // Returns a col slice from the matrix.
    Eigen::TensorMap<Eigen::Tensor<const float, 1, Eigen::RowMajor>> Col()
        const {
      return Eigen::TensorMap<Eigen::Tensor<const float, 1, Eigen::RowMajor>>(
          data_matrix.data(), data_matrix.dimension(0));
    }

    // Returns a row slice as a 1 * F matrix, where F is the number of features.
    Eigen::TensorMap<Eigen::Tensor<const float, 2, Eigen::RowMajor>>
    ColAsMatrix() const {
      return Eigen::TensorMap<Eigen::Tensor<const float, 2, Eigen::RowMajor>>(
          data_matrix.data(), data_matrix.dimension(0), 1);
    }

    // Returns a row slice as a 1 * F matrix, where F is the number of features.
    Eigen::TensorMap<Eigen::Tensor<const float, 2, Eigen::RowMajor>>
    RowAsMatrix() const {
      return Eigen::TensorMap<Eigen::Tensor<const float, 2, Eigen::RowMajor>>(
          data_matrix.data() + row_index * data_matrix.dimension(1), 1,
          data_matrix.dimension(1));
    }

    // data_matrix contains all of the information of dense features.
    // shape: num_examples * num_dense_features
    // For each DenseVector instance, the data_matrix points to same matrix.
    const TTypes<float>::ConstMatrix data_matrix;
    const int64 row_index;
  };

  const std::vector<std::unique_ptr<DenseVector> >& dense_vectors() const{
    return dense_vectors_;
  }

  const Example::SparseFeatures& sparse_feature(const int sf_idx) const {
    return sparse_features_[sf_idx];
  }

  // Return the sprase value of `sparse_feature_index`-th feature with `indices`
  // float sparse_value(int sparse_feature_index, int64 indices) const{
  //   // TODO: It seems better to compute the information of the indices at the 
  //   // beginning of initialization.
  //   const SparseFeatures& sparse_feature = 
  //           sparse_features_[sparse_feature_index];

  //   return sparse_feature.query_value(indices);
  // }
  ////////////////////////////////////////////////////////////////////////////

  // float feature(int i) const {
  //   float tmp = dense_vectors_[i]->RowAsMatrix()(0, 0);
  //   return tmp;
  // }
  ////////////////////////////////////////////////////////////////////////////

 private:
  std::vector<SparseFeatures> sparse_features_;
  std::vector<std::unique_ptr<DenseVector>> dense_vectors_;

  float example_label_ = 0;
  float example_weight_ = 0;
  double squared_norm_ = 0;  // sum squared norm of the features.

  // Examples fills Example in a multi-threaded way.
  friend class Examples;

  // ModelWeights use each example for model update w += \alpha * x_{i};
  friend class ModelWeights;
};

// Weights related to features. For example, say you have two sets of sparse
// features i.e. age bracket and country, then FeatureWeightsDenseStorage hold
// the parameters for it. We keep track of the original weight passed in and the
// delta weight which the optimizer learns in each call to the optimizer.
class FeatureWeightsDenseStorage {
 public:
  FeatureWeightsDenseStorage(const TTypes<const float>::Matrix nominals,
                             TTypes<float>::Matrix deltas)
      : nominals_(nominals), deltas_(deltas) {
    CHECK(deltas.rank() > 1);
  }

  // Check if a feature index is with-in the bounds.
  bool IndexValid(const int64 index) const {
    return index >= 0 && index < deltas_.dimension(1);
  }

  // Nominals here are the original weight matrix.
  TTypes<const float>::Matrix nominals() const { return nominals_; }

  // Delta weights durining mini-batch updates.
  TTypes<float>::Matrix deltas() const { return deltas_; }

  // Updates delta weights based on active dense features in the example and
  // the corresponding dual residual.
  void UpdateDenseDeltaWeights(
      const Eigen::ThreadPoolDevice& device,
      const Example::DenseVector& dense_vector,
      const std::vector<double>& normalized_bounded_dual_delta);

  float weight() const { return nominals_(0,0) + deltas_(0,0);}
  // Update delta weights using dual alpha
  void UpdateDenseDeltaWeights(const Eigen::ThreadPoolDevice& device,
    double delta_alpha);

 private:
  // The nominal value of the weight for a feature (indexed by its id).
  const TTypes<const float>::Matrix nominals_;
  // The accumulated delta weight for a feature (indexed by its id).
  TTypes<float>::Matrix deltas_;
};

// Similar to FeatureWeightsDenseStorage, but the underlying weights are stored
// in an unordered map.
class FeatureWeightsSparseStorage {
  // `FeatureWeightsSparseStorage` stores weights to a sparse feature group.
  // Suppose a sparse feature has N categories, then there are N weight in this
  // storage instance. The length of `indices` equals the columns of nominals 
  // equals N.

  // The `indices` are the index of weight within the feature group. For 
  // FeatureColumns like `sparse_column_with_keys`, the `index` equals to `id`.
  // For `hash_bucket`, the `indices` can be very large while `id` is the
  // actual index within the storage.
 public:
  FeatureWeightsSparseStorage(const TTypes<const int64>::Vec indices,
                              const TTypes<const float>::Matrix nominals,
                              TTypes<float>::Matrix deltas)
      : nominals_(nominals), deltas_(deltas) {
    // Create a map from sparse index to the dense index of the underlying
    // storage.
    for (int64 j = 0; j < indices.size(); ++j) {
      indices_to_id_[indices(j)] = j;
      id_to_indices_[j] = indices(j);
    }
  }

  // Check if a feature index exists.
  bool IndexValid(const int64 index) const {
    return indices_to_id_.find(index) != indices_to_id_.end();
  }

  // Nominal value at a particular feature index and class label.
  float nominals(const int class_id, const int64 index) const {
    auto it = indices_to_id_.find(index);
    return nominals_(class_id, it->second);
  }

  // Delta weights durining mini-batch updates.
  float deltas(const int class_id, const int64 index) const {
    auto it = indices_to_id_.find(index);
    return deltas_(class_id, it->second);
  }

  // Updates delta weights based on active sparse features in the example and
  // the corresponding dual residual.
  void UpdateSparseDeltaWeights(
      const Eigen::ThreadPoolDevice& device,
      const Example::SparseFeatures& sparse_features,
      const std::vector<double>& normalized_bounded_dual_delta);

  void UpdateSparseDeltaWeights(
    const Eigen::ThreadPoolDevice& device,
    const int64 sf_indices,
    const double delta_alpha);

  // The difference of the following tow class with the above two is that 
  float nominals_by_id(const int class_id, const int64 id) const {
    return nominals_(class_id, id);
  }

  float deltas_by_id(const int class_id, const int64 id) const {
    return deltas_(class_id, id);
  }

  int64 id_to_indices(int64 id) const {
    return id_to_indices_.at(id);
  }

  int64 indices_to_id(int64 indices) const {
    return indices_to_id_.at(indices);
  }

  int num_weights() const{
    return nominals_.dimension(1);
  }

 private:
   std::unordered_map<int64, int64> id_to_indices_;
 private:
  // The nominal value of the weight for a feature (indexed by its id).
  const TTypes<const float>::Matrix nominals_;
  // The accumulated delta weight for a feature (indexed by its id).
  TTypes<float>::Matrix deltas_;
  // Map from feature index to an index to the dense vector.
  std::unordered_map<int64, int64> indices_to_id_;
};

// Weights in the model, wraps both current weights, and the delta weights
// for both sparse and dense features.
class ModelWeights {
 public:
  ModelWeights() {}

  bool SparseIndexValid(const int col, const int64 index) const {
    return sparse_weights_[col].IndexValid(index);
  }

  bool DenseIndexValid(const int col, const int64 index) const {
    return dense_weights_[col].IndexValid(index);
  }

  // Go through all the features present in the example, and update the
  // weights based on the dual delta.
  void UpdateDeltaWeights(
      const Eigen::ThreadPoolDevice& device, const Example& example,
      const std::vector<double>& normalized_bounded_dual_delta);

  Status Initialize(OpKernelContext* const context);

  const std::vector<FeatureWeightsSparseStorage>& sparse_weights() const {
    return sparse_weights_;
  }

  const std::vector<FeatureWeightsDenseStorage>& dense_weights() const {
    return dense_weights_;
  }

  // Update i-th weight with delta dual weight.
  void UpdateDenseDeltaWeights(const Eigen::ThreadPoolDevice& device, 
    const float delta_alpha, int i);

  void UpdateSparseDeltaWeights(
    const Eigen::ThreadPoolDevice& device,
    const float delta_alpha, int sf_id, int sf_indices);
  ////////////////////////////////////////////////
 private:
  std::vector<FeatureWeightsSparseStorage> sparse_weights_;
  std::vector<FeatureWeightsDenseStorage> dense_weights_;

  TF_DISALLOW_COPY_AND_ASSIGN(ModelWeights);
};

// Examples contains all the training examples that SDCA uses for a mini-batch.
class Examples {
 public:
  Examples() {}

  // Returns the Example at |example_index|.
  const Example& example(const int example_index) const {
    return examples_.at(example_index);
  }

  int sampled_index(const int id, const bool adaptative) const {
    if (adaptative) return sampled_index_[id];
    return id;
  }

  // Adaptive SDCA in the current implementation only works for
  // binary classification, where the input argument for num_weight_vectors
  // is 1.
  Status SampleAdaptativeProbabilities(
      const int num_loss_partitions, const Regularizations& regularization,
      const ModelWeights& model_weights,
      const TTypes<float>::Matrix example_state_data,
      const std::unique_ptr<DualLossUpdater>& loss_updater,
      const int num_weight_vectors);

  int num_examples() const { return examples_.size(); }

  int num_features() const { return num_features_; }

  // Initialize() must be called immediately after construction.
  Status Initialize(OpKernelContext* const context, const ModelWeights& weights,
                    int num_sparse_features,
                    int num_sparse_features_with_values,
                    int num_dense_features);

  //////////////////////////////////////////////////////////////////////////////
  // Compute squared norm of i-th dense column of A
  float DenseFeatureSquaredNorm (int i) const;

  // Return i-th column of Matrix A
  // Eigen::TensorMap<Eigen::Tensor<const float, 1, Eigen::RowMajor>> Ai(int i) const{
  //   return example(0).dense_vectors_[i]->Col();
  // }

  Eigen::TensorMap<Eigen::Tensor<const float, 2, Eigen::RowMajor>>
    DenseFeatureAsMatrix(int i) const{
    return example(0).dense_vectors_[i]->ColAsMatrix();
  }

  TTypes<float>::ConstMatrix labels()const{
    return TTypes<float>::ConstMatrix(labels_->data(), labels_->dimension(0),
     labels_->dimension(1));
  }

 private:
  std::unique_ptr<Eigen::Tensor<float, 2, Eigen::RowMajor> > labels_;

 private:
  // Reads the input tensors, and builds the internal representation for sparse
  // features per example. This function modifies the |examples| passed in
  // to build the sparse representations.
  static Status CreateSparseFeatureRepresentation(
      const DeviceBase::CpuWorkerThreads& worker_threads, int num_examples,
      int num_sparse_features, const ModelWeights& weights,
      const OpInputList& sparse_example_indices_inputs,
      const OpInputList& sparse_feature_indices_inputs,
      const OpInputList& sparse_feature_values_inputs,
      std::vector<Example>* const examples);

  // Reads the input tensors, and builds the internal representation for dense
  // features per example. This function modifies the |examples| passed in
  // to build the sparse representations.
  static Status CreateDenseFeatureRepresentation(
      const DeviceBase::CpuWorkerThreads& worker_threads, int num_examples,
      int num_dense_features, const ModelWeights& weights,
      const OpInputList& dense_features_inputs,
      std::vector<Example>* const examples);

  // Computes squared example norm per example i.e |x|^2. This function modifies
  // the |examples| passed in and adds the squared norm per example.
  static void ComputeSquaredNormPerExample(
      const DeviceBase::CpuWorkerThreads& worker_threads, int num_examples,
      int num_sparse_features, int num_dense_features,
      std::vector<Example>* const examples);

  // All examples in the batch.
  std::vector<Example> examples_;

  // Adaptative sampling variables
  std::vector<float> probabilities_;
  std::vector<int> sampled_index_;
  std::vector<int> sampled_count_;

  int num_features_ = 0;

  TF_DISALLOW_COPY_AND_ASSIGN(Examples);
};

}  // namespace sdca
}  // namespace tensorflow

#endif  // TENSORFLOW_KERNELS_SDCA_INTERNAL_H_
