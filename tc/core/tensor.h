/**
 * Copyright (c) 2017-present, Facebook, Inc.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
#pragma once

#include <iostream>
#include <memory>
#include <string>
#include <vector>

#include <dlpack/dlpack.h>

#include "tc/proto/compcache.pb.h"

/**
 * Various tensor utilities used in the Tensor Comprehensions compiler.
 * At a high-level we use the DLTensor type from dlpack and we add a
 * DLConstTensor to propagate const-correctness.
 * DLTensor is used as an ML-framework and backend-agnostic representation to
 * interact with user-level tensors.
 * DLTensorUPtr and DLConstTensorUPtr are used as metadata-owning pointers.
 * The underlying concrete device pointer is borrowed when calling
 * ExecutionEngine::compile and ExecutionEngine::run.
 *
 * Internally, TC uses TensorInfo for compiling and interacting with the
 * compilation caches. TensorInfo is backed by protobuf and stored directly in
 * the caches.
 */

/**
 * DLPack is missing a ConstTensor. Add it here for now.
 * In the future, this would be contributed back to dlpack/include/dlpack.h.
 */
typedef struct {
  const void* data;
  DLContext ctx;
  int ndim;
  DLDataType dtype;
  int64_t* shape;
  int64_t* strides;
  uint64_t byte_offset;
} DLConstTensor;

namespace tc {
/**
 * C++ unique_ptr abstraction on top of DLTensor/DLConstTensor
 */
struct DLTensorDeleter {
  template <typename DLTensorType>
  void operator()(const DLTensorType* t);
};
using DLTensorUPtr = std::unique_ptr<DLTensor, DLTensorDeleter>;
using DLConstTensorUPtr = std::unique_ptr<DLConstTensor, DLTensorDeleter>;

// Return non-owning raw pointers to DLTensor from metadata-owning DLTensorUPtr
std::vector<const DLConstTensor*> extractRawPtrs(
    const std::vector<DLConstTensorUPtr>& uptrs);
std::vector<const DLTensor*> extractRawPtrs(
    const std::vector<DLTensorUPtr>& uptrs);

/**
 * TensorInfo wraps the necessary tensor information to compile TCs and
 * interact with the various caches.
 * Notably, it contains alignment information but no underlying data pointer.
 * It is serializable to protobuf and stored directly in the cache.
 */
struct TensorInfo {
  DLDataType dtype;
  uint64_t alignment;
  std::vector<int64_t> shape;
  std::vector<int64_t> strides;

  TensorInfo(
      DLDataType dtype,
      uint64_t alignment,
      const std::vector<int64_t>& shape,
      const std::vector<int64_t>& strides);
  explicit TensorInfo(const DLTensor* t);
  explicit TensorInfo(const DLConstTensor* t);
  explicit TensorInfo(const TensorInfoProto& buf);

  bool operator==(const TensorInfo& t) const;
  TensorInfoProto toProtobuf() const;
};

/// Given sizes, this returns strides that can be used to construct a
/// tensor with a contiguous memory-layout (Torch7 nomenclature).
/// A contiguous tensor is on which indexing with an ordered multi-dimensional
/// index ranging from (0, ..., 0) to sizes results in a strictly monotonic
/// traversal of the underlying memory, without holes.
/// For instance, given sizes=[3, 4, 5], the strides compatible with a
/// contiguous memory layout are [20, 5, 1].
/// Note that general stride expressions can result in multiple different
/// behavior (e.g. stride 0 along a certain dimension will traverse the same
/// values multiple times).
template <typename T>
std::vector<T> makeStridesFromSizes(const std::vector<T>& sizes);

// Specializes for DLTensor, DLConstTensor
template <typename DLTensorPtrType>
std::vector<TensorInfo> makeTensorInfoVector(
    const std::vector<DLTensorPtrType>& ts);
std::vector<TensorInfo> makeTensorInfoVector(
    const google::protobuf::RepeatedPtrField<TensorInfoProto>& buf);

// Basic support functions for DLTensors
DLContext getCPUDLContext();
DLContext getGPUDLContext(int device_id = 0);
bool operator==(const DLDataType& t1, const DLDataType& t2);

// Print the metadata for DLDataType and DLTensor
std::string toString(const DLDataType& t);
std::ostream& operator<<(std::ostream& os, const DLDataType& t);
std::ostream& operator<<(std::ostream& os, const DLTensor& t);

// Basic metadata-owning DLTensor, only copies the underlying raw pointer.
DLTensorUPtr makeDLTensor(const DLTensor* ptr);
template <typename DLTensorType>
DLConstTensorUPtr makeDLConstTensor(const DLTensorType* ptr);
template <typename T>
inline DLTensorUPtr makeDLTensor(
    DLContext ctx,
    DLDataType dtype,
    const std::vector<T>& sizes,
    const std::vector<T>& strides = std::vector<T>(),
    const void* data = nullptr,
    uint64_t byteOffset = 0);
template <typename T>
inline DLConstTensorUPtr makeDLConstTensor(
    DLContext ctx,
    DLDataType dtype,
    const std::vector<T>& sizes,
    const std::vector<T>& strides = std::vector<T>(),
    const void* data = nullptr,
    uint64_t byteOffset = 0);
// A metadata-owning DLTensor reconstructed from TensorInfo does not have a
// meaningful pointer or DLContext: pointer is  nullptr and ctx is kDLCPU.
DLTensorUPtr makeDLTensor(const TensorInfo& tensor);
DLConstTensorUPtr makeDLConstTensor(const TensorInfo& tensor);

// Specializes for const DLTensor*, const DLConstTensor* and TensorInfo
template <typename T>
std::vector<DLTensorUPtr> makeDLTensorVector(const std::vector<T>& ptrs);
template <typename T>
std::vector<DLConstTensorUPtr> makeDLConstTensorVector(
    const std::vector<T>& ptrs);
} // namespace tc

#include "tc/core/tensor-inl.h"
