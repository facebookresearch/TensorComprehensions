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
#include "tc/core/cuda/cuda_compilation_cache.h"

#include <version.h>

#include <cstdint>
#include <fstream>
#include <numeric>
#include <tuple>

#include "tc/core/cuda/cuda_mapping_options.h"
#include "tc/core/utils/math.h"

namespace tc {

namespace {
uint64_t GetDLTensorAlignment(const DLTensor* t) {
  return (reinterpret_cast<std::uintptr_t>(t->data) + t->byte_offset) % 256;
}
} // namespace

detail::TensorInfo::TensorInfo(const DLTensor* t)
    : alignment{GetDLTensorAlignment(t)}, dType(t->dtype) {
  shape.reserve(t->ndim);
  std::copy(t->shape, t->shape + t->ndim, std::back_inserter(shape));
  if (not t->strides) {
    return;
  }
  strides.reserve(t->ndim);
  std::copy(t->strides, t->strides + t->ndim, std::back_inserter(strides));
}

detail::TensorInfo::TensorInfo(const TensorInfoProto& buf)
    : shape{buf.shape().begin(), buf.shape().end()},
      strides{buf.strides().begin(), buf.strides().end()},
      alignment{buf.alignment()},
      dType{static_cast<uint8_t>(buf.dtype().code()),
            static_cast<uint8_t>(buf.dtype().bits()),
            static_cast<uint16_t>(buf.dtype().lanes())} {}

TensorInfoProto detail::TensorInfo::toProtobuf() const {
  TensorInfoProto buf;
  buf.mutable_shape()->Reserve(shape.size());
  std::copy(
      shape.begin(),
      shape.end(),
      google::protobuf::RepeatedFieldBackInserter(buf.mutable_shape()));
  buf.mutable_strides()->Reserve(strides.size());
  std::copy(
      strides.begin(),
      strides.end(),
      google::protobuf::RepeatedFieldBackInserter(buf.mutable_strides()));
  buf.set_alignment(alignment);
  buf.mutable_dtype()->set_code(dType.code);
  buf.mutable_dtype()->set_bits(dType.bits);
  buf.mutable_dtype()->set_lanes(dType.lanes);
  return buf;
}

bool detail::TensorInfo::operator==(const DLTensor* t) const {
  if (t->ndim != static_cast<int>(shape.size())) {
    return false;
  }

  auto res = std::mismatch(shape.begin(), shape.end(), t->shape);
  if (res.first != shape.end() || res.second != t->shape + t->ndim) {
    return false;
  }

  if (t->strides == nullptr) {
    if (strides.size() > 0) {
      return false;
    }
  } else {
    if (t->ndim != static_cast<int>(strides.size())) {
      return false;
    }

    res = std::mismatch(strides.begin(), strides.end(), t->strides);
    if (res.first != strides.end() || res.second != t->strides + t->ndim) {
      return false;
    }
  }

  /*This should be enabled when/if tc starts using alignment information
   *if (GetDLTensorAlignment(t) != alignment) {
   *  return false;
   *}
   */
  return std::tie(t->dtype.code, t->dtype.bits, t->dtype.lanes) ==
      std::tie(dType.code, dType.bits, dType.lanes);
}

bool operator==(const DLDataType& a, const DLDataType& b) {
  return a.code == b.code and a.bits == b.bits and a.lanes == b.lanes;
}

bool operator<(const DLDataType& a, const DLDataType& b) {
  return a.code < b.code and a.bits < b.bits and a.lanes < b.lanes;
}

bool detail::TensorInfo::operator==(const TensorInfo& t) const {
  return alignment == t.alignment and dType == t.dType and shape == t.shape and
      strides == t.strides;
}

bool detail::TensorInfo::operator<(const TensorInfo& t) const {
  return alignment < t.alignment and dType < t.dType and shape < t.shape and
      strides < t.strides;
}

} // namespace tc
