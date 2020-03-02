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

#include <version.h>

#include <cstdint>
#include <fstream>
#include <numeric>
#include <tuple>

#include "tc/core/tensor.h"
#include "tc/core/utils/math.h"

namespace tc {
namespace detail {
template <typename DLTensorType>
uint64_t getDLTensorAlignment(const DLTensorType* t) {
  return (reinterpret_cast<std::uintptr_t>(t->data) + t->byte_offset) %
      TensorInfo::kAlignment;
}

std::vector<int64_t> toIntVector(const int64_t* ptr, size_t ndim) {
  if (!ptr) {
    return {};
  }
  std::vector<int64_t> res;
  res.reserve(ndim);
  std::copy(ptr, ptr + ndim, std::back_inserter(res));
  return res;
}
} // namespace detail

TensorInfo::TensorInfo(
    DLDataType t,
    uint64_t align,
    const std::vector<int64_t>& sz,
    const std::vector<int64_t>& st)
    : dtype(t), alignment(align), shape(sz), strides(st) {}

TensorInfo::TensorInfo(const DLTensor* t)
    : TensorInfo(
          t->dtype,
          detail::getDLTensorAlignment(t),
          detail::toIntVector(t->shape, t->ndim),
          detail::toIntVector(t->strides, t->ndim)) {}

TensorInfo::TensorInfo(const DLTensorUPtr& ptr) : TensorInfo(ptr.get()) {}

TensorInfo::TensorInfo(const DLConstTensor* t)
    : TensorInfo(
          t->dtype,
          detail::getDLTensorAlignment(t),
          detail::toIntVector(t->shape, t->ndim),
          detail::toIntVector(t->strides, t->ndim)) {}

TensorInfo::TensorInfo(const DLConstTensorUPtr& ptr) : TensorInfo(ptr.get()) {}

TensorInfo::TensorInfo(const TensorInfoProto& buf)
    : dtype{static_cast<uint8_t>(buf.dtype().code()),
            static_cast<uint8_t>(buf.dtype().bits()),
            static_cast<uint16_t>(buf.dtype().lanes())},
      alignment{buf.alignment()},
      shape{buf.shape().begin(), buf.shape().end()},
      strides{buf.strides().begin(), buf.strides().end()} {}

TensorInfoProto TensorInfo::toProtobuf() const {
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
  buf.mutable_dtype()->set_code(dtype.code);
  buf.mutable_dtype()->set_bits(dtype.bits);
  buf.mutable_dtype()->set_lanes(dtype.lanes);
  return buf;
}

bool operator==(const DLDataType& a, const DLDataType& b) {
  return a.code == b.code and a.bits == b.bits and a.lanes == b.lanes;
}

bool TensorInfo::operator==(const TensorInfo& t) const {
  return alignment == t.alignment and dtype == t.dtype and shape == t.shape and
      strides == t.strides;
}

bool TensorInfo::operator!=(const TensorInfo& t) const {
  return !(*this == t);
}

std::ostream& operator<<(std::ostream& os, const TensorInfo& t) {
  auto ndim = t.shape.size();
  if (ndim == 0) {
    return os;
  }
  os << "Dimensions: [";
  if (ndim) {
    for (size_t i = 0; i < ndim - 1; ++i) {
      os << t.shape[i] << ", ";
    }
    os << t.shape[ndim - 1] << "]";
    os << " Strides: [";
    if (t.strides.size()) {
      for (size_t i = 0; i < ndim - 1; ++i) {
        os << t.strides[i] << ", ";
      }
      os << t.strides[ndim - 1] << "] ";
    }
  }
  os << "Type: ";
  os << toString(t.dtype) << " ";
  return os;
}

std::ostream& operator<<(std::ostream& os, const DLTensor& t) {
  auto t_info = TensorInfo(&t);
  os << t_info;
  os << "Byte Offset: ";
  os << t.byte_offset << " ";
  os << "Base Pointer: " << t.data << " ";
  return os;
}

std::ostream& operator<<(std::ostream& os, const DLConstTensor& t) {
  auto t_info = TensorInfo(&t);
  os << t_info;
  os << "Byte Offset: ";
  os << t.byte_offset << " ";
  os << "Base Pointer: " << t.data << " ";
  return os;
}

std::vector<TensorInfo> makeTensorInfoVector(
    const google::protobuf::RepeatedPtrField<TensorInfoProto>& buf) {
  std::vector<TensorInfo> iis;
  iis.reserve(buf.size());
  std::transform(
      buf.begin(),
      buf.end(),
      std::back_inserter(iis),
      [](const TensorInfoProto& iip) { return TensorInfo{iip}; });
  return iis;
}
} // namespace tc
