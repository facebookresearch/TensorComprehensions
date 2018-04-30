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

#include <algorithm>
#include <sstream>

#include <glog/logging.h>
#include <google/protobuf/text_format.h>

#include "tc/proto/compcache.pb.h"

namespace tc {
template <typename T>
std::vector<T> makeStridesFromSizes(const std::vector<T>& sizes) {
  auto ndim = sizes.size();
  if (ndim == 0) {
    return std::vector<T>();
  }
  std::vector<T> strides(sizes.size(), 0);
  strides[ndim - 1] = 1;
  for (int i = ndim - 2; i >= 0; --i) {
    strides[i] = strides[i + 1] * sizes[i + 1];
  }
  return strides;
}

inline DLContext getCPUDLContext() {
  DLContext res;
  res.device_id = 0;
  res.device_type = DLDeviceType::kDLCPU;
  return res;
}

inline DLContext getGPUDLContext(int device_id) {
  DLContext res;
  res.device_id = device_id;
  res.device_type = DLDeviceType::kDLGPU;
  return res;
}

template <typename DLTensorType>
void DLTensorDeleter::operator()(const DLTensorType* t) {
  if (t->shape) {
    delete[] t->shape;
  }
  if (t->strides) {
    delete[] t->strides;
  }
  delete t;
};

namespace detail {
template <typename DLTensorType, typename T>
inline std::unique_ptr<DLTensorType, DLTensorDeleter> makeDLTensor(
    DLContext ctx,
    DLDataType dtype,
    const std::vector<T>& sizes,
    const std::vector<T>& strides = std::vector<T>(),
    decltype(DLTensorType().data) data = nullptr,
    uint64_t byteOffset = 0) {
  static_assert(
      std::is_convertible<T, int64_t>::value,
      "Template type not convertible to int64_t");
  std::unique_ptr<DLTensorType, DLTensorDeleter> res(new DLTensorType);
  res->data = data;
  res->ctx = ctx;
  auto ndim = sizes.size();
  res->ndim = ndim;
  res->dtype = dtype;
  res->shape = new int64_t[ndim];
  for (size_t i = 0; i < ndim; ++i) {
    res->shape[i] = sizes[i];
  }
  res->strides = new int64_t[ndim];
  std::vector<T> st(strides);
  if (st.size() == 0) {
    st = makeStridesFromSizes(sizes);
  }
  for (size_t i = 0; i < ndim; ++i) {
    res->strides[i] = st[i];
  }
  res->byte_offset = byteOffset;
  return res;
}

template <typename DLTensorType>
inline std::unique_ptr<DLTensorType, DLTensorDeleter> makeDLTensorHelper(
    const DLTensor* ptr) {
  std::vector<int64_t> sizes(ptr->ndim, 0);
  std::copy(ptr->shape, ptr->shape + ptr->ndim, sizes.begin());
  std::vector<int64_t> strides(ptr->ndim, 0);
  std::copy(ptr->strides, ptr->strides + ptr->ndim, strides.begin());
  return makeDLTensor<DLTensorType>(
      ptr->ctx, ptr->dtype, sizes, strides, ptr->data, ptr->byte_offset);
}
} // namespace detail

inline DLTensorUPtr makeDLTensor(const DLTensor* ptr) {
  return detail::makeDLTensorHelper<DLTensor>(ptr);
}

inline DLTensorUPtr makeDLTensor(const TensorInfo& tensor) {
  return detail::makeDLTensor<DLTensor>(
      DLContext{kDLCPU, 0},
      tensor.dtype,
      tensor.shape,
      tensor.strides,
      nullptr,
      tensor.alignment);
}

template <typename T>
inline DLTensorUPtr makeDLTensor(
    DLContext ctx,
    DLDataType dtype,
    const std::vector<T>& sizes,
    const std::vector<T>& strides,
    void* data,
    uint64_t byteOffset) {
  return detail::makeDLTensor<DLTensor>(
      ctx, dtype, sizes, strides, data, byteOffset);
}

template <typename DLTensorType>
inline DLConstTensorUPtr makeDLConstTensor(const DLTensorType* ptr) {
  return detail::makeDLTensorHelper<DLConstTensor>(ptr);
}

inline DLConstTensorUPtr makeDLConstTensor(const TensorInfo& tensor) {
  return detail::makeDLTensor<DLConstTensor>(
      DLContext{kDLCPU, 0},
      tensor.dtype,
      tensor.shape,
      tensor.strides,
      nullptr,
      tensor.alignment);
}

template <typename T>
inline DLConstTensorUPtr makeDLConstTensor(
    DLContext ctx,
    DLDataType dtype,
    const std::vector<T>& sizes,
    const std::vector<T>& strides,
    const void* data,
    uint64_t byteOffset) {
  return detail::makeDLTensor<DLConstTensor>(
      ctx, dtype, sizes, strides, data, byteOffset);
}

// Specializes for const DLTensor*, const DLConstTensor* and TensorInfo
template <typename T>
std::vector<DLTensorUPtr> makeDLTensorVector(const std::vector<T>& ptrs) {
  std::vector<DLTensorUPtr> res;
  for (auto p : ptrs) {
    res.push_back(makeDLTensor(p));
  }
  return res;
}

template <typename T>
std::vector<DLConstTensorUPtr> makeDLConstTensorVector(
    const std::vector<T>& ptrs) {
  std::vector<DLConstTensorUPtr> res;
  res.reserve(ptrs.size());
  for (auto p : ptrs) {
    res.push_back(makeDLConstTensor(p));
  }
  return res;
}

template <typename DLTensorPtrType>
std::vector<TensorInfo> makeTensorInfoVector(
    const std::vector<DLTensorPtrType>& ts) {
  std::vector<TensorInfo> res;
  res.reserve(ts.size());
  std::transform(
      ts.begin(), ts.end(), std::back_inserter(res), [](DLTensorPtrType t) {
        return TensorInfo(t);
      });
  return res;
}

inline std::vector<const DLTensor*> extractRawPtrs(
    const std::vector<DLTensorUPtr>& uptrs) {
  std::vector<const DLTensor*> res(uptrs.size(), nullptr);
  for (size_t i = 0; i < uptrs.size(); ++i) {
    res[i] = uptrs[i].get();
  }
  return res;
}

inline std::vector<const DLConstTensor*> extractRawPtrs(
    const std::vector<DLConstTensorUPtr>& uptrs) {
  std::vector<const DLConstTensor*> res(uptrs.size(), nullptr);
  for (size_t i = 0; i < uptrs.size(); ++i) {
    res[i] = uptrs[i].get();
  }
  return res;
}

inline std::string toString(const DLDataType& t) {
  if (t.lanes != 1) {
    CHECK(false) << "NYI: toString for >1 lanes";
  }
  switch (t.code) {
    case DLDataTypeCode::kDLFloat:
      switch (t.bits) {
        case 16:
          return "Half";
        case 32:
          return "float";
        case 64:
          return "double";
      }
      break;
    case DLDataTypeCode::kDLInt:
      switch (t.bits) {
        case 8:
          return "int8_t";
        case 16:
          return "int16_t";
        case 32:
          return "int";
        case 64:
          return "int64_t";
      }
      break;
    case DLDataTypeCode::kDLUInt:
      switch (t.bits) {
        case 8:
          return "uint8_t";
      }
      break;
  }
  CHECK(false) << "NYI: toString for type: " << t.code << ", bits: " << t.bits;
  return "";
}

inline std::string toString(const DLTensor& t) {
  std::stringstream ss;
  ss << "DLTensor@" << t.data << ":\n";
  std::string res;
  google::protobuf::TextFormat::PrintToString(
      TensorInfo(&t).toProtobuf(), &res);
  ss << res;
  return ss.str();
}
} // namespace tc
