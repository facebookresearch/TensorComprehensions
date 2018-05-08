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

#include <functional>
#include <sstream>
#include <string>
#include <vector>

#include "tc/c2/context.h"
#include "tc/core/tensor.h"

#include "caffe2/core/common.h"

namespace caffe2 {
namespace dlpack {

template <typename C2Context>
DLContext getDLContext();

template <>
inline DLContext getDLContext<CPUContext>() {
  return tc::getCPUDLContext();
}

// Can't have a CUDAContext object, how do we get the GPU id of a C2 Tensor?
template <>
inline DLContext getDLContext<CUDAContext>() {
  return tc::getGPUDLContext(0 /*ctx ? ctx->cuda_gpu_id() : 0*/);
}

inline DLDataType getDLDataType(const TypeMeta& meta) {
  DLDataType res;
  if (meta.Match<float>()) {
    res.code = DLDataTypeCode::kDLFloat;
  } else if (meta.Match<int>()) {
    res.code = DLDataTypeCode::kDLInt;
  } else {
    CHECK(false) << "NYI: getDLDataType(caffe2::Meta::Make<" << meta.name()
                 << ">))";
  }
  res.bits = 32;
  res.lanes = 1;
  return res;
}

template <typename C2Context>
tc::DLConstTensorUPtr makeDLConstTensor(
    const caffe2::Tensor<C2Context>& tensor) {
  return tc::makeDLConstTensor(
      getDLContext<C2Context>(),
      getDLDataType(tensor.meta()),
      tensor.dims(),
      std::vector<TIndex>(),
      tensor.raw_data(),
      0);
}

template <typename C2Context>
tc::DLTensorUPtr makeDLTensor(caffe2::Tensor<C2Context>& tensor) {
  return tc::makeDLTensor(
      getDLContext<C2Context>(),
      getDLDataType(tensor.meta()),
      tensor.dims(),
      std::vector<TIndex>(),
      tensor.raw_mutable_data(),
      0);
}
} // namespace dlpack
} // namespace caffe2
