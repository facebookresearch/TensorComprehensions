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

#include <string>
#include <vector>

#include <ATen/ATen.h>
#include <ATen/DLConvertor.h>

#include "tc/core/tensor.h"

namespace tc {
namespace aten {

// Stolen from ATen, get rid of our copy when ATen exposes the functionality
// Unfortunately we need to wait for updated conda packages so we just copy
// for now.
inline DLDataType getDLDataType(const at::Type& type) {
  using at::ScalarType;

  DLDataType dtype;
  dtype.lanes = 1;
  dtype.bits = type.elementSizeInBytes() * 8;
  switch (type.scalarType()) {
    case ScalarType::Byte:
      dtype.code = DLDataTypeCode::kDLUInt;
      break;
    case ScalarType::Char:
      dtype.code = DLDataTypeCode::kDLInt;
      break;
    case ScalarType::Double:
      dtype.code = DLDataTypeCode::kDLFloat;
      break;
    case ScalarType::Float:
      dtype.code = DLDataTypeCode::kDLFloat;
      break;
    case ScalarType::Int:
      dtype.code = DLDataTypeCode::kDLInt;
      break;
    case ScalarType::Long:
      dtype.code = DLDataTypeCode::kDLInt;
      break;
    case ScalarType::Short:
      dtype.code = DLDataTypeCode::kDLInt;
      break;
    case ScalarType::Half:
      dtype.code = DLDataTypeCode::kDLFloat;
      break;
    case ScalarType::Undefined:
      throw std::logic_error("Undefined is not a valid ScalarType");
    case ScalarType::NumOptions:
      throw std::logic_error("NumOptions is not a valid ScalarType");
  }
  return dtype;
}

inline TensorInfo toTensorInfo(const at::Tensor& t) {
  return TensorInfo(
      getDLDataType(t.type()),
      reinterpret_cast<std::uintptr_t>(t.data_ptr()) % TensorInfo::kAlignment,
      t.sizes(),
      t.strides());
}

inline std::vector<DLTensorUPtr> makeDLTensors(
    const std::vector<at::Tensor>& tensors) {
  std::vector<DLTensorUPtr> dlTensors;
  for (auto tensor : tensors) {
    auto dlMTensor = at::toDLPack(tensor);
    dlTensors.push_back(makeDLTensor(&(dlMTensor->dl_tensor)));
    dlMTensor->deleter(dlMTensor);
  }
  return dlTensors;
}

inline std::vector<DLConstTensorUPtr> makeDLConstTensors(
    const std::vector<at::Tensor>& tensors) {
  std::vector<DLConstTensorUPtr> dlTensors;
  for (auto tensor : tensors) {
    auto dlMTensor = at::toDLPack(tensor);
    dlTensors.push_back(makeDLConstTensor(&(dlMTensor->dl_tensor)));
    dlMTensor->deleter(dlMTensor);
  }
  return dlTensors;
}

inline void setAtenSeed(uint64_t seed, at::Backend backend) {
  at::Generator& gen = at::globalContext().defaultGenerator(backend);
  gen.manualSeed(seed);
}

inline uint64_t getAtenSeed(at::Backend backend) {
  at::Generator& gen = at::globalContext().defaultGenerator(backend);
  return gen.seed();
}
} // namespace aten
} // namespace tc
