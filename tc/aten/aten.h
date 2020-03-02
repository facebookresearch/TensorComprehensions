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

#include "tc/core/tensor.h"

namespace tc {
namespace aten {

inline TensorInfo toTensorInfo(const at::Tensor&);

inline std::vector<DLTensorUPtr> makeDLTensors(
    const std::vector<at::Tensor>& tensors);

inline std::vector<DLConstTensorUPtr> makeDLConstTensors(
    const std::vector<at::Tensor>& tensors);

inline void setAtenSeed(uint64_t seed, at::Backend backend);
inline uint64_t getAtenSeed(at::Backend backend);

} // namespace aten
} // namespace tc

#include "tc/aten/aten-inl.h"
