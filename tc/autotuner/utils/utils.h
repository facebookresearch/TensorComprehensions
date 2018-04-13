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
#include <vector>

#include <ATen/ATen.h>

#include "tc/core/cuda/cuda.h"
#include "tc/core/cuda/cuda_mapping_options.h"
#include "tc/core/utils/dlpack.h"
#include "tc/lang/canonicalize.h"
#include "tc/lang/tree.h"

#include <llvm/ADT/Optional.h>

namespace tc {
namespace autotune {

/// Returns all the powers of 2 up to the first one that is larger than val
/// and the result of ceil(val/pow2) for each of those powers of 2 (except for
/// the larger one)
std::vector<std::size_t> powers2andCeilDivisors(std::size_t val);

template <typename Vector, typename... Vectors>
Vector mergeVectors(Vector&& v, Vectors&&... vs);

/// The following API allows interacting with the autotuner caches.
/// Caches generally take arbitrary strings for keys.
/// The autotuner uses a canonicalized TC expression to load / store into
/// caches. Add a layer of type safety to interact with these.
std::vector<CudaMappingOptions> restoreCandidates(
    const lang::CanonicalTcString& tc,
    const std::vector<const DLTensor*>& inputs,
    const std::vector<const DLTensor*>& outputs);

llvm::Optional<CudaMappingOptions> getBestOptions(
    const lang::CanonicalTcString& id,
    const std::vector<const DLTensor*>& inputs,
    const std::vector<const DLTensor*>& outputs);

struct OptionsWithMedianTime {
  CudaMappingOptions options;
  Duration medianRuntime;
};

std::vector<OptionsWithMedianTime> getOptionsAndMedianRuntimes(
    const lang::CanonicalTcString& id,
    const std::vector<const DLTensor*>& inputs);
double mean(std::vector<double>& v);
double stdv(std::vector<double>& v, double mean);
void normalizeVector(std::vector<double>& v);
void sigmaScale(std::vector<double>& v);

} // namespace autotune
} // namespace tc

#include "tc/autotuner/utils/utils-inl.h"
