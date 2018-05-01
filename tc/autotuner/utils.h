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

#include <atomic>
#include <iostream>
#include <mutex>
#include <numeric>
#include <thread>
#include <vector>

#include <llvm/ADT/Optional.h>

#include "tc/core/cuda/cuda_compilation_cache.h"
#include "tc/core/cuda/cuda_mapping_options.h"
#include "tc/core/utils/dlpack.h"
#include "tc/core/utils/time.h"
#include "tc/lang/canonicalize.h"

namespace tc {
namespace autotune {

/// Returns all the powers of 2 up to the first one that is larger than val
/// and the result of ceil(val/pow2) for each of those powers of 2 (except for
/// the larger one)
std::vector<std::size_t> powers2andCeilDivisors(std::size_t val);

template <typename Vector, typename... Vectors>
Vector mergeVectors(Vector&& v, Vectors&&... vs);

std::vector<CudaMappingOptions> restoreCandidates(
    const lang::CanonicalTcString& tc,
    const std::vector<const DLTensor*>& inputs,
    const std::vector<const DLTensor*>& outputs);

llvm::Optional<CudaMappingOptions> getBestOptions(
    const lang::CanonicalTcString& id,
    const std::vector<const DLTensor*>& inputs,
    const std::vector<const DLTensor*>& outputs);

/**
 * Helper class to pretty print autotuning progress
 */
class Printer {
 public:
  Printer(
      size_t iteration,
      size_t total,
      const std::atomic_size_t& currentCompilationJob,
      const std::atomic_size_t& numEvaluations);
  ~Printer();

  void record(Duration runtime);
  void stop();

  void printAll();

 private:
  void printLoop();

  size_t iteration_;
  std::vector<Duration> runtimes_;
  mutable std::mutex runtimesMtx_;

  std::atomic_bool stopPrinting_{false};
  std::thread printerThread_;

  const size_t total_;
  const std::atomic_size_t& currentCompilationJob_;
  const std::atomic_size_t& numEvaluations_;
};
} // namespace autotune
} // namespace tc

#include "tc/autotuner/utils-inl.h"
