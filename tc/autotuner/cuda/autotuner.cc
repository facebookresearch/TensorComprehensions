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

#include "tc/autotuner/autotuner.h"

#include <atomic>
#include <chrono>
#include <numeric>
#include <thread>

#include <cuda_runtime_api.h>
#include <glog/stl_logging.h>

#include "tc/autotuner/utils.h"
#include "tc/core/compiler.h"
#include "tc/core/cuda/cuda.h"
#include "tc/core/cuda/cuda_mapping_options_cpp_printer.h"
#include "tc/core/cuda/cuda_tc_executor.h"
#include "tc/core/flags.h"
#include "tc/core/polyhedral/cuda/mapping_types.h"
#include "tc/core/scope_guard.h"
#include "tc/core/tensor.h"
#include "tc/core/utils/math.h"
#include "tc/lang/canonicalize.h"

namespace tc {
namespace autotune {
namespace detail {

using CudaTuningHarness = class TuningHarness<tc::CudaBackend>;

template <>
typename CudaBackend::MappingOptionsType makeOptions<CudaBackend>(
    const typename CudaBackend::MappingOptionsType& baseMapping,
    const CandidateConfiguration& c) {
  auto options = baseMapping;
  c.configuration.applyToCudaMappingOptions(options);
  return options;
}

template <>
TuningConfiguration makeTuningConfiguration<CudaBackend>(
    const typename CudaBackend::MappingOptionsType& options,
    const TuningConfiguration& configuration) {
  TuningConfiguration conf = configuration;
  conf.fromCudaMappingOptions(options);
  return conf;
}

// This function is ran on a single pre-determined GPU, in a single thread
// It takes the input/output DLTensor objects that reside on that GPU
//
// We pass the bestTimeSoFar as an option to avoid taking locks in this
// function. This trades off a bit of conservativeness for code sanity.
//
// The function returns true if purning is possible and we can skip poorly
// performing versions early.
template <>
bool skipExecutionOrWarmup<CudaBackend>(
    typename CudaBackend::ExecutorType& executor,
    const std::vector<const DLTensor*>& outputs,
    const std::vector<const DLConstTensor*>& inputs,
    size_t bestTimeSoFar) {
  // 1. Prune based on the number of threads: if you don't hit at least k warps
  // (default k = 8; 256 total threads, controlled by
  // FLAGS_tuner_min_launch_total_threads) then it's likely the kernel is not
  // performing great.
  // This may be completely off but is a good first initial rule of thumb
  // for stress-testing autotuning.
  auto debugTuner = FLAGS_debug_tuner;
  auto minThreads = FLAGS_tuner_min_launch_total_threads;
  USING_MAPPING_SHORT_NAMES(BX, BY, BZ, TX, TY, TZ);
  auto block = executor.block();
  auto nThreads =
      mappingSize(TX, block) * mappingSize(TY, block) * mappingSize(TZ, block);
  auto grid = executor.grid();
  auto nBlocks =
      mappingSize(BX, grid) * mappingSize(BY, grid) * mappingSize(BZ, grid);
  if (nBlocks * nThreads < minThreads) {
    LOG_IF(INFO, debugTuner)
        << "Skip configuration: too few threads " << grid << " / " << block;
    return true;
  } else {
    LOG_IF(INFO, debugTuner)
        << "Run configuration launch bounds blocks: " << grid
        << " and threads: " << block << "\n";
  }

  // 2. Perform a first run which may have one of 2 behaviors:
  //   2.a. return a very slow first execution time, we should stop
  //     early. This is akin to pruning but in this case we have run once,
  //   2.b. return a reasonable execution time, in which case we proceed with
  //     warmup.
  auto timings = executor.profile(inputs, outputs);
  // 2.a.
  constexpr size_t kCatastrophicPerfFactor = 100;
  if (bestTimeSoFar < std::numeric_limits<size_t>::max() and
      timings.kernelRuntime >= std::chrono::microseconds(
                                   (kCatastrophicPerfFactor * bestTimeSoFar))) {
    return true;
  }
  // 2.b. during autotuning we don't want to spend too much time executing,
  // use a reduced number of iterations for warmup.
  constexpr int kReducedWarmupIterations = 2;
  for (size_t i = 1; i < kReducedWarmupIterations - 1; ++i) {
    executor.profile(inputs, outputs);
  }

  // 3. After reasonable warmup, look at the performance and prune if
  // catastrophically bad.
  constexpr int kEarlyPruneFactor = 5;
  timings = executor.profile(inputs, outputs);
  if (bestTimeSoFar < std::numeric_limits<size_t>::max() and
      timings.kernelRuntime >=
          std::chrono::microseconds((kEarlyPruneFactor * bestTimeSoFar))) {
    return true;
  }

  // 4. If we get here then the kernel is good to be benchmarked
  return false;
}

template <>
void handleDeviceRuntimeError<CudaBackend>(
    size_t device,
    typename CudaBackend::MappingOptionsType& options) {
  while (cudaGetLastError() != cudaSuccess) {
    // In case of errors in the generated, we cannot rely on deviceReset to
    // set the device in a clean state. So instead we just pop and discard
    // all the errors accumulated on the device until we get to a clean
    // slate (i.e. cudaSuccess).
    ;
  }
  try {
    // Some errors, such as illegal memory access, cannot be recovered from
    // without a cudaDeviceReset (i.e. because user protection)
    // In those cases we have no choice than to fail hard.
    TC_CUDA_RUNTIMEAPI_ENFORCE(cudaDeviceSynchronize());
  } catch (const std::exception& e) {
    LOG(FATAL) << "[FATAL] cuda error on device " << device << ": " << e.what()
               << "\n"
               << CudaBackend::MappingOptionsAsCpp(options);
  }
}

template <>
std::vector<size_t> parseDevices<CudaBackend>(const std::string& devices) {
  std::stringstream ss(devices);
  size_t device;
  std::vector<size_t> res;
  while (ss >> device) {
    res.push_back(device);
    if (ss.peek() == ',') {
      ss.ignore();
    }
  }
  return res;
}
} // namespace detail
} // namespace autotune
} // namespace tc
