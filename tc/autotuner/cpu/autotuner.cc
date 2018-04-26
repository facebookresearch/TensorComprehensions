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

#include <glog/stl_logging.h>

#include "tc/autotuner/utils.h"
#include "tc/core/compiler.h"
#include "tc/core/cpu/cpu_mapping_options_cpp_printer.h"
#include "tc/core/cpu/cpu_tc_executor_new_api.h"
#include "tc/core/flags.h"
#include "tc/core/scope_guard.h"
#include "tc/core/tensor.h"
#include "tc/core/utils/math.h"
#include "tc/lang/canonicalize.h"

namespace tc {
namespace autotune {
namespace detail {
template <>
typename CpuBackend::MappingOptionsType makeOptions<CpuBackend>(
    const typename CpuBackend::MappingOptionsType& baseMapping,
    const CandidateConfiguration& c) {
  auto options = baseMapping;
  c.configuration.applyToCpuMappingOptions(options);
  return options;
}

template <>
TuningConfiguration makeTuningConfiguration<CpuBackend>(
    const typename CpuBackend::MappingOptionsType& options,
    const TuningConfiguration& configuration) {
  TuningConfiguration conf = configuration;
  conf.fromCpuMappingOptions(options);
  return conf;
}

template <>
bool skipExecutionOrWarmup<CpuBackend>(
    typename CpuBackend::ExecutorType& executor,
    const std::vector<const DLTensor*>& outputs,
    const std::vector<const DLConstTensor*>& inputs,
    size_t bestTimeSoFar) {
  LOG(ERROR) << "NYI: skipExecutionOrWarmup<CpuBackend>";
  return false;
}

template <>
void handleDeviceRuntimeError<CpuBackend>(
    size_t device,
    typename CpuBackend::MappingOptionsType& options) {
  LOG(ERROR) << "NYI: handleDeviceRuntimeError<CpuBackend>";
}

template <>
std::vector<size_t> parseDevices<CpuBackend>(const std::string& devices) {
  LOG(ERROR) << "NYI: handleDeviceRuntimeError<CpuBackend>";
  return {};
}
} // namespace detail
} // namespace autotune
} // namespace tc
