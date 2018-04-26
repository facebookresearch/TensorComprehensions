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
#include "tc/aten/aten_compiler_new_api.h"

#include <iostream>
#include <string>
#include <vector>

#include "tc/aten/aten.h"
#include "tc/core/compiler.h"
#include "tc/core/tc_executor_new_api.h"
#include "tc/core/tensor.h"

namespace tc {
namespace aten {
template <typename Backend>
std::unique_ptr<typename Backend::ExecutorType> compile(
    const std::string& tc,
    const std::string& entryPoint,
    const std::vector<at::Tensor>& inputs,
    const typename Backend::MappingOptionsType& options) {
  auto inputDLTensors = makeDLConstTensors(inputs);
  return tc::compile<Backend>(
      tc, entryPoint, extractRawPtrs(inputDLTensors), options);
}

template <typename Executor>
void run(
    const Executor& executor,
    const std::vector<at::Tensor>& inputs,
    std::vector<at::Tensor>& outputs) {
  auto inputDLTensors = makeDLConstTensors(inputs);
  auto outputDLTensors = makeDLTensors(outputs);
  return executor.run(
      extractRawPtrs(inputDLTensors), extractRawPtrs(outputDLTensors));
}

template <typename Executor>
ProfilingInfo profile(
    const Executor& executor,
    const std::vector<at::Tensor>& inputs,
    std::vector<at::Tensor>& outputs) {
  auto start = std::chrono::system_clock::now();
  auto inputDLTensors = makeDLConstTensors(inputs);
  auto outputDLTensors = makeDLTensors(outputs);
  ProfilingInfo pi(executor.profile(
      extractRawPtrs(inputDLTensors), extractRawPtrs(outputDLTensors)));

  // The total CPU overhead is the total time minus the (synchronized) kernel
  // runtime
  auto end = std::chrono::system_clock::now();
  Duration cpuOverhead(end - start);
  cpuOverhead = cpuOverhead - pi.kernelRuntime;
  return ProfilingInfo{cpuOverhead, pi.kernelRuntime};
}

template <typename Executor>
void uncheckedRun(
    const Executor& executor,
    const std::vector<at::Tensor>& inputs,
    std::vector<at::Tensor>& outputs) {
  CHECK_GE(outputs.size(), 1u);
  std::vector<const void*> rawInputs(inputs.size(), nullptr);
  std::vector<void*> rawOutputs(outputs.size(), nullptr);
  for (size_t i = 0; i < inputs.size(); ++i) {
    rawInputs[i] = inputs[i].data_ptr();
  }
  for (size_t i = 0; i < outputs.size(); ++i) {
    rawOutputs[i] = outputs[i].data_ptr();
  }
  return executor.uncheckedRun(rawInputs, rawOutputs);
}
} // namespace aten
} // namespace tc
