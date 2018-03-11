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

#include "tc/autotuner/genetic_autotuner_aten.h"

#include <chrono>
#include <csignal>
#include <thread>

#include "tc/core/cuda/cuda.h"
#include "tc/core/cuda/cuda_compilation_cache.h"
#include "tc/core/cuda/cuda_tc_executor.h"
#include "tc/core/flags.h"
#include "tc/core/scope_guard.h"
#include "tc/lang/parser.h"

namespace tc {
namespace autotune {

GeneticAutotunerATen::GeneticAutotunerATen(const std::string tc) : tc_(tc) {
  geneticAutotuner_ = std::unique_ptr<detail::GeneticAutotuner>(
      new detail::GeneticAutotuner(tc));
}

std::vector<MappingOptions> GeneticAutotunerATen::load(
    const std::string& cacheFileName,
    const std::string& tcName,
    const std::vector<at::Tensor> inputs,
    const size_t numCandidates) {
  auto inputsPair = tc::toConstDlpackTensors(inputs);
  tc::ScopeGuard g([&]() { tc::deleteDlmTensors(inputsPair.second); });
  return geneticAutotuner_->load(
      cacheFileName, tcName, inputsPair.first, numCandidates);
}

namespace {
void deleteGpuDlmTensors(
    std::unordered_map<size_t, std::vector<DLManagedTensor*>>
        managedGpuTensors) {
  for (auto& gpuTensors : managedGpuTensors) {
    for (auto& tensor : gpuTensors.second) {
      tensor->deleter(tensor);
    }
  }
}

std::vector<at::Tensor> cloneTensors(const std::vector<at::Tensor>& inputs) {
  std::vector<at::Tensor> copies;
  copies.reserve(inputs.size());
  for (const auto& t : inputs) {
    copies.push_back(t.clone());
  }
  return copies;
}

} // namespace

llvm::Optional<MappingOptions> GeneticAutotunerATen::tune(
    const std::string& cacheFileName,
    const std::string& tcName,
    const std::vector<at::Tensor>& inputs,
    MappingOptions baseMapping,
    std::vector<MappingOptions> startingPoints,
    const TuningParameterFixer& fixedParams) {
  // create instance of ATenCompilationUnit so that we can get the outputsInfo
  // and convert those outputs to DLTensors.
  tc::ATenCompilationUnit<CudaTcExecutor> atCompl;
  atCompl.define(tc_);
  auto handle = atCompl.compile(tcName, inputs, baseMapping);
  std::vector<at::Tensor> outputs;
  atCompl.run(tcName, inputs, outputs, handle);

  // first parse the gpus, clone the inputs on each gpu, pass that inputs to the
  // geneticAutotuner_->tune() call
  auto gpus = tc::autotune::detail::parseGpus();
  std::unordered_map<size_t, std::vector<const DLTensor*>> inputsPerGpu;
  std::unordered_map<size_t, std::vector<DLManagedTensor*>> managedInputsPerGpu;
  std::unordered_map<size_t, std::vector<DLTensor*>> outputsPerGpu;
  std::unordered_map<size_t, std::vector<DLManagedTensor*>>
      managedOutputsPerGpu;
  for (auto gpu : gpus) {
    WithDevice wd(gpu);
    auto gpuInputs = cloneTensors(inputs);
    auto gpuInputDLTensorsPair = tc::toConstDlpackTensors(gpuInputs);
    inputsPerGpu.emplace(gpu, gpuInputDLTensorsPair.first);
    managedInputsPerGpu.emplace(gpu, gpuInputDLTensorsPair.second);

    auto gpuOutputs = cloneTensors(outputs);
    auto gpuOutputDLTensorsPair = tc::toDlpackTensors(gpuOutputs);
    outputsPerGpu.emplace(gpu, gpuOutputDLTensorsPair.first);
    managedOutputsPerGpu.emplace(gpu, gpuOutputDLTensorsPair.second);
  }

  tc::ScopeGuard g1([&]() { deleteGpuDlmTensors(managedInputsPerGpu); });
  tc::ScopeGuard g2([&]() { deleteGpuDlmTensors(managedOutputsPerGpu); });

  return geneticAutotuner_->tune(
      cacheFileName,
      tcName,
      inputsPerGpu,
      outputsPerGpu,
      baseMapping,
      startingPoints,
      fixedParams);
}

} // namespace autotune
} // namespace tc
