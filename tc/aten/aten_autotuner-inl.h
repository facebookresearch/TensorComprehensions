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

#include "tc/aten/aten.h"
#include "tc/aten/aten_compiler.h"
#include "tc/autotuner/utils.h"
#include "tc/core/compiler.h"
#include "tc/core/flags.h"
#include "tc/core/scope_guard.h"
#include "tc/core/tensor.h"
#include "tc/core/utils/math.h"
#include "tc/lang/canonicalize.h"

namespace tc {
namespace aten {
template <typename Backend, typename Search>
ATenAutotuner<Backend, Search>::ATenAutotuner(const std::string& tc)
    : tc::autotune::Autotuner<Backend, Search>(tc), tc_(tc) {}

std::vector<at::Tensor> cloneTensors(const std::vector<at::Tensor>& inputs) {
  std::vector<at::Tensor> copies;
  copies.reserve(inputs.size());
  for (const auto& t : inputs) {
    copies.push_back(t.clone());
  }
  return copies;
}

template <typename Backend, typename Search>
std::vector<typename Backend::MappingOptionsType>
ATenAutotuner<Backend, Search>::tune(
    const std::string& tcName,
    const std::vector<at::Tensor>& inputs,
    const typename Backend::MappingOptionsType& baseMapping,
    const std::string& cacheFileName,
    const tc::autotune::TuningParameterFixer& fixedParams) {
  // TODO: some checks that inputs memory lives on the proper Backend device

  // prepare outputs of the proper shape
  auto outputs = tc::aten::prepareOutputs(tc_, tcName, inputs);

  // first parse the devices
  auto devices =
      tc::autotune::detail::parseDevices<Backend>(FLAGS_tuner_devices);
  // clone the inputs/outputs on each device
  // TODO: this takes twice the space it should, alternatives are:
  // 1. enforce inputs and outputs live on the CPU in the first place so we
  //    don't spuriously run out of device memory (assuming CPU memory is
  //    infinite for now);
  // 2. if 1. is not reasonable, detect the device on which each tensor lives
  //    and point to the raw data for that (device, tensor) pair.
  std::unordered_map<size_t, std::vector<DLConstTensorUPtr>> inputsPerDevice;
  std::unordered_map<size_t, std::vector<const DLConstTensor*>>
      rawInputsPerDevice;
  std::unordered_map<size_t, std::vector<DLTensorUPtr>> outputsPerDevice;
  std::unordered_map<size_t, std::vector<const DLTensor*>> rawOutputsPerDevice;
  for (auto device : devices) {
    typename Backend::WithDevice wd(device);
    auto deviceInputs = cloneTensors(inputs);
    inputsPerDevice.emplace(device, toDLConstTensors(deviceInputs));
    rawInputsPerDevice.emplace(
        device, extractRawPtrs(inputsPerDevice.at(device)));
    auto deviceOutputs = cloneTensors(outputs);
    outputsPerDevice.emplace(device, makeDLTensors(deviceOutputs));
    rawOutputsPerDevice.emplace(
        device, extractRawPtrs(outputsPerDevice.at(device)));
  }
  return tc::autotune::Autotuner<Backend, Search>::tune(
      tcName,
      rawInputsPerDevice,
      rawOutputsPerDevice,
      baseMapping,
      cacheFileName,
      fixedParams);
}
} // namespace aten
} // namespace tc
