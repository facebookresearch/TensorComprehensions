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

#include "tc/core/cuda/cuda.h"
#include "tc/core/cuda/cuda_mapping_options.h"
#include "tc/core/cuda/cuda_mapping_options_cpp_printer.h"
#include "tc/core/cuda/cuda_rtc.h"
#include "tc/core/halide_utils.h"
#include "tc/core/tensor.h"

namespace tc {

struct CudaCompilationResult {
  std::string source;
  std::string specializedName;
  std::vector<int> parameters;
  Grid grid;
  Block block;
};

struct CudaTcExecutor;

struct CudaBackend {
  using ExecutorType = CudaTcExecutor;
  using MappingOptionsType = CudaMappingOptions;
  using CompilationResultType = CudaCompilationResult;
  using OptionsCacheProtoType = CudaOptionsCacheProto;
  using OptionsCacheValueProtoType = CudaOptionsCacheValueProto;
  using RTCFunctionType = CudaRTCFunction;

  using WithDevice = tc::WithDevice;
  using MappingOptionsAsCpp = tc::CudaMappingOptionsAsCpp;
  using MappingOptionsCppPrinter = tc::CudaMappingOptionsCppPrinter;

  static std::string deviceString() {
    return CudaGPUInfo::GPUInfo().getCudaDeviceStr();
  }
  static std::string makeDeviceFilename(const std::string& fn) {
    return fn + ".cuda";
  }
  static CompilationResultType compileWithTcMapper(
      const std::string& tcName,
      tc2halide::HalideComponents halideComponents,
      const std::vector<const DLConstTensor*>& inputs,
      /* TODO: in the future also pass outputs for stride and alignment */
      const MappingOptionsType& options);
};
} // namespace tc
