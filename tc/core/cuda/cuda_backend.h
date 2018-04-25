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
/**
 * Information returned by polyhedral compilation. In particular, since we use
 * tightening of loop bounds for CUDA kernels, it includes the actual grid and
 * block sizes needed at runtime.
 */
struct CudaCompilationResult {
  std::string source;
  std::string specializedName;
  std::vector<long> parameters;
  Grid grid;
  Block block;
  bool useGridSync;
};

/**
 * Information that can be set at runtime to control placement and
 * synchronization information of a kernel.
 */
struct CudaRuntimeInformation {
 public:
  CudaRuntimeInformation() : stream(0) {}

 public:
  cudaStream_t stream;
};

struct CudaTcExecutor;

/**
 * This type declares the dependent types and static functions needed to
 * autotune, compile and run for the CPU backend.
 */
struct CudaBackend {
  using ExecutorType = CudaTcExecutor;
  using MappingOptionsType = CudaMappingOptions;
  using CompilationResultType = CudaCompilationResult;
  using OptionsCacheProtoType = CudaOptionsCacheProto;
  using OptionsCacheValueProtoType = CudaOptionsCacheValueProto;
  using RTCFunctionType = CudaRTCFunction;

  using WithDevice = WithCudaDevice;
  using RuntimeInformation = CudaRuntimeInformation;
  using MappingOptionsAsCpp = CudaMappingOptionsAsCpp;
  using MappingOptionsCppPrinter = CudaMappingOptionsCppPrinter;

  static inline std::string backendString() {
    return CudaGPUInfo::GPUInfo().getCudaDeviceStr();
  }
  static inline std::string makeDeviceFilename(const std::string& fn) {
    return fn + ".cuda";
  }

  /// Main entry point for polyhedral compilation
  static CompilationResultType compileWithTcMapper(
      const std::string& tcName,
      tc2halide::HalideComponents halideComponents,
      const std::vector<const DLConstTensor*>& inputs,
      /* TODO: in the future also pass outputs for stride and alignment */
      const MappingOptionsType& options);
};
} // namespace tc
