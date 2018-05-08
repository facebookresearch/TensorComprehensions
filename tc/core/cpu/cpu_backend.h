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

#include <glog/logging.h>

#include "tc/core/cpu/cpu_mapping_options.h"
#include "tc/core/cpu/cpu_mapping_options_cpp_printer.h"
#include "tc/core/cpu/cpu_rtc.h"
#include "tc/core/halide_utils.h"
#include "tc/core/tensor.h"

namespace tc {
/**
 * RAII placement support, modeled on CUDA getDevice / setDevice support in
 * WithCudaDevice.
 * TODO: Impl me (NUMA domain/pinned to core/nothing?)
 */
struct WithCpuDevice {
  WithCpuDevice(size_t g) {}
};

/**
 * Information returned by polyhedral compilation
 */
struct CpuCompilationResult {
  std::string source;
  std::string specializedName;
  std::vector<long> parameters;
};

/**
 * Information that can be set at runtime to control placement and
 * synchronization information of a kernel.
 */
struct CpuRuntimeInformation {};

struct CpuTcExecutor;

/**
 * This type declares the dependent types and static functions needed to
 * autotune, compile and run for the CPU backend.
 */
struct CpuBackend {
  using ExecutorType = CpuTcExecutor;
  using MappingOptionsType = CpuMappingOptions;
  using CompilationResultType = CpuCompilationResult;
  using OptionsCacheProtoType = CpuOptionsCacheProto;
  using OptionsCacheValueProtoType = CpuOptionsCacheValueProto;
  using RTCFunctionType = CpuRTCFunction;

  using WithDevice = WithCpuDevice;
  using RuntimeInformation = CpuRuntimeInformation;
  using MappingOptionsAsCpp = CpuMappingOptionsAsCpp;
  using MappingOptionsCppPrinter = CpuMappingOptionsCppPrinter;

  static inline std::string backendString() {
    LOG(ERROR) << "NYI: CpuBackend::backendString";
    return "SOME_CPU";
  }
  static inline std::string makeDeviceFilename(const std::string& fn) {
    return fn + ".cpu";
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
