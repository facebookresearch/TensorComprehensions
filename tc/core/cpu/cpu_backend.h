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

#include "tc/core/cpu/cpu_mapping_options.h"
#include "tc/core/cpu/cpu_mapping_options_cpp_printer.h"
// TODO: when ready
// #include "tc/core/cpu/cpu_rtc.h"
#include "tc/core/halide_utils.h"
#include "tc/core/tensor.h"

namespace tc {
struct CpuRTCFunction {
  void clear() {}
};

struct WithCpuDevice {
  WithCpuDevice(size_t g) {}
};

struct CpuCompilationResult {
  std::string source;
  std::string specializedName;
  std::vector<int> parameters;
};

struct CpuTcExecutor;

struct CpuBackend {
  using ExecutorType = CpuTcExecutor;
  using MappingOptionsType = CpuMappingOptions;
  using CompilationResultType = CpuCompilationResult;
  using OptionsCacheProtoType = CpuOptionsCacheProto;
  using OptionsCacheValueProtoType = CpuOptionsCacheValueProto;
  using RTCFunctionType = CpuRTCFunction;

  // TODO: specialize for proper CPU device (NUMA domain/pinned to
  // core/nothing?)
  using WithDevice = tc::WithCpuDevice;
  using MappingOptionsAsCpp = tc::CpuMappingOptionsAsCpp;
  using MappingOptionsCppPrinter = tc::CpuMappingOptionsCppPrinter;

  static std::string deviceString() {
    return "SOME_CPU";
  }
  static std::string makeDeviceFilename(const std::string& fn) {
    return fn + ".cpu";
  }

  static CompilationResultType compileWithTcMapper(
      const std::string& tcName,
      tc2halide::HalideComponents halideComponents,
      const std::vector<const DLConstTensor*>& inputs,
      /* TODO: in the future also pass outputs for stride and alignment */
      const MappingOptionsType& options);
};
} // namespace tc
