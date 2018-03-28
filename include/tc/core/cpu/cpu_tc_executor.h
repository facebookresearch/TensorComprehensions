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

#include <dlpack/dlpack.h>

#include "tc/core/cpu/cpu_mapping_options.h"
#include "tc/core/halide_utils.h"
#include "tc/core/polyhedral/scop.h"
#include "tc/core/tc_executor.h"
#include "tc/core/utils/dlpack.h"
#include "tc/lang/parser.h"

namespace tc {

struct CpuRTCFunction {
  void clear() {}
};

class CpuTcExecutor : public ::tc::TcExecutor {
 public:
  using MappingOptionsType = CpuMappingOptions;

  CpuTcExecutor(
      std::string id,
      const std::vector<const DLTensor*>& inputsInfo,
      const std::string& options,
      lang::TreeRef tcDefinition)
      : TcExecutor(id, inputsInfo, options, tcDefinition) {}

  ~CpuTcExecutor() {}

  CpuTcExecutor(CpuTcExecutor&&) = delete;
  CpuTcExecutor& operator=(CpuTcExecutor&&) = delete;
  CpuTcExecutor(const CpuTcExecutor&) = delete;
  CpuTcExecutor& operator=(const CpuTcExecutor&) = delete;

  // Can only be called once with specific kernel options.  Input sizes are
  // set up as constructor argument and output sizes are inferred.
  //
  // If you need another kernel for another Tc or another inputs, outputs,
  // options then just instantiate another CpuTcExecutor.
  // This is because for the time being we fully specialize all the sizes and
  // strides at runtime.
  // @{
  void compile(const std::string& options) override {
    compile(MappingOptionsType(options));
  }
  void compile(const tc::CpuMappingOptions& options);
  // @}

  // Run can be called multiple times given a compilation, inputs are allowed
  // to change in that their data pointer is allowed to change.
  // Sizes and strides must remain constant otherwise this is an error
  // The only thing that is allowed to change across runs is the input
  // and output pointers base address.
  // It is the caller's responsibility to ensure proper non-aliasing (or
  // advanced aliasing) properties of the input and output tensors.
  // if profile is set the kernel runtime (nanoseconds) is returned
  Duration run(
      const std::vector<const DLTensor*>& inputs,
      const std::vector<DLTensor*>& outputs,
      bool profile = false) const override;

  // This is the "low-latency" mode in which we just propagate raw pointers to
  // data in GPU address space.
  // No tensor-related information can be checked so it is the user's
  // responsibility to ensure that shapes and strides match. If the user
  // doesn't then segfault will likely occur.
  void uncheckedRun(
      const std::vector<const void*>& inputs,
      const std::vector<void*>& outputs) const override;

  bool hasRuntimeCompiledFunction() override {
    return rtcFunction.get() != nullptr;
  }

  // It is necessary to clear the RTC manually because it can throw and we
  // can't have that in the destructor.
  void clearRuntimeCompiledFunction() override {
    if (!hasRuntimeCompiledFunction()) {
      return;
    }
    rtcFunction->clear();
  }

  std::string kernelName() const {
    return executionInfo_.kernelName;
  }

 private:
  void compileWithTcMapper();

 public:
  std::string kernelSpecializedName;
  std::string cpuSource;

 protected:
  std::shared_ptr<CpuRTCFunction> rtcFunction;
};

} // namespace tc
