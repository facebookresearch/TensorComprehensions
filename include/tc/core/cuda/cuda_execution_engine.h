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

#include <map>
#include <memory>
#include <string>
#include <vector>

#include <dlpack/dlpack.h>

#include "tc/core/cuda/cuda_tc_executor.h"
#include "tc/core/execution_engine.h"
#include "tc/core/mapping_options.h"

namespace tc {

/// The goal for this new shiny API is to provide a different pathway for being
/// able to execute the kernels for multiple TC i.e. given the language which
/// can have multiple TCs, people should be able to run things by just calling
/// out the run function with the name of function and the inputs to run on.
class CudaExecutionEngine : public ExecutionEngine {
 public:
  struct CudaExecutorInfo : public ExecutionEngine::ExecutorInfo {
    CudaExecutorInfo(
        std::string id,
        std::vector<const DLTensor*> inputsInfo,
        const MappingOptions& options,
        lang::TreeRef tc,
        size_t handle)
        : ExecutionEngine::ExecutorInfo(
              id,
              inputsInfo,
              options.toProtobufSerializedString(),
              tc,
              handle) {
      exec =
          std::unique_ptr<CudaTcExecutor>(new CudaTcExecutor(tc, inputsInfo));
    }

    void clear() {
      static_cast<CudaTcExecutor&>(*exec).clearRTC();
    }
  };

  CudaExecutionEngine() = default;

  void addTC(const std::string& tc);

  lang::TreeRef treeForFunction(const std::string& name) {
    return tcNameMap_.at(name);
  }

  // TODO: Pass autotuning info (none by default, otherwise some struct with
  //       maxtime and other things)

  /// returns a handle for the compiled kernel
  /// @{
  size_t compile(
      const std::string& name,
      const std::vector<const DLTensor*>& inputs,
      const std::string& mappingOptions) override {
    return compile(name, inputs, MappingOptions(mappingOptions));
  }
  size_t compile(
      const std::string& name,
      const std::vector<const DLTensor*>& inputs,
      const MappingOptions& options);
  /// @}

  // TODO: sanity check on name and input / output sizes.
  /// Run a TC specified by its name on the given tensor inputs and fill the
  /// outputs with the result.
  /// The TC is looked up by its handle.
  /// If profile is set, the kernel runtime is returned.
  ///
  /// The pruning function returns true if the run should not proceed (e.g. if
  /// there are too few threads mapped that would likely result in catastrophic
  /// performance). In this case, return Duration::max().
  /// @{
  Duration run(
      size_t handle,
      const std::vector<const DLTensor*>& inputs,
      const std::vector<DLTensor*>& outputs,
      bool profile = false) override {
    return run(handle, inputs, outputs, profile, [](const CudaExecutorInfo*) {
      return false;
    });
  }
  Duration run(
      size_t handle,
      const std::vector<const DLTensor*>& inputs,
      const std::vector<DLTensor*>& outputs,
      bool profile,
      std::function<bool(const CudaExecutorInfo*)> pruningFunction);
  /// @}

  /// This is the "low-latency" mode in which we just propagate raw pointers to
  /// data in GPU address space.
  /// No tensor-related information can be checked so it is the user's
  /// responsibility to ensure that shapes and strides match. If the user
  /// doesn't then segfault will likely occur.
  void uncheckedRun(
      size_t handle,
      const std::vector<const void*>& inputs,
      const std::vector<void*>& outputs) override;

  void clear(size_t handle) override;

 private:
  size_t getHandle(
      const std::string& name,
      const std::vector<const DLTensor*>& inputsInfo,
      const MappingOptions& options);
  std::unique_ptr<ExecutorInfo> makeExecutorInfo(
      const std::string& name,
      const std::vector<const DLTensor*>& inputsInfo,
      const MappingOptions& options);
};

} // namespace tc
