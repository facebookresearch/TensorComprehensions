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

#include "tc/core/cuda/cuda_mapping_options.h"
#include "tc/core/tc_executor.h"
#include "tc/core/utils/dlpack.h"
#include "tc/core/utils/time.h"
#include "tc/lang/tree.h"

namespace tc {
template <typename ExecutorType>
class ExecutionEngine {
 public:
  ExecutionEngine() = default;

  lang::TreeRef treeForFunction(const std::string& name) {
    return tcNameMap_.at(name);
  }

  /// Parse TC definitions provided as string, store parsed trees internally.
  void define(const std::string& language);

  /// Store the provided parsed trees internally.
  void define(const std::vector<lang::TreeRef>& treeRefs);

  /// For a given TC kernel name, compute the shapes of the output tensors
  /// provided the shapes of the input TC tensors.  The caller can use the
  /// computed shapes to allocate memory for outputs.  Values inside tensors
  /// are left unmodified.
  std::vector<const DLTensor*> inferOutputTensorInfo(
      const std::string& name,
      const std::vector<const DLTensor*>& inTensorPtrs);

  /// JIT-compile given the TC kernel name, the shapes of the input tensors and
  /// the compilation options.  Must be overridden by a specific
  /// ExecutionEngine, which also interprets the options as it sees fit.
  /// \returns opaque handle of a compiled kernel.
  size_t compile(
      const std::string& name,
      const std::vector<const DLTensor*>& inputs,
      const std::string& options);

  /// Run a compiled TC kernel given its handle, on the given input tensors and
  /// fill in the outputs.  All tensors must be allocated and have appropriate
  /// shapes (inputs same as for copmilation, outputs same as returned by
  /// inferOutputTensorInfo).
  /// \returns The kernel runtime if profile is set, Duration::max() otherwise.
  Duration run(
      size_t handle,
      const std::vector<const DLTensor*>& inputs,
      const std::vector<DLTensor*>& outputs,
      bool profile = false,
      std::function<bool(const ExecutorType*)> pruningFunction =
          [](const ExecutorType*) { return false; });

  /// "Low-latency" execution mode in which we just propagate raw pointers to
  /// data in GPU address space.
  /// No tensor-related information can be checked so it is the user's
  /// responsibility to ensure that shapes and strides match.
  void uncheckedRun(
      size_t handle,
      const std::vector<const void*>& inputs,
      const std::vector<void*>& outputs);

  /// Clear the compilation result for the given handle.
  void clear(size_t handle);

 protected:
  size_t emplaceExecutor(std::unique_ptr<ExecutorType> p);

  size_t getHandle(
      const std::string& name,
      const std::vector<const DLTensor*>& inputsInfo,
      const std::string& optionsStr);

  /// For thread-safety perform all cheap operations under lock.
  std::mutex tcExecutorMutex_;

  /// Parsed TC trees.
  std::map<std::string, lang::TreeRef> tcNameMap_;

  /// List of executors, indexed by handle.  Derived ExecutionEngines can also
  /// derive TcExecutor.
  std::vector<std::unique_ptr<ExecutorType>> executors_;

  size_t uidCounter = 0;
};
} // namespace tc

#include "tc/core/execution_engine-inl.h"
