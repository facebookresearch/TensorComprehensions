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

#include "tc/core/tc_executor.h"
#include "tc/core/utils/dlpack.h"
#include "tc/core/utils/time.h"
#include "tc/lang/tree.h"

namespace tc {
class ExecutionEngine {
 public:
  struct ExecutorInfo {
    ExecutorInfo(
        std::string id,
        std::vector<const DLTensor*> inputsInfo,
        const std::string& options,
        lang::TreeRef tc,
        size_t handle)
        : identifier(id),
          inputsInfo(dlutils::makeDLTensorVector(inputsInfo)),
          options(options),
          exec(new TcExecutor(tc, inputsInfo)),
          objectLocalHandle(handle) {}

    std::string identifier;
    std::vector<dlutils::DLTensorUPtr> inputsInfo;
    std::string options;
    std::unique_ptr<TcExecutor> exec;
    /// When run is called this is used to find the most recently compiled
    /// version.
    size_t objectLocalHandle;
  };

  ExecutionEngine() = default;

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
  virtual size_t compile(
      const std::string& name,
      const std::vector<const DLTensor*>& inputs,
      const std::string& options) = 0;

  /// Run a compiled TC kernel given its handle, on the given input tensors and
  /// fill in the outputs.  All tensors must be allocated and have appropriate
  /// shapes (inputs same as for copmilation, outputs same as returned by
  /// inferOutputTensorInfo).
  /// \returns The kernel runtime if profile is set, Duration::max() otherwise.
  virtual Duration run(
      size_t handle,
      const std::vector<const DLTensor*>& inputs,
      const std::vector<DLTensor*>& outputs,
      bool profile = false) = 0;

  /// "Low-latency" execution mode in which we just propagate raw pointers to
  /// data in GPU address space.
  /// No tensor-related information can be checked so it is the user's
  /// responsibility to ensure that shapes and strides match.
  virtual void uncheckedRun(
      size_t handle,
      const std::vector<const void*>& inputs,
      const std::vector<void*>& outputs) = 0;

  /// Clear the compilation result for the given handle.
  virtual void clear(size_t handle) {}

 protected:
  size_t emplaceExecutor(std::unique_ptr<ExecutorInfo> p);

  /// For thread-safety perform all cheap operations under lock.
  std::mutex executorInfoMutex;

  /// Parsed TC trees.
  std::map<std::string, lang::TreeRef> tcNameMap_;

  /// List of executors, indexed by handle.  Derived ExecutionEngines can also
  /// derive ExecutorInfo.
  std::vector<std::unique_ptr<ExecutorInfo>> executors_;

  size_t uidCounter = 0;
};

} // namespace tc
