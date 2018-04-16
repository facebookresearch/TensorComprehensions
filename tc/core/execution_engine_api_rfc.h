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

#include "tc/core/compilation_cache.h"
#include "tc/core/tc_executor.h"
#include "tc/core/tensor.h"
#include "tc/core/utils/time.h"
#include "tc/lang/tree.h"

namespace tc {

/**
 * The ExecutionEngine is the entry point to using TC in C++.
 * It provides simple backend-agnostic capabilities to:
 *   1. construct the tree IR for parsed TCs from
 *      a string representation and store (a shared_ptr to) it
 *   2. infer actual tmp/output tensor shapes given input tensor shapes;
 *   3. compile and run TCs on backend-specific TcExecutors;
 *   4. store the TcExecutors for memoization of compilations.
 *      The correspondance is:
 *      1 TcExecutor <-> 1 tuple<TC function, input shapes, MappingOptions>
 *
 * ExecutionEngine is templated by the Backend type.
 * For each backend, the specific Backend type lives in
 *   backend/backed_tc_executor.h and declares all the required dependent
 *   **concrete** types.
 * For example:
 *   CudaBackend is declared in core/cuda/cuda_tc_executor.h
 *
 * struct CudaBackend {
 *   using ExecutorType = CudaTcExecutor;
 *   using MappingOptionsType = CudaMappingOptions;
 *   using ProtobufType = CudaCacheProto;
 *   using CacheType = CudaCache;
 *   using CacheEntryType = CudaCacheEntry;
 *   using CacheKeyType = typename CacheEntryType::Key;
 *   using CacheValueType = typename CacheEntryType::Values;
 *   using RTCFunctionType = CudaRTCFunction;
 * };
 *
 * Sketching usage resembles:
 *   std::string someTc = "...";
 *   ExecutionEngine<CudaBackend> engine;
 *   engine.define(someTc)
 *   auto handle = engine.compile(someTc, inputs, mappingOptions)
 *   auto cpuKernelProfilingInfo = engine.run(handle, inputs, outputs, true);
 *   // alternatively:
 *   // auto kernelTiming = engine.uncheckedRun(handle, inputs, outputs, true);
 */
template <typename Backend>
class ExecutionEngine {
 public:
  using BackendType = Backend;
  using ExecutorType = typename BackendType::ExecutorType;
  using MappingOptionsType = typename BackendType::MappingOptionsType;

  ExecutionEngine() = default;

  /// Parse TC definitions provided as string, store parsed trees internally.
  void define(const std::string& language);

  /// Store the provided parsed trees internally.
  void define(const std::vector<lang::TreeRef>& treeRefs);

  /// For a given TC kernel name, compute the shapes of the output tensors
  /// provided the shapes of the input TC tensors.  The caller can use the
  /// computed shapes to allocate memory for outputs.
  std::vector<TensorInfo> inferOutputTensorInfo(
      const std::string& name,
      const std::vector<const DLConstTensor*>& inTensorPtrs);

  /// JIT-compile given the TC kernel name, the shapes of the input tensors and
  /// the compilation options.  Must be overridden by a specific
  /// ExecutionEngine, which also interprets the options as it sees fit.
  /// \returns opaque handle of a compiled kernel.
  size_t compile(
      const std::string& name,
      const std::vector<const DLConstTensor*>& inputs,
      const MappingOptionsType& options);

  /// Run a compiled TC kernel given its handle, on the given input tensors and
  /// fill in the outputs.  All tensors must be allocated and have appropriate
  /// shapes (inputs same as for copmilation, outputs same as returned by
  /// inferOutputTensorInfo).
  /// \returns The pair of (cpu,kernel) runtimes. If profile is set, the
  /// kernel is profiled otherwise its duration is set to Duration::max().
  ProfilingInfo run(
      size_t handle,
      const std::vector<const DLConstTensor*>& inputs,
      const std::vector<const DLTensor*>& outputs,
      bool profileKernel = false,
      std::function<bool(const ExecutorType*)> pruningFunction =
          [](const ExecutorType*) { return false; });

  /// "Low-latency" execution mode in which we just propagate raw pointers to
  /// data in GPU address space.
  /// No tensor-related information can be checked so it is the user's
  /// responsibility to ensure that shapes and strides match.
  Duration uncheckedRun(
      size_t handle,
      const std::vector<const void*>& inputs,
      const std::vector<void*>& outputs,
      bool profileKernel = false);

  /// Clear the compilation result for the given handle.
  void clear(size_t handle);

 private:
  size_t emplaceExecutor(std::unique_ptr<ExecutorType>&& p);

  size_t getHandle(
      const std::string& name,
      const std::vector<const DLConstTensor*>& inputs,
      const MappingOptionsType& options);

 private:
  /// For thread-safety perform all cheap operations under lock.
  std::mutex tcExecutorMutex_;

  /// Parsed TC trees.
  std::map<std::string, lang::TreeRef> tcNameMap_;

  /// List of executors, indexed by handle.
  std::vector<std::unique_ptr<ExecutorType>> executors_;

  size_t nextHandle = 0;
};
} // namespace tc

#include "tc/core/execution_engine-inl.h"
