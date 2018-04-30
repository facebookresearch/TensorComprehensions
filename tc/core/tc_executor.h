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

#include "tc/core/tc2halide.h"
#include "tc/core/tensor.h"
#include "tc/core/utils/time.h"
#include "tc/lang/tree.h"

namespace tc {
/**
 * TcExecutor is a backend-agnostic abstraction that provides the base
 * functionality for an object returned by compilation to run.
 * TcExecutor is templated by the Backend type.
 * When a derived executor inherits from TcExecutor<Backend>, it
 * provides support for running a compiled TC on the particular backend.
 *
 * A TcExecutor mixes templating and inheritance:
 *   1. templating is necessary because we want type-safety with proper
 *      dependent types (e.g. Backend::CompilationResultType)
 *   2. inheritance is necessary because executors require specific
 *      additional information for running on different backends
 *      (e.g. Grid/Block for CUDA)
 *   3. virtual functions seem to be more confusing to people so we don't use
 *      them here. Still derived classes of TcExecutor<Backend> **must**
 *      implement the following functions (or errors will occur at **all call
 *      sites**):
 *
 * /// This is the "low-latency" mode in which we just propagate raw pointers to
 * /// data in the address space where kernel is executed.
 * /// No tensor-related information can be checked so it is the user's
 * /// responsibility to ensure that shapes and strides match. If the user
 * /// doesn't then segfault will likely occur.
 * void uncheckedRun(
 *     const std::vector<const void*>& inputs,
 *     const std::vector<void*>& outputs) const;
 * /// Calls uncheckedRun and profiles the cpu overhead and kernel runtime
 * /// (microseconds).
 * /// \returns profiling information (see: tc/core/utils/time.h)
 * ProfilingInfo profileUnchecked(
 *     const std::vector<const void*>& inputs,
 *     const std::vector<void*>& outputs) const;
 *
 * As a reminder: for each backend, the specific Backend type lives in
 *   core/backendname/backendname_backend.h and declares all the required
 *   dependent **derived** types.
 * For example:
 *   CudaBackend is declared in core/cuda/cuda_backend.h
 *
 * struct CudaBackend {
 *   using ExecutorType = CudaTcExecutor;
 *   using MappingOptionsType = CudaMappingOptions;
 *   using CompilationResultType = CudaCompilationResult;
 *   using RTCFunctionType = CudaRTCFunction;
 * };
 *
 * The correspondence is 1 TcExecutor for 1 compiled
 *   tuple<TC function, input shapes, Backend::MappingOptions>.
 */
template <typename Backend>
class TcExecutor {
 protected:
  TcExecutor(
      const std::vector<TensorInfo>& inputsInfo,
      const std::vector<TensorInfo>& outputsInfo,
      const tc2halide::HalideComponents& halideComponents,
      const typename Backend::CompilationResultType& compilationResult);

 public:
  TcExecutor(TcExecutor&&) = delete;
  TcExecutor& operator=(TcExecutor&&) = delete;
  TcExecutor(const TcExecutor&) = delete;
  TcExecutor& operator=(const TcExecutor&) = delete;

  /// Run can be called multiple times given a compilation, inputs are allowed
  /// to change in that their data pointer is allowed to change.
  /// Sizes and strides must remain constant otherwise this is an error.
  /// The only thing that is allowed to change across runs is the input
  /// and output pointers base address.
  /// It is the caller's responsibility to ensure proper non-aliasing (or
  /// advanced aliasing) properties of the input and output tensors.
  void run(
      const std::vector<const DLConstTensor*>& inputs,
      const std::vector<const DLTensor*>& outputs) const;

  /// Calls run and profiles the cpu overhead and kernel runtime (microseconds).
  /// \returns profiling information (see: tc/core/utils/time.h)
  ProfilingInfo profile(
      const std::vector<const DLConstTensor*>& inputs,
      const std::vector<const DLTensor*>& outputs) const;

  /// It may be necessary to clear the RTC manually because it can throw and
  /// we can't have that in the RTC destructor.
  void clearRuntimeCompiledFunction();

 public:
  // For inspection purposes
  const std::string compiledSource;

 protected:
  /// Used to check proper metadata when calling run
  std::vector<TensorInfo> inputsInfo_;
  std::vector<TensorInfo> outputsInfo_;
  tc2halide::HalideComponents halideComponents_;

  /// The following are initialized as a result of compilation with the TcMapper
  std::vector<int> parameters_;
  std::unique_ptr<typename Backend::RTCFunctionType> rtcFun_;
};
} // namespace tc

#include "tc/core/tc_executor-inl.h"
