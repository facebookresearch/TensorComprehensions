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
 * TcExecutor is a backend-agnostic abstraction that is the result of
 * compilation. When a concrete executor specializes TcExecutor<Backend> it
 * provides support for running a compiled TC on the particular backend.
 *
 * TcExecutor is templated by the Backend type.
 * For each backend, the specific Backend type lives in
 *   core/backend/backend_tc_executor.h and declares all the required dependent
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
 *   using CompilationResultType = typename CacheEntryType::Values;
 *   using RTCFunctionType = CudaRTCFunction;
 * };
 *
 * The correspondence is 1 TcExecutor for 1
 *   tuple<TC function, input shapes, Backend::MappingOptions>.
 *
 * Backend-specific Executors should specialize TcExecutor<Backend> and
 * implement the uncheckedRun and setRuntimeCompiledFunction.
 *
 * Specializing a TcExecutor<Backend> consists in overriding 2 types of methods:
 * 1. uncheckedRun/profileUnchecked which provides the low-latency path on
 *    void* data
 * 2. setupRuntimeCompiledFunction/clearRuntimeCompiledFunction which takes
 *    the results of compileWithTcMapper to:
 *       a. save backend-specific information required at runtime (e.g. grid and
 *          block sizes for the CUDA backend)
 *       b. create the in-memory object resulting from JIT-compilation that
 *          implements the TC
 */
template <typename Backend>
class TcExecutor {
 protected:
  TcExecutor(
      lang::TreeRef tcDefinition,
      const std::vector<const DLConstTensor*>& inputs);

 public:
  virtual ~TcExecutor();

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

  /// Calls run and profiles the kernel runtime (microseconds).
  /// \returns profiling information (see: tc/core/utils/time.h)
  ProfilingInfo profile(
      const std::vector<const DLConstTensor*>& inputs,
      const std::vector<const DLTensor*>& outputs) const;

  /*****************************************************************************
   * The following pure virtual methods are the minimal set of required
   * backend-dependent functions.
   ****************************************************************************/
  /// This is the "low-latency" mode in which we just propagate raw pointers to
  /// data in the address space where kernel is executed.
  /// No tensor-related information can be checked so it is the user's
  /// responsibility to ensure that shapes and strides match. If the user
  /// doesn't then segfault will likely occur.
  virtual void uncheckedRun(
      const std::vector<const void*>& inputs,
      const std::vector<void*>& outputs) const = 0;
  virtual ProfilingInfo profileUnchecked(
      const std::vector<const void*>& inputs,
      const std::vector<void*>& outputs) const = 0;

 protected:
  /// setupRuntimeCompiledFunction takes a compilationResult (returned by
  /// the specialized TC mapper) and:
  ///   a. saves backend-specific information required at runtime
  ///     (e.g. grid and block sizes for the CUDA backend)
  ///   b. creates the in-memory object resulting from JIT-compilation that
  ///      implements the TC
  virtual void setupRuntimeCompiledFunction(
      const typename Backend::CompilationResultType& compilationResult) = 0;

  /// It may be necessary to clear the RTC manually because it can throw and
  /// we can't have that in the RTC destructor.
  virtual void clearRuntimeCompiledFunction() = 0;

 protected:
  /// Used to check proper metadata when calling run
  std::vector<TensorInfo> inputsInfo_;
  std::vector<TensorInfo> outputsInfo_;
  tc2halide::HalideComponents halideComponents_;

  /// The following are initialized as a result of compilation with the TcMapper
  std::vector<int> parameters_;
  std::shared_ptr<typename Backend::RTCFunctionType> rtcFun_;

 public:
  template <typename BackendType>
  friend std::unique_ptr<typename BackendType::ExecutorType> compile(
      lang::TreeRef tcDefinition,
      const std::vector<const DLConstTensor*>& inputs,
      /* TODO: in the future also pass outputs for stride and alignment info */
      const typename BackendType::MappingOptionsType& options);
};
} // namespace tc

#include "tc/core/tc_executor-inl.h"
