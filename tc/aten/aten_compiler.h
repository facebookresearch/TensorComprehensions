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

#include <chrono>
#include <string>
#include <vector>

#include <ATen/ATen.h>
#include <ATen/DLConvertor.h>

#include "tc/aten/utils.h"
#include "tc/core/execution_engine.h"
#include "tc/lang/parser.h"

namespace tc {
/// This provides the basic interface for writing ATen style tensor operations
/// based on Tensor Comprehensions.

template <typename ExecutorType>
class ATenCompilationUnit {
 public:
  explicit ATenCompilationUnit();

  /// Define a database from input TC language where language can have many
  /// tc strings. This database is used to run any TC just by its name
  /// by passing it to the run function.
  void define(const std::string& language);

  /// Given a TC name, compile the TC
  // TODO: Pass struct to allow autotuning
  size_t compile(
      const std::string& name,
      const std::vector<at::Tensor>& inputs,
      const typename ExecutorType::MappingOptionsType& options);

  /// Get the output Tensor info
  std::vector<const DLTensor*> inferOutputTensorInfo(
      const std::string& name,
      const std::vector<at::Tensor>& inputs);

  /// Given a TC name, run the TC and fill the outputs vector the results if
  /// profile is set it returns the runtime in nanoseconds.
  /// Compilation must have already occured.
  Duration run(
      const std::string& name,
      const std::vector<at::Tensor>& inputs,
      std::vector<at::Tensor>& outputs,
      size_t handle,
      bool profile = false);

  typename ExecutorType::ProfilingInfoType profile(
      const std::string& name,
      const std::vector<at::Tensor>& inputs,
      std::vector<at::Tensor>& outputs,
      size_t handle);

  /// This is the "low-latency" mode in which we just propagate ATen tensors
  /// Sizes are not checked and it is the user's responsibility to ensure that
  /// they match. If the user doesn't then segfault will likely occur.
  void uncheckedRun(
      const std::vector<at::Tensor>& inputs,
      std::vector<at::Tensor>& outputs,
      size_t handle);

 private:
  std::unique_ptr<ExecutionEngine<ExecutorType>> executionEngine_;
};
} // namespace tc

#include "tc/aten/aten_compiler-inl.h"
