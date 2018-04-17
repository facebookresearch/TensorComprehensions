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

#include "tc/aten/aten.h"
#include "tc/core/utils/time.h"
#include "tc/lang/tree.h"

namespace tc {
/// Run inference on the TC tree resulting from parsing a TC function and
/// applied to the specified input shapes.
/// This is used for manual processing of outptu metadata on the ATen user
/// side.
/// \returns the (contiguous) output tensor shapes as a metadata-owning
/// DLTensorUPtr.
std::vector<DLTensorUPtr> inferOutputTensorInfo(
  lang::TreeRef tcDefinition,
  const std::vector<at::Tensor>& inputs);

/// Allocate fresh new output tensors.
/// If one wants inplace/resize behavior, one can implement it using
/// inferOutputTensorInfo.
/// \returns the (contiguous) output tensors with properly inferred sizes.
std::vector<at::Tensor> prepareOutputs(
  lang::TreeRef tcDefinition,
  const std::vector<at::Tensor>& inputs);

/// Given a TC tree resulting from parsing a TC function, compile the TC for
/// the specified input shapes with the prescribed options.
/// \returns an Executor for the specified backend with an underlying
/// JIT-compiled callable function.
template <typename Backend>
std::unique_ptr<typename Backend::ExecutorType> compile(
    lang::TreeRef tcDefinition,
    const std::vector<at::Tensor>& inputs,
    const typename Backend::MappingOptionsType& options);

/// Given an executor resulting from compiling a TC, run the TC and fill the
/// outputs vector with the results.
/// \returns ProfilingInfo (cpuOverhead + kernelRuntime) in microseconds.
/// If profile=false, kernelRuntime is set to Duration::max.
template <typename Executor>
ProfilingInfo run(
  const Executor& executor,
  const std::vector<at::Tensor>& inputs,
  std::vector<at::Tensor>& outputs,
  bool profile = false);

/// This is the "low-latency" mode in which we just propagate ATen tensors
/// Sizes are not checked and it is the user's responsibility to ensure that
/// they match. If the user doesn't then segfault will likely occur.
/// \returns the kernel runtime in microseconds if profile=true.
/// If profile=false, kernelRuntime is set to Duration::max.
template <typename Executor>
Duration uncheckedRun(
    const Executor& executor,
    const std::vector<at::Tensor>& inputs,
    std::vector<at::Tensor>& outputs,
    bool profile = false);
} // namespace tc

#include "tc/aten/aten_compiler-inl.h"