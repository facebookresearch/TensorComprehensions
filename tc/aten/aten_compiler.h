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

namespace tc {
namespace aten {
/// Given a TC string with multiple functions defined and a TC function name
/// entryPoint, this runs inference, applied to the specified input shapes.
/// This is used for manual processing of output metadata on the ATen user
/// side.
/// \returns the (contiguous) output tensor shapes as a metadata-owning
/// DLTensorUPtr.
std::vector<DLTensorUPtr> inferOutputTensorInfo(
    const std::string& tc,
    const std::string& entryPoint,
    const std::vector<at::Tensor>& inputs);

/// Given a TC string with multiple functions defined and a TC function name
/// entryPoint, this runs inference, applied to the specified input shapes and
/// allocates fresh new output tensors.
/// If one wants inplace/resize behavior, one can implement it using
/// inferOutputTensorInfo.
/// \returns the (contiguous) output tensors with properly inferred sizes.
std::vector<at::Tensor> prepareOutputs(
    const std::string& tc,
    const std::string& entryPoint,
    const std::vector<at::Tensor>& inputs);

/// Given a TC string with multiple functions defined and a TC function name
/// entryPoint, compile the TC for the specified input shapes with the
/// prescribed options.
/// \returns an Executor for the specified backend with an underlying
/// JIT-compiled callable function.
template <typename Backend>
std::unique_ptr<typename Backend::ExecutorType> compile(
    const std::string& tc,
    const std::string& entryPoint,
    const std::vector<at::Tensor>& inputs,
    const typename Backend::MappingOptionsType& options);

/// Given an executor resulting from compiling a TC, run the TC and fill the
/// outputs vector with the results.
template <typename Executor>
void run(
    const Executor& executor,
    const std::vector<at::Tensor>& inputs,
    std::vector<at::Tensor>& outputs);

/// Given an executor resulting from compiling a TC, run the TC and fill the
/// outputs vector with the results.
/// \returns ProfilingInfo (cpuOverhead + kernelRuntime) in microseconds.
template <typename Executor>
ProfilingInfo profile(
    const Executor& executor,
    const std::vector<at::Tensor>& inputs,
    std::vector<at::Tensor>& outputs);

/// This is the "low-latency" mode in which we just propagate ATen tensors
/// Sizes are not checked and it is the user's responsibility to ensure that
/// they match. If the user doesn't then segfault will likely occur.
template <typename Executor>
void uncheckedRun(
    const Executor& executor,
    const std::vector<at::Tensor>& inputs,
    std::vector<at::Tensor>& outputs);
} // namespace aten
} // namespace tc

#include "tc/aten/aten_compiler-inl.h"
