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

#include "tc/core/cpu/cpu_backend.h"
#include "tc/core/cpu/cpu_mapping_options.h"
#include "tc/core/tc_executor.h"
#include "tc/core/tensor.h"

namespace tc {
class CpuTcExecutor : public TcExecutor<CpuBackend> {
 public:
  CpuTcExecutor(
      const std::vector<TensorInfo>& inputsInfo,
      const std::vector<TensorInfo>& outputsInfo,
      const tc2halide::HalideComponents& halideComponents,
      const typename CpuBackend::CompilationResultType& compilationResult);

  /// This is the "low-latency" mode in which we just propagate raw pointers to
  /// data in the address space where kernel is executed.
  /// No tensor-related information can be checked so it is the user's
  /// responsibility to ensure that shapes and strides match. If the user
  /// doesn't then segfault will likely occur.
  void uncheckedRun(
      const std::vector<const void*>& inputs,
      const std::vector<void*>& outputs) const;

  /// Calls uncheckedRun and profiles the cpu overhead and kernel runtime
  /// (microseconds).
  /// \returns profiling information (see: tc/core/utils/time.h)
  ProfilingInfo profileUnchecked(
      const std::vector<const void*>& inputs,
      const std::vector<void*>& outputs) const;
};
} // namespace tc
