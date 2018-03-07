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
#include "tc/core/cuda/cuda_execution_engine.h"
#include "tc/core/polyhedral/mapping_types.h"
#include "tc/core/utils/memory.h"

#include "tc/lang/parser.h"

namespace tc {

using namespace dlutils;

// Steal ExecutorInfo and give it back under lock
// Run outside of lock on owning ExecutorInfo.
Duration CudaExecutionEngine::run(
    size_t handle,
    const std::vector<const DLTensor*>& inputs,
    const std::vector<DLTensor*>& outputs,
    bool profile,
    std::function<bool(const CudaTcExecutor*)> pruningFunction) {
  std::unique_ptr<TcExecutor> p(nullptr);
  {
    std::lock_guard<std::mutex> lg(tcExecutorMutex_);
    std::swap(p, executors_[handle]);
  }

  // It turns out someone else may already be running this configuration in
  // some unexpected cases: there is no guarantee of no-redundancy in
  // compilation options. In that case, we swapped 2 nullptrs and we just
  // exit.
  Duration res(Duration::max());
  if (p) {
    if (pruningFunction(static_cast<CudaTcExecutor*>(p.get()))) {
      return Duration::max();
    }
    CHECK(p->hasRuntimeCompiledFunction());
    try {
      // Must catch and swap to avoid exception in destructor!
      res = p->run(inputs, outputs, profile);
    } catch (std::exception& e) {
      std::lock_guard<std::mutex> lg(tcExecutorMutex_);
      std::swap(p, executors_[handle]);
      throw;
    }
    {
      std::lock_guard<std::mutex> lg(tcExecutorMutex_);
      std::swap(p, executors_[handle]);
    }
  }
  return res;
}

// Steal ExecutorInfo and give it back under lock
// Run outside of lock on owning ExecutorInfo.
void CudaExecutionEngine::uncheckedRun(
    size_t handle,
    const std::vector<const void*>& inputs,
    const std::vector<void*>& outputs) {
  std::unique_ptr<TcExecutor> p(nullptr);
  {
    std::lock_guard<std::mutex> lg(tcExecutorMutex_);
    std::swap(p, executors_[handle]);
  }

  // It turns out someone else may already be running this configuration in
  // some unexpected cases: there is no guarantee of no-redundancy in
  // compilation options. In that case, we swapped 2 nullptrs and we just
  // exit.
  if (p) {
    CHECK(p->hasRuntimeCompiledFunction());
    try {
      // Must catch and swap to avoid exception in destructor!
      p->uncheckedRun(inputs, outputs);
    } catch (std::exception& e) {
      std::lock_guard<std::mutex> lg(tcExecutorMutex_);
      std::swap(p, executors_[handle]);
      throw;
    }
    {
      std::lock_guard<std::mutex> lg(tcExecutorMutex_);
      std::swap(p, executors_[handle]);
    }
  }
}

// Steal ExecutorInfo, clear the underlying RTC object and give it back under
// lock.
void CudaExecutionEngine::clear(size_t handle) {
  std::lock_guard<std::mutex> lg(tcExecutorMutex_);
  auto executor = static_cast<CudaTcExecutor*>(executors_[handle].get());
  executor->clearRuntimeCompiledFunction();
  executors_[handle] = std::unique_ptr<TcExecutor>(nullptr);
}

size_t CudaExecutionEngine::compile(
    const std::string& name,
    const std::vector<const DLTensor*>& inputs,
    const MappingOptions& options) {
  // Check if we already have a handle for this name+size+options combination.
  // If so, return it.
  size_t handle = getHandle(name, inputs, options.toProtobufSerializedString());
  if (handle != TcExecutor::InvalidHandle) {
    return handle;
  }

  // Otherwise we need to compile.
  std::unique_ptr<CudaTcExecutor> p(new CudaTcExecutor(
      name, inputs, options, tcNameMap_.at(name), TcExecutor::InvalidHandle));
  CHECK(p);
  p->compile(options);
  CHECK(p->hasRuntimeCompiledFunction());

  handle = emplaceExecutor(std::move(p));
  return handle;
}

} // namespace tc
