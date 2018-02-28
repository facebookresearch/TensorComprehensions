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
    std::function<bool(const CudaExecutorInfo*)> pruningFunction) {
  std::unique_ptr<ExecutionEngine::ExecutorInfo> p(nullptr);
  {
    std::lock_guard<std::mutex> lg(executorInfoMutex);
    std::swap(p, executors_[handle]);
  }

  // It turns out someone else may already be running this configuration in
  // some unexpected cases: there is no guarantee of no-redundancy in
  // compilation options. In that case, we swapped 2 nullptrs and we just
  // exit.
  Duration res(Duration::max());
  if (p.get()) {
    auto& execInfo = static_cast<CudaExecutorInfo&>(*p);
    auto exec = static_cast<CudaTcExecutor*>(execInfo.exec.get());
    if (pruningFunction(&execInfo)) {
      return Duration::max();
    }
    CHECK(exec->hasRTCFun());
    try {
      // Must catch and swap to avoid exception in destructor!
      res = exec->run(inputs, outputs, profile);
    } catch (std::exception& e) {
      std::lock_guard<std::mutex> lg(executorInfoMutex);
      std::swap(p, executors_[handle]);
      throw;
    }
    {
      std::lock_guard<std::mutex> lg(executorInfoMutex);
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
  std::unique_ptr<ExecutionEngine::ExecutorInfo> p(nullptr);
  {
    std::lock_guard<std::mutex> lg(executorInfoMutex);
    std::swap(p, executors_[handle]);
  }

  // It turns out someone else may already be running this configuration in
  // some unexpected cases: there is no guarantee of no-redundancy in
  // compilation options. In that case, we swapped 2 nullptrs and we just
  // exit.
  if (p.get()) {
    auto exec = static_cast<CudaTcExecutor*>(p->exec.get());
    CHECK(exec->hasRTCFun());
    try {
      // Must catch and swap to avoid exception in destructor!
      exec->uncheckedRun(inputs, outputs);
    } catch (std::exception& e) {
      std::lock_guard<std::mutex> lg(executorInfoMutex);
      std::swap(p, executors_[handle]);
      throw;
    }
    {
      std::lock_guard<std::mutex> lg(executorInfoMutex);
      std::swap(p, executors_[handle]);
    }
  }
}

// Steal ExecutorInfo, clear the underlying RTC object and give it back under
// lock.
void CudaExecutionEngine::clear(size_t handle) {
  std::lock_guard<std::mutex> lg(executorInfoMutex);
  auto executor = static_cast<CudaExecutorInfo*>(executors_[handle].get());
  executor->clear();
  executors_[handle] = std::unique_ptr<ExecutionEngine::ExecutorInfo>(nullptr);
}

size_t CudaExecutionEngine::getHandle(
    const std::string& name,
    const std::vector<const DLTensor*>& inputsInfo,
    const MappingOptions& options) {
  std::lock_guard<std::mutex> lg(executorInfoMutex);
  auto ei = std::find_if(
      executors_.begin(),
      executors_.end(),
      [&](const std::unique_ptr<ExecutionEngine::ExecutorInfo>& ei) {
        return ei && // UPtrs get stolen by run to avoid underlying vector
                     // realloc issues, guard against that
            name == ei->identifier &&
            compareDLTensorVectorMetadata(
                   extractRawPtrs(ei->inputsInfo), inputsInfo) &&
            ei->options != "" && MappingOptions(ei->options) == options;
      });
  if (ei != executors_.end()) {
    return (*ei)->objectLocalHandle;
  }
  return TcExecutor::InvalidHandle;
}

std::unique_ptr<ExecutionEngine::ExecutorInfo>
CudaExecutionEngine::makeExecutorInfo(
    const std::string& name,
    const std::vector<const DLTensor*>& inputsInfo,
    const MappingOptions& options) {
  CHECK_EQ(tcNameMap_.count(name), 1)
      << "TC function " << name << " not defined";
  return std::unique_ptr<ExecutionEngine::ExecutorInfo>(new CudaExecutorInfo(
      name,
      inputsInfo,
      options,
      tcNameMap_.at(name),
      TcExecutor::InvalidHandle));
}

size_t CudaExecutionEngine::compile(
    const std::string& name,
    const std::vector<const DLTensor*>& inputs,
    const MappingOptions& options) {
  // Check if we already have a handle for this name+size+options combination.
  // If so, return it.
  size_t handle = getHandle(name, inputs, options);
  if (handle != TcExecutor::InvalidHandle) {
    return handle;
  }

  // Otherwise we need to compile.
  std::unique_ptr<ExecutionEngine::ExecutorInfo> p =
      makeExecutorInfo(name, inputs, options);
  auto exec = static_cast<CudaTcExecutor*>(p->exec.get());
  CHECK(exec);
  exec->compile(options);
  CHECK(exec->hasRTCFun());

  handle = emplaceExecutor(std::move(p));
  return handle;
}

} // namespace tc
