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

// Under object lock, fill parse the language and fill the underlying map
void ExecutionEngine::define(const std::string& language) {
  lang::Parser parser(language);
  std::lock_guard<std::mutex> lg(executorInfoMutex);
  while (parser.L.cur().kind != lang::TK_EOF) {
    auto treeRef = parser.parseFunction();
    auto name = lang::Def(treeRef).name().name();
    tcNameMap_.emplace(std::make_pair(name, treeRef));
  }
}

// support define if we pass the parsed TreeRefs.
void ExecutionEngine::define(const std::vector<lang::TreeRef>& treeRefs) {
  std::lock_guard<std::mutex> lg(executorInfoMutex);
  for (auto& ref : treeRefs) {
    auto name = lang::Def(ref).name().name();
    tcNameMap_.emplace(std::make_pair(name, ref));
  }
}

// Under object lock, retrieve the TreeRef at name and infer the output
// tensors informations
std::vector<const DLTensor*> ExecutionEngine::inferOutputTensorInfo(
    const std::string& name,
    const std::vector<const DLTensor*>& inputs) {
  {
    std::lock_guard<std::mutex> lg(executorInfoMutex);
    CHECK_EQ(1, tcNameMap_.count(name))
        << "attempting to access undefined function " << name;
    // If we have already compiled for the given inputs, regardless of
    // the options, we can get sizes from a corresponding TcExecutor.
    auto ei = std::find_if(
        executors_.begin(),
        executors_.end(),
        [&](const std::unique_ptr<ExecutorInfo>& ei) {
          return ei && name == ei->identifier &&
              compareDLTensorVectorMetadata(
                     extractRawPtrs(ei->inputsInfo), inputs);
        });
    if (ei != executors_.end()) {
      return (*ei)->exec.inferOutputTensorInfo();
    }
  }

  // Otherwise, create a new executor and add it to executor_ with
  // null options.  It will be used for further size queries but
  // will fail if somebody attempts to run it.
  auto execInfo = tc::make_unique<ExecutorInfo>(
      name,
      inputs,
      std::unique_ptr<MappingOptions>(nullptr),
      tcNameMap_.at(name),
      TcExecutor::InvalidHandle);
  auto outputsInfo = execInfo->exec.inferOutputTensorInfo();
  emplaceExecutor(std::move(execInfo));
  return outputsInfo;
}

// Steal ExecutorInfo and give it back under lock
// Run outside of lock on owning ExecutorInfo.
Duration ExecutionEngine::run(
    size_t handle,
    const std::vector<const DLTensor*>& inputs,
    const std::vector<DLTensor*>& outputs,
    bool profile,
    std::function<bool(const ExecutorInfo*)> pruningFunction) {
  std::unique_ptr<ExecutorInfo> p(nullptr);
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
    if (pruningFunction(p.get())) {
      return Duration::max();
    }
    CHECK(p->exec.hasRTCFun());
    try {
      // Must catch and swap to avoid exception in destructor!
      res = p->exec.run(inputs, outputs, profile);
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
void ExecutionEngine::uncheckedRun(
    size_t handle,
    const std::vector<const void*>& inputs,
    const std::vector<void*>& outputs) {
  std::unique_ptr<ExecutorInfo> p(nullptr);
  {
    std::lock_guard<std::mutex> lg(executorInfoMutex);
    std::swap(p, executors_[handle]);
  }

  // It turns out someone else may already be running this configuration in
  // some unexpected cases: there is no guarantee of no-redundancy in
  // compilation options. In that case, we swapped 2 nullptrs and we just
  // exit.
  if (p.get()) {
    CHECK(p->exec.hasRTCFun());
    try {
      // Must catch and swap to avoid exception in destructor!
      p->exec.uncheckedRun(inputs, outputs);
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
void ExecutionEngine::clear(size_t handle) {
  std::lock_guard<std::mutex> lg(executorInfoMutex);
  executors_[handle]->clear();
  executors_[handle] = std::unique_ptr<ExecutorInfo>(nullptr);
}

size_t ExecutionEngine::getHandle(
    const std::string& name,
    const std::vector<const DLTensor*>& inputsInfo,
    const MappingOptions& options) {
  std::lock_guard<std::mutex> lg(executorInfoMutex);
  auto ei = std::find_if(
      executors_.begin(),
      executors_.end(),
      [&](const std::unique_ptr<ExecutorInfo>& ei) {
        return ei && // UPtrs get stolen by run to avoid underlying vector
                     // realloc issues, guard against that
            name == ei->identifier &&
            compareDLTensorVectorMetadata(
                   extractRawPtrs(ei->inputsInfo), inputsInfo) &&
            ei->options && *ei->options == options;
      });
  if (ei != executors_.end()) {
    return (*ei)->objectLocalHandle;
  }
  return TcExecutor::InvalidHandle;
}

std::unique_ptr<ExecutionEngine::ExecutorInfo>
ExecutionEngine::makeExecutorInfo(
    const std::string& name,
    const std::vector<const DLTensor*>& inputsInfo,
    const MappingOptions& options) {
  CHECK_EQ(tcNameMap_.count(name), 1)
      << "TC function " << name << " not defined";
  return tc::make_unique<ExecutorInfo>(
      name,
      inputsInfo,
      std::unique_ptr<MappingOptions>(new MappingOptions(options)),
      tcNameMap_.at(name),
      TcExecutor::InvalidHandle);
}

size_t ExecutionEngine::emplaceExecutor(std::unique_ptr<ExecutorInfo> p) {
  // Insert in vector under lock
  std::lock_guard<std::mutex> lg(executorInfoMutex);
  size_t handle = uidCounter++;
  p->objectLocalHandle = handle;
  // This may trigger reallocs and moves of the underlying vector, fun!
  executors_.emplace_back(std::move(p));
  // This is really the invariant we enforce
  CHECK_EQ(executors_.size(), uidCounter);
  return handle;
}

size_t ExecutionEngine::compile(
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
  auto p = makeExecutorInfo(name, inputs, options);
  p->exec.compile(options);
  CHECK(p->exec.hasRTCFun());

  handle = emplaceExecutor(std::move(p));
  return handle;
}

} // namespace tc
