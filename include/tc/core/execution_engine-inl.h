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

#include "tc/core/polyhedral/cuda/cuda_mapping_types.h"
#include "tc/core/utils/memory.h"

#include "tc/lang/parser.h"

namespace tc {

namespace {
const size_t InvalidHandle = std::numeric_limits<size_t>::max();

std::vector<lang::TreeRef> parseDefs(const std::string& language) {
  lang::Parser parser(language);
  std::vector<lang::TreeRef> res;
  while (parser.L.cur().kind != lang::TK_EOF) {
    res.push_back(parser.parseFunction());
  }
  return res;
}
} // namespace

// Under object lock, fill parse the language and fill the underlying map
template <typename ExecutorType>
void ExecutionEngine<ExecutorType>::define(const std::string& language) {
  define(parseDefs(language));
}

// support define if we pass the parsed TreeRefs.
template <typename ExecutorType>
void ExecutionEngine<ExecutorType>::define(
    const std::vector<lang::TreeRef>& treeRefs) {
  std::lock_guard<std::mutex> lg(tcExecutorMutex_);
  for (auto& ref : treeRefs) {
    auto name = lang::Def(ref).name().name();
    tcNameMap_.emplace(std::make_pair(name, ref));
  }
}

// Under object lock, retrieve the TreeRef at name and infer the output
// tensors informations
template <typename ExecutorType>
std::vector<const DLTensor*>
ExecutionEngine<ExecutorType>::inferOutputTensorInfo(
    const std::string& name,
    const std::vector<const DLTensor*>& inputs) {
  {
    std::lock_guard<std::mutex> lg(tcExecutorMutex_);
    CHECK_EQ(1, tcNameMap_.count(name))
        << "attempting to access undefined function " << name;
    // If we have already compiled for the given inputs, regardless of
    // the options, we can get sizes from a corresponding ExecutorType.
    auto e = std::find_if(
        executors_.begin(),
        executors_.end(),
        [&](const std::unique_ptr<ExecutorType>& e) {
          return e && name == e->identifier &&
              compareDLTensorVectorMetadata(
                     extractRawPtrs(e->inputsInfo), inputs);
        });
    if (e != executors_.end()) {
      return (*e)->inferOutputTensorInfo();
    }
  }

  // Otherwise, create a new executor and add it to executor_ with
  // null options. It will be used for further size queries but
  // will fail if somebody attempts to run it.
  auto executor =
      tc::make_unique<ExecutorType>(name, inputs, "", tcNameMap_.at(name));
  auto outputsInfo = executor->inferOutputTensorInfo();
  emplaceExecutor(std::move(executor));
  return outputsInfo;
}

template <typename ExecutorType>
size_t ExecutionEngine<ExecutorType>::compile(
    const std::string& name,
    const std::vector<const DLTensor*>& inputs,
    const std::string& options) {
  // Check if we already have a handle for this name+size+options combination.
  // If so, return it.
  size_t handle = getHandle(name, inputs, options);
  if (handle != InvalidHandle) {
    return handle;
  }

  // Otherwise we need to compile.
  std::unique_ptr<ExecutorType> executorUPtr(
      new ExecutorType(name, inputs, options, tcNameMap_.at(name)));
  CHECK(executorUPtr);
  executorUPtr->compile(options);
  CHECK(executorUPtr->hasRuntimeCompiledFunction());

  handle = emplaceExecutor(std::move(executorUPtr));
  return handle;
}

// Steal the executor instance and give it back under lock.
// Run outside of lock on owning ExecutorType.
template <typename ExecutorType>
Duration ExecutionEngine<ExecutorType>::run(
    size_t handle,
    const std::vector<const DLTensor*>& inputs,
    const std::vector<DLTensor*>& outputs,
    bool profile,
    std::function<bool(const ExecutorType*)> pruningFunction) {
  std::unique_ptr<ExecutorType> executorUPtr(nullptr);
  {
    std::lock_guard<std::mutex> lg(tcExecutorMutex_);
    std::swap(executorUPtr, executors_[handle]);
  }

  // It turns out someone else may already be running this configuration in
  // some unexpected cases: there is no guarantee of no-redundancy in
  // compilation options. In that case, we swapped 2 nullptrs and we just
  // exit.
  Duration res(Duration::max());
  if (executorUPtr) {
    if (pruningFunction(static_cast<ExecutorType*>(executorUPtr.get()))) {
      return Duration::max();
    }
    CHECK(executorUPtr->hasRuntimeCompiledFunction());
    try {
      // Must catch and swap to avoid exception in destructor!
      res = executorUPtr->run(inputs, outputs, profile);
    } catch (std::exception& e) {
      std::lock_guard<std::mutex> lg(tcExecutorMutex_);
      std::swap(executorUPtr, executors_[handle]);
      throw;
    }
    {
      std::lock_guard<std::mutex> lg(tcExecutorMutex_);
      std::swap(executorUPtr, executors_[handle]);
    }
  }
  return res;
}

// Steal ExecutorType and give it back under lock
// Run outside of lock on owning ExecutorType.
template <typename ExecutorType>
void ExecutionEngine<ExecutorType>::uncheckedRun(
    size_t handle,
    const std::vector<const void*>& inputs,
    const std::vector<void*>& outputs) {
  std::unique_ptr<ExecutorType> executorUPtr(nullptr);
  {
    std::lock_guard<std::mutex> lg(tcExecutorMutex_);
    std::swap(executorUPtr, executors_[handle]);
  }

  // It turns out someone else may already be running this configuration in
  // some unexpected cases: there is no guarantee of no-redundancy in
  // compilation options. In that case, we swapped 2 nullptrs and we just
  // exit.
  if (executorUPtr) {
    CHECK(executorUPtr->hasRuntimeCompiledFunction());
    try {
      // Must catch and swap to avoid exception in destructor!
      executorUPtr->uncheckedRun(inputs, outputs);
    } catch (std::exception& e) {
      std::lock_guard<std::mutex> lg(tcExecutorMutex_);
      std::swap(executorUPtr, executors_[handle]);
      throw;
    }
    {
      std::lock_guard<std::mutex> lg(tcExecutorMutex_);
      std::swap(executorUPtr, executors_[handle]);
    }
  }
}

// Clear the underlying RTC object and executor under lock.
template <typename ExecutorType>
void ExecutionEngine<ExecutorType>::clear(size_t handle) {
  std::lock_guard<std::mutex> lg(tcExecutorMutex_);
  executors_[handle]->clearRuntimeCompiledFunction();
  executors_[handle] = std::unique_ptr<ExecutorType>(nullptr);
}

template <typename ExecutorType>
size_t ExecutionEngine<ExecutorType>::emplaceExecutor(
    std::unique_ptr<ExecutorType> executorUPtr) {
  // Insert in vector under lock
  std::lock_guard<std::mutex> lg(tcExecutorMutex_);
  size_t handle = uidCounter++;
  // This may trigger reallocs and moves of the underlying vector, fun!
  executors_.emplace_back(std::move(executorUPtr));
  // This is really the invariant we enforce
  CHECK_EQ(executors_.size(), uidCounter);
  return handle;
}

template <typename ExecutorType>
size_t ExecutionEngine<ExecutorType>::getHandle(
    const std::string& name,
    const std::vector<const DLTensor*>& inputsInfo,
    const std::string& optionsStr) {
  std::lock_guard<std::mutex> lg(tcExecutorMutex_);
  typename ExecutorType::MappingOptionsType options(optionsStr);
  auto it = std::find_if(
      executors_.begin(),
      executors_.end(),
      [&](const std::unique_ptr<ExecutorType>& e) {
        return e && // UPtrs get stolen by run to avoid underlying vector
                    // realloc issues, guard against that
            name == e->identifier &&
            compareDLTensorVectorMetadata(
                   extractRawPtrs(e->inputsInfo), inputsInfo) &&
            e->options != "" &&
            typename ExecutorType::MappingOptionsType(e->options) == options;
      });
  if (it != executors_.end()) {
    return it - executors_.begin();
  }
  return InvalidHandle;
}
} // namespace tc
