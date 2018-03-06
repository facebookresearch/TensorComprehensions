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
#include "tc/core/execution_engine.h"
#include "tc/core/utils/memory.h"

#include "tc/lang/parser.h"

namespace tc {

// Under object lock, fill parse the language and fill the underlying map
void ExecutionEngine::define(const std::string& language) {
  lang::Parser parser(language);
  std::lock_guard<std::mutex> lg(tcExecutorMutex_);
  while (parser.L.cur().kind != lang::TK_EOF) {
    auto treeRef = parser.parseFunction();
    auto name = lang::Def(treeRef).name().name();
    tcNameMap_.emplace(std::make_pair(name, treeRef));
  }
}

// support define if we pass the parsed TreeRefs.
void ExecutionEngine::define(const std::vector<lang::TreeRef>& treeRefs) {
  std::lock_guard<std::mutex> lg(tcExecutorMutex_);
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
    std::lock_guard<std::mutex> lg(tcExecutorMutex_);
    CHECK_EQ(1, tcNameMap_.count(name))
        << "attempting to access undefined function " << name;
    // If we have already compiled for the given inputs, regardless of
    // the options, we can get sizes from a corresponding TcExecutor.
    auto e = std::find_if(
        executors_.begin(),
        executors_.end(),
        [&](const std::unique_ptr<TcExecutor>& e) {
          return e && name == e->identifier &&
              compareDLTensorVectorMetadata(
                     extractRawPtrs(e->inputsInfo), inputs);
        });
    if (e != executors_.end()) {
      return (*e)->inferOutputTensorInfo();
    }
  }

  // Otherwise, create a new executor and add it to executor_ with
  // null options.  It will be used for further size queries but
  // will fail if somebody attempts to run it.
  auto executor = tc::make_unique<TcExecutor>(
      name, inputs, "", tcNameMap_.at(name), TcExecutor::InvalidHandle);
  auto outputsInfo = executor->inferOutputTensorInfo();
  emplaceExecutor(std::move(executor));
  return outputsInfo;
}

size_t ExecutionEngine::emplaceExecutor(std::unique_ptr<TcExecutor> p) {
  // Insert in vector under lock
  std::lock_guard<std::mutex> lg(tcExecutorMutex_);
  size_t handle = uidCounter++;
  // This may trigger reallocs and moves of the underlying vector, fun!
  executors_.emplace_back(std::move(p));
  // This is really the invariant we enforce
  CHECK_EQ(executors_.size(), uidCounter);
  return handle;
}

size_t ExecutionEngine::getHandle(
    const std::string& name,
    const std::vector<const DLTensor*>& inputsInfo,
    const std::string& optionsStr) {
  std::lock_guard<std::mutex> lg(tcExecutorMutex_);
  MappingOptions options(optionsStr);
  auto it = std::find_if(
      executors_.begin(),
      executors_.end(),
      [&](const std::unique_ptr<TcExecutor>& e) {
        return e && // UPtrs get stolen by run to avoid underlying vector
                    // realloc issues, guard against that
            name == e->identifier &&
            compareDLTensorVectorMetadata(
                   extractRawPtrs(e->inputsInfo), inputsInfo) &&
            e->options != "" && MappingOptions(e->options) == options;
      });
  if (it != executors_.end()) {
    return it - executors_.begin();
  }
  return TcExecutor::InvalidHandle;
}

}; // namespace tc
