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
#include "tc/aten/aten_compiler.h"

#include <iostream>
#include <string>
#include <vector>

#include "tc/core/scope_guard.h"

namespace tc {

ATenCompilationUnit::ATenCompilationUnit() {
  executionEngine_ =
      std::unique_ptr<CudaExecutionEngine>(new CudaExecutionEngine());
}

void ATenCompilationUnit::define(const std::string& language) {
  executionEngine_->define(language);
}

void deleteDlmTensors(std::vector<DLManagedTensor*>& tensors) {
  for (auto& tensor : tensors) {
    tensor->deleter(tensor);
  }
}

namespace {

// given the tensor shape and DLType, allocate storage for the tensor output
// type.
void prepareOutputs(
    lang::TreeRef func,
    const std::vector<const DLTensor*> tensorInfo,
    const at::Backend& backend,
    std::vector<at::Tensor>& outputs) {
  // prereqs for reusing CUDA memory, just allocate the first time then resize
  // (if needed). Most of the time should do nothing
  if (outputs.size() != 0 && outputs.size() != tensorInfo.size()) {
    throw lang::ErrorReport(func) << "expected " << tensorInfo.size()
                                  << " outputs but found " << outputs.size();
  }
  for (int i = 0; i < tensorInfo.size(); ++i) {
    auto info = tensorInfo[i];
    auto stype = at::toScalarType(info->dtype);
    if (outputs.size() < tensorInfo.size()) {
      outputs.push_back(at::getType(backend, stype)
                            .tensor(at::IntList(info->shape, info->ndim)));
      // TODO: we just malloc'ed I guess we can pay a memset
      outputs.back().zero_();
    } else {
      // In-place ATen operators have a trailing _
      std::vector<int64_t> shape(info->shape, info->shape + info->ndim);
      outputs[i].resize_(shape);
      // TODO: zero on shape increase? Not clear it's needed ..
      // outputs.back().zero_();
    }
  }
}

} // namespace

std::pair<std::vector<DLTensor*>, std::vector<DLManagedTensor*>>
toDlpackTensors(const std::vector<at::Tensor>& tensors) {
  std::vector<DLTensor*> dlTensors;
  std::vector<DLManagedTensor*> dlMTensors;
  for (auto tensor : tensors) {
    auto dlMTensor = at::toDLPack(tensor);
    dlTensors.push_back(&(dlMTensor->dl_tensor));
    dlMTensors.push_back(dlMTensor);
  }
  return make_pair(dlTensors, dlMTensors);
}

std::pair<std::vector<const DLTensor*>, std::vector<DLManagedTensor*>>
toConstDlpackTensors(const std::vector<at::Tensor>& tensors) {
  std::vector<const DLTensor*> dlTensors;
  std::vector<DLManagedTensor*> dlMTensors;
  for (auto tensor : tensors) {
    auto dlMTensor = at::toDLPack(tensor);
    dlTensors.push_back(&(dlMTensor->dl_tensor));
    dlMTensors.push_back(dlMTensor);
  }
  return make_pair(dlTensors, dlMTensors);
}

size_t ATenCompilationUnit::compile(
    const std::string& name,
    const std::vector<at::Tensor>& inputs,
    const MappingOptions& options) {
  auto inputDLTensorsPair = toConstDlpackTensors(inputs);
  ScopeGuard g([&]() { deleteDlmTensors(inputDLTensorsPair.second); });
  return executionEngine_->compile(name, inputDLTensorsPair.first, options);
}

std::vector<const DLTensor*> ATenCompilationUnit::inferOutputTensorInfo(
    const std::string& name,
    const std::vector<at::Tensor>& inputs) {
  auto inputDLTensorsPair = toConstDlpackTensors(inputs);
  ScopeGuard g([&]() { deleteDlmTensors(inputDLTensorsPair.second); });
  return executionEngine_->inferOutputTensorInfo(
      name, inputDLTensorsPair.first);
}

Duration ATenCompilationUnit::run(
    const std::string& name,
    const std::vector<at::Tensor>& inputs,
    std::vector<at::Tensor>& outputs,
    size_t handle,
    bool profile) {
  at::Backend backend = inputs[0].type().backend();
  auto inputDLTensorsPair = toConstDlpackTensors(inputs);
  ScopeGuard g1([&]() { deleteDlmTensors(inputDLTensorsPair.second); });
  auto outTensorInfo =
      executionEngine_->inferOutputTensorInfo(name, inputDLTensorsPair.first);
  prepareOutputs(
      executionEngine_->treeForFunction(name), outTensorInfo, backend, outputs);
  auto outputDLTensorsPair = toDlpackTensors(outputs);
  ScopeGuard g2([&]() { deleteDlmTensors(outputDLTensorsPair.second); });
  return executionEngine_->run(
      handle, inputDLTensorsPair.first, outputDLTensorsPair.first, profile);
}

void ATenCompilationUnit::uncheckedRun(
    const std::vector<at::Tensor>& inputs,
    std::vector<at::Tensor>& outputs,
    size_t handle) {
  CHECK_LT(0, outputs.size());

  constexpr auto kReservedSize = 8;
  std::vector<const void*> I(kReservedSize, nullptr);
  std::vector<void*> O(kReservedSize, nullptr);
  int i;
  for (i = 0; i < inputs.size(); ++i) {
    if (i < kReservedSize) {
      I[i] = inputs[i].data_ptr();
    } else {
      I.push_back(inputs[i].data_ptr());
    }
  }
  I.resize(i);
  for (i = 0; i < outputs.size(); ++i) {
    if (i < kReservedSize) {
      O[i] = outputs[i].data_ptr();
    } else {
      O.push_back(outputs[i].data_ptr());
    }
  }
  O.resize(i);

  executionEngine_->uncheckedRun(handle, I, O);
}

} // namespace tc
