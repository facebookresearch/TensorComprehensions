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
#include "tc/core/tc_executor.h"

#include <sstream>
#include <string>

#include "tc/core/compiler.h"
#include "tc/core/flags.h"
#include "tc/core/tensor.h"
#include "tc/lang/canonicalize.h"

namespace tc {
namespace {
int toTypeToken(DLDataType dtype) {
  return lang::TypeInfo(lang::TypeInfo::Code(dtype.code), dtype.bits)
      .toScalarToken();
}

template <typename DLTensorType, typename ParamType>
void checkSizesAndStridesAreCompliant(
    const std::vector<const DLTensorType*>& actualVec,
    const std::vector<TensorInfo>& expectedVec,
    const ParamType& dbg) {
  if (actualVec.size() != expectedVec.size()) {
    throw lang::ErrorReport(dbg)
        << "expected " << expectedVec.size() << " parameters but found "
        << actualVec.size() << " parameters";
  }

  for (size_t i = 0; i < actualVec.size(); ++i) {
    auto actual = actualVec[i];
    auto expected = expectedVec[i];
    if (static_cast<size_t>(actual->ndim) != expected.shape.size()) {
      throw lang::ErrorReport(dbg) << "expected " << expected.shape.size()
                                   << " dimensions but found tensor with "
                                   << actual->ndim << " dimensions";
    }
    auto atype = toTypeToken(actual->dtype);
    auto etype = toTypeToken(expected.dtype);
    if (atype != etype) {
      throw lang::ErrorReport(dbg)
          << "expected " << lang::kindToString(etype) << " but found "
          << lang::kindToString(atype);
    }
    for (size_t i = 0; i < actual->ndim; ++i) {
      if (actual->shape[i] != expected.shape[i]) {
        throw lang::ErrorReport(dbg)
            << "expected size " << expected.shape[i] << " for dim " << i
            << " but found " << actual->shape[i];
      }
    }
  }
}
} // namespace

template <typename Backend>
TcExecutor<Backend>::TcExecutor(
    const std::vector<TensorInfo>& inputsInfo,
    const std::vector<TensorInfo>& outputsInfo,
    const tc2halide::HalideComponents& halideComponents,
    const typename Backend::CompilationResultType& compilationResult)
    : compiledSource(compilationResult.source),
      inputsInfo_(inputsInfo),
      outputsInfo_(outputsInfo),
      halideComponents_(halideComponents),
      // Parameters (currently) depend on compileWithTcMapper but are
      // not backend-specific, so we store them in the executor
      // TODO: revisit this later once we have strides and parametric kernels
      // with more legitimate uses of parameters.
      parameters_(compilationResult.parameters) {}

template <typename Backend>
void TcExecutor<Backend>::run(
    const std::vector<const DLConstTensor*>& inputs,
    const std::vector<const DLTensor*>& outputs) const {
  checkSizesAndStridesAreCompliant(
      inputs, inputsInfo_, halideComponents_.getDef().params());
  checkSizesAndStridesAreCompliant(
      outputs, outputsInfo_, halideComponents_.getDef().returns());
  std::vector<const void*> I;
  std::vector<void*> O;
  for (size_t i = 0; i < inputs.size(); ++i) {
    I.push_back(inputs[i]->data);
  }
  for (size_t i = 0; i < outputs.size(); ++i) {
    O.push_back(outputs[i]->data);
  }
  // Static dispatch instead of virtual functions requires this cast.
  static_cast<const typename Backend::ExecutorType&>(*this).uncheckedRun(I, O);
}

template <typename Backend>
ProfilingInfo TcExecutor<Backend>::profile(
    const std::vector<const DLConstTensor*>& inputs,
    const std::vector<const DLTensor*>& outputs) const {
  auto start = std::chrono::system_clock::now();
  checkSizesAndStridesAreCompliant(
      inputs, inputsInfo_, halideComponents_.getDef().params());
  checkSizesAndStridesAreCompliant(
      outputs, outputsInfo_, halideComponents_.getDef().returns());

  std::vector<const void*> I;
  std::vector<void*> O;
  for (size_t i = 0; i < inputs.size(); ++i) {
    I.push_back(inputs[i]->data);
  }
  for (size_t i = 0; i < outputs.size(); ++i) {
    O.push_back(outputs[i]->data);
  }

  // Launch kernel and get **the kernel** time (without CPU overhead)
  ProfilingInfo pi(
      // Static dispatch instead of virtual functions requires this cast.
      static_cast<const typename Backend::ExecutorType&>(*this)
          .profileUnchecked(I, O));

  // The total CPU overhead is the total time minus the (synchronized) kernel
  // runtime
  auto end = std::chrono::system_clock::now();
  Duration cpuOverhead(end - start);
  cpuOverhead = cpuOverhead - pi.kernelRuntime;
  return ProfilingInfo{cpuOverhead, pi.kernelRuntime};
}

template <typename Backend>
void TcExecutor<Backend>::clearRuntimeCompiledFunction() {
  if (!rtcFun_.get()) {
    return;
  }
  rtcFun_->clear();
  rtcFun_ = nullptr;
}
} // namespace tc
