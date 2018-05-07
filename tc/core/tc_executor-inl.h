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
          << "expected " << lang::kindToString(etype) << " for dim " << i
          << " but found " << lang::kindToString(atype);
    }
    for (int ii = 0; ii < actual->ndim; ++ii) {
      if (actual->shape[ii] != expected.shape[ii]) {
        throw lang::ErrorReport(dbg)
            << "expected size " << expected.shape[ii] << " for dim " << ii
            << " but found " << actual->shape[ii];
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
    const lang::TreeRef& tc,
    const typename Backend::MappingOptionsType& options,
    const typename Backend::CompilationResultType& compilationResult)
    : compiledSource(compilationResult.source),
      inputsInfo_(inputsInfo),
      outputsInfo_(outputsInfo),
      halideComponents_(halideComponents),
      // Parameters (currently) depend on compileWithTcMapper but are
      // not backend-specific, so we store them in the executor
      // TODO: revisit this later once we have strides and parametric kernels
      // with more legitimate uses of parameters.
      parameters_(compilationResult.parameters),
      tc_(tc),
      options_(options) {}

namespace detail {
inline std::pair<std::vector<const void*>, std::vector<void*>> prepareRun(
    const std::vector<const DLConstTensor*>& inputs,
    const std::vector<const DLTensor*>& outputs,
    const std::vector<TensorInfo>& inputsInfo,
    const std::vector<TensorInfo>& outputsInfo,
    const tc2halide::HalideComponents& halideComponents) {
  std::vector<const void*> rawInputs;
  std::vector<void*> rawOutputs;
  checkSizesAndStridesAreCompliant(
      inputs, inputsInfo, halideComponents.getDef().params());
  checkSizesAndStridesAreCompliant(
      outputs, outputsInfo, halideComponents.getDef().returns());
  for (size_t i = 0; i < inputs.size(); ++i) {
    rawInputs.push_back(inputs[i]->data);
  }
  for (size_t i = 0; i < outputs.size(); ++i) {
    rawOutputs.push_back(outputs[i]->data);
  }
  return std::make_pair(rawInputs, rawOutputs);
}
} // namespace detail

template <typename Backend>
void TcExecutor<Backend>::run(
    const std::vector<const DLConstTensor*>& inputs,
    const std::vector<const DLTensor*>& outputs) const {
  std::vector<const void*> rawInputs;
  std::vector<void*> rawOutputs;
  std::tie(rawInputs, rawOutputs) = detail::prepareRun(
      inputs, outputs, inputsInfo_, outputsInfo_, halideComponents_);

  // Static dispatch instead of virtual functions requires this cast.
  static_cast<const typename Backend::ExecutorType&>(*this).uncheckedRun(
      rawInputs, rawOutputs);
}

template <typename Backend>
ProfilingInfo TcExecutor<Backend>::profile(
    const std::vector<const DLConstTensor*>& inputs,
    const std::vector<const DLTensor*>& outputs) const {
  auto start = std::chrono::system_clock::now();
  std::vector<const void*> rawInputs;
  std::vector<void*> rawOutputs;
  std::tie(rawInputs, rawOutputs) = detail::prepareRun(
      inputs, outputs, inputsInfo_, outputsInfo_, halideComponents_);

  // Launch kernel and get **the kernel** time (without CPU overhead)
  ProfilingInfo pi(
      // Static dispatch instead of virtual functions requires this cast.
      static_cast<const typename Backend::ExecutorType&>(*this)
          .profileUnchecked(rawInputs, rawOutputs));

  // The total CPU overhead is the total time minus the (synchronized) kernel
  // runtime
  Duration cpuOverhead(Duration::since(start));
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

template <typename Backend>
const std::vector<TensorInfo>& TcExecutor<Backend>::inputsInfo() const {
  return inputsInfo_;
}

template <typename Backend>
const std::vector<TensorInfo>& TcExecutor<Backend>::outputsInfo() const {
  return outputsInfo_;
}

template <typename Backend>
std::string TcExecutor<Backend>::deviceName() const {
  return rtcFun_->deviceName();
}

template <typename Backend>
const lang::TreeRef& TcExecutor<Backend>::tc() const {
  return tc_;
}

template <typename Backend>
const typename Backend::MappingOptionsType& TcExecutor<Backend>::options()
    const {
  return options_;
}
} // namespace tc
