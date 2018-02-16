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

#include <limits>

#include <dlpack/dlpack.h>

#include "tc/core/halide_utils.h"
#include "tc/core/tc2halide.h"
#include "tc/core/utils/dlpack.h"
#include "tc/core/utils/time.h"

#include "tc/lang/parser.h"

// TODO: remove me
#include "tc/core/mapping_options.h"

namespace tc {

using namespace dlutils;

class TcExecutor {
 public:
  TcExecutor(
      const std::string& TcDefinition,
      const std::vector<const DLTensor*>& inputsInfo);
  TcExecutor(
      lang::TreeRef TcDefinition,
      const std::vector<const DLTensor*>& inputsInfo);
  virtual ~TcExecutor();

  TcExecutor(TcExecutor&&) = delete;
  TcExecutor& operator=(TcExecutor&&) = delete;
  TcExecutor(const TcExecutor&) = delete;
  TcExecutor& operator=(const TcExecutor&) = delete;

  // Given a Tc and a list of input tensors that match the definition in the
  // Tc in positional order, this generates the output tensor infos issued
  // from forward inference.
  // The typical flow is to infer output sizes, allocate/resize them within
  // you favorite ML framework / tensor library and then call compile.
  std::vector<const DLTensor*> inferOutputTensorInfo();

  // Can only be called once with specific kernel options.  Input sizes are
  // set up as constructor argument and output sizes are inferred.
  //
  // If you need another kernel for another Tc or another inputs, outputs,
  // options then just instantiate another TcExecutor.
  // This is because for the time being we fully specialize all the sizes and
  // strides at runtime.
  virtual void compile(const std::string& options) {
    LOG(FATAL) << "TcExecutor::compile is abstract and should not be called";
  }

  // Run can be called multiple times given a compilation, inputs are allowed
  // to change in that their data pointer is allowed to change.
  // Sizes and strides must remain constant otherwise this is an error
  // The only thing that is allowed to change across runs is the input
  // and output pointers base address.
  // It is the caller's responsibility to ensure proper non-aliasing (or
  // advanced aliasing) properties of the input and output tensors.
  // if profile is set the kernel runtime (nanoseconds) is returned
  virtual Duration run(
      const std::vector<const DLTensor*>& inputs,
      const std::vector<DLTensor*>& outputs,
      bool profile = false) const {
    LOG(FATAL) << "TcExecutor::run is abstract and should not be called";
  }

  // This is the "low-latency" mode in which we just propagate raw pointers to
  // data in the address space where kernel is executed.
  // No tensor-related information can be checked so it is the user's
  // responsibility to ensure that shapes and strides match. If the user
  // doesn't then segfault will likely occur.
  virtual void uncheckedRun(
      const std::vector<const void*>& inputs,
      const std::vector<void*>& outputs) const {
    LOG(FATAL)
        << "TcExecutor::uncheckedRun is abstract and should not be called";
  }

  const static size_t InvalidHandle = std::numeric_limits<size_t>::max();

 protected:
  struct TcExecutionInfo {
    std::string kernelName;
    std::vector<dlutils::DLTensorUPtr> inputsInfo;
    std::vector<dlutils::DLTensorUPtr> outputsInfo;
    std::vector<int> kernelParams;
    std::string kernelSpecializedName;
    std::string options;
    std::string cudaSource;
    Grid grid{{0, 0, 0}};
    Block block{{0, 0, 0}};
    std::shared_ptr<CudaRTCFunction> rtcFun;
  };

  void checkSizesAndStridesAreCompliant(
      const DLTensor* actual,
      const DLTensor* expected,
      const lang::Param& dbg) const;

  template <typename T>
  void checkSizesAndStridesAreCompliant(
      const std::vector<T*>& dlTensors,
      const std::vector<DLTensorUPtr>& tensorInfos,
      const lang::ListView<lang::Param>& dbgInfo) const;

  void checkInputsCompliant(
      const std::vector<const DLTensor*>& inputsInfo) const;

  tc2halide::HalideComponents halideComponents_;
  TcExecutionInfo execInfo_;
  lang::TreeRef tcTree_;
  mutable isl::ctx ctx_;
};

// templating to match both const and non-const DLTensor pointers
template <typename T>
void TcExecutor::checkSizesAndStridesAreCompliant(
    const std::vector<T*>& dlTensors,
    const std::vector<DLTensorUPtr>& tensorInfos,
    const lang::ListView<lang::Param>& dbgInfo) const {
  if (tensorInfos.size() != dlTensors.size()) {
    throw lang::ErrorReport(dbgInfo)
        << "expected " << tensorInfos.size() << " values but found "
        << dlTensors.size();
  }
  for (size_t i = 0; i < tensorInfos.size(); ++i) {
    checkSizesAndStridesAreCompliant(
        dlTensors[i], tensorInfos[i].get(), dbgInfo[i]);
  }
}

} // namespace tc
