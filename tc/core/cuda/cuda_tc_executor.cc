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
#include "tc/core/cuda/cuda_tc_executor.h"

#include "tc/core/cuda/cuda_compilation_cache.h"
#include "tc/core/cuda/cuda_mapping_options_cpp_printer.h"
#include "tc/core/halide_utils.h"
#include "tc/core/polyhedral/cuda/mapped_scop.h"
#include "tc/core/tc2halide.h"
#include "tc/core/utils/dlpack.h"

#include "tc/lang/parser.h"
#include "tc/lang/sema.h"

#include <version.h>
#include <utility>

namespace tc {

using namespace dlutils;

namespace {

std::string appendOptionsAndGitHash(
    const std::string& source,
    const CudaMappingOptions& options) {
  std::stringstream ss;
  ss << source << "\n/*\nMapping Options:\n"
     << CudaMappingOptionsAsCpp(options) << "TC version: " << git_version
     << "\n*/\n";
  return ss.str();
}

} // namespace

void CudaTcExecutor::compile(const tc::CudaMappingOptions& options) {
  if (rtcFun) {
    throw std::runtime_error{
        "CudaTcExecutor::compile cannot be called multiple tines."};
  }
  executionInfo_.options = options.toProtobufSerializedString();

  auto cachedOp = [&]() -> std::unique_ptr<CudaCacheRetrievalResult> {
    if (ManualCudaCache::cacheEnabled()) {
      auto rr = ManualCudaCache::getCache()->retrieveKernel(
          cacheKeyId_,
          extractRawPtrs(executionInfo_.inputsInfo),
          extractRawPtrs(executionInfo_.outputsInfo));
      if (rr) {
        return rr;
      }
    }

    if (not CudaCache::cacheEnabled()) {
      return nullptr;
    }
    CHECK_NE(executionInfo_.options, "")
        << "options string is empty, are you trying compile "
        << "a dummy CudaTcExecutor?";
    return CudaCache::getCache()->retrieveKernel(
        cacheKeyId_,
        options,
        extractRawPtrs(executionInfo_.inputsInfo),
        extractRawPtrs(executionInfo_.outputsInfo));
  }();

  if (cachedOp) {
    cudaSource = cachedOp->source;
    grid = cachedOp->grid;
    block = cachedOp->block;
    executionInfo_.kernelParams = cachedOp->parameters;
    kernelSpecializedName = cachedOp->specializedName;
    LOG_IF(INFO, FLAGS_debug_tc_mapper) << "generatedCuda: " << cudaSource;
    LOG_IF(INFO, FLAGS_debug_tc_mapper) << "retrieved grid: " << grid;
    LOG_IF(INFO, FLAGS_debug_tc_mapper) << "retrieved block: " << block;
  } else {
    compileWithTcMapper();
    cudaSource = appendOptionsAndGitHash(cudaSource, options);
    if (CudaCache::cacheEnabled()) {
      LOG_IF(INFO, FLAGS_debug_tc_mapper) << "original grid: " << grid;
      LOG_IF(INFO, FLAGS_debug_tc_mapper) << "original block: " << block;
      CudaCache::getCache()->cacheKernel(CudaCachedEntry(
          cacheKeyId_,
          kernelSpecializedName,
          executionInfo_.kernelParams,
          grid,
          block,
          options,
          extractRawPtrs(executionInfo_.inputsInfo),
          extractRawPtrs(executionInfo_.outputsInfo),
          cudaSource,
          CudaGPUInfo::GPUInfo().GetCudaDeviceStr()));
    }
  }

  rtcFun = nullptr; // force unloading in case we
  // NVRTC the same name / input with different options.
  auto t0 = std::chrono::high_resolution_clock::now();
  rtcFun = CudaRTCFunction::Compile(kernelSpecializedName, cudaSource);
  auto t1 = std::chrono::high_resolution_clock::now();
  LOG_IF(INFO, FLAGS_debug_tc_mapper)
      << "[COMPILE] Compiling with nvrtc took: "
      << std::chrono::duration_cast<std::chrono::milliseconds>(t1 - t0).count()
      << "ms" << std::endl;
}

namespace {

// Append ordered values to the kernel name, separated by "_".
std::string specializeKernelName(
    const std::string& kernelName,
    std::vector<int> params) {
  std::stringstream ss;
  ss << kernelName;
  for (auto i : params) {
    ss << "_" << i;
  }
  return ss.str();
}

std::vector<int> narrowParamsVector(const std::vector<long>& params) {
  std::vector<int> result;
  result.reserve(params.size());
  for (auto l : params) {
    CHECK_GE(l, std::numeric_limits<int>::min()) << "parameter underflows int";
    CHECK_LE(l, std::numeric_limits<int>::max()) << "parameter overflows int";
    result.push_back(static_cast<int>(l));
  }
  return result;
}
} // namespace

void CudaTcExecutor::compileWithTcMapper() {
  // A bit chicken-and-eggy, need scop from TC to have the space to build the
  // context to specialize the scop..
  auto scopTmp = polyhedral::Scop::makeScop(
      isl::with_exceptions::globalIslCtx(), halideComponents_);
  auto globalParameterContext =
      scopTmp->makeContextFromInputs(extractRawPtrs(executionInfo_.inputsInfo));
  scopTmp = polyhedral::Scop::makeSpecializedScop(
      *scopTmp,
      globalParameterContext.intersect(scopTmp->globalParameterContext));
  LOG_IF(INFO, FLAGS_debug_tc_mapper)
      << CudaMappingOptions(executionInfo_.options);
  LOG_IF(INFO, FLAGS_debug_tc_mapper) << *(scopTmp->scheduleRoot());

  // Now we can build stuff
  auto mappedScop =
      polyhedral::MappedScop::makeWithOuterBlockInnerThreadStrategy(
          std::move(scopTmp), CudaMappingOptions(executionInfo_.options));
  LOG_IF(INFO, FLAGS_debug_tc_mapper) << "Mapped schedule:" << std::endl
                                      << *(mappedScop->schedule());

  executionInfo_.kernelParams = narrowParamsVector(
      mappedScop->scop().getParameterValues(globalParameterContext));
  kernelSpecializedName = specializeKernelName(
      executionInfo_.kernelName, executionInfo_.kernelParams);

  // This updates the launch bounds with the actual result from compilation
  // with tightening of launch_bounds.
  // What you get is not what you asked for, the autotuner should adapt to
  // that.
  std::tie(cudaSource, grid, block) =
      mappedScop->codegen(kernelSpecializedName);
  LOG_IF(INFO, FLAGS_dump_cuda) << "generatedCuda: " << cudaSource;
}

void CudaTcExecutor::preRunChecks(
    const std::vector<const DLTensor*>& inputs,
    const std::vector<DLTensor*>& outputs) const {
  CHECK(rtcFun) << "Can't launch uncompiled: " << executionInfo_.kernelName;
  CHECK_NE(executionInfo_.options, "");
  checkSizesAndStridesAreCompliant(
      inputs, executionInfo_.inputsInfo, halideComponents_.getDef().params());
  checkSizesAndStridesAreCompliant(
      outputs,
      executionInfo_.outputsInfo,
      halideComponents_.getDef().returns());

  CHECK_NE(grid.view[0], 0) << "Grid dims are not set up";
  CHECK_NE(block.view[0], 0) << "Block dims are not set up";
}

std::pair<std::vector<const void*>, std::vector<void*>>
CudaTcExecutor::prepareCudaArgs(
    const std::vector<const DLTensor*>& inputs,
    const std::vector<DLTensor*>& outputs) const {
  std::vector<const void*> I;
  std::vector<void*> O;
  for (size_t i = 0; i < inputs.size(); ++i) {
    I.push_back(inputs[i]->data);
  }
  for (size_t i = 0; i < outputs.size(); ++i) {
    O.push_back(outputs[i]->data);
  }
  return std::make_pair(std::move(I), std::move(O));
}

Duration CudaTcExecutor::run(
    const std::vector<const DLTensor*>& inputs,
    const std::vector<DLTensor*>& outputs,
    bool profile) const {
  preRunChecks(inputs, outputs);
  auto IO = prepareCudaArgs(inputs, outputs);
  cudaStream_t stream = 0;
  CHECK_NE(grid.view[0], 0u) << "Grid dims are not set up";
  CHECK_NE(block.view[0], 0u) << "Block dims are not set up";
  auto res = rtcFun->Launch(
      grid.view.extractDefaultedArray(),
      block.view.extractDefaultedArray(),
      0,
      stream,
      executionInfo_.kernelParams,
      IO.second,
      IO.first,
      profile);

  if (profile and OptionsCache::cacheEnabled()) {
    OptionsCache::getCache()->recordRuntime(
        cacheKeyId_,
        CudaMappingOptions(executionInfo_.options),
        inputs,
        constPtrs(outputs),
        res);
  }
  return res;
}

CudaProfilingInfo CudaTcExecutor::profile(
    const std::vector<const DLTensor*>& inputs,
    const std::vector<DLTensor*>& outputs) const {
  preRunChecks(inputs, outputs);
  auto IO = prepareCudaArgs(inputs, outputs);
  cudaStream_t stream = 0;
  auto res = rtcFun->Profile(
      grid.view.extractDefaultedArray(),
      block.view.extractDefaultedArray(),
      0,
      stream,
      executionInfo_.kernelParams,
      IO.second,
      IO.first);
  if (OptionsCache::cacheEnabled()) {
    OptionsCache::getCache()->recordProfilingInfo(
        cacheKeyId_,
        CudaMappingOptions(executionInfo_.options),
        inputs,
        constPtrs(outputs),
        res);
  }
  return res;
}

void CudaTcExecutor::uncheckedRun(
    const std::vector<const void*>& inputs,
    const std::vector<void*>& outputs) const {
  cudaStream_t stream = 0;
  CHECK_NE(grid.view[0], 0u) << "Grid dims are not set up";
  CHECK_NE(block.view[0], 0u) << "Block dims are not set up";
  bool profile = false;
  rtcFun->Launch(
      grid.view.extractDefaultedArray(),
      block.view.extractDefaultedArray(),
      0,
      stream,
      executionInfo_.kernelParams,
      outputs,
      inputs,
      profile);
}

} // namespace tc
