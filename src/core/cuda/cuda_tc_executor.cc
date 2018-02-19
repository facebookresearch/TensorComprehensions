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

#include "tc/core/compilation_cache.h"
#include "tc/core/halide_utils.h"
#include "tc/core/mapping_options_cpp_printer.h"
#include "tc/core/polyhedral/mapped_scop.h"
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
    const MappingOptions& options) {
  std::stringstream ss;
  ss << source << "\n/*\nMapping Options:\n"
     << MappingOptionsAsCpp(options) << "TC version: " << git_version
     << "\n*/\n";
  return ss.str();
}

} // namespace

void CudaTcExecutor::compile(const tc::MappingOptions& options) {
  if (rtcFun) {
    throw std::runtime_error{
        "CudaTcExecutor::compile cannot be called multiple tines."};
  }
  execInfo_.options = options.toProtobufSerializedString();

  auto cachedOp = [&]() -> std::unique_ptr<CudaCache::RetrievalResult> {
    if (ManualCudaCache::cacheEnabled()) {
      auto rr = ManualCudaCache::getCache()->retrieveKernel(
          // TODO:replace this with pretty printed TC
          execInfo_.kernelName,
          extractRawPtrs(execInfo_.inputsInfo),
          extractRawPtrs(execInfo_.outputsInfo));
      if (rr) {
        return rr;
      }
    }

    if (not CudaCache::cacheEnabled()) {
      return nullptr;
    }
    CHECK_NE(execInfo_.options, "")
        << "options string is empty, are you trying compile "
        << "a dummy CudaTcExecutor?";
    return CudaCache::getCache()->retrieveKernel(
        execInfo_.kernelName, // TODO:replace this with pretty printed TC
        options,
        extractRawPtrs(execInfo_.inputsInfo),
        extractRawPtrs(execInfo_.outputsInfo));
  }();

  if (cachedOp) {
    cudaSource = cachedOp->source;
    grid = cachedOp->grid;
    block = cachedOp->block;
    execInfo_.kernelParams = cachedOp->parameters;
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
      CudaCache::getCache()->cacheKernel(
          execInfo_.kernelName, // TODO:replace this with pretty printed TC
          options,
          extractRawPtrs(execInfo_.inputsInfo),
          extractRawPtrs(execInfo_.outputsInfo),
          kernelSpecializedName,
          execInfo_.kernelParams,
          cudaSource,
          grid,
          block);
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
  auto scopTmp = polyhedral::Scop::makeScop(ctx_, halideComponents_);
  auto globalParameterContext =
      scopTmp->makeContextFromInputs(extractRawPtrs(execInfo_.inputsInfo));
  scopTmp = polyhedral::Scop::makeSpecializedScop(
      *scopTmp,
      globalParameterContext.intersect(scopTmp->globalParameterContext));
  LOG_IF(INFO, FLAGS_debug_tc_mapper) << MappingOptions(execInfo_.options);
  LOG_IF(INFO, FLAGS_debug_tc_mapper) << *(scopTmp->scheduleRoot());

  // Now we can build stuff
  auto mappedScop =
      polyhedral::MappedScop::makeWithOuterBlockInnerThreadStrategy(
          std::move(scopTmp), MappingOptions(execInfo_.options));
  LOG_IF(INFO, FLAGS_debug_tc_mapper) << "Mapped schedule:" << std::endl
                                      << *(mappedScop->schedule());

  execInfo_.kernelParams = narrowParamsVector(
      mappedScop->scop().getParameterValues(globalParameterContext));
  kernelSpecializedName =
      specializeKernelName(execInfo_.kernelName, execInfo_.kernelParams);

  // This updates the launch bounds with the actual result from compilation
  // with tightening of launch_bounds.
  // What you get is not what you asked for, the autotuner should adapt to
  // that.
  std::tie(cudaSource, grid, block) =
      mappedScop->codegen(kernelSpecializedName);
  LOG_IF(INFO, FLAGS_dump_cuda) << "generatedCuda: " << cudaSource;
}

Duration CudaTcExecutor::run(
    const std::vector<const DLTensor*>& inputs,
    const std::vector<DLTensor*>& outputs,
    bool profile) const {
  CHECK(rtcFun) << "Can't launch uncompiled: " << execInfo_.kernelName;
  CHECK_NE(execInfo_.options, "");
  checkSizesAndStridesAreCompliant(
      inputs, execInfo_.inputsInfo, halideComponents_.getDef().params());
  checkSizesAndStridesAreCompliant(
      outputs, execInfo_.outputsInfo, halideComponents_.getDef().returns());

  std::vector<const void*> I;
  std::vector<void*> O;
  for (int i = 0; i < inputs.size(); ++i) {
    I.push_back(inputs[i]->data);
  }
  for (int i = 0; i < outputs.size(); ++i) {
    O.push_back(outputs[i]->data);
  }
  cudaStream_t stream = 0;
  CHECK_NE(grid[0], 0) << "Grid dims are not set up";
  CHECK_NE(block[0], 0) << "Block dims are not set up";
  auto res = rtcFun->Launch(
      grid.extractDefaultedArray(),
      block.extractDefaultedArray(),
      0,
      stream,
      execInfo_.kernelParams,
      O,
      I,
      profile);
  if (profile and OptionsCache::cacheEnabled()) {
    OptionsCache::getCache()->recordRuntime(
        // TODO:replace this with pretty printed TC
        execInfo_.kernelName,
        MappingOptions(execInfo_.options),
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
  CHECK_NE(grid[0], 0) << "Grid dims are not set up";
  CHECK_NE(block[0], 0) << "Block dims are not set up";
  bool profile = false;
  rtcFun->Launch(
      grid.extractDefaultedArray(),
      block.extractDefaultedArray(),
      0,
      stream,
      execInfo_.kernelParams,
      outputs,
      inputs,
      profile);
}

} // namespace tc
