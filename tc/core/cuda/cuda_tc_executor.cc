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

#include "tc/core/cuda/cuda_mapping_options_cpp_printer.h"
#include "tc/core/halide_utils.h"
#include "tc/core/polyhedral/cuda/mapped_scop.h"
#include "tc/core/tc2halide.h"
#include "tc/core/tensor.h"

#include "tc/lang/parser.h"
#include "tc/lang/sema.h"

#include <utility>

namespace tc {
namespace {
// Append ordered values to the kernel name, separated by "_".
std::string specializeKernelName(
    const std::string& tcName,
    std::vector<int> params) {
  std::stringstream ss;
  ss << tcName;
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

CudaTcExecutor::CudaTcExecutor(
    const std::vector<TensorInfo>& inputsInfo,
    const std::vector<TensorInfo>& outputsInfo,
    const tc2halide::HalideComponents& halideComponents,
    const typename CudaBackend::CompilationResultType& compilationResult)
    : TcExecutor<CudaBackend>(
          inputsInfo,
          outputsInfo,
          halideComponents,
          compilationResult) {
  auto t0 = std::chrono::high_resolution_clock::now();
  // force unloading in case we JIT with the same name/input/outputs with
  // different options.
  this->clearRuntimeCompiledFunction();
  rtcFun_ = CudaRTCFunction::Compile(
      compilationResult.specializedName, compilationResult.source);
  grid_ = compilationResult.grid;
  block_ = compilationResult.block;
  auto t1 = std::chrono::high_resolution_clock::now();
  LOG_IF(INFO, FLAGS_debug_tc_mapper)
      << "[COMPILE] Compiling with host JIT compiler took: "
      << std::chrono::duration_cast<std::chrono::milliseconds>(t1 - t0).count()
      << "ms" << std::endl;
}

CudaCompilationResult CudaBackend::compileWithTcMapper(
    const std::string& tcName,
    tc2halide::HalideComponents halideComponents,
    const std::vector<const DLConstTensor*>& inputs,
    /* TODO: in the future also pass outputs for stride and alignment info */
    const CudaMappingOptions& options) {
  // A bit chicken-and-eggy, need scop from TC to have the space to build the
  // context to specialize the scop..
  auto scopTmp = polyhedral::Scop::makeScop(
      isl::with_exceptions::globalIslCtx(), halideComponents);
  auto globalParameterContext = scopTmp->makeContextFromInputs(inputs);
  scopTmp = polyhedral::Scop::makeSpecializedScop(
      *scopTmp,
      globalParameterContext.intersect(scopTmp->globalParameterContext));
  LOG_IF(INFO, FLAGS_debug_tc_mapper) << options;
  LOG_IF(INFO, FLAGS_debug_tc_mapper) << *(scopTmp->scheduleRoot());

  // Now we can build stuff
  auto mappedScop =
      polyhedral::MappedScop::makeWithOuterBlockInnerThreadStrategy(
          std::move(scopTmp), options);
  LOG_IF(INFO, FLAGS_debug_tc_mapper) << "Mapped schedule:" << std::endl
                                      << *(mappedScop->schedule());

  auto parameters = narrowParamsVector(
      mappedScop->scop().getParameterValues(globalParameterContext));
  auto specializedName = specializeKernelName(tcName, parameters);

  // This updates the launch bounds with the actual result from compilation
  // with tightening of launch_bounds.
  // What you get is not what you asked for, the autotuner should adapt to
  // that.
  std::string source;
  Grid grid;
  Block block;
  std::tie(source, grid, block) = mappedScop->codegen(specializedName);
  LOG_IF(INFO, FLAGS_dump_cuda) << "generatedCuda: " << source << "\n"
                                << "grid: " << grid << " block: " << block;

  return CudaCompilationResult{
      source, specializedName, parameters, grid, block};
}

void CudaTcExecutor::uncheckedRun(
    const std::vector<const void*>& inputs,
    const std::vector<void*>& outputs) const {
  CHECK(rtcFun_) << "No rtcFun_ attached, cannot launch";
  cudaStream_t stream = 0;
  CHECK_NE(grid_.view[0], 0u) << "Grid dims are not set up";
  CHECK_NE(block_.view[0], 0u) << "Block dims are not set up";
  rtcFun_->Launch(
      grid_.view.extractDefaultedArray(),
      block_.view.extractDefaultedArray(),
      0,
      stream,
      parameters_,
      outputs,
      inputs);
}

ProfilingInfo CudaTcExecutor::profileUnchecked(
    const std::vector<const void*>& inputs,
    const std::vector<void*>& outputs) const {
  auto start = std::chrono::system_clock::now();
  CHECK(rtcFun_) << "No rtcFun_ attached, cannot launch";
  cudaStream_t stream = 0;
  CHECK_NE(grid_.view[0], 0u) << "Grid dims are not set up";
  CHECK_NE(block_.view[0], 0u) << "Block dims are not set up";
  Duration kernelRuntime(rtcFun_->Launch(
      grid_.view.extractDefaultedArray(),
      block_.view.extractDefaultedArray(),
      0,
      stream,
      parameters_,
      outputs,
      inputs,
      true));
  // The CPU overhead is the total time minus the (synchronized) kernel runtime
  auto end = std::chrono::system_clock::now();
  Duration cpuOverhead(end - start);
  cpuOverhead = cpuOverhead - kernelRuntime;
  return ProfilingInfo{cpuOverhead, kernelRuntime};
}
} // namespace tc
