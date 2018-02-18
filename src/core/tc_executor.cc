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

#include "tc/core/compilation_cache.h"
#include "tc/core/halide2pencil.h"
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

const size_t TcExecutor::InvalidHandle;

lang::TreeRef parseOneFunction(const std::string& def) {
  lang::Parser parser(def);
  auto r = parser.parseFunction();
  if (parser.L.cur().kind != lang::TK_EOF) {
    throw lang::ErrorReport(parser.L.cur().range)
        << "More than one TCs were passed to TcExecutor.";
  }
  return r;
}

int toTypeToken(DLDataType dtype) {
  return lang::TypeInfo(lang::TypeInfo::Code(dtype.code), dtype.bits)
      .toScalarToken();
}

TcExecutor::TcExecutor(
    const std::string& TcDefinition,
    const std::vector<const DLTensor*>& inputsInfo)
    : TcExecutor(parseOneFunction(TcDefinition), inputsInfo) {}

// TODO: make sure that the empty stride arrays (in DLTensor) are not a problem
void checkSizesAndStridesAreCompliant(
    const DLTensor* actual,
    const DLTensor* expected,
    const lang::Param& dbg) {
  if (actual->ndim != expected->ndim) {
    throw lang::ErrorReport(dbg)
        << "expected " << expected->ndim << " dimensions but found tensor with "
        << actual->ndim << " dimensions";
  }
  auto atype = toTypeToken(actual->dtype);
  auto etype = toTypeToken(expected->dtype);
  if (atype != etype) {
    throw lang::ErrorReport(dbg) << "expected " << lang::kindToString(etype)
                                 << " but found " << lang::kindToString(atype);
  }
  std::vector<int64_t> shapeA(actual->shape, actual->shape + actual->ndim);
  std::vector<int64_t> shapeE(
      expected->shape, expected->shape + expected->ndim);
  for (int i = 0; i < shapeA.size(); ++i) {
    if (shapeA[i] != shapeE[i]) {
      throw lang::ErrorReport(dbg)
          << "expected size " << shapeE[i] << " for dim " << i << " but found "
          << shapeA[i];
    }
  }
}

// templating to match both const and non-const DLTensor pointers
template <typename T>
void checkSizesAndStridesAreCompliant(
    const std::vector<T*>& dlTensors,
    const std::vector<DLTensorUPtr>& tensorInfos,
    const lang::ListView<lang::Param>& dbgInfo) {
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

void TcExecutor::checkInputsCompliant(
    const std::vector<const DLTensor*>& inputsInfo) const {
  if (inputsInfo.size() != halideComponents_.inputs.size()) {
    throw lang::ErrorReport(halideComponents_.getDef())
        << "expected " << halideComponents_.inputs.size()
        << " inputs but found " << inputsInfo.size();
  }
  for (size_t i = 0; i < inputsInfo.size(); ++i) {
    auto dltype_ = inputsInfo[i]->dtype;
    auto htype_ = halideComponents_.inputs[i].type();
    // we have three type representations here: (1) halide Type (2) DLTensor
    // type, and (3) the token representing the type in the frontend (e.g.
    // TK_FLOAT) we need to translate to (3) to report user facing errors
    auto dltype =
        lang::TypeInfo(lang::TypeInfo::Code(dltype_.code), dltype_.bits)
            .toScalarToken();
    auto htype =
        lang::TypeInfo(lang::TypeInfo::Code(htype_.code()), htype_.bits())
            .toScalarToken();
    if (dltype != htype) {
      throw lang::ErrorReport(halideComponents_.getDef().params()[i])
          << "expected type " << lang::kindToString(htype) << " but found "
          << lang::kindToString(dltype);
    }
    int edim = halideComponents_.inputs[i].dimensions();
    int adim = inputsInfo[i]->ndim;
    if (adim != edim) {
      throw lang::ErrorReport(halideComponents_.getDef().params()[i])
          << "expected a tensor with " << edim << " dimensions but found "
          << adim << " dimensions.";
    }
  }
}

TcExecutor::TcExecutor(
    lang::TreeRef TcDefinition,
    const std::vector<const DLTensor*>& inputsInfo)
    : tcTree_(TcDefinition), ctx_(isl_ctx_alloc()) {
  execInfo_.kernelName = lang::Def(tcTree_).name().name();
  halideComponents_ = tc2halide::translate(ctx_, tcTree_);
  checkInputsCompliant(inputsInfo);
  execInfo_.inputsInfo = makeDLTensorVector(inputsInfo);
  // TODO: check if this is wrong, packed tensors may  have 0 strides stored
  execInfo_.outputsInfo =
      tc::inferOutputTensorInfo(halideComponents_, inputsInfo);
}

TcExecutor::~TcExecutor() {
  isl_ctx_free(ctx_.release());
}

std::vector<const DLTensor*> TcExecutor::inferOutputTensorInfo() {
  return extractRawPtrs(execInfo_.outputsInfo);
}

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

void TcExecutor::compile(const tc::MappingOptions& options) {
  if (execInfo_.rtcFun) {
    throw std::runtime_error{
        "TcExecutor::compile cannot be called multiple tines."};
  }
  execInfo_.options =
      std::unique_ptr<MappingOptions>(new MappingOptions(options));

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
    CHECK(execInfo_.options)
        << "Isl Kernel options are NULL, are you trying compile "
        << "a dummy TcExecutor?";
    return CudaCache::getCache()->retrieveKernel(
        execInfo_.kernelName, // TODO:replace this with pretty printed TC
        *execInfo_.options,
        extractRawPtrs(execInfo_.inputsInfo),
        extractRawPtrs(execInfo_.outputsInfo));
  }();

  if (cachedOp) {
    execInfo_.cudaSource = cachedOp->source;
    execInfo_.grid = cachedOp->grid;
    execInfo_.block = cachedOp->block;
    execInfo_.kernelParams = cachedOp->parameters;
    execInfo_.kernelSpecializedName = cachedOp->specializedName;
    LOG_IF(INFO, FLAGS_debug_tc_mapper)
        << "generatedCuda: " << execInfo_.cudaSource;
    LOG_IF(INFO, FLAGS_debug_tc_mapper) << "retrieved grid: " << execInfo_.grid;
    LOG_IF(INFO, FLAGS_debug_tc_mapper)
        << "retrieved block: " << execInfo_.block;
  } else {
    compileWithTcMapper();
    execInfo_.cudaSource =
        appendOptionsAndGitHash(execInfo_.cudaSource, *execInfo_.options);
    if (CudaCache::cacheEnabled()) {
      LOG_IF(INFO, FLAGS_debug_tc_mapper)
          << "original grid: " << execInfo_.grid;
      LOG_IF(INFO, FLAGS_debug_tc_mapper)
          << "original block: " << execInfo_.block;
      CudaCache::getCache()->cacheKernel(
          execInfo_.kernelName, // TODO:replace this with pretty printed TC
          *execInfo_.options,
          extractRawPtrs(execInfo_.inputsInfo),
          extractRawPtrs(execInfo_.outputsInfo),
          execInfo_.kernelSpecializedName,
          execInfo_.kernelParams,
          execInfo_.cudaSource,
          execInfo_.grid,
          execInfo_.block);
    }
  }

  execInfo_.rtcFun = nullptr; // force unloading in case we
  // NVRTC the same name / input with different options.
  auto t0 = std::chrono::high_resolution_clock::now();
  execInfo_.rtcFun = CudaRTCFunction::Compile(
      execInfo_.kernelSpecializedName, execInfo_.cudaSource);
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

void TcExecutor::compileWithTcMapper() {
  // A bit chicken-and-eggy, need scop from TC to have the space to build the
  // context to specialize the scop..
  auto scopTmp = polyhedral::Scop::makeScop(ctx_, halideComponents_);
  auto globalParameterContext =
      scopTmp->makeContextFromInputs(extractRawPtrs(execInfo_.inputsInfo));
  scopTmp = polyhedral::Scop::makeSpecializedScop(
      *scopTmp,
      globalParameterContext.intersect(scopTmp->globalParameterContext));
  LOG_IF(INFO, FLAGS_debug_tc_mapper) << *(execInfo_.options);
  LOG_IF(INFO, FLAGS_debug_tc_mapper) << *(scopTmp->scheduleRoot());

  // Now we can build stuff
  auto mappedScop =
      polyhedral::MappedScop::makeWithOuterBlockInnerThreadStrategy(
          std::move(scopTmp), *execInfo_.options);
  LOG_IF(INFO, FLAGS_debug_tc_mapper) << "Mapped schedule:" << std::endl
                                      << *(mappedScop->schedule());

  execInfo_.kernelParams = narrowParamsVector(
      mappedScop->scop().getParameterValues(globalParameterContext));
  execInfo_.kernelSpecializedName =
      specializeKernelName(execInfo_.kernelName, execInfo_.kernelParams);

  // This updates the launch bounds with the actual result from compilation
  // with tightening of launch_bounds.
  // What you get is not what you asked for, the autotuner should adapt to
  // that.
  std::tie(execInfo_.cudaSource, execInfo_.grid, execInfo_.block) =
      mappedScop->codegen(execInfo_.kernelSpecializedName);
  LOG_IF(INFO, FLAGS_dump_cuda) << "generatedCuda: " << execInfo_.cudaSource;
}

Duration TcExecutor::run(
    const std::vector<const DLTensor*>& inputs,
    const std::vector<DLTensor*>& outputs,
    bool profile) const {
  CHECK(execInfo_.rtcFun) << "Can't launch uncompiled: "
                          << execInfo_.kernelName;
  CHECK(execInfo_.options);
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
  CHECK_NE(execInfo_.grid[0], 0) << "Grid dims are not set up";
  CHECK_NE(execInfo_.block[0], 0) << "Block dims are not set up";
  auto res = execInfo_.rtcFun->Launch(
      execInfo_.grid.extractDefaultedArray(),
      execInfo_.block.extractDefaultedArray(),
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
        *execInfo_.options,
        inputs,
        constPtrs(outputs),
        res);
  }
  return res;
}

void TcExecutor::uncheckedRun(
    const std::vector<const void*>& inputs,
    const std::vector<void*>& outputs) const {
  cudaStream_t stream = 0;
  CHECK_NE(execInfo_.grid[0], 0) << "Grid dims are not set up";
  CHECK_NE(execInfo_.block[0], 0) << "Block dims are not set up";
  bool profile = false;
  execInfo_.rtcFun->Launch(
      execInfo_.grid.extractDefaultedArray(),
      execInfo_.block.extractDefaultedArray(),
      0,
      stream,
      execInfo_.kernelParams,
      outputs,
      inputs,
      profile);
}

} // namespace tc
