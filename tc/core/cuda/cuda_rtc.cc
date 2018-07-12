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
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

#include <cuda_runtime.h>
#include <nvrtc.h>

#include "tc/core/check.h"
#include "tc/core/cuda/cuda.h"
#include "tc/core/cuda/cuda_rtc.h"
#include "tc/core/flags.h"
#include "tc/core/scope_guard.h"

namespace tc {
std::mutex nvrtc_mutex;

CudaRTCFunction::CudaRTCFunction() {}

CudaRTCFunction::~CudaRTCFunction() {
  if (!cleared_) {
    // XXX:this may throw
    clear();
  }
}

void CudaRTCFunction::clear() {
  if (!cleared_) {
    for (auto kvp : perGpuModule_) {
      WithCudaDevice(kvp.first);
      TC_CUDA_DRIVERAPI_ENFORCE(cuModuleUnload(kvp.second));
    }
    cleared_ = true;
  }
}

void checkOrCreateContext() {
  static thread_local bool created = false;
  if (!created) {
    created = true;
    CUcontext ctx;
    TC_CUDA_DRIVERAPI_ENFORCE(cuCtxGetCurrent(&ctx));
    if (!ctx) {
      TC_CUDA_RUNTIMEAPI_ENFORCE(cudaDeviceSynchronize());
    }
  }
}

namespace {
static std::tuple<int, int, int> getCudaArchitecture() {
  int device, major, minor;
  CUdevice deviceHandle;
  TC_CUDA_RUNTIMEAPI_ENFORCE(cudaGetDevice(&device));
  TC_CUDA_DRIVERAPI_ENFORCE(cuDeviceGet(&deviceHandle, device));
  TC_CUDA_DRIVERAPI_ENFORCE(cuDeviceGetAttribute(
      &major, CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR, deviceHandle));
  TC_CUDA_DRIVERAPI_ENFORCE(cuDeviceGetAttribute(
      &minor, CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR, deviceHandle));
  return std::tuple<int, int, int>(device, major, minor);
}

static std::string llvmCompile(
    const std::string& name,
    const std::string& source) {
  int device, major, minor;
  std::tie(device, major, minor) = getCudaArchitecture();

  std::string pat("/tmp/cudaXXXXXX");
  std::vector<char> ifn(pat.begin(), pat.end());
  TC_CHECK_GE(mkstemp(ifn.data()), 0); // string.c_str is const char*
  std::string inputFileName(ifn.begin(), ifn.end());
  // cstdio's std::remove to delete files
  tc::ScopeGuard sgi([&]() { std::remove(inputFileName.c_str()); });
  {
    std::ofstream ostream(inputFileName, std::ios::binary);
    ostream << source;
  }

  std::string arch = "sm_" + std::to_string(major) + std::to_string(minor);
  std::string outputClangFile = inputFileName + "-clang.ll";
  std::string outputLinkFile = inputFileName + "-link.ll";
  std::string outputOptFile = inputFileName + "-opt.ll";
  std::string outputPtxFile = inputFileName + ".s";
  tc::ScopeGuard sgo([&]() {
    // cstdio's std::remove to delete files
    std::remove(outputClangFile.c_str());
    std::remove(outputLinkFile.c_str());
    std::remove(outputOptFile.c_str());
    std::remove(outputPtxFile.c_str());
  });

  std::string cmdLlvmIr = std::string(TC_STRINGIFY(TC_LLVM_BIN_DIR)) +
      "/clang++ -x cuda " + inputFileName + " " + "--cuda-device-only " +
      "--cuda-gpu-arch=" + arch + " " +
      "--cuda-path=" + TC_STRINGIFY(TC_CUDA_TOOLKIT_ROOT_DIR) + " " + "-I" +
      TC_STRINGIFY(TC_CUDA_INCLUDE_DIR) + " " + "-I" +
      TC_STRINGIFY(TC_CUB_INCLUDE_DIR) + " " + tc::FLAGS_llvm_flags +
      "  -DNVRTC_CUB=1 " + "-nocudalib -S -emit-llvm " + "-o " +
      outputClangFile;
  TC_CHECK_EQ(std::system(cmdLlvmIr.c_str()), 0) << cmdLlvmIr;

  std::string cmdLlvmLink = std::string(TC_STRINGIFY(TC_LLVM_BIN_DIR)) +
      "/llvm-link " + outputClangFile + " " +
      TC_STRINGIFY(TC_CUDA_TOOLKIT_ROOT_DIR) +
      "/nvvm/libdevice/libdevice.*.bc " + "-S -o " + outputLinkFile;
  TC_CHECK_EQ(std::system(cmdLlvmLink.c_str()), 0) << cmdLlvmLink;

  std::string cmdOpt = std::string(TC_STRINGIFY(TC_LLVM_BIN_DIR)) + "/opt " +
      "-internalize -internalize-public-api-list=" + name + " " +
      "-nvvm-reflect -O3 " + outputLinkFile + " -S -o " + outputOptFile;
  TC_CHECK_EQ(std::system(cmdOpt.c_str()), 0) << cmdOpt;

  std::string cmdPtx = std::string(TC_STRINGIFY(TC_LLVM_BIN_DIR)) +
      "/llc -mcpu=" + arch + " " + outputOptFile + " -o " + outputPtxFile;
  TC_CHECK_EQ(std::system(cmdPtx.c_str()), 0) << cmdPtx;

  std::ifstream stream(outputPtxFile);
  return std::string(
      (std::istreambuf_iterator<char>(stream)),
      std::istreambuf_iterator<char>());
}

static std::string nvccCompile(
    const std::string& name,
    const std::string& source) {
  int device, major, minor;
  std::tie(device, major, minor) = getCudaArchitecture();

  std::string pat("/tmp/cudaXXXXXX");
  std::vector<char> ifn(pat.begin(), pat.end());
  TC_CHECK_GE(mkstemp(ifn.data()), 0); // string.c_str is const char*
  std::string inputFileName(ifn.begin(), ifn.end());
  // cstdio's std::remove to delete files
  tc::ScopeGuard sgi([&]() { std::remove(inputFileName.c_str()); });
  {
    std::ofstream ostream(inputFileName, std::ios::binary);
    ostream << source;
  }

  std::string arch = "sm_" + std::to_string(major) + std::to_string(minor);
  std::string outputPtxFile = inputFileName + ".ptx";
  // cstdio's std::remove to delete files
  tc::ScopeGuard sgo([&]() { std::remove(outputPtxFile.c_str()); });

  std::string cmdPtx = std::string(TC_STRINGIFY(TC_CUDA_TOOLKIT_ROOT_DIR)) +
      "/bin/nvcc -x cu " + inputFileName + " --gpu-architecture=" + arch + " " +
      "--ptx " + "-I" + TC_STRINGIFY(TC_CUDA_INCLUDE_DIR) + " " + "-I" +
      TC_STRINGIFY(TC_CUB_INCLUDE_DIR) + " " + tc::FLAGS_nvcc_flags + " -o " +
      outputPtxFile;
  TC_CHECK_EQ(std::system(cmdPtx.c_str()), 0) << cmdPtx;

  std::ifstream stream(outputPtxFile);
  return std::string(
      (std::istreambuf_iterator<char>(stream)),
      std::istreambuf_iterator<char>());
}

static std::string nvrtcCompile(
    const std::string& name,
    const std::string& source) {
  int device, major, minor;
  std::tie(device, major, minor) = getCudaArchitecture();

  nvrtcProgram prog;
  TC_NVRTC_CHECK(
      nvrtcCreateProgram(&prog, source.c_str(), nullptr, 0, nullptr, nullptr));

  std::stringstream arch_param;
  arch_param << "--gpu-architecture=compute_" << major << minor;
  std::string arch = arch_param.str();

  // Compile the program.
  const char* nvrtc_debug_opts[] = {"-G", "-lineinfo"};
  std::string cudaHome =
      std::string("-I ") + std::string(TC_STRINGIFY(TC_CUDA_INCLUDE_DIR));
  std::string cubHome =
      std::string("-I ") + std::string(TC_STRINGIFY(TC_CUB_INCLUDE_DIR));
  std::vector<const char*> nvrtcts = {arch.c_str(),
                                      "--use_fast_math",
                                      "-std=c++11",
                                      "-default-device",
                                      "-DNVRTC_CUB=1",
                                      cudaHome.c_str(),
                                      cubHome.c_str()};
  if (FLAGS_debug_cuda) {
    nvrtcts.push_back(nvrtc_debug_opts[0]);
    nvrtcts.push_back(nvrtc_debug_opts[1]);
  }
  // TODO: Use me
  // const char* nvrtc_tune_options = {"--maxregcount=32"};
  nvrtcResult compile_result =
      nvrtcCompileProgram(prog, nvrtcts.size(), nvrtcts.data());
  if (compile_result != NVRTC_SUCCESS) {
    size_t log_size;
    TC_NVRTC_CHECK(nvrtcGetProgramLogSize(prog, &log_size));
    std::vector<char> nvrtc_log(log_size);
    TC_NVRTC_CHECK(nvrtcGetProgramLog(prog, nvrtc_log.data()));
    LOG(WARNING) << "Compilation failure for nvrtc("
                 << nvrtcGetErrorString(compile_result) << "): \n"
                 << nvrtc_log.data() << " source:" << source;
    LOG(ERROR) << "Compilation failure for nvrtc("
               << nvrtcGetErrorString(compile_result) << "): \n"
               << nvrtc_log.data() << " source:" << source;
    throw std::runtime_error("Could not compile function");
  }
  size_t ptx_size;
  TC_NVRTC_CHECK(nvrtcGetPTXSize(prog, &ptx_size));
  std::vector<char> res(ptx_size);
  TC_NVRTC_CHECK(nvrtcGetPTX(prog, res.data()));
  TC_NVRTC_CHECK(nvrtcDestroyProgram(&prog));
  return std::string(res.begin(), res.end());
}
} // namespace

std::unique_ptr<CudaRTCFunction> CudaRTCFunction::Compile(
    const std::string& name,
    const std::string& source) {
  std::unique_ptr<CudaRTCFunction> res(new CudaRTCFunction());
  res->specializedName = name;
  res->cleared_ = false;
  if (FLAGS_debug_tc_mapper) {
    LOG(INFO) << "NVRTC function source:\n" << source;
  }
  if (FLAGS_cuda_compiler == "nvrtc") {
    res->ptx = nvrtcCompile(name, source);
  } else if (FLAGS_cuda_compiler == "llvm") {
    res->ptx = llvmCompile(name, source);
  } else if (FLAGS_cuda_compiler == "nvcc") {
    res->ptx = nvccCompile(name, source);
  } else {
    CHECK(false) << "Unknown CUDA compiler: " << FLAGS_cuda_compiler;
  }
  if (FLAGS_dump_ptx) {
    LOG(INFO) << "PTX:\n" << res->ptx;
  }
  return res;
}

namespace {

template <typename T>
std::ostream& operator<<(std::ostream& os, const std::array<T, 3>& a) {
  os << "annotation: " << a[0] << " " << a[1] << " " << a[2] << " ";
  return os;
}

} // namespace

Duration CudaRTCFunction::Launch(
    const std::array<size_t, 3>& grid,
    const std::array<size_t, 3>& block,
    unsigned int shared_mem,
    cudaStream_t stream,
    std::vector<long> params,
    std::vector<void*> outputs,
    std::vector<const void*> inputs,
    bool profile) const {
  int dev;
  TC_CUDA_RUNTIMEAPI_ENFORCE(cudaGetDevice(&dev));
  if (perGpuModule_.count(dev) == 0) {
    CUmodule module;
    CUfunction function;
    // Checking that a CUDA context exists for the current thread is necessary
    // when benchmarking the backward of a PyTorch gradient operator:
    // the backward is called on a different thread whose context may not have
    // been initialized explicitly.
    // This call to cudaDeviceSynchronize implicitly creates a new context if
    // one is not bound to the current CPU.
    checkOrCreateContext();
    auto res = cuModuleLoadData(&module, ptx.c_str());
    if (res != CUDA_SUCCESS) {
      LOG(ERROR) << "Invalid PTX: " << ptx;
    }
    TC_CUDA_DRIVERAPI_ENFORCE(res);
    perGpuModule_.emplace(dev, module);
    TC_CUDA_DRIVERAPI_ENFORCE(
        cuModuleGetFunction(&function, module, specializedName.c_str()));
    perGpuKernel_.emplace(dev, function);
  }

  constexpr size_t kNumMaxParameters = 100;
  std::array<void*, kNumMaxParameters> args_voidp{0};
  TC_CHECK_GE(
      kNumMaxParameters, params.size() + outputs.size() + inputs.size());
  int ind = 0;
  for (auto& p : params) {
    args_voidp[ind++] = &p;
  }
  for (auto& o : outputs) {
    args_voidp[ind++] = &o;
  }
  for (auto& i : inputs) {
    args_voidp[ind++] = static_cast<void*>(&i);
  }
  // TODO: some sanity checks before launching such that we don't make clear
  // mistakes
  unsigned int gx = grid[0];
  unsigned int gy = grid[1];
  unsigned int gz = grid[2];
  unsigned int bx = block[0];
  unsigned int by = block[1];
  unsigned int bz = block[2];
  auto launch = [&]() {
    TC_CUDA_DRIVERAPI_ENFORCE(cuLaunchKernel(
        perGpuKernel_.at(dev),
        gx,
        gy,
        gz,
        bx,
        by,
        bz,
        shared_mem,
        stream,
        args_voidp.data(),
        0));
  };

  if (not profile) {
    launch();
    return Duration::max();
  }

  cudaEvent_t start, stop;

  TC_CUDA_RUNTIMEAPI_ENFORCE(cudaEventCreate(&start));
  TC_CUDA_RUNTIMEAPI_ENFORCE(cudaEventCreate(&stop));
  TC_CUDA_RUNTIMEAPI_ENFORCE(cudaEventRecord(start, stream));
  launch();
  TC_CUDA_RUNTIMEAPI_ENFORCE(cudaEventRecord(stop, stream));

  TC_CUDA_RUNTIMEAPI_ENFORCE(cudaEventSynchronize(stop));
  float milliseconds = 0;
  TC_CUDA_RUNTIMEAPI_ENFORCE(cudaEventElapsedTime(&milliseconds, start, stop));
  TC_CUDA_RUNTIMEAPI_ENFORCE(cudaEventDestroy(start));
  TC_CUDA_RUNTIMEAPI_ENFORCE(cudaEventDestroy(stop));
  return Duration::fromMicroSeconds(milliseconds * 1000);
}
} // namespace tc
