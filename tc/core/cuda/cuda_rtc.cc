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
#include <sstream>
#include <string>
#include <vector>

#include <cuda_runtime.h>
#include <nvrtc.h>

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
      WithDevice(kvp.first);
      TC_CUDA_DRIVERAPI_ENFORCE(cuModuleUnload(kvp.second));
    }
    cleared_ = true;
  }
}

std::unique_ptr<CudaRTCFunction> CudaRTCFunction::Compile(
    const std::string& name,
    const std::string& source) {
  std::unique_ptr<CudaRTCFunction> res(new CudaRTCFunction());
  res->specializedName = name;
  res->cleared_ = false;

  if (FLAGS_debug_tc_mapper) {
    LOG(INFO) << "NVRTC function source:\n" << source;
  }
  // Actually do the compiling.
  nvrtcProgram prog;
  TC_NVRTC_CHECK(
      nvrtcCreateProgram(&prog, source.c_str(), nullptr, 0, nullptr, nullptr));

  // Get the architecture of the current device.
  int device, minor, major;
  CUdevice deviceHandle;
  TC_CUDA_RUNTIMEAPI_ENFORCE(cudaGetDevice(&device));
  TC_CUDA_DRIVERAPI_ENFORCE(cuDeviceGet(&deviceHandle, device));
  TC_CUDA_DRIVERAPI_ENFORCE(cuDeviceGetAttribute(
      &major, CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR, deviceHandle));
  TC_CUDA_DRIVERAPI_ENFORCE(cuDeviceGetAttribute(
      &minor, CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR, deviceHandle));

  std::stringstream arch_param;
  arch_param << "--gpu-architecture=compute_" << major << minor;
  std::string arch = arch_param.str();

  // Compile the program.
  const char* nvrtc_debug_opts[] = {"-G", "-lineinfo"};
  std::string cudaHome = std::string("-I ") + std::string(CUDA_HOME);
  std::string cubHome = std::string("-I ") + std::string(CUB_HOME);
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
  res->nvrtc_ptx = std::vector<char>(ptx_size);
  TC_NVRTC_CHECK(nvrtcGetPTX(prog, res->nvrtc_ptx.data()));
  TC_NVRTC_CHECK(nvrtcDestroyProgram(&prog));

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
    std::vector<int> params,
    std::vector<void*> outputs,
    std::vector<const void*> inputs,
    bool profile) const {
  int dev;
  TC_CUDA_RUNTIMEAPI_ENFORCE(cudaGetDevice(&dev));
  if (perGpuModule_.count(dev) == 0) {
    CUmodule module;
    CUfunction function;
    TC_CUDA_DRIVERAPI_ENFORCE(
        cuModuleLoadDataEx(&module, nvrtc_ptx.data(), 0, 0, 0));
    perGpuModule_.emplace(dev, module);
    TC_CUDA_DRIVERAPI_ENFORCE(
        cuModuleGetFunction(&function, module, specializedName.c_str()));
    perGpuKernel_.emplace(dev, function);
  }

  constexpr size_t kNumMaxParameters = 100;
  std::array<void*, kNumMaxParameters> args_voidp{0};
  CHECK_GE(kNumMaxParameters, params.size() + outputs.size() + inputs.size());
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
  return std::chrono::microseconds(static_cast<int64_t>(milliseconds * 1000));
}
} // namespace tc
