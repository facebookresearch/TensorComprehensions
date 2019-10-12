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

#include "tc/tc_config.h"

#ifndef TC_CUDA_INCLUDE_DIR
#error "TC_CUDA_INCLUDE_DIR must be defined"
#endif // TC_CUDA_INCLUDE_DIR

#ifndef TC_CUB_INCLUDE_DIR
#error "TC_CUB_INCLUDE_DIR must be defined"
#endif // TC_CUB_INCLUDE_DIR

#include <sstream>
#include <stdexcept>

#include <cuda.h>
#include <cuda_profiler_api.h>
#include <cuda_runtime.h>

#include <gflags/gflags.h>
#include <glog/logging.h>

#define TC_CUDA_DRIVERAPI_ENFORCE(condition)                            \
  do {                                                                  \
    CUresult result = condition;                                        \
    if (result != CUDA_SUCCESS) {                                       \
      const char* msg;                                                  \
      cuGetErrorName(result, &msg);                                     \
      std::stringstream ss;                                             \
      ss << "Error at: " << __FILE__ << ":" << __LINE__ << ": " << msg; \
      LOG(WARNING) << ss.str();                                         \
      throw std::runtime_error(ss.str().c_str());                       \
    }                                                                   \
  } while (0)

#define TC_NVRTC_CHECK(condition)                               \
  do {                                                          \
    nvrtcResult result = condition;                             \
    if (result != NVRTC_SUCCESS) {                              \
      std::stringstream ss;                                     \
      ss << "Error at: " << __FILE__ << ":" << __LINE__ << ": " \
         << nvrtcGetErrorString(result);                        \
      LOG(WARNING) << ss.str();                                 \
      throw std::runtime_error(ss.str().c_str());               \
    }                                                           \
  } while (0)

#define TC_CUDA_RUNTIMEAPI_ENFORCE(condition)                   \
  do {                                                          \
    cudaError_t result = condition;                             \
    if (result != cudaSuccess) {                                \
      std::stringstream ss;                                     \
      ss << "Error at: " << __FILE__ << ":" << __LINE__ << ": " \
         << cudaGetErrorString(result);                         \
      LOG(WARNING) << ss.str();                                 \
      throw std::runtime_error(ss.str().c_str());               \
    }                                                           \
  } while (0)

namespace tc {

DECLARE_bool(use_nvprof);

struct WithCudaDevice {
  WithCudaDevice(size_t g) : newGpu(g) {
    int dev;
    TC_CUDA_RUNTIMEAPI_ENFORCE(cudaGetDevice(&dev));
    oldGpu = dev;
    TC_CUDA_RUNTIMEAPI_ENFORCE(cudaSetDevice(newGpu));
  }
  ~WithCudaDevice() noexcept(false) {
    TC_CUDA_RUNTIMEAPI_ENFORCE(cudaSetDevice(oldGpu));
  }
  size_t oldGpu;
  size_t newGpu;
};

//
// This functionality in this type of class has been rewritten over and over
// again. Here we just provide a static singleton and basic properties.
// Consider lifting stuff up from fbcuda rather than reinventing the wheel
//
class CudaGPUInfo {
  CudaGPUInfo(
      const std::vector<std::string>& gpuNames,
      const std::vector<size_t>& sharedMemSizes,
      const std::vector<size_t>& registersPerBlock)
      : gpuNames_(gpuNames),
        sharedMemSizes_(sharedMemSizes),
        registersPerBlock_(registersPerBlock) {}

 public:
  static CudaGPUInfo& GPUInfo();

  // These functions require init to have been run, they are thus members of
  // the singleton object and not static functions.
  int NumberGPUs() const;
  int CurrentGPUId() const;
  void SynchronizeCurrentGPU() const;
  std::string GetGPUName(int id = -1) const;
  std::string getCudaDeviceStr() const;
  size_t SharedMemorySize() const;
  size_t RegistersPerBlock() const;

  std::vector<std::string> gpuNames_;
  std::vector<size_t> sharedMemSizes_;
  std::vector<size_t> registersPerBlock_;
};

struct CudaProfiler {
  CudaProfiler() {
    if (FLAGS_use_nvprof) {
      cudaProfilerStart();
    }
  }
  ~CudaProfiler() {
    if (FLAGS_use_nvprof) {
      cudaProfilerStop();
    }
  }
};

} // namespace tc
