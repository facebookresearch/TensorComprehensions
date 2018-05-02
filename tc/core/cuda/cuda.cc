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

#include "tc/core/cuda/cuda.h"

#include <memory>
#include <stdexcept>
#include <tuple>
#include <vector>

#include <cuda_runtime_api.h>

#include "tc/core/flags.h"

namespace tc {
DEFINE_bool(use_nvprof, false, "Start / stop nvprof");

namespace {

std::tuple<std::vector<std::string>, std::vector<size_t>> init() {
  int deviceCount = 0;
  auto err_id = cudaGetDeviceCount(&deviceCount);
  if (err_id == 35 or err_id == 30) {
    // Cuda driver not available?
    LOG(INFO) << "Cuda driver not available.\n";
    return {};
  }
  TC_CUDA_RUNTIMEAPI_ENFORCE(err_id);
  if (deviceCount == 0) {
    return {};
  }
  std::vector<std::string> gpuNames;
  std::vector<size_t> sharedMemSizes;
  gpuNames.reserve(deviceCount);
  for (int i = 0; i < deviceCount; ++i) {
    cudaDeviceProp deviceProp;
    TC_CUDA_RUNTIMEAPI_ENFORCE(cudaGetDeviceProperties(&deviceProp, i));
    gpuNames.emplace_back(deviceProp.name);
    sharedMemSizes.emplace_back(deviceProp.sharedMemPerBlock);
  }
  return std::make_tuple(gpuNames, sharedMemSizes);
}

} // namespace

CudaGPUInfo& CudaGPUInfo::GPUInfo() {
  static thread_local std::unique_ptr<CudaGPUInfo> pInfo;
  static thread_local bool inited = false;
  if (!inited) {
    auto infos = init();
    pInfo = std::unique_ptr<CudaGPUInfo>(
        new CudaGPUInfo(std::get<0>(infos), std::get<1>(infos)));
    inited = true;
  }
  return *pInfo;
}

int CudaGPUInfo::NumberGPUs() const {
  return gpuNames_.size();
}

std::string CudaGPUInfo::GetGPUName(int id) const {
  if (id < 0) {
    return gpuNames_.at(CurrentGPUId());
  }
  return gpuNames_.at(id);
}

int CudaGPUInfo::CurrentGPUId() const {
  int deviceID = 0;
  TC_CUDA_RUNTIMEAPI_ENFORCE(cudaGetDevice(&deviceID));
  return deviceID;
}

void CudaGPUInfo::SynchronizeCurrentGPU() const {
  TC_CUDA_RUNTIMEAPI_ENFORCE(cudaDeviceSynchronize());
}

std::string CudaGPUInfo::getCudaDeviceStr() const {
  if (NumberGPUs() == 0) {
    throw std::runtime_error("No GPUs found.");
  }
  return GetGPUName(CurrentGPUId());
}

size_t CudaGPUInfo::SharedMemorySize() const {
  if (NumberGPUs() == 0) {
    return 0; // no shared memory if no GPUs
  }
  return sharedMemSizes_.at(CurrentGPUId());
}
} // namespace tc
