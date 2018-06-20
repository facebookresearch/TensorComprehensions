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

std::tuple<
    std::vector<std::string>,
    std::vector<size_t>,
    std::vector<size_t>,
    std::vector<size_t>,
    std::vector<size_t>,
    std::vector<size_t>>
init() {
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
  std::vector<size_t> sharedMemSizesPerSM;
  std::vector<size_t> blocksPerSM;
  std::vector<size_t> threadsPerSM;
  std::vector<size_t> nbOfSM;
  gpuNames.reserve(deviceCount);
  for (int i = 0; i < deviceCount; ++i) {
    cudaDeviceProp deviceProp;
    TC_CUDA_RUNTIMEAPI_ENFORCE(cudaGetDeviceProperties(&deviceProp, i));
    gpuNames.emplace_back(deviceProp.name);
    sharedMemSizes.emplace_back(deviceProp.sharedMemPerBlock);
    sharedMemSizesPerSM.emplace_back(deviceProp.sharedMemPerMultiprocessor);

    // There is currently no way to get the number of blocks per sm
    // with the CUDA api. The only relevant solution is to compute it
    // with the compute capability.
    // the formula works if the number of blocks per sm is nondecreasing after
    // the 6.0 compute capability.
    auto major = deviceProp.major;
    blocksPerSM.emplace_back(major < 3 ? 8 : (major < 4 ? 16 : 32));

    threadsPerSM.emplace_back(deviceProp.maxThreadsPerMultiProcessor);
    nbOfSM.emplace_back(deviceProp.multiProcessorCount);
  }
  return std::make_tuple(
      gpuNames,
      sharedMemSizes,
      sharedMemSizesPerSM,
      blocksPerSM,
      threadsPerSM,
      nbOfSM);
}

} // namespace

CudaGPUInfo& CudaGPUInfo::GPUInfo() {
  static thread_local std::unique_ptr<CudaGPUInfo> pInfo;
  static thread_local bool inited = false;
  if (!inited) {
    auto infos = init();
    pInfo = std::unique_ptr<CudaGPUInfo>(new CudaGPUInfo(
        std::get<0>(infos),
        std::get<1>(infos),
        std::get<2>(infos),
        std::get<3>(infos),
        std::get<4>(infos),
        std::get<5>(infos)));
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

size_t CudaGPUInfo::SharedMemorySizePerSM() const {
  if (NumberGPUs() == 0) {
    return 0; // no shared memory per sm if no GPUs
  }
  return sharedMemSizesPerSM_.at(CurrentGPUId());
}

size_t CudaGPUInfo::BlocksPerSM() const {
  if (NumberGPUs() == 0) {
    return 0; // no blocks per sm if no GPUs
  }
  return blocksPerSM_.at(CurrentGPUId());
}

size_t CudaGPUInfo::ThreadsPerSM() const {
  if (NumberGPUs() == 0) {
    return 0; // no threads per sm if no GPUs
  }
  return threadsPerSM_.at(CurrentGPUId());
}

size_t CudaGPUInfo::NbOfSM() const {
  if (NumberGPUs() == 0) {
    return 0; // no sm if no GPUs
  }
  return nbOfSM_.at(CurrentGPUId());
}

} // namespace tc
