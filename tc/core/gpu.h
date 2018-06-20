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

// Conditionally include CUDA-specific headers.  This file should compile even
// without them.
#ifdef CUDA_HOME
#include "tc/core/cuda/cuda.h"
#endif

namespace tc {

/// Get the shared memory size of the GPU device active in the current thread.
/// The call is forwarded to the appropriate GPU driver (CUDA in particular).
/// If a thread has no associated GPU device, return 0.
inline size_t querySharedMemorySize() {
#ifdef CUDA_HOME
  return CudaGPUInfo::GPUInfo().SharedMemorySize();
#else
  return 0;
#endif
}

/// Get the shared memory size per sm of the GPU device active in the current
/// thread.
/// The call is forwarded to the appropriate GPU driver (CUDA in particular).
/// If a thread has no associated GPU device, return 0.
inline size_t querySharedMemorySizePerSM() {
#ifdef CUDA_HOME
  return CudaGPUInfo::GPUInfo().SharedMemorySizePerSM();
#else
  return 0;
#endif
}

/// Get the maximum number of blocks per sm of the GPU device active
/// in the current thread.
/// The call is forwarded to the appropriate GPU driver (CUDA in particular).
/// If a thread has no associated GPU device, return 0.
inline size_t queryBlocksPerSM() {
#ifdef CUDA_HOME
  return CudaGPUInfo::GPUInfo().BlocksPerSM();
#else
  return 0;
#endif
}

/// Get the maximum number of threads per sm of the GPU device active
/// in the current thread.
/// The call is forwarded to the appropriate GPU driver (CUDA in particular).
/// If a thread has no associated GPU device, return 0.
inline size_t queryThreadsPerSM() {
#ifdef CUDA_HOME
  return CudaGPUInfo::GPUInfo().ThreadsPerSM();
#else
  return 0;
#endif
}

/// Get the number of sm on the GPU device active in the current thread.
/// The call is forwarded to the appropriate GPU driver (CUDA in particular).
/// If a thread has no associated GPU device, return 0.
inline size_t queryNbOfSM() {
#ifdef CUDA_HOME
  return CudaGPUInfo::GPUInfo().NbOfSM();
#else
  return 0;
#endif
}

} // namespace tc
