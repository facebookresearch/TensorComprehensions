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

// Conditionally include CUDA-specific headers.  This file should compile even
// without them.
#if TC_WITH_CUDA
#include "tc/core/cuda/cuda.h"
#endif

namespace tc {

/// Get the shared memory size of the GPU device active in the current thread.
/// The call is forwarded to the appropriate GPU driver (CUDA in particular).
/// If a thread has no associated GPU device, return 0.
inline size_t querySharedMemorySize() {
#if TC_WITH_CUDA && !defined(NO_CUDA_SDK)
  return CudaGPUInfo::GPUInfo().SharedMemorySize();
#else
  return 0;
#endif
}

/// Get the maximum number of registers per block provided by the GPU device
/// active in the current thread.  The call is forwarded to the GPU driver.
/// If the thread has no associated GPU, return 0.
inline size_t queryRegistersPerBlock() {
#if TC_WITH_CUDA && !defined(NO_CUDA_SDK)
  return CudaGPUInfo::GPUInfo().RegistersPerBlock();
#else
  return 0;
#endif
}

} // namespace tc
