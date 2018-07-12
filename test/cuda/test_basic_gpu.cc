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
#include <cassert>
#include <functional>
#include <iostream>
#include <sstream>
#include <string>
#include <unordered_set>
#include <vector>

#include <cuda.h>
#include <cuda_runtime.h>
#include <gtest/gtest.h>
#include <nvrtc.h>

#include "tc/core/cuda/cuda.h"

std::vector<char> jitCompile(
    std::string cuda,
    std::vector<const char*> extraCompileOptions = std::vector<const char*>{}) {
  // Actually do the compiling.
  nvrtcProgram prog;
  TC_NVRTC_CHECK(
      nvrtcCreateProgram(&prog, cuda.c_str(), nullptr, 0, nullptr, nullptr));

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
  std::string cudaHome =
      std::string("-I ") + std::string(TC_STRINGIFY(TC_CUDA_INCLUDE_DIR));
  std::vector<const char*> nvrtcts = {arch.c_str(),
                                      "--use_fast_math",
                                      "-std=c++11",
                                      "-DNVRTC_CUB=1",
                                      "-lineinfo",
                                      cudaHome.c_str()};
  for (auto o : extraCompileOptions) {
    nvrtcts.push_back(o);
  }

  nvrtcResult compile_result =
      nvrtcCompileProgram(prog, nvrtcts.size(), nvrtcts.data());
  if (compile_result != NVRTC_SUCCESS) {
    size_t log_size;
    TC_NVRTC_CHECK(nvrtcGetProgramLogSize(prog, &log_size));
    std::vector<char> nvrtc_log(log_size);
    TC_NVRTC_CHECK(nvrtcGetProgramLog(prog, nvrtc_log.data()));
    std::cerr << "Compilation failure for nvrtc("
              << nvrtcGetErrorString(compile_result) << "): \n"
              << nvrtc_log.data() << " source:" << cuda;
    throw std::runtime_error("Could not compile function");
  }
  size_t ptx_size;
  TC_NVRTC_CHECK(nvrtcGetPTXSize(prog, &ptx_size));
  std::vector<char> PTX(ptx_size);
  TC_NVRTC_CHECK(nvrtcGetPTX(prog, PTX.data()));
  TC_NVRTC_CHECK(nvrtcDestroyProgram(&prog));

  return PTX;
}

void loadUnload(const std::string& ptx) {
  CUdevice cuDevice;
  CUcontext context;
  TC_CUDA_DRIVERAPI_ENFORCE(cuInit(0));
  TC_CUDA_DRIVERAPI_ENFORCE(cuDeviceGet(&cuDevice, 0));
  TC_CUDA_DRIVERAPI_ENFORCE(cuCtxCreate(&context, 0, cuDevice));

  CUmodule m;
  CUfunction k;
  TC_CUDA_DRIVERAPI_ENFORCE(cuModuleLoadDataEx(&m, ptx.c_str(), 0, 0, 0));
  TC_CUDA_DRIVERAPI_ENFORCE(cuModuleGetFunction(&k, m, "foo"));
  TC_CUDA_DRIVERAPI_ENFORCE(cuModuleUnload(m));
}

TEST(BasicGpuTest, Nvrtc) {
  TC_CUDA_RUNTIMEAPI_ENFORCE(cudaFree(0));
  auto PTX = jitCompile(
      R"CUDA(
extern "C" {
__global__ void foo(int N)
{
  assert(N == 1);
}
})CUDA",
      {"-G"});

  std::string ptx(PTX.data());
  loadUnload(ptx);
  auto expected = R"E(.visible .entry foo()E";
  EXPECT_NE(std::string::npos, ptx.find(expected));
}

TEST(BasicGpuTest, CubReduce) {
  std::string path(TC_STRINGIFY(TC_CUB_INCLUDE_DIR));
  std::string include = std::string("-I ") + path;
  auto PTX = jitCompile(
      R"CUDA(
#include "cub/nvrtc_cub.cuh"

extern "C" {

__global__ void foo(float* o, const float* i) {
  typedef cub::WarpReduce<float, 4> WarpReduce;
  __shared__ typename WarpReduce::TempStorage s;
  int tx = threadIdx.x;
  if (tx >= 567) { return; }
  o[tx] = WarpReduce(s).Sum(i[tx]);
}

__global__ void bar(float* o, const float* i) {
  typedef cub::BlockReduce<float, 4> BlockReduce;
  __shared__ typename BlockReduce::TempStorage s;
  int tx = threadIdx.x;
  if (tx >= 567) { return; }
  o[tx] = BlockReduce(s).Sum(i[tx]);
}

})CUDA",
      {include.c_str(), "-default-device", "-G"});

  std::string ptx(PTX.data());
  loadUnload(ptx);
  auto expected = R"E(.visible .entry foo()E";
  EXPECT_NE(std::string::npos, ptx.find(expected));
  expected = R"E(.visible .entry bar()E";
  EXPECT_NE(std::string::npos, ptx.find(expected));
}

namespace {
// Mark the function argument as __restrict__ depending on the flag.
std::string makeFuncWithOptionalRestrict(bool useRestrict) {
  std::stringstream ss;
  ss
      << (useRestrict ? "__global__ void func(float* __restrict__ pO2) {"
                      : "__global__ void func(float* pO2) {");
  ss << R"CUDA(int b0 = blockIdx.x;
  int t0 = threadIdx.x;
  float (*O2)[2] = reinterpret_cast<float (*)[2]>(pO2);
  O2[b0][t0] = 0.000000f;  // S1
  __syncthreads();
  if (t0 == 0) {
    for (int c3 = 0; c3 <= 1; c3 += 1) {
      O2[b0][c3] = 12865.0f; // S2
    }
  }
  __syncthreads();
  O2[b0][t0] = fmax(O2[b0][t0], 0);  // S3
})CUDA";
  return ss.str();
}
} // namespace

TEST(BasicGpuTest, Restrict) {
  // CUDA has a particular behavior when __restrict__ keyword is provided for a
  // function argument: beyond assuming that function arguments do not alias,
  // it further seems to assume the argument marked __restrict__ will not be
  // aliased *inside* the function.  The code produced by TC uses "array views"
  // that intentionally alias the argument (but the argument is never used
  // directly).  As a result, when __restrict__ is provided, nvcc/nvrtc assume
  // array elements accessed via different subscripts do not point to the same
  // value, which is wrong, e.g., in case when value are first accessed by
  // different threads, and then sequentially in a loop by a single thread.
  //
  // In the function produced by makeFuncWithOptionalRestrict, the compiler
  // optimizes away the load from global memory in S3 if __restrict__ is
  // provided, so S3 does not see the value written by S2.  This causes
  // incorrect results even in some basic operators, and may cause
  // additional problems when more threads are used for copying to/from shared
  // memory than for the actual computation.
  //
  // This test illustrates the removal of the load from global memory when
  // __restrict__ is provided. In particular, it insepcts the PTX output and
  // looks for a global load instructon after the second barrier
  // synchronization, so that the read can only stem from S3.

  auto PTX = jitCompile(makeFuncWithOptionalRestrict(true));

  std::string ptx(PTX.data());
  auto syncInstr = "bar.sync";
  auto loadInstr = "ld.global";
  auto firstSyncPos = ptx.find(syncInstr);
  auto secondSyncPos = ptx.find(syncInstr, firstSyncPos + 1);
  auto loadPos = ptx.find(loadInstr, secondSyncPos + 1);
  ASSERT_TRUE(firstSyncPos != std::string::npos);
  ASSERT_TRUE(secondSyncPos != std::string::npos);
  ASSERT_TRUE(loadPos == std::string::npos);

  PTX = jitCompile(makeFuncWithOptionalRestrict(false));
  ptx = std::string(PTX.data());
  firstSyncPos = ptx.find(syncInstr);
  secondSyncPos = ptx.find(syncInstr, firstSyncPos + 1);
  loadPos = ptx.find(loadInstr, secondSyncPos + 1);
  ASSERT_TRUE(firstSyncPos != std::string::npos);
  ASSERT_TRUE(secondSyncPos != std::string::npos);
  ASSERT_TRUE(loadPos != std::string::npos);
}

int main(int argc, char** argv) {
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
