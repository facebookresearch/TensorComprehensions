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
#include <gtest/gtest.h>
#include <iostream>
#include <string>

#include <ATen/ATen.h>

#include "tc/aten/aten_compiler.h"
#include "tc/core/cuda/cuda.h"
#include "tc/core/cuda/cuda_tc_executor.h"
#include "tc/core/flags.h"
#include "tc/core/polyhedral/exceptions.h"

#include "test_harness_aten_cuda.h"

#include <nvrtc.h>

using namespace std;
using namespace tc;

///////////////////////////////////////////////////////////////////////////////
// This file is where bugs are reported.
// They should be clearly named "XXXBug" if they are expected to fail.
// When they are fixed, they graduate to legit unit tests, they shed their
// "Bug" suffix and fly away in the sun.
///////////////////////////////////////////////////////////////////////////////

std::string makeUniqueName(const std::string& name) {
  static int count = 0;
  return name + std::string("_cnt") + std::to_string(++count);
}

////////////////////////////////////////////////////////////////////////////////
// TensorDot bugs found during autotuning
////////////////////////////////////////////////////////////////////////////////
struct TensorDot_32_512_8_2_28_28 : public ::testing::Test {
  std::vector<at::Tensor> inputs;
  std::vector<at::Tensor> outputsNaive;
  bool inited = false;

  void Init() {
    if (inited) {
      return;
    }
    inited = true;

    at::Tensor I0 = at::CUDA(at::kFloat).rand({32, 512, 8, 28, 28});
    at::Tensor I1 = at::CUDA(at::kFloat).rand({32, 8, 2, 28, 28});
    inputs = std::vector<at::Tensor>{I0, I1};

    // Build naive options baseline to check correctness
    // Make naive compile first to better see debug spew
    auto TC = std::string(R"TC(
def tensordot_naive(float(N, C1, C2, H, W) I0, float(N, C2, C3, H, W) I1) -> (O)
{
    O(n, c1, c3, h, w) +=! I0(n, c1, r_c2, h, w) * I1(n, r_c2, c3, h, w)
}
  )TC");

    // If running cuda-gdb only run on test code, not reference: things are
    // slow in this mode
    if (!FLAGS_debug_cuda) {
      auto mappingOptions = tc::CudaMappingOptions::makeNaiveMappingOptions();
      tc::ATenCompilationUnit<tc::CudaTcExecutor> atCompl;
      atCompl.define(TC);
      auto handle = atCompl.compile("tensordot_naive", inputs, mappingOptions);
      atCompl.run("tensordot_naive", inputs, outputsNaive, handle);
    }
  }

  std::vector<at::Tensor> Check(
      const tc::CudaMappingOptions& mappingOptions,
      bool runCuda = true) {
    auto name = makeUniqueName("tensordot");
    auto TC = std::string("def ") + name +
        std::string(R"TC((float(N, C1, C2, H, W) I0,
                            float(N, C2, C3, H, W) I1)
-> (O)
{
    O(n, c1, c3, h, w) +=! I0(n, c1, r_c2, h, w) * I1(n, r_c2, c3, h, w)
}
  )TC");

    std::vector<at::Tensor> outputs;
    tc::ATenCompilationUnit<tc::CudaTcExecutor> atCompl;
    atCompl.define(TC);
    auto handle = atCompl.compile(name, inputs, mappingOptions);
    if (runCuda) {
      atCompl.run(name, inputs, outputs, handle);
      if (!FLAGS_debug_cuda) {
        checkRtol(outputsNaive[0].sub(outputs[0]), inputs, 8, 5e-7);
      }
    }
    return outputs;
  }
};

TEST_F(TensorDot_32_512_8_2_28_28, BaseCorrect) {
  auto options = tc::CudaMappingOptions::makeConvolutionMappingOptions();
  Init();
  Check(options);
}

// This test exercises the unrolling and reduction matching which used to fail
TEST_F(TensorDot_32_512_8_2_28_28, ReductionUnroll) {
  Init();
  auto options =
      tc::CudaMappingOptions::makeNaiveMappingOptions()
          .outerScheduleFusionStrategy(tc::FusionStrategy::Preserve3Coincident)
          .outerScheduleAllowSkewing(false)
          .outerSchedulePositiveOrthant(true)
          .intraTileScheduleFusionStrategy(tc::FusionStrategy::Max)
          .intraTileScheduleAllowSkewing(false)
          .intraTileSchedulePositiveOrthant(true)
          .tile(28, 14, 7, 32, 28, 256)
          .mapToBlocks(1)
          .mapToThreads(7, 28)
          .unroll(64)
          .tileImperfectlyNested(false)
          .useSharedMemory(true)
          .usePrivateMemory(false)
          .unrollCopyShared(true)
          .matchLibraryCalls(true);

  // Tiling by 28 (problem size after scheduling) and setting
  // threadIdx.y to 28 seems to work fine.
  Check(options
            //                tile for y
            //                   ~~
            .tile(28, 14, 7, 32, 28, 256)
            //              map y
            //               ~~
            .mapToThreads(7, 28));
}

TEST_F(TensorDot_32_512_8_2_28_28, Reduction1) {
  Init();
  auto options =
      tc::CudaMappingOptions::makeNaiveMappingOptions()
          .outerScheduleFusionStrategy(tc::FusionStrategy::Preserve3Coincident)
          .outerScheduleAllowSkewing(false)
          .outerSchedulePositiveOrthant(true)
          .intraTileScheduleFusionStrategy(tc::FusionStrategy::Max)
          .intraTileScheduleAllowSkewing(false)
          .intraTileSchedulePositiveOrthant(true)
          .mapToBlocks(1)
          .tileImperfectlyNested(false)
          .useSharedMemory(true)
          .usePrivateMemory(false)
          .unrollCopyShared(true)
          .matchLibraryCalls(true);

  Check(options
            //                tile for y
            //                   ~~
            .tile(28, 14, 7, 32, 28, 256)
            //              map y
            //               ~~
            .mapToThreads(7, 28));
}

TEST_F(TensorDot_32_512_8_2_28_28, Reduction2) {
  Init();
  auto options =
      tc::CudaMappingOptions::makeNaiveMappingOptions()
          .outerScheduleFusionStrategy(tc::FusionStrategy::Preserve3Coincident)
          .outerScheduleAllowSkewing(false)
          .outerSchedulePositiveOrthant(true)
          .intraTileScheduleFusionStrategy(tc::FusionStrategy::Max)
          .intraTileScheduleAllowSkewing(false)
          .intraTileSchedulePositiveOrthant(true)
          .mapToBlocks(1)
          .tileImperfectlyNested(false)
          .useSharedMemory(true)
          .usePrivateMemory(false)
          .unrollCopyShared(true)
          .matchLibraryCalls(true);

  Check(options
            //                tile for y
            //                   ~
            .tile(28, 14, 7, 32, 9, 256)
            //               map y
            //                ~~
            .mapToThreads(64, 10));
}

TEST_F(TensorDot_32_512_8_2_28_28, Reduction3) {
  Init();
  auto options =
      tc::CudaMappingOptions::makeNaiveMappingOptions()
          .outerScheduleFusionStrategy(tc::FusionStrategy::Preserve3Coincident)
          .outerScheduleAllowSkewing(false)
          .outerSchedulePositiveOrthant(true)
          .intraTileScheduleFusionStrategy(tc::FusionStrategy::Max)
          .intraTileScheduleAllowSkewing(false)
          .intraTileSchedulePositiveOrthant(true)
          .mapToBlocks(256, 256, 256)
          .tileImperfectlyNested(false)
          .useSharedMemory(true)
          .usePrivateMemory(false)
          .unrollCopyShared(true)
          .matchLibraryCalls(true);

  Check(options
            //                tile for y
            //                   ~
            .tile(28, 14, 7, 32, 28, 256)
            //              map y
            //               ~~
            .mapToThreads(7, 32));
}

TEST_F(TensorDot_32_512_8_2_28_28, FormerSharedIllegalAddress) {
  Init();
  auto options =
      tc::CudaMappingOptions::makeNaiveMappingOptions()
          .outerScheduleFusionStrategy(tc::FusionStrategy::Preserve3Coincident)
          .outerScheduleAllowSkewing(false)
          .outerSchedulePositiveOrthant(true)
          .intraTileScheduleFusionStrategy(tc::FusionStrategy::Min)
          .intraTileScheduleAllowSkewing(false)
          .intraTileSchedulePositiveOrthant(true)
          .tile(14, 4, 28, 7, 1)
          .mapToThreads(256)
          .mapToBlocks(7, 4, 256)
          .unroll(16)
          .tileImperfectlyNested(false)
          .useSharedMemory(true)
          .usePrivateMemory(true)
          .unrollCopyShared(true)
          .matchLibraryCalls(false);
  Check(options);
}

TEST_F(TensorDot_32_512_8_2_28_28, NoUnroll) {
  Init();
  auto options =
      tc::CudaMappingOptions::makeNaiveMappingOptions()
          .outerScheduleFusionStrategy(tc::FusionStrategy::Preserve3Coincident)
          .outerScheduleAllowSkewing(false)
          .outerSchedulePositiveOrthant(true)
          .intraTileScheduleFusionStrategy(tc::FusionStrategy::Min)
          .intraTileScheduleAllowSkewing(false)
          .intraTileSchedulePositiveOrthant(true)
          .tile(1, 7)
          .mapToThreads(4, 16, 4)
          .mapToBlocks(14, 14, 128)
          .unroll(256)
          .tileImperfectlyNested(false)
          .useSharedMemory(true)
          .usePrivateMemory(true)
          .unrollCopyShared(false)
          .matchLibraryCalls(false);
  Check(options, false);
}

// Reduction init separation was creating empty filters when there was no init
// statement.  Tightening was choking on such empty filters.
TEST_F(TensorDot_32_512_8_2_28_28, EmptyLeaf) {
  Init();
  auto options = tc::CudaMappingOptions::makeNaiveMappingOptions()
                     .outerScheduleFusionStrategy(tc::FusionStrategy::Min)
                     .outerScheduleAllowSkewing(false)
                     .outerSchedulePositiveOrthant(true)
                     .intraTileScheduleFusionStrategy(tc::FusionStrategy::Min)
                     .intraTileScheduleAllowSkewing(false)
                     .intraTileSchedulePositiveOrthant(true)
                     .tile(256, 64)
                     .mapToThreads(32)
                     .mapToBlocks(128, 14)
                     .unroll(8)
                     .tileImperfectlyNested(false)
                     .useSharedMemory(false)
                     .usePrivateMemory(false)
                     .unrollCopyShared(false)
                     .matchLibraryCalls(true);
  // Just compiling is enough for the test, run results in catastrophically
  // bad perf so skip it
  Check(options, false);
}

TEST_F(TensorDot_32_512_8_2_28_28, FormerIllegalAccess) {
  Init();
  auto options =
      tc::CudaMappingOptions::makeNaiveMappingOptions()
          .outerScheduleFusionStrategy(tc::FusionStrategy::Preserve3Coincident)
          .outerScheduleAllowSkewing(false)
          .outerSchedulePositiveOrthant(true)
          .intraTileScheduleFusionStrategy(
              tc::FusionStrategy::Preserve3Coincident)
          .intraTileScheduleAllowSkewing(false)
          .intraTileSchedulePositiveOrthant(true)
          .tile(1, 7, 14)
          .mapToThreads(28, 16)
          .mapToBlocks(128, 16, 28)
          .unroll(64)
          .tileImperfectlyNested(false)
          .useSharedMemory(true)
          .usePrivateMemory(false)
          .unrollCopyShared(true)
          .matchLibraryCalls(false);
  Check(options);
}

////////////////////////////////////////////////////////////////////////////////
// GroupConvolution bugs found during autotuning
////////////////////////////////////////////////////////////////////////////////
struct GroupConvolution_32_32_4_4_56_56_3_3 : public ::testing::Test {
  std::vector<at::Tensor> inputs;
  std::vector<at::Tensor> outputsNaive;
  bool inited = false;

  void Init() {
    if (inited) {
      return;
    }
    inited = true;

    at::Tensor I = at::CUDA(at::kFloat).rand({32, 32, 4, 56, 56});
    at::Tensor W = at::CUDA(at::kFloat).rand({32, 4, 4, 3, 3});
    at::Tensor B = at::CUDA(at::kFloat).rand({32, 4});
    inputs = std::vector<at::Tensor>{I, W, B};

    // Build naive options baseline to check correctness
    // Make naive compile first to better see debug spew
    auto TC = std::string(R"TC(
def group_convolution_naive(float(N,G,C,H,W) I, float(G,F,C,KH,KW) W_, float(G,F) B)
-> (O)
{
    O(n, g, f, h, w) +=!
        I(n, g, r_c, h + r_kh, w + r_kw) * W_(g, f, r_c, r_kh, r_kw)
    O(n, g, f, h, w)  = O(n, g, f, h, w) + B(g, f)
}
)TC");

    // If running cuda-gdb only run on test code, not reference: things are
    // slow in this mode
    if (!FLAGS_debug_cuda) {
      auto mappingOptions = tc::CudaMappingOptions::makeNaiveMappingOptions();
      tc::ATenCompilationUnit<tc::CudaTcExecutor> atCompl;
      atCompl.define(TC);
      auto handle =
          atCompl.compile("group_convolution_naive", inputs, mappingOptions);
      atCompl.run("group_convolution_naive", inputs, outputsNaive, handle);
    }
  }

  std::vector<at::Tensor> Check(
      const tc::CudaMappingOptions& mappingOptions,
      bool runCuda = true) {
    auto name = makeUniqueName("group_convolution");
    auto TC = std::string("def ") + name +
        std::string(
                  R"TC((float(N,G,C,H,W) I, float(G,F,C,KH,KW) W_, float(G,F) B)
-> (O)
{
    O(n, g, f, h, w) +=!
        I(n, g, r_c, h + r_kh, w + r_kw) * W_(g, f, r_c, r_kh, r_kw)
    O(n, g, f, h, w) = O(n, g, f, h, w) + B(g, f)
}
)TC");

    std::vector<at::Tensor> outputs;
    tc::ATenCompilationUnit<tc::CudaTcExecutor> atCompl;
    atCompl.define(TC);
    auto handle = atCompl.compile(name, inputs, mappingOptions);
    if (runCuda) {
      atCompl.run(name, inputs, outputs, handle);
      if (!FLAGS_debug_cuda) {
        checkRtol(outputsNaive[0].sub(outputs[0]), inputs, 3 * 3 * 4, 5e-7);
      }
    }
    return outputs;
  }
};

TEST_F(GroupConvolution_32_32_4_4_56_56_3_3, FormerSharedIllegalAddress) {
  Init();
  auto options = tc::CudaMappingOptions::makeNaiveMappingOptions()
                     .outerScheduleFusionStrategy(tc::FusionStrategy::Max)
                     .outerScheduleAllowSkewing(false)
                     .outerSchedulePositiveOrthant(true)
                     .intraTileScheduleFusionStrategy(tc::FusionStrategy::Min)
                     .intraTileScheduleAllowSkewing(false)
                     .intraTileSchedulePositiveOrthant(true)
                     .tile(14, 1, 1, 1, 1)
                     .mapToThreads(14, 28)
                     .mapToBlocks(32, 56, 1)
                     .unroll(1)
                     .tileImperfectlyNested(false)
                     .useSharedMemory(true)
                     .usePrivateMemory(false)
                     .unrollCopyShared(true)
                     .matchLibraryCalls(false);
  Check(options);
}

// Autotuner says isl_scheduler.c:5529: unable to carry dependences but cannot
// repro standalone. Still keeping the unit test.
TEST_F(
    GroupConvolution_32_32_4_4_56_56_3_3,
    FakeIslSchedulerUnableToCarryDependences) {
  Init();
  auto options = tc::CudaMappingOptions::makeNaiveMappingOptions()
                     .outerScheduleFusionStrategy(tc::FusionStrategy::Max)
                     .outerScheduleAllowSkewing(false)
                     .outerSchedulePositiveOrthant(true)
                     .intraTileScheduleFusionStrategy(
                         tc::FusionStrategy::Preserve3Coincident)
                     .intraTileScheduleAllowSkewing(false)
                     .intraTileSchedulePositiveOrthant(true)
                     .tile(14, 8, 1, 64, 7, 4)
                     .mapToThreads(32)
                     .mapToBlocks(8, 32)
                     .unroll(1)
                     .tileImperfectlyNested(false)
                     .useSharedMemory(true)
                     .usePrivateMemory(true)
                     .unrollCopyShared(false)
                     .matchLibraryCalls(false);
  Check(options);
}

////////////////////////////////////////////////////////////////////////////////
// C3 bugs found during autotuning
////////////////////////////////////////////////////////////////////////////////
struct C3_128_1000_1024 : public ::testing::Test {
  std::vector<at::Tensor> inputs;
  std::vector<at::Tensor> outputsNaive;
  bool inited = false;

  void Init() {
    if (inited) {
      return;
    }
    inited = true;

    at::Tensor I = at::CUDA(at::kFloat).rand({128, 1000});
    at::Tensor W = at::CUDA(at::kFloat).rand({1024, 1000});
    inputs = std::vector<at::Tensor>{I, W};

    // Build naive options baseline to check correctness
    // Make naive compile first to better see debug spew
    auto TC = std::string(R"TC(
def _C3_naive(float(B,WX) I, float(WY, WX) W) -> (C3) {
    C3(b, wy) +=! I(b, r_wx) * W(wy, r_wx)
}
)TC");

    // If running cuda-gdb only run on test code, not reference: things are
    // slow in this mode
    if (!FLAGS_debug_cuda) {
      auto mappingOptions = tc::CudaMappingOptions::makeNaiveMappingOptions();
      tc::ATenCompilationUnit<tc::CudaTcExecutor> atCompl;
      atCompl.define(TC);
      auto handle = atCompl.compile("_C3_naive", inputs, mappingOptions);
      atCompl.run("_C3_naive", inputs, outputsNaive, handle);
    }
  }

  std::vector<at::Tensor> Check(
      const tc::CudaMappingOptions& mappingOptions,
      bool runCuda = true) {
    auto name = makeUniqueName("_C3");
    auto TC = std::string("def ") + name +
        std::string(
                  R"TC((float(B,WX) I, float(WY, WX) W) -> (C3) {
    C3(b, wy) +=! I(b, r_wx) * W(wy, r_wx)
}
)TC");

    std::vector<at::Tensor> outputs;
    tc::ATenCompilationUnit<tc::CudaTcExecutor> atCompl;
    atCompl.define(TC);
    auto handle = atCompl.compile(name, inputs, mappingOptions);
    if (runCuda) {
      atCompl.run(name, inputs, outputs, handle);
      if (!FLAGS_debug_cuda) {
        checkRtol(outputsNaive[0].sub(outputs[0]), inputs, 1000, 1e-6);
      }
    }
    return outputs;
  }
};

// This used to complain about INVALID_PTX but really was using more shared
// memory than available.  Plan a conservative buffer when mixing reductions
// with shared memory.
TEST_F(C3_128_1000_1024, InvalidPtx) {
  Init();
  auto options = tc::CudaMappingOptions::makeNaiveMappingOptions()
                     .outerScheduleFusionStrategy(tc::FusionStrategy::Max)
                     .outerScheduleAllowSkewing(false)
                     .outerSchedulePositiveOrthant(true)
                     .intraTileScheduleFusionStrategy(tc::FusionStrategy::Min)
                     .intraTileScheduleAllowSkewing(false)
                     .intraTileSchedulePositiveOrthant(true)
                     .tile(8, 500)
                     .mapToThreads(125)
                     .mapToBlocks(128, 1000, 500)
                     .unroll(256)
                     .tileImperfectlyNested(false)
                     .useSharedMemory(true)
                     .usePrivateMemory(true)
                     .unrollCopyShared(true)
                     .matchLibraryCalls(true);
  Check(options);
}

TEST_F(C3_128_1000_1024, IllegalAccess) {
  Init();
  auto options = tc::CudaMappingOptions::makeNaiveMappingOptions()
                     .outerScheduleFusionStrategy(tc::FusionStrategy::Max)
                     .outerScheduleAllowSkewing(false)
                     .outerSchedulePositiveOrthant(true)
                     .intraTileScheduleFusionStrategy(
                         tc::FusionStrategy::Preserve3Coincident)
                     .intraTileScheduleAllowSkewing(false)
                     .intraTileSchedulePositiveOrthant(true)
                     .tile(32, 32, 250)
                     .mapToThreads(4, 32)
                     .mapToBlocks(63, 128)
                     .unroll(16)
                     .tileImperfectlyNested(false)
                     .useSharedMemory(true)
                     .usePrivateMemory(true)
                     .unrollCopyShared(false)
                     .matchLibraryCalls(false);
  // Supposedly nvcc bug, appears intermittently with 8.0.61, disappears when
  // compiled with 9.*.
  int nvrtcVersionMajor;
  int nvrtcVersionMinor;
  TC_NVRTC_CHECK(nvrtcVersion(&nvrtcVersionMajor, &nvrtcVersionMinor));
  if (nvrtcVersionMajor > 8) {
    Check(options);
  }
}

////////////////////////////////////////////////////////////////////////////////
// TMM bugs found during autotuning
////////////////////////////////////////////////////////////////////////////////
struct TMM_128_1024_1024 : public ::testing::Test {
  std::vector<at::Tensor> inputs;
  std::vector<at::Tensor> outputsNaive;
  bool inited = false;

  void Init() {
    if (inited) {
      return;
    }
    inited = true;

    at::Tensor I = at::CUDA(at::kFloat).rand({128, 1024});
    at::Tensor W = at::CUDA(at::kFloat).rand({1024, 1024});
    inputs = std::vector<at::Tensor>{I, W};

    // Build naive options baseline to check correctness
    // Make naive compile first to better see debug spew
    auto TC = std::string(R"TC(
def tmm_naive(float(B, X) I, float(Y, X) W) -> (O) {
    O(b, y) +=! I(b, r_x) * W(y, r_x)
}
)TC");

    // If running cuda-gdb only run on test code, not reference: things are
    // slow in this mode
    if (!FLAGS_debug_cuda) {
      auto mappingOptions = tc::CudaMappingOptions::makeNaiveMappingOptions();
      tc::ATenCompilationUnit<tc::CudaTcExecutor> atCompl;
      atCompl.define(TC);
      auto handle = atCompl.compile("tmm_naive", inputs, mappingOptions);
      atCompl.run("tmm_naive", inputs, outputsNaive, handle);
    }
  }

  std::vector<at::Tensor> Check(
      const tc::CudaMappingOptions& mappingOptions,
      bool runCuda = true) {
    auto name = makeUniqueName("tmm");
    auto TC = std::string("def ") + name +
        std::string(
                  R"TC((float(B, X) I, float(Y, X) W) -> (O) {
    O(b, y) +=! I(b, r_x) * W(y, r_x)
}
)TC");

    std::vector<at::Tensor> outputs;
    tc::ATenCompilationUnit<tc::CudaTcExecutor> atCompl;
    atCompl.define(TC);
    auto handle = atCompl.compile(name, inputs, mappingOptions);
    if (runCuda) {
      atCompl.run(name, inputs, outputs, handle);
      if (!FLAGS_debug_cuda) {
        checkRtol(outputsNaive[0].sub(outputs[0]), inputs, 1024, 1e-6);
      }
    }
    return outputs;
  }
};

// For this one we need to relax the precision to 1e-6
TEST_F(TMM_128_1024_1024, TooStrictPrecisionAfterTuner) {
  Init();
  auto options = tc::CudaMappingOptions::makeNaiveMappingOptions()
                     .outerScheduleFusionStrategy(tc::FusionStrategy::Max)
                     .outerScheduleAllowSkewing(false)
                     .outerSchedulePositiveOrthant(true)
                     .intraTileScheduleFusionStrategy(tc::FusionStrategy::Max)
                     .intraTileScheduleAllowSkewing(false)
                     .intraTileSchedulePositiveOrthant(true)
                     .tile(2)
                     .mapToThreads(128)
                     .mapToBlocks(64)
                     .unroll(16)
                     .tileImperfectlyNested(false)
                     .useSharedMemory(false)
                     .usePrivateMemory(false)
                     .unrollCopyShared(true)
                     .matchLibraryCalls(true);
  Check(options);
}

// This exercises a former MappingFilter leaf with a union_set of > 1 spaces
TEST_F(TMM_128_1024_1024, Tightening) {
  Init();
  auto options = tc::CudaMappingOptions::makeNaiveMappingOptions()
                     .outerScheduleFusionStrategy(tc::FusionStrategy::Max)
                     .outerScheduleAllowSkewing(false)
                     .outerSchedulePositiveOrthant(true)
                     .intraTileScheduleFusionStrategy(tc::FusionStrategy::Min)
                     .intraTileScheduleAllowSkewing(false)
                     .intraTileSchedulePositiveOrthant(true)
                     .tile(1, 1)
                     .mapToThreads(32)
                     .mapToBlocks(128, 128)
                     .unroll(64)
                     .tileImperfectlyNested(false)
                     .useSharedMemory(true)
                     .usePrivateMemory(true)
                     .unrollCopyShared(true)
                     .matchLibraryCalls(true);
  Check(options);
}

TEST(LayerNorm, ReferenceBelongsToTwoGroups) {
  at::Tensor mat1 = at::CUDA(at::kFloat).rand({7, 32, 64});
  std::vector<at::Tensor> inputs = {mat1};
  std::vector<at::Tensor> outputs;

  static constexpr auto TC = R"TC(
def layernorm(float(T, B, C) I) -> (O, mean, centered, var) {
        mean(t, b)    +=! I(t, b, r_c) / C
    centered(t, b, c)  =  I(t, b,   c) - mean(t, b)

    var(t, b) +=! centered(t, b, r_c) * centered(t, b, r_c)
    var(t, b)  =       var(t, b) / C
    O(t, b, c) =  centered(t, b, c) / rsqrt(var(t, b))
}
  )TC";
  auto options = tc::CudaMappingOptions::makeNaiveMappingOptions()
                     .outerScheduleFusionStrategy(tc::FusionStrategy::Max)
                     .outerScheduleAllowSkewing(false)
                     .outerSchedulePositiveOrthant(true)
                     .intraTileScheduleFusionStrategy(
                         tc::FusionStrategy::Preserve3Coincident)
                     .intraTileScheduleAllowSkewing(false)
                     .intraTileSchedulePositiveOrthant(true)
                     .tile(4, 16)
                     .mapToThreads(64)
                     .mapToBlocks(128, 32, 4)
                     .unroll(2)
                     .tileImperfectlyNested(false)
                     .useSharedMemory(true)
                     .usePrivateMemory(true)
                     .unrollCopyShared(false)
                     .matchLibraryCalls(false);

  tc::ATenCompilationUnit<tc::CudaTcExecutor> atCompl;
  atCompl.define(TC);
  // Expecting this to compile without dying.
  atCompl.compile("layernorm", inputs, options);
}

// This case was observed when running the autotuner on example_MLP_model
// (#200).  It calls code generation on a schedule tree containing a
// disjunctive filter, which results in expression with more than one disjunct
// that was not handed properly.
// TODO: the disjunctive filter in the schedule is unexpected and its origin
// should be identified and explained.
TEST(TMM_128_1024_1000, DisjunctiveFilter) {
  at::Tensor I = at::CUDA(at::kFloat).rand({128, 1024});
  at::Tensor W = at::CUDA(at::kFloat).rand({1000, 1024});
  std::vector<at::Tensor> inputs = {I, W};
  std::vector<at::Tensor> outputs;

  auto TC = std::string(R"TC(
def tmm_naive(float(B, X) I, float(Y, X) W) -> (O) {
    O(b, y) +=! I(b, r_x) * W(y, r_x)
}
)TC");
  auto options =
      tc::CudaMappingOptions::makeNaiveMappingOptions()
          .outerScheduleFusionStrategy(tc::FusionStrategy::Preserve3Coincident)
          .outerScheduleAllowSkewing(false)
          .outerSchedulePositiveOrthant(true)
          .intraTileScheduleFusionStrategy(tc::FusionStrategy::Min)
          .intraTileScheduleAllowSkewing(false)
          .intraTileSchedulePositiveOrthant(true)
          .tile(1, 32, 63)
          .mapToThreads(2, 32)
          .mapToBlocks(64, 128, 1024)
          .unroll(128)
          .tileImperfectlyNested(false)
          .useSharedMemory(false)
          .usePrivateMemory(false)
          .unrollCopyShared(false)
          .matchLibraryCalls(true);

  tc::ATenCompilationUnit<tc::CudaTcExecutor> atCompl;
  atCompl.define(TC);
  // Expecting this to compile without dying.
  atCompl.compile("tmm_naive", inputs, options);
}

TEST(Halide2Isl, MinInUpperBound) {
  at::Tensor mat1 = at::CUDA(at::kFloat).rand({1, 100, 184, 184});
  at::Tensor mat1_pad = at::CUDA(at::kFloat).rand({1, 100, 186, 186});
  at::Tensor mat2 = at::CUDA(at::kFloat).rand({3, 3});
  std::vector<at::Tensor> inputs = {mat1, mat1_pad, mat2};

  static constexpr auto TC = R"TC(
def graph2(float(N, C, H, W) I, float(N, C, R, T) J, float(KH, KW) W1) -> (O, Out) {
      O(n, c, h, w) +=! J(n,  c, h + r_kh, w + r_kw) * W1(r_kh, r_kw)
    Out(c0, c1)     +=! I(n, c0,        h,        w) *  O(   n,   c1, h, w)
}
  )TC";
  auto options = tc::CudaMappingOptions::makeNaiveMappingOptions();

  tc::ATenCompilationUnit<tc::CudaTcExecutor> atCompl;
  atCompl.define(TC);
  atCompl.compile("graph2", inputs, options);
}

// Check that nested expressions are properly formatted.
// In particular, as soon as the tensor size X is larger than the tile size,
// the expression for "xp" is a sum of multiple loop iterators
// in the generated code.  Parentheses need to be placed around
// these expressions to ensure the end result is, say, "-(c1 + c3)"
// rather than "-c1 + c3".
// The actual convolution is one where the output is equal to the input.
TEST(Convolution, NestedExpressions) {
  auto convolution = "convolution";
  auto TC = std::string(R"TC(
  def convolution(float(X) A, float(Xp) K) -> (B) {
      B(x) +=! A(xp) * K(X - 1 + x - xp) where xp in 0:X
  }
  )TC");
  int X = 33;
  at::Tensor A = at::CUDA(at::kFloat).zeros({X});
  at::Tensor K = at::CUDA(at::kFloat).zeros({2 * X - 1});
  A[10] = 1;
  K[X - 1] = 1;
  std::vector<at::Tensor> inputs = {A, K};
  std::vector<at::Tensor> outputs;
  tc::ATenCompilationUnit<tc::CudaTcExecutor> cu;
  cu.define(TC);
  auto options = tc::CudaMappingOptions::makeNaiveMappingOptions();
  auto handle = cu.compile(convolution, inputs, options);
  cu.run(convolution, inputs, outputs, handle);
  auto B = outputs[0];
  CHECK_EQ(at::Scalar(B[10]).toFloat(), 1);
}

int main(int argc, char** argv) {
  ::testing::InitGoogleTest(&argc, argv);
  ::gflags::ParseCommandLineFlags(&argc, &argv, true);
  ::google::InitGoogleLogging(argv[0]);
  setAtenSeed(tc::initRandomSeed(), at::Backend::CUDA);
  return RUN_ALL_TESTS();
}
