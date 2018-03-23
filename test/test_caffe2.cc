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
#include <iostream>
#include <string>
#include <vector>

#include <gtest/gtest.h>

#include "tc/c2/context.h"
#include "tc/core/cuda/cuda_mapping_options.h"
#include "tc/core/flags.h"

#include "isl_cli_strategy.h"
#include "test_harness.h"

using namespace std;
using namespace caffe2;
using namespace tc;

struct Caffe2Test : public ::testing::Test {
  static constexpr uint32_t B = 32, M = 200, N = 64, K = 64;
  static constexpr uint32_t O = 32, P = 2, Q = 23, R = 31;
  static constexpr uint32_t vB = 100, vD = 64, vL = 50, vE = 1e5;
  static constexpr uint32_t vL1 = 50, vE1 = 1e5, vL2 = 50, vE2 = 1e5;
  static constexpr uint32_t NN = 32, G = 32, C = 4, F = 4, W = 56, H = 56;
  static constexpr uint32_t KW = 3, KH = 3, SW = 2, SH = 2;

  Caffe2Test() {}
};

struct Caffe2CopyTest : public Caffe2Test {
  using Caffe2Test::M;
  using Caffe2Test::N;
  using Caffe2Test::P;
  using Caffe2Test::Q;
  using Caffe2Test::R;

  Argument strategyArg;
  Caffe2CopyTest() {
    CudaMappingOptions options = tc::makeBaseCliStrategy()
                                     .tile(32, 32)
                                     .mapToThreads({32, 32})
                                     .mapToBlocks({32, 32, 32});
    strategyArg = MakeArgument<string>(
        "mappingOptions",
        tc::makeCliStrategy(options).toProtobufSerializedString());
  }
};

TEST_F(Caffe2CopyTest, TcCopyOp_Default1D) {
  auto init_ws = [&](Workspace& w) {
    auto AddInput =
        TestHarness::AddDeterministicallyRandomInput<float, CUDAContext>;
    AddInput(w, {M}, "I");
  };
  OperatorDef def =
      TestHarness::ConfigureCUDA("TcCopyOp", {"I"}, {"O"}, {strategyArg});
  TestHarness::BasicCorrectnessTest(def, init_ws);
}

TEST_F(Caffe2CopyTest, DISABLED_TcCopyOp_Gradient1D) {
  auto init_ws = [&](Workspace& w) {
    auto AddInput =
        TestHarness::AddDeterministicallyRandomInput<float, CUDAContext>;
    AddInput(w, {M}, "I");
    AddInput(w, {M}, "O_grad");
  };
  OperatorDef def =
      TestHarness::ConfigureCUDA("TcCopyOp", {"I"}, {"O"}, {strategyArg});
  TestHarness::BasicGradientCorrectnessTest(def, init_ws);
}

TEST_F(Caffe2CopyTest, TcCopyOp_Default2D) {
  auto init_ws = [&](Workspace& w) {
    auto AddInput =
        TestHarness::AddDeterministicallyRandomInput<float, CUDAContext>;
    AddInput(w, {M, N}, "I");
  };
  OperatorDef def =
      TestHarness::ConfigureCUDA("TcCopyOp", {"I"}, {"O"}, {strategyArg});
  TestHarness::BasicCorrectnessTest(def, init_ws);
}

TEST_F(Caffe2CopyTest, DISABLED_TcCopyOp_Gradient2D) {
  auto init_ws = [&](Workspace& w) {
    auto AddInput =
        TestHarness::AddDeterministicallyRandomInput<float, CUDAContext>;
    AddInput(w, {M, N}, "I");
    AddInput(w, {M, N}, "O_grad");
  };
  OperatorDef def =
      TestHarness::ConfigureCUDA("TcCopyOp", {"I"}, {"O"}, {strategyArg});
  TestHarness::BasicGradientCorrectnessTest(def, init_ws);
}

TEST_F(Caffe2CopyTest, TcCopyOp_Default3D) {
  auto init_ws = [&](Workspace& w) {
    auto AddInput =
        TestHarness::AddDeterministicallyRandomInput<float, CUDAContext>;
    AddInput(w, {M, N, P}, "I");
  };
  OperatorDef def =
      TestHarness::ConfigureCUDA("TcCopyOp", {"I"}, {"O"}, {strategyArg});
  TestHarness::BasicCorrectnessTest(def, init_ws);
}

TEST_F(Caffe2CopyTest, DISABLED_TcCopyOp_Gradient3D) {
  auto init_ws = [&](Workspace& w) {
    auto AddInput =
        TestHarness::AddDeterministicallyRandomInput<float, CUDAContext>;
    AddInput(w, {M, N, P}, "I");
    AddInput(w, {M, N, P}, "O_grad");
  };
  OperatorDef def =
      TestHarness::ConfigureCUDA("TcCopyOp", {"I"}, {"O"}, {strategyArg});
  TestHarness::BasicGradientCorrectnessTest(def, init_ws);
}

TEST_F(Caffe2CopyTest, TcCopyOp_Default4D) {
  auto init_ws = [&](Workspace& w) {
    auto AddInput =
        TestHarness::AddDeterministicallyRandomInput<float, CUDAContext>;
    AddInput(w, {M, N, P, Q}, "I");
  };
  OperatorDef def =
      TestHarness::ConfigureCUDA("TcCopyOp", {"I"}, {"O"}, {strategyArg});
  TestHarness::BasicCorrectnessTest(def, init_ws);
}

TEST_F(Caffe2CopyTest, DISABLED_TcCopyOp_Gradient4D) {
  auto init_ws = [&](Workspace& w) {
    auto AddInput =
        TestHarness::AddDeterministicallyRandomInput<float, CUDAContext>;
    AddInput(w, {M, N, P, Q}, "I");
    AddInput(w, {M, N, P, Q}, "O_grad");
  };
  OperatorDef def =
      TestHarness::ConfigureCUDA("TcCopyOp", {"I"}, {"O"}, {strategyArg});
  TestHarness::BasicGradientCorrectnessTest(def, init_ws);
}

TEST_F(Caffe2CopyTest, TcCopyOp_Default5D) {
  auto init_ws = [&](Workspace& w) {
    auto AddInput =
        TestHarness::AddDeterministicallyRandomInput<float, CUDAContext>;
    AddInput(w, {M, N, P, Q, R}, "I");
  };
  OperatorDef def =
      TestHarness::ConfigureCUDA("TcCopyOp", {"I"}, {"O"}, {strategyArg});
  TestHarness::BasicCorrectnessTest(def, init_ws);
}

TEST_F(Caffe2CopyTest, DISABLED_TcCopyOp_Gradient5D) {
  auto init_ws = [&](Workspace& w) {
    auto AddInput =
        TestHarness::AddDeterministicallyRandomInput<float, CUDAContext>;
    AddInput(w, {M, N, P, Q, R}, "I");
    AddInput(w, {M, N, P, Q, R}, "O_grad");
  };
  OperatorDef def =
      TestHarness::ConfigureCUDA("TcCopyOp", {"I"}, {"O"}, {strategyArg});
  TestHarness::BasicGradientCorrectnessTest(def, init_ws);
}

TEST_F(Caffe2Test, TcMatMulOp) {
  auto init_ws = [&](Workspace& w) {
    auto AddInput =
        TestHarness::AddDeterministicallyRandomInput<float, CUDAContext>;
    AddInput(w, {M, K}, "I");
    AddInput(w, {K, N}, "W");
  };

  CudaMappingOptions options = tc::makeBaseCliStrategy()
                                   .tile(32, 32, 32)
                                   .mapToThreads({4, 32})
                                   .mapToBlocks({32, 32, 32});
  Argument strategyArg = MakeArgument<string>(
      "mappingOptions",
      tc::makeCliStrategy(options).toProtobufSerializedString());
  OperatorDef def = TestHarness::ConfigureCUDA(
      "TcMatMulOp", {"I", "W"}, {"O"}, {strategyArg});
  TestHarness::BasicCorrectnessTest(def, init_ws, 1e-6);
}

TEST_F(Caffe2Test, DISABLED_TcMatMulOp_Gradient) {
  auto init_ws = [&](Workspace& w) {
    auto AddInput =
        TestHarness::AddDeterministicallyRandomInput<float, CUDAContext>;
    AddInput(w, {M, K}, "I");
    AddInput(w, {K, N}, "W");
    AddInput(w, {M, N}, "O_grad");
  };

  CudaMappingOptions options = tc::makeBaseCliStrategy()
                                   .tile(32, 32, 32)
                                   .mapToThreads({4, 32})
                                   .mapToBlocks({32, 32, 32});
  Argument strategyArg = MakeArgument<string>(
      "mappingOptions",
      tc::makeCliStrategy(options).toProtobufSerializedString());
  OperatorDef def = TestHarness::ConfigureCUDA(
      "TcMatMulOp", {"I", "W"}, {"O"}, {strategyArg});
  TestHarness::BasicGradientCorrectnessTest(def, init_ws);
}

TEST_F(Caffe2Test, TcLUTOp) {
  auto init_ws = [=](Workspace& w) {
    TestHarness::AddDeterministicallyRandomInput<float, CUDAContext>(
        w, {vE, vD}, "LUT");
    TestHarness::AddDeterministicallyRandomInputWithRange<int, CUDAContext>(
        w, {vB, vL}, "I", 0, vE - 1);

    TestHarness::AddConstInput<int, CUDAContext>(w, {vB}, vL, "__lengths");
  };

  CudaMappingOptions options =
      tc::makeBaseCliStrategy().tile(32, 32).mapToThreads({32, 32}).mapToBlocks(
          {32, 32, 32});
  Argument strategyArg = MakeArgument<string>(
      "mappingOptions",
      tc::makeCliStrategy(options)
          .useSharedMemory(false) // NYI: shared indirection
          .usePrivateMemory(false)
          .unrollCopyShared(false)
          .toProtobufSerializedString());
  OperatorDef def =
      TestHarness::ConfigureCUDA("TcLUTOp", {"LUT", "I"}, {"O"}, {strategyArg});
  TestHarness::BasicCorrectnessTest(def, init_ws, 1e-6);
}

TEST_F(Caffe2Test, Tc2LUTOp) {
  auto init_ws = [=](Workspace& w) {
    TestHarness::AddDeterministicallyRandomInput<float, CUDAContext>(
        w, {vE1, vD}, "LUT1");
    TestHarness::AddDeterministicallyRandomInputWithRange<int, CUDAContext>(
        w, {vB, vL1}, "IDX1", 0, vE1 - 1);
    TestHarness::AddDeterministicallyRandomInput<float, CUDAContext>(
        w, {vE2, vD}, "LUT2");
    TestHarness::AddDeterministicallyRandomInputWithRange<int, CUDAContext>(
        w, {vB, vL2}, "IDX2", 0, vE2 - 1);

    TestHarness::AddConstInput<int, CUDAContext>(w, {vB}, vL1, "__lengths1");
    TestHarness::AddConstInput<int, CUDAContext>(w, {vB}, vL2, "__lengths2");
  };

  CudaMappingOptions options =
      tc::makeBaseCliStrategy().tile(1, 32).mapToThreads({1, 32}).mapToBlocks(
          {100, 100});
  Argument strategyArg = MakeArgument<string>(
      "mappingOptions",
      tc::makeCliStrategy(options)
          .useSharedMemory(false) // NYI: shared indirection
          .usePrivateMemory(false)
          .unrollCopyShared(false)
          .toProtobufSerializedString());
  OperatorDef def = TestHarness::ConfigureCUDA(
      "Tc2LUTOp",
      {"LUT1", "IDX1", "LUT2", "IDX2"},
      {"O1", "O2"},
      {strategyArg});
  auto opTesterUPtr = TestHarness::BasicCorrectnessTest(def, init_ws, 1e-6);

  // TODO: FIXME
  // The code below is an example of injecting manually modified CUDA code in
  // the compilation cache.
  // In the presence of large lookup tables, TLB misses make data dependences
  // on the LUT content significantly higher latency in the innermost loop.
  // Here we prefetch the portion of LUT1 that we need in shared memory (we may
  // want to try L1/2 cache also, esp on Volta).
  // We don't prefetch both in shared memory because the cost becomes
  // prohibitive and  adversely impacts latency hiding due to reduced occupancy.
  //
  // Run like this:
  //   CUDA_VISIBLE_DEVICES=1 /usr/local/cuda/bin/nvprof --print-gpu-trace
  //   ./build/test/test_2lut --vE1=10000000 --vE2=10000000
  //
  // Results on Maxwell: 115us -> 60 us
  //

  //   std::string cudaSource = R"CUDA(
  // __global__ void lut_100_64_50_vE1_50_vE2(int B, int D, int L1, int E1, int
  // L2, int E2, float* __restrict__ O1, float* __restrict__ O2, const float*
  // __restrict__ LUT1, const int*
  std::string cudaSource = R"CUDA(
__global__ void islanon_50_50_100_64_vE1_vE2(int B, int D, int L1, int E1, int L2, int E2, float* __restrict__ O1, float* __restrict__ O2, const float* __restrict__ LUT1, const int* __restrict__ Idx1, const float* __restrict__ LUT2, const int* __restrict__ Idx2)
{
    int b0 = blockIdx.y, b1 = blockIdx.x;
    int t0 = threadIdx.y, t1 = threadIdx.x;
    __shared__ int shared_Idx1[1][51];
    __shared__ int shared_Idx2[1][51];
    __shared__ float shared_LUT1[50][32];
    // __shared__ float shared_LUT2[50][32];
    float private_O1[1][1];
    float private_O2[1][1];

    {
      shared_Idx1[0][t1] = Idx1[b0 * 50 + t1];
      if (t1 <= 17)
        shared_Idx1[0][t1 + 32] = Idx1[b0 * 50 + (t1 + 32)];
      shared_Idx2[0][t1] = Idx2[b0 * 50 + t1];
      if (t1 <= 17)
        shared_Idx2[0][t1 + 32] = Idx2[b0 * 50 + (t1 + 32)];
      __syncthreads();
      for (int c4 = 0; c4 <= 49; c4 += 1) {
        shared_LUT1[c4][t1] = LUT1[(0 * vE1 + shared_Idx1[0][c4]) * 64 + (32 * b1 + t1)];
        // shared_LUT2[c4][t1] = LUT2[(0 * vE2 + shared_Idx2[0][c4]) * 64 + (32 * b1 + t1)];
      }
      __syncthreads();
      private_O1[0][0] = 0.00000f;
      private_O2[0][0] = 0.00000f;
      for (int c4 = 0; c4 <= 49; c4 += 1) {
        private_O1[0][0] = (private_O1[0][0] + shared_LUT1[c4][t1]);
        // private_O1[0][0] = (private_O1[0][0] + LUT1[(0 * vE1 + shared_Idx1[0][c4]) * 64 + (32 * b1 + t1)]);
        // private_O2[0][0] = (private_O2[0][0] + shared_LUT2[c4][t1]);
        private_O2[0][0] = (private_O2[0][0] + LUT2[(0 * vE2 + shared_Idx2[0][c4]) * 64 + (32 * b1 + t1)]);
      }
      __syncthreads();
      O2[b0 * 64 + (32 * b1 + t1)] = private_O2[0][0];
      O1[b0 * 64 + (32 * b1 + t1)] = private_O1[0][0];
      __syncthreads();
    }
}
)CUDA";

  auto E1 = vE1;
  auto E2 = vE2;
  for (auto p : {std::pair<std::string, uint32_t>("vE1", E1),
                 std::pair<std::string, uint32_t>("vE2", E2)}) {
    auto pos = std::string::npos;
    while ((pos = cudaSource.find(p.first)) != std::string::npos) {
      cudaSource.replace(pos, 3, std::to_string(p.second));
    }
  }
  // auto pOp =
  //     static_cast<TcOp<float, CUDAContext>*>(opTesterUPtr->op_test.get());
  // TODO(ttheodor) FIXME
  // pOp->InjectVersionInCache(
  //     cudaSource, tc::GridDims{2, 100, 1}, tc::BlockDims{32, 1, 1});
  // pOp->Run();
  // try all output of all ops in original net as they are preserved
  // for (auto out : def.output()) {
  //   TestHarness::CheckEqual(
  //       opTesterUPtr->w_ref, opTesterUPtr->w_test, out, 0.01);
  // }
}

TEST_F(Caffe2Test, TcFCReluOp) {
  auto init_ws = [&](Workspace& w) {
    auto AddInput =
        TestHarness::AddDeterministicallyRandomInput<float, CUDAContext>;
    AddInput(w, {B, M}, "I");
    AddInput(w, {N, M}, "W1");
    AddInput(w, {N}, "B1");
  };

  CudaMappingOptions options = tc::makeBaseCliStrategy()
                                   .tile(32, 32, 32)
                                   .mapToThreads({32, 4})
                                   .mapToBlocks({32, 32});
  Argument strategyArg = MakeArgument<string>(
      "mappingOptions",
      tc::makeCliStrategy(options).toProtobufSerializedString());
  OperatorDef def = TestHarness::ConfigureCUDA(
      "TcFCReluOp", {"I", "W1", "B1"}, {"O1"}, {strategyArg});
  TestHarness::BasicCorrectnessTest(def, init_ws, 1e-6);
}

TEST_F(Caffe2Test, Tc2FCReluOp) {
  auto init_ws = [&](Workspace& w) {
    auto AddInput =
        TestHarness::AddDeterministicallyRandomInput<float, CUDAContext>;
    AddInput(w, {B, M}, "I");
    AddInput(w, {N, M}, "W1");
    AddInput(w, {N}, "B1");
    AddInput(w, {O, N}, "W2");
    AddInput(w, {O}, "B2");
  };

  CudaMappingOptions options = tc::makeBaseCliStrategy()
                                   .scheduleFusionStrategy("Max")
                                   .tile(1)
                                   .mapToThreads({128})
                                   .mapToBlocks({128})
                                   .useSharedMemory(false)
                                   .usePrivateMemory(false);
  Argument strategyArg = MakeArgument<string>(
      "mappingOptions",
      tc::makeCliStrategy(options).toProtobufSerializedString());
  OperatorDef def = TestHarness::ConfigureCUDA(
      "Tc2FCReluOp",
      {"I", "W1", "B1", "W2", "B2"},
      {"O1", "O2"},
      {strategyArg});
  TestHarness::BasicCorrectnessTest(def, init_ws, 1e-6);
}

TEST_F(Caffe2Test, Tc3FCReluOp) {
  auto AddConstInput = TestHarness::AddConstInput<float, CUDAContext>;
  auto init_ws = [&](Workspace& w) {
    AddConstInput(w, vector<TIndex>{B, M}, 1., "I");
    AddConstInput(w, vector<TIndex>{N, M}, 1., "W1");
    AddConstInput(w, vector<TIndex>{N}, 1., "B1");
    AddConstInput(w, vector<TIndex>{O, N}, 1., "W2");
    AddConstInput(w, vector<TIndex>{O}, 1., "B2");
    AddConstInput(w, vector<TIndex>{P, O}, 1., "W3");
    AddConstInput(w, vector<TIndex>{P}, 1., "B3");
  };

  CudaMappingOptions options = tc::makeBaseCliStrategy()
                                   .scheduleFusionStrategy("Max")
                                   .tile(1)
                                   .mapToThreads({200})
                                   .mapToBlocks({32});
  Argument strategyArg = MakeArgument<string>(
      "mappingOptions",
      tc::makeCliStrategy(options).toProtobufSerializedString());
  OperatorDef def = TestHarness::ConfigureCUDA(
      "Tc3FCReluOp",
      {"I", "W1", "B1", "W2", "B2", "W3", "B3"},
      {"O1", "O2", "O3"},
      {strategyArg});
  TestHarness::BasicCorrectnessTest(def, init_ws, 1e-6);
}

TEST_F(Caffe2Test, Tc4FCReluOp) {
  auto AddConstInput = TestHarness::AddConstInput<float, CUDAContext>;
  auto init_ws = [&](Workspace& w) {
    AddConstInput(w, vector<TIndex>{B, M}, 1., "I");
    AddConstInput(w, vector<TIndex>{N, M}, 1., "W1");
    AddConstInput(w, vector<TIndex>{N}, 1., "B1");
    AddConstInput(w, vector<TIndex>{O, N}, 1., "W2");
    AddConstInput(w, vector<TIndex>{O}, 1., "B2");
    AddConstInput(w, vector<TIndex>{P, O}, 1., "W3");
    AddConstInput(w, vector<TIndex>{P}, 1., "B3");
    AddConstInput(w, vector<TIndex>{Q, P}, 1., "W4");
    AddConstInput(w, vector<TIndex>{Q}, 1., "B4");
  };

  CudaMappingOptions options = tc::makeBaseCliStrategy()
                                   .scheduleFusionStrategy("Max")
                                   .tile(1)
                                   .mapToThreads({128})
                                   .mapToBlocks({128});
  Argument strategyArg = MakeArgument<string>(
      "mappingOptions",
      tc::makeCliStrategy(options).toProtobufSerializedString());
  OperatorDef def = TestHarness::ConfigureCUDA(
      "Tc4FCReluOp",
      {"I", "W1", "B1", "W2", "B2", "W3", "B3", "W4", "B4"},
      {"O1", "O2", "O3", "O4"},
      {strategyArg});
  TestHarness::BasicCorrectnessTest(def, init_ws, 1e-6);
}

TEST_F(Caffe2Test, TcGroupConvolutionOp) {
  auto init_ws = [&](Workspace& w) {
    auto AddInput = TestHarness::AddConstInput<float, CUDAContext>;
    AddInput(w, vector<TIndex>{NN, G * C, W, H}, 1., "I");
    AddInput(w, vector<TIndex>{G * F, C, KW, KH}, 1., "W");
    AddInput(w, {G * F}, 1., "B");
  };

  CudaMappingOptions options = tc::makeBaseCliStrategy()
                                   .scheduleFusionStrategy("Max")
                                   .tile(1)
                                   .mapToThreads({128})
                                   .mapToBlocks({128});
  Argument strategyArg = MakeArgument<string>(
      "mappingOptions",
      tc::makeCliStrategy(options).toProtobufSerializedString());

  // NetDef defined in caffe2.proto, see caffe2/build/caffe2/proto/caffe2.pb.h
  // Can be created with strings but give the C++ ptoto API a shot
  Workspace w1;
  init_ws(w1);
  Argument groupArg = MakeArgument<int>("group", G);
  Argument kernelHArg = MakeArgument<int>("kernel_h", KH);
  Argument kernelWArg = MakeArgument<int>("kernel_w", KW);
  OperatorDef ndef = TestHarness::ConfigureCUDA(
      "Conv",
      {"I", "W", "B"},
      {"O"},
      {strategyArg, groupArg, kernelHArg, kernelWArg});

  unique_ptr<OperatorBase> net(CreateOperator(ndef, &w1));
  ASSERT_TRUE(net.get());
  {
    CudaProfiler p;
    ASSERT_TRUE(net->Run());
  }

  auto init_ws2 = [&](Workspace& w) {
    auto AddInput = TestHarness::AddConstInput<float, CUDAContext>;
    AddInput(w, vector<TIndex>{NN, G, C, W, H}, 1., "I");
    AddInput(w, vector<TIndex>{G, F, C, KW, KH}, 1., "W");
    AddInput(w, {G, F}, 1., "B");
  };

  Workspace w2;
  init_ws2(w2);
  // groupArg = MakeArgument<int>("group", G);
  // Argument reshapeArg = MakeArgument<int>("reshape", 1);
  OperatorDef def = TestHarness::ConfigureCUDA(
      "TcGroupConvolutionOp", {"I", "W", "B"}, {"O"}, {strategyArg});
  // {strategyArg, groupArg, reshapeArg});
  unique_ptr<OperatorBase> op(CreateOperator(def, &w2));
  ASSERT_TRUE(op.get());
  {
    CudaProfiler p;
    ASSERT_TRUE(op->Run());
  }

  TC_CUDA_RUNTIMEAPI_ENFORCE(cudaDeviceSynchronize());

  TestHarness::CheckEqual(w1, w2, "O");
}

TEST_F(Caffe2Test, TcConvolutionOp) {
  auto init_ws = [&](Workspace& w) {
    auto AddInput =
        TestHarness::AddDeterministicallyRandomInput<float, CUDAContext>;
    AddInput(w, {NN, C, H, W}, "I");
    AddInput(w, {F, C, KH, KW}, "filter");
    AddInput(w, {F}, "bias");
  };

  CudaMappingOptions options =
      tc::makeBaseCliStrategy().tile(32, 32).mapToThreads({32, 32}).mapToBlocks(
          {32, 32, 32});
  Argument strategyArg = MakeArgument<string>(
      "mappingOptions",
      tc::makeCliStrategy(options).toProtobufSerializedString());
  Argument strideHArg = MakeArgument<int>("stride_h", SH);
  Argument strideWArg = MakeArgument<int>("stride_w", SW);
  OperatorDef def = TestHarness::ConfigureCUDA(
      "TcConvolutionOp",
      {"I", "filter", "bias"},
      {"H_test"},
      {strategyArg, strideHArg, strideWArg});
  auto kh = KH;
  auto kw = KW;
  auto sh = SH;
  auto sw = SW;
  TestHarness::BasicCorrectnessTest(
      def,
      init_ws,
      1e-6,
      {{"kernel_h", kh}, {"kernel_w", kw}, {"stride_h", sh}, {"stride_w", sw}});
}

TEST_F(Caffe2Test, DISABLED_TcConvolutionOp_Gradient) {
  auto init_ws = [&](Workspace& w) {
    auto AddInput =
        TestHarness::AddDeterministicallyRandomInput<float, CUDAContext>;
    /*
    auto AddConst =
    TestHarness::AddConstInput<float, CUDAContext>;
    */
    AddInput(w, {NN, C, H, W}, "I");
    AddInput(w, {F, C, KH, KW}, "filter");
    AddInput(w, {F}, "bias");
    AddInput(w, {NN, F, H - KH + 1, W - KW + 1}, "H_grad");
  };

  CudaMappingOptions options =
      tc::makeBaseCliStrategy().tile(32, 32).mapToThreads({32, 32}).mapToBlocks(
          {32, 32, 32});
  Argument strategyArg = MakeArgument<string>(
      "mappingOptions",
      tc::makeCliStrategy(options).toProtobufSerializedString());
  Argument strideHArg = MakeArgument<int>("stride_h", SH);
  Argument strideWArg = MakeArgument<int>("stride_w", SW);
  OperatorDef def = TestHarness::ConfigureCUDA(
      "TcConvolutionOp",
      {"I", "filter", "bias"},
      {"H"},
      {strategyArg, strideHArg, strideWArg});

  auto kh = KH;
  auto kw = KW;
  auto sh = SH;
  auto sw = SW;
  TestHarness::BasicGradientCorrectnessTest(
      def,
      init_ws,
      {{"kernel_h", kh}, {"kernel_w", kw}, {"stride_h", sh}, {"stride_w", sw}});
}

TEST_F(Caffe2Test, TcBatchMatmul) {
  auto init_ws = [&](Workspace& w) {
    auto AddInput =
        TestHarness::AddDeterministicallyRandomInput<float, CUDAContext>;
    AddInput(w, {B, N, M}, "X");
    AddInput(w, {B, M, K}, "Y");
  };

  Workspace w_ref;
  init_ws(w_ref);
  OperatorDef ref_def =
      TestHarness::ConfigureCUDA("BatchMatMul", {"X", "Y"}, {"Z"});
  auto ref_op = CreateOperator(ref_def, &w_ref);
  ASSERT_TRUE(ref_op.get());
  ASSERT_TRUE(ref_op->Run());

  auto tc = R"TC(
def fun(float(B, N, M) X, float(B, M, K) Y) -> (Z)
{
   Z(b, i, j) +=! X(b, i, k) * Y(b, k, j)
}
)TC";

  Workspace w_test;
  init_ws(w_test);
  Argument tcArg = MakeArgument<string>("tcDef", tc);
  Argument tcNameArg = MakeArgument<string>("tcName", "fun");
  CudaMappingOptions options = tc::makeBaseCliStrategy()
                                   .tile(1)
                                   .mapToThreads({128})
                                   .mapToBlocks({1000})
                                   .unroll(1024);
  Argument strategyArg = MakeArgument<string>(
      "mappingOptions",
      tc::makeCliStrategy(options).toProtobufSerializedString());
  auto op_def = TestHarness::ConfigureCUDA(
      "TcOp", {"X", "Y"}, {"Z"}, {tcArg, tcNameArg, strategyArg});
  auto op = CreateOperator(op_def, &w_test);
  ASSERT_TRUE(op.get());
  ASSERT_TRUE(op->Run());

  TC_CUDA_RUNTIMEAPI_ENFORCE(cudaDeviceSynchronize());

  TestHarness::CheckEqual(w_ref, w_test, "Z", 1e-6);
}

// TODO:
TEST_F(Caffe2Test, DISABLED_TcGather) {
  auto init_ws = [&](Workspace& w) {
    auto AddInput =
        TestHarness::AddDeterministicallyRandomInput<float, CUDAContext>;
    AddInput(w, {12}, "X");
    auto* I = w.CreateBlob("I")->GetMutable<TensorCUDA>();
    I->Resize(3, 4);
    auto ptr = I->raw_mutable_data(TypeMeta::Make<int32_t>());
    int32_t data[] = {8, 9, 10, 11, 0, 1, 2, 3, 4, 5, 6, 7};
    cudaMemcpy(ptr, data, I->nbytes(), cudaMemcpyHostToDevice);
  };

  Workspace w_ref;
  init_ws(w_ref);
  OperatorDef ref_def = TestHarness::ConfigureCUDA("Gather", {"X", "I"}, {"Z"});
  auto ref_op = CreateOperator(ref_def, &w_ref);
  ASSERT_TRUE(ref_op.get());
  ASSERT_TRUE(ref_op->Run());

  Workspace w_test;
  init_ws(w_test);
  auto tc = R"TC(
def fun(float(N) X, int32(A,B) I) -> (Z) {
   Z(i,j) = X(I(i,j))
}
)TC";

  Argument tcArg = MakeArgument<string>("tcDef", tc);
  Argument tcNameArg = MakeArgument<string>("tcName", "fun");
  CudaMappingOptions options = tc::makeBaseCliStrategy()
                                   .tile(1)
                                   .mapToThreads({128})
                                   .mapToBlocks({1000})
                                   .unroll(1024);
  Argument strategyArg = MakeArgument<string>(
      "mappingOptions",
      tc::makeCliStrategy(options).toProtobufSerializedString());
  auto op_def = TestHarness::ConfigureCUDA(
      "TcOp", {"X", "I"}, {"Z"}, {tcArg, tcNameArg, strategyArg});
  auto op = CreateOperator(op_def, &w_test);
  ASSERT_TRUE(op.get());
  ASSERT_TRUE(op->Run());
  TestHarness::CheckEqual(w_ref, w_test, "Z");
};

TEST_F(Caffe2Test, TcFunctions) {
  auto init_ws = [&](Workspace& w) {
    auto AddInput =
        TestHarness::AddDeterministicallyRandomInput<float, CUDAContext>;
    AddInput(w, {B, N, M}, "X");
  };

  Workspace w_ref;
  init_ws(w_ref);
  OperatorDef ref_def = TestHarness::ConfigureCUDA("Sigmoid", {"X"}, {"Z"});
  auto ref_op = CreateOperator(ref_def, &w_ref);
  ASSERT_TRUE(ref_op.get());
  ASSERT_TRUE(ref_op->Run());

  auto tc = R"TC(
def fun(float(B, N, M) X) -> (Z) {
   Z(b, i, j) = 1 / (1 + exp(-X(b, i, j))) <=> sigmoid(X(b, i, j))
}
)TC";

  Workspace w_test;
  init_ws(w_test);
  Argument tcArg = MakeArgument<string>("tcDef", tc);
  Argument tcNameArg = MakeArgument<string>("tcName", "fun");
  CudaMappingOptions options = tc::makeBaseCliStrategy()
                                   .tile(1)
                                   .mapToThreads({128})
                                   .mapToBlocks({1000})
                                   .unroll(1024);
  Argument strategyArg = MakeArgument<string>(
      "mappingOptions",
      tc::makeCliStrategy(options).toProtobufSerializedString());
  auto op_def = TestHarness::ConfigureCUDA(
      "TcOp", {"X"}, {"Z"}, {tcArg, tcNameArg, strategyArg});
  auto op = CreateOperator(op_def, &w_test);
  ASSERT_TRUE(op.get());
  ASSERT_TRUE(op->Run());
  TestHarness::CheckEqual(w_ref, w_test, "Z", 1e-6);
};

int main(int argc, char** argv) {
  ::testing::InitGoogleTest(&argc, argv);
  ::gflags::ParseCommandLineFlags(&argc, &argv, true);
  ::google::InitGoogleLogging(argv[0]);
  return RUN_ALL_TESTS();
}
