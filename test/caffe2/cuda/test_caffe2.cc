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

#include "../test_harness.h"
#include "isl_cli_strategy.h"
#include "test_harness.h"

using namespace std;
using namespace caffe2;
using namespace caffe2::detail;
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

  Argument strategy_arg;
  Caffe2CopyTest() {
    CudaMappingOptions options = tc::makeBaseCliStrategy()
                                     .tile(32, 32)
                                     .mapToThreads({32, 32})
                                     .mapToBlocks({32, 32, 32});
    strategy_arg = MakeArgument<string>(
        "mapping_options",
        tc::makeCliStrategy(options).toProtobufSerializedString());
  }
};

TEST_F(Caffe2CopyTest, TcCopyOp_Default1D) {
  auto init_ws = [&](Workspace& w) {
    auto AddInput = AddDeterministicallyRandomInput<caffe2::CUDABackend, float>;
    AddInput(w, {M}, "I");
  };
  OperatorDef def = MakeOperatorDef<caffe2::CUDABackend>(
      "TcCopyOp", {"I"}, {"O"}, {strategy_arg});
  BasicCorrectnessTest(def, init_ws);
}

TEST_F(Caffe2CopyTest, TcCopyOp_Default2D) {
  auto init_ws = [&](Workspace& w) {
    auto AddInput = AddDeterministicallyRandomInput<caffe2::CUDABackend, float>;
    AddInput(w, {M, N}, "I");
  };
  OperatorDef def = MakeOperatorDef<caffe2::CUDABackend>(
      "TcCopyOp", {"I"}, {"O"}, {strategy_arg});
  BasicCorrectnessTest(def, init_ws);
}

TEST_F(Caffe2CopyTest, TcCopyOp_Default3D) {
  auto init_ws = [&](Workspace& w) {
    auto AddInput = AddDeterministicallyRandomInput<caffe2::CUDABackend, float>;
    AddInput(w, {M, N, P}, "I");
  };
  OperatorDef def = MakeOperatorDef<caffe2::CUDABackend>(
      "TcCopyOp", {"I"}, {"O"}, {strategy_arg});
  BasicCorrectnessTest(def, init_ws);
}

TEST_F(Caffe2CopyTest, TcCopyOp_Default4D) {
  auto init_ws = [&](Workspace& w) {
    auto AddInput = AddDeterministicallyRandomInput<caffe2::CUDABackend, float>;
    AddInput(w, {M, N, P, Q}, "I");
  };
  OperatorDef def = MakeOperatorDef<caffe2::CUDABackend>(
      "TcCopyOp", {"I"}, {"O"}, {strategy_arg});
  BasicCorrectnessTest(def, init_ws);
}

TEST_F(Caffe2CopyTest, TcCopyOp_Default5D) {
  auto init_ws = [&](Workspace& w) {
    auto AddInput = AddDeterministicallyRandomInput<caffe2::CUDABackend, float>;
    AddInput(w, {M, N, P, Q, R}, "I");
  };
  OperatorDef def = MakeOperatorDef<caffe2::CUDABackend>(
      "TcCopyOp", {"I"}, {"O"}, {strategy_arg});
  BasicCorrectnessTest(def, init_ws);
}

TEST_F(Caffe2Test, TcMatMulOp) {
  auto init_ws = [&](Workspace& w) {
    auto AddInput = AddDeterministicallyRandomInput<caffe2::CUDABackend, float>;
    AddInput(w, {M, K}, "I");
    AddInput(w, {K, N}, "W");
  };

  CudaMappingOptions options = tc::makeBaseCliStrategy()
                                   .tile(32, 32, 32)
                                   .mapToThreads({4, 32})
                                   .mapToBlocks({32, 32, 32});
  Argument strategy_arg = MakeArgument<string>(
      "mapping_options",
      tc::makeCliStrategy(options).toProtobufSerializedString());
  OperatorDef def = MakeOperatorDef<caffe2::CUDABackend>(
      "TcMatMulOp", {"I", "W"}, {"O"}, {strategy_arg});
  BasicCorrectnessTest(def, init_ws, 1e-6);
}

// In Caffe2, one does not specify the inputs/outputs for the gradients.
// Instead, one runs the forward pass with the input/output blobs and then
// uses the input_map_arg, output_map_arg and input_to_compute_grad_op_map_arg
// to map the correspondence between input/output, gradInput/gradOutput and the
// actual parameters of the function. So we only declare what we need for the
// forward pass and the rest will appear automagically when running.
TEST_F(Caffe2Test, TcMatMulOp_Gradient) {
  // 1. Function to initialize tensors in a workspace.
  // Will be applied to both reference and tested workspaces.
  auto init_ws = [&](Workspace& w) {
    auto AddInput = AddDeterministicallyRandomInput<caffe2::CUDABackend, float>;
    AddInput(w, {M, N}, "I");
    AddInput(w, {N, K}, "W");
  };

  // 2. Make arguments for TcOp execution:
  //    a. MappingOptions for forward and gradient
  CudaMappingOptions options = tc::makeBaseCliStrategy()
                                   .tile(32, 32, 32)
                                   .mapToThreads({4, 32})
                                   .mapToBlocks({32, 32, 32});
  Argument mapping_options_arg = MakeArgument<string>(
      "mapping_options",
      tc::makeCliStrategy(options).toProtobufSerializedString());
  Argument mapping_options_grad_arg = MakeArgument<string>(
      "grad_mapping_options",
      tc::makeCliStrategy(options).toProtobufSerializedString());
  //    b. TC string and entry point name for both forward and gradient
  Argument tc_def_arg = MakeArgument<string>(
      "tc_def",
      R"TC(
def matmul(float(M, N) I, float(N, K) W) -> (O) {
    O(m, k) +=! I(m, r_n) * W(r_n, k)
}
)TC");
  Argument tc_name_arg = MakeArgument<string>("tc_name", "matmul");
  Argument tc_grad_def_arg = MakeArgument<string>(
      "tc_grad_def",
      R"TC(
def matmul_grad(float(M, N) I, float(N, K) W, float(M, K) d_O) -> (d_I, d_W) {
    d_I(m, n) +=! d_O(m, r_k) * W(n, r_k)
    d_W(n, k) +=! d_O(r_m, k) * I(r_m, n)
}
)TC");
  Argument tc_grad_name_arg =
      MakeArgument<string>("tc_grad_name", "matmul_grad");
  //    c. input and output maps to properly connect the input/ouput tensors
  //       for a given operator to the gradient operator.
  Argument input_map_arg = MakeArgument<std::vector<int>>(
      "inputs_used_by_gradient", std::vector<int>{0, 1});
  Argument output_map_arg = MakeArgument<std::vector<int>>(
      "output_gradients_used_by_gradient", std::vector<int>{0});
  Argument input_to_compute_grad_op_map_arg = MakeArgument<std::vector<int>>(
      "inputs_to_compute_gradients_of", std::vector<int>{0, 1});

  // 3. Create the TC operator we want to test with all the parameters above
  OperatorDef def = MakeOperatorDef<caffe2::CUDABackend>(
      "TcOp",
      {"I", "W"},
      {"O"},
      {mapping_options_arg,
       tc_def_arg,
       tc_name_arg,
       mapping_options_grad_arg,
       tc_grad_def_arg,
       tc_grad_name_arg,
       input_map_arg,
       output_map_arg,
       input_to_compute_grad_op_map_arg});

  // 4. Since we are using the generic TcOp, we need to explicitly build a
  //    reference using existing caffe2 operators.
  ReferenceImplementationBuilder referenceMatMul = [](const OperatorDef& op_def,
                                                      NetDef* net_def) {
    DeviceOption option;
    option.set_device_type(CUDA);
    net_def->add_op()->CopyFrom(
        CreateOperatorDef("MatMul", "", {"I", "W"}, {"O"}, option, "CUDA"));
  };

  // For some reason, undefined references occur without an extra variable
  auto m = M;
  auto k = K;
  // 5. Now we can run the correctness test: both forward and backward
  BasicGradientCorrectnessTest<caffe2::CUDABackend>(
      def,
      init_ws,
      1e-7 * std::max(m, k), // number of reductions * 1e-7 relative precision
      // TODO: It seems Caffe2 creates blobs with _grad appended for tensors
      // whose gradient is computed. Is there an easy idiomatic way to access
      // those or do we consider that this is reasonable enough for small tests?
      std::vector<std::string>{"I_grad", "W_grad"},
      {},
      referenceMatMul);
}

TEST_F(Caffe2Test, TcLUTOp) {
  auto init_ws = [=](Workspace& w) {
    AddDeterministicallyRandomInput<caffe2::CUDABackend, float>(
        w, {vE, vD}, "LUT");

    AddDeterministicallyRandomInputWithRange<caffe2::CUDABackend, int>(
        w, {vB, vL}, "I", 0, vE - 1);

    AddConstInput<caffe2::CUDABackend, int>(w, {vB}, vL, "__lengths");
  };

  CudaMappingOptions options =
      tc::makeBaseCliStrategy().tile(32, 32).mapToThreads({32, 32}).mapToBlocks(
          {32, 32, 32});
  Argument strategy_arg = MakeArgument<string>(
      "mapping_options",
      tc::makeCliStrategy(options)
          .useSharedMemory(false) // NYI: shared indirection
          .usePrivateMemory(false)
          .unrollCopyShared(false)
          .toProtobufSerializedString());
  OperatorDef def = MakeOperatorDef<caffe2::CUDABackend>(
      "TcLUTOp", {"LUT", "I"}, {"O"}, {strategy_arg});
  BasicCorrectnessTest(def, init_ws, 1e-6);
}

TEST_F(Caffe2Test, TcFCReluOp) {
  auto init_ws = [&](Workspace& w) {
    auto AddInput = AddDeterministicallyRandomInput<caffe2::CUDABackend, float>;
    AddInput(w, {B, M}, "I");
    AddInput(w, {N, M}, "W1");
    AddInput(w, {N}, "B1");
  };

  CudaMappingOptions options = tc::makeBaseCliStrategy()
                                   .tile(32, 32, 32)
                                   .mapToThreads({32, 4})
                                   .mapToBlocks({32, 32});
  Argument strategy_arg = MakeArgument<string>(
      "mapping_options",
      tc::makeCliStrategy(options).toProtobufSerializedString());
  OperatorDef def = MakeOperatorDef<caffe2::CUDABackend>(
      "TcFCReluOp", {"I", "W1", "B1"}, {"O1"}, {strategy_arg});
  BasicCorrectnessTest(def, init_ws, 1e-6);
}

TEST_F(Caffe2Test, Tc2FCReluOp) {
  auto init_ws = [&](Workspace& w) {
    auto AddInput = AddDeterministicallyRandomInput<caffe2::CUDABackend, float>;
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
  Argument strategy_arg = MakeArgument<string>(
      "mapping_options",
      tc::makeCliStrategy(options).toProtobufSerializedString());
  OperatorDef def = MakeOperatorDef<caffe2::CUDABackend>(
      "Tc2FCReluOp",
      {"I", "W1", "B1", "W2", "B2"},
      {"O1", "O2"},
      {strategy_arg});
  BasicCorrectnessTest(def, init_ws, 1e-6);
}

TEST_F(Caffe2Test, Tc3FCReluOp) {
  auto AddInput = AddConstInput<caffe2::CUDABackend, float>;
  auto init_ws = [&](Workspace& w) {
    AddInput(w, vector<TIndex>{B, M}, 1., "I");
    AddInput(w, vector<TIndex>{N, M}, 1., "W1");
    AddInput(w, vector<TIndex>{N}, 1., "B1");
    AddInput(w, vector<TIndex>{O, N}, 1., "W2");
    AddInput(w, vector<TIndex>{O}, 1., "B2");
    AddInput(w, vector<TIndex>{P, O}, 1., "W3");
    AddInput(w, vector<TIndex>{P}, 1., "B3");
  };

  CudaMappingOptions options = tc::makeBaseCliStrategy()
                                   .scheduleFusionStrategy("Max")
                                   .tile(1)
                                   .mapToThreads({200})
                                   .mapToBlocks({32});
  Argument strategy_arg = MakeArgument<string>(
      "mapping_options",
      tc::makeCliStrategy(options).toProtobufSerializedString());
  OperatorDef def = MakeOperatorDef<caffe2::CUDABackend>(
      "Tc3FCReluOp",
      {"I", "W1", "B1", "W2", "B2", "W3", "B3"},
      {"O1", "O2", "O3"},
      {strategy_arg});
  BasicCorrectnessTest(def, init_ws, 1e-6);
}

TEST_F(Caffe2Test, Tc4FCReluOp) {
  auto AddInput = AddConstInput<caffe2::CUDABackend, float>;
  auto init_ws = [&](Workspace& w) {
    AddInput(w, vector<TIndex>{B, M}, 1., "I");
    AddInput(w, vector<TIndex>{N, M}, 1., "W1");
    AddInput(w, vector<TIndex>{N}, 1., "B1");
    AddInput(w, vector<TIndex>{O, N}, 1., "W2");
    AddInput(w, vector<TIndex>{O}, 1., "B2");
    AddInput(w, vector<TIndex>{P, O}, 1., "W3");
    AddInput(w, vector<TIndex>{P}, 1., "B3");
    AddInput(w, vector<TIndex>{Q, P}, 1., "W4");
    AddInput(w, vector<TIndex>{Q}, 1., "B4");
  };

  CudaMappingOptions options = tc::makeBaseCliStrategy()
                                   .scheduleFusionStrategy("Max")
                                   .tile(1)
                                   .mapToThreads({128})
                                   .mapToBlocks({128});
  Argument strategy_arg = MakeArgument<string>(
      "mapping_options",
      tc::makeCliStrategy(options).toProtobufSerializedString());
  OperatorDef def = MakeOperatorDef<caffe2::CUDABackend>(
      "Tc4FCReluOp",
      {"I", "W1", "B1", "W2", "B2", "W3", "B3", "W4", "B4"},
      {"O1", "O2", "O3", "O4"},
      {strategy_arg});
  BasicCorrectnessTest(def, init_ws, 1e-6);
}

TEST_F(Caffe2Test, TcGroupConvolutionOp) {
  auto init_ws = [&](Workspace& w) {
    auto AddInput = AddConstInput<caffe2::CUDABackend, float>;
    AddInput(w, vector<TIndex>{NN, G * C, W, H}, 1., "I");
    AddInput(w, vector<TIndex>{G * F, C, KW, KH}, 1., "W");
    AddInput(w, {G * F}, 1., "B");
  };

  CudaMappingOptions options = tc::makeBaseCliStrategy()
                                   .scheduleFusionStrategy("Max")
                                   .tile(1)
                                   .mapToThreads({128})
                                   .mapToBlocks({128});
  Argument strategy_arg = MakeArgument<string>(
      "mapping_options",
      tc::makeCliStrategy(options).toProtobufSerializedString());

  // NetDef defined in caffe2.proto, see caffe2/build/caffe2/proto/caffe2.pb.h
  // Can be created with strings but give the C++ ptoto API a shot
  Workspace w1;
  init_ws(w1);
  Argument groupArg = MakeArgument<int>("group", G);
  Argument kernelHArg = MakeArgument<int>("kernel_h", KH);
  Argument kernelWArg = MakeArgument<int>("kernel_w", KW);
  OperatorDef ndef = MakeOperatorDef<caffe2::CUDABackend>(
      "Conv",
      {"I", "W", "B"},
      {"O"},
      {strategy_arg, groupArg, kernelHArg, kernelWArg});

  unique_ptr<OperatorBase> net(CreateOperator(ndef, &w1));
  ASSERT_TRUE(net.get());
  {
    CudaProfiler p;
    ASSERT_TRUE(net->Run());
  }

  auto init_ws2 = [&](Workspace& w) {
    auto AddInput = AddConstInput<caffe2::CUDABackend, float>;
    AddInput(w, vector<TIndex>{NN, G, C, W, H}, 1., "I");
    AddInput(w, vector<TIndex>{G, F, C, KW, KH}, 1., "W");
    AddInput(w, {G, F}, 1., "B");
  };

  Workspace w2;
  init_ws2(w2);
  // groupArg = MakeArgument<int>("group", G);
  // Argument reshapeArg = MakeArgument<int>("reshape", 1);
  OperatorDef def = MakeOperatorDef<caffe2::CUDABackend>(
      "TcGroupConvolutionOp", {"I", "W", "B"}, {"O"}, {strategy_arg});
  // {strategy_arg, groupArg, reshapeArg});
  unique_ptr<OperatorBase> op(CreateOperator(def, &w2));
  ASSERT_TRUE(op.get());
  {
    CudaProfiler p;
    ASSERT_TRUE(op->Run());
  }

  TC_CUDA_RUNTIMEAPI_ENFORCE(cudaDeviceSynchronize());

  CheckEqual(w1, w2, "O");
}

TEST_F(Caffe2Test, TcConvolutionOp) {
  auto init_ws = [&](Workspace& w) {
    auto AddInput = AddDeterministicallyRandomInput<caffe2::CUDABackend, float>;
    AddInput(w, {NN, C, H, W}, "I");
    AddInput(w, {F, C, KH, KW}, "filter");
    AddInput(w, {F}, "bias");
  };

  CudaMappingOptions options =
      tc::makeBaseCliStrategy().tile(32, 32).mapToThreads({32, 32}).mapToBlocks(
          {32, 32, 32});
  Argument strategy_arg = MakeArgument<string>(
      "mapping_options",
      tc::makeCliStrategy(options).toProtobufSerializedString());
  Argument strideHArg = MakeArgument<int>("stride_h", SH);
  Argument strideWArg = MakeArgument<int>("stride_w", SW);
  OperatorDef def = MakeOperatorDef<caffe2::CUDABackend>(
      "TcConvolutionOp",
      {"I", "filter", "bias"},
      {"H_test"},
      {strategy_arg, strideHArg, strideWArg});
  auto kh = KH;
  auto kw = KW;
  auto sh = SH;
  auto sw = SW;
  BasicCorrectnessTest(
      def,
      init_ws,
      1e-6,
      {{"kernel_h", kh}, {"kernel_w", kw}, {"stride_h", sh}, {"stride_w", sw}});
}

TEST_F(Caffe2Test, TcBatchMatmul) {
  auto init_ws = [&](Workspace& w) {
    auto AddInput = AddDeterministicallyRandomInput<caffe2::CUDABackend, float>;
    AddInput(w, {B, N, M}, "X");
    AddInput(w, {B, M, K}, "Y");
  };

  Workspace w_ref;
  init_ws(w_ref);
  OperatorDef ref_def =
      MakeOperatorDef<caffe2::CUDABackend>("BatchMatMul", {"X", "Y"}, {"Z"});
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
  Argument tc_arg = MakeArgument<string>("tc_def", tc);
  Argument tc_name_arg = MakeArgument<string>("tc_name", "fun");
  CudaMappingOptions options = tc::makeBaseCliStrategy()
                                   .tile(1)
                                   .mapToThreads({128})
                                   .mapToBlocks({1000})
                                   .unroll(1024);
  Argument strategy_arg = MakeArgument<string>(
      "mapping_options",
      tc::makeCliStrategy(options).toProtobufSerializedString());
  auto op_def = MakeOperatorDef<caffe2::CUDABackend>(
      "TcOp", {"X", "Y"}, {"Z"}, {tc_arg, tc_name_arg, strategy_arg});
  auto op = CreateOperator(op_def, &w_test);
  ASSERT_TRUE(op.get());
  ASSERT_TRUE(op->Run());

  TC_CUDA_RUNTIMEAPI_ENFORCE(cudaDeviceSynchronize());

  CheckEqual(w_ref, w_test, "Z", 1e-6);
}

TEST_F(Caffe2Test, TcBatchNormNoGrid) {
  static constexpr uint32_t N = 32, C = 4, H = 256, W = 256;
  auto init_ws = [&](Workspace& w) {
    auto AddInput = AddDeterministicallyRandomInput<caffe2::CUDABackend, float>;
    AddInput(w, {N, C, H, W}, "I");
    AddInput(w, {C}, "rMeanIn");
    AddInput(w, {C}, "rVarIn");
  };

  auto tc = R"TC(
  def batchnormnogrid(float(N,C,H,W) I, float(C) rMeanIn, float(C) rVarIn) -> (O, rMeanOut, rVarOut, mean, centered, variance, expectedVariance, normalizedOut) {
    mean(c) +=! I(nn, c, hh, ww)
    mean(c)  = mean(c) / (N * H * W)
    rMeanOut(c) = (1 - 0.2) * rMeanIn(c) + 0.2 * mean(c)
    centered(n, c, h, w) =        I(n, c, h, w) - rMeanOut(c)
    variance(n, c, h, w) = centered(n, c, h, w) * centered(n, c, h, w)
    expectedVariance(c) +=! (variance(n, c, h, w) + 0.001) / (N * H * W)
    rVarOut(c) = rsqrt((1 - 0.2) * rVarIn(c) + 0.2 * expectedVariance(c))
    O(n, c, h, w) = centered(n, c, h, w) / rVarOut(c)
    normalizedOut(n, c, h, w) = O(n, c, h, w)
  }
)TC";

  Workspace w_test;
  init_ws(w_test);
  Argument tcArg = MakeArgument<string>("tc_def", tc);
  Argument tcNameArg = MakeArgument<string>("tc_name", "batchnormnogrid");
  CudaMappingOptions options =
      tc::makeBaseCliStrategy()
          .outerScheduleFusionStrategy(tc::FusionStrategy::Max)
          .outerScheduleAllowSkewing(false)
          .outerSchedulePositiveOrthant(true)
          .intraTileScheduleFusionStrategy(tc::FusionStrategy::Min)
          .intraTileScheduleAllowSkewing(false)
          .intraTileSchedulePositiveOrthant(true)
          .tile(2, 1, 8, 1)
          .unroll(1)
          .tileImperfectlyNested(false)
          .matchLibraryCalls(true)
          .mapToThreads(256, 2)
          .mapToBlocks(256)
          .useSharedMemory(true)
          .usePrivateMemory(false)
          .unrollCopyShared(true);
  Argument strategyArg = MakeArgument<string>(
      "mappingOptions",
      tc::makeCliStrategy(options).toProtobufSerializedString());
  auto op_def = MakeOperatorDef<caffe2::CUDABackend>(
      "TcOp",
      {"I", "rMeanIn", "rVarIn"},
      {"O",
       "rMeanOut",
       "rVarOut",
       "mean",
       "centered",
       "variance",
       "expectedVariance",
       "normalizedOut"},
      {tcArg, tcNameArg, strategyArg});
  auto op = CreateOperator(op_def, &w_test);
  ASSERT_TRUE(op.get());
  ASSERT_TRUE(op->Run());

  TC_CUDA_RUNTIMEAPI_ENFORCE(cudaDeviceSynchronize());

  // CheckEqual(w_ref, w_test, "Z", 1e-6);
}

TEST_F(Caffe2Test, TcBatchNormGrid) {
  static constexpr uint32_t N = 32, C = 4, H = 256, W = 256;
  auto init_ws = [&](Workspace& w) {
    auto AddInput = AddDeterministicallyRandomInput<caffe2::CUDABackend, float>;
    AddInput(w, {N, C, H, W}, "I");
    AddInput(w, {C}, "rMeanIn");
    AddInput(w, {C}, "rVarIn");
  };

  auto tc = R"TC(
  def batchnormgrid(float(N,C,H,W) I, float(C) rMeanIn, float(C) rVarIn) -> (O, rMeanOut, rVarOut, mean, centered, variance, expectedVariance, normalizedOut) {
    mean(c) +=! I(nn, c, hh, ww)
    mean(c)  = mean(c) / (N * H * W)
    rMeanOut(c) = (1 - 0.2) * rMeanIn(c) + 0.2 * mean(c)
    centered(n, c, h, w) =        I(n, c, h, w) - rMeanOut(c)
    variance(n, c, h, w) = centered(n, c, h, w) * centered(n, c, h, w)
    expectedVariance(c) +=! (variance(n, c, h, w) + 0.001) / (N * H * W)
    rVarOut(c) = rsqrt((1 - 0.2) * rVarIn(c) + 0.2 * expectedVariance(c))
    O(n, c, h, w) = centered(n, c, h, w) * rVarOut(c)
    normalizedOut(n, c, h, w) = O(n, c, h, w)
  }
)TC";

  Workspace w_test;
  init_ws(w_test);
  Argument tcArg = MakeArgument<string>("tc_def", tc);
  Argument tcNameArg = MakeArgument<string>("tc_name", "batchnormgrid");
  CudaMappingOptions options =
      tc::makeBaseCliStrategy()
          .outerScheduleFusionStrategy(tc::FusionStrategy::Preserve3Coincident)
          .outerScheduleAllowSkewing(false)
          .outerSchedulePositiveOrthant(true)
          .intraTileScheduleFusionStrategy(tc::FusionStrategy::Min)
          .intraTileScheduleAllowSkewing(false)
          .intraTileSchedulePositiveOrthant(true)
          .tile(2, 64)
          .unroll(2)
          .tileImperfectlyNested(false)
          .matchLibraryCalls(true)
          .mapToThreads(32, 2)
          .mapToBlocks(128)
          .useSharedMemory(false)
          .usePrivateMemory(true)
          .unrollCopyShared(false);
  Argument strategyArg = MakeArgument<string>(
      "mappingOptions",
      tc::makeCliStrategy(options).toProtobufSerializedString());
  auto op_def = MakeOperatorDef<caffe2::CUDABackend>(
      "TcOp",
      {"I", "rMeanIn", "rVarIn"},
      {"O",
       "rMeanOut",
       "rVarOut",
       "mean",
       "centered",
       "variance",
       "expectedVariance",
       "normalizedOut"},
      {tcArg, tcNameArg, strategyArg});
  auto op = CreateOperator(op_def, &w_test);
  ASSERT_TRUE(op.get());
  ASSERT_TRUE(op->Run());

  TC_CUDA_RUNTIMEAPI_ENFORCE(cudaDeviceSynchronize());

  // CheckEqual(w_ref, w_test, "Z", 1e-6);
}

// TODO:
TEST_F(Caffe2Test, DISABLED_TcGather) {
  auto init_ws = [&](Workspace& w) {
    auto AddInput = AddDeterministicallyRandomInput<caffe2::CUDABackend, float>;
    AddInput(w, {12}, "X");
    auto* I = w.CreateBlob("I")->GetMutable<TensorCUDA>();
    I->Resize(3, 4);
    auto ptr = I->raw_mutable_data(TypeMeta::Make<int32_t>());
    int32_t data[] = {8, 9, 10, 11, 0, 1, 2, 3, 4, 5, 6, 7};
    cudaMemcpy(ptr, data, I->nbytes(), cudaMemcpyHostToDevice);
  };

  Workspace w_ref;
  init_ws(w_ref);
  OperatorDef ref_def =
      MakeOperatorDef<caffe2::CUDABackend>("Gather", {"X", "I"}, {"Z"});
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

  Argument tc_arg = MakeArgument<string>("tc_def", tc);
  Argument tc_name_arg = MakeArgument<string>("tc_name", "fun");
  CudaMappingOptions options = tc::makeBaseCliStrategy()
                                   .tile(1)
                                   .mapToThreads({128})
                                   .mapToBlocks({1000})
                                   .unroll(1024);
  Argument strategy_arg = MakeArgument<string>(
      "mapping_options",
      tc::makeCliStrategy(options).toProtobufSerializedString());
  auto op_def = MakeOperatorDef<caffe2::CUDABackend>(
      "TcOp", {"X", "I"}, {"Z"}, {tc_arg, tc_name_arg, strategy_arg});
  auto op = CreateOperator(op_def, &w_test);
  ASSERT_TRUE(op.get());
  ASSERT_TRUE(op->Run());
  CheckEqual(w_ref, w_test, "Z");
};

TEST_F(Caffe2Test, TcFunctions) {
  auto init_ws = [&](Workspace& w) {
    auto AddInput = AddDeterministicallyRandomInput<caffe2::CUDABackend, float>;
    AddInput(w, {B, N, M}, "X");
  };

  Workspace w_ref;
  init_ws(w_ref);
  OperatorDef ref_def =
      MakeOperatorDef<caffe2::CUDABackend>("Sigmoid", {"X"}, {"Z"});
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
  Argument tc_arg = MakeArgument<string>("tc_def", tc);
  Argument tc_name_arg = MakeArgument<string>("tc_name", "fun");
  CudaMappingOptions options = tc::makeBaseCliStrategy()
                                   .tile(1)
                                   .mapToThreads({128})
                                   .mapToBlocks({1000})
                                   .unroll(1024);
  Argument strategy_arg = MakeArgument<string>(
      "mapping_options",
      tc::makeCliStrategy(options).toProtobufSerializedString());
  auto op_def = MakeOperatorDef<caffe2::CUDABackend>(
      "TcOp", {"X"}, {"Z"}, {tc_arg, tc_name_arg, strategy_arg});
  auto op = CreateOperator(op_def, &w_test);
  ASSERT_TRUE(op.get());
  ASSERT_TRUE(op->Run());
  CheckEqual(w_ref, w_test, "Z", 1e-6);
};

int main(int argc, char** argv) {
  ::testing::InitGoogleTest(&argc, argv);
  ::gflags::ParseCommandLineFlags(&argc, &argv, true);
  ::google::InitGoogleLogging(argv[0]);
  return RUN_ALL_TESTS();
}
