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

#include <gflags/gflags.h>
#include <glog/logging.h>
#include <gtest/gtest.h>

#include <ATen/ATen.h>

#include "tc/aten/aten_compiler.h"
#include "tc/core/cuda/cuda_mapping_options.h"

#include "../test/test_harness.h"
#include "../test/test_harness_aten_cuda.h"
#include "benchmark_fixture.h"

#include "tc/c2/context.h"
#include "tc/core/cuda/cuda.h"
#include "tc/core/flags.h"

using namespace caffe2;

DEFINE_uint32(N, 32, "Batch size (NCHW notation)");
DEFINE_uint32(G, 32, "Number of groups (NCHW notation)");
DEFINE_uint32(C, 4, "Input channels (NCHW notation)");
DEFINE_uint32(F, 4, "Output filters (NCHW notation)");
DEFINE_uint32(H, 56, "Image width (NCHW notation)");
DEFINE_uint32(W, 56, "Image height (NCHW notation)");
DEFINE_uint32(KH, 3, "Kernel width (NCHW notation)");
DEFINE_uint32(KW, 3, "Kernel height (NCHW notation)");

class GroupConvolution : public Benchmark {
 public:
  void runGroupConvolution(
      uint32_t N,
      uint32_t G,
      uint32_t C,
      uint32_t F,
      uint32_t H,
      uint32_t W,
      uint32_t KH,
      uint32_t KW,
      const tc::CudaMappingOptions& options,
      bool useFlags = false);
};

void GroupConvolution::runGroupConvolution(
    uint32_t N,
    uint32_t G,
    uint32_t C,
    uint32_t F,
    uint32_t H,
    uint32_t W,
    uint32_t KH,
    uint32_t KW,
    const tc::CudaMappingOptions& options,
    bool useFlags) {
  Workspace w;
  auto AddInput =
      TestHarness::AddDeterministicallyRandomInput<float, CUDAContext>;
  AddInput(w, vector<TIndex>{N, G * C, H, W}, "I");
  AddInput(w, vector<TIndex>{G * F, C, KH, KW}, "W");
  AddInput(w, {G * F}, "B");

  Argument groupArg = MakeArgument<int>("group", G);
  Argument kernelHArg = MakeArgument<int>("kernel_h", KH);
  Argument kernelWArg = MakeArgument<int>("kernel_w", KW);
  OperatorDef op_def = TestHarness::ConfigureCUDA(
      "Conv", {"I", "W", "B"}, {"O"}, {groupArg, kernelHArg, kernelWArg});

  std::unique_ptr<OperatorBase> net(CreateOperator(op_def, &w));
  ASSERT_TRUE(net.get());
  net->Run();
  caffe2::Tensor<caffe2::CUDAContext> expectedBlob(
      w.GetBlob("O")->Get<caffe2::TensorCUDA>());

  at::Tensor refOutput =
      makeATenTensor(expectedBlob, at::Backend::CUDA, at::kFloat)
          .resize_({N, G, F, H - KH + 1, W - KW + 1});

  auto checkFun = [&, refOutput](
                      const std::vector<at::Tensor>& inputs,
                      const std::vector<at::Tensor>& outputs) {
    TC_CUDA_RUNTIMEAPI_ENFORCE(cudaDeviceSynchronize());
    double prec = 1e-6; // relax precision to account for CUDNN Winograd kernels
    std::cout << "Checking expected output relative precision @" << prec;
    checkRtol(outputs[0].sub(refOutput), inputs, C * KH * KW, prec);
    return true;
  };

  // Use the underlying C2 tensors CUDA pointers
  at::Tensor tI = makeATenTensor(
                      w.GetBlob("I")->Get<caffe2::TensorCUDA>(),
                      at::Backend::CUDA,
                      at::kFloat)
                      .resize_({N, G, C, H, W});
  at::Tensor tW = makeATenTensor(
                      w.GetBlob("W")->Get<caffe2::TensorCUDA>(),
                      at::Backend::CUDA,
                      at::kFloat)
                      .resize_({G, F, C, KH, KW});
  at::Tensor tB = makeATenTensor(
                      w.GetBlob("B")->Get<caffe2::TensorCUDA>(),
                      at::Backend::CUDA,
                      at::kFloat)
                      .resize_({G, F});
  std::vector<at::Tensor> inputs = {tI, tW, tB};
  std::string tc = R"(
def group_convolution(float(N,G,C,H,W) I, float(G,F,C,KH,KW) W1, float(G,F) B)
-> (O)
{
    O(n, g, f, h, w) +=!
        I(n, g, r_c, h + r_kh, w + r_kw) * W1(g, f, r_c, r_kh, r_kw)
    O(n, g, f, h, w)  = O(n, g, f, h, w) + B(g, f)
}
)";

  std::string suffix = std::string("_N_") + std::to_string(FLAGS_N) +
      std::string("_G_") + std::to_string(FLAGS_G) + std::string("_C_") +
      std::to_string(FLAGS_C) + std::string("_F_") + std::to_string(FLAGS_F) +
      std::string("_W_") + std::to_string(FLAGS_W) + std::string("_H_") +
      std::to_string(FLAGS_H) + std::string("_KW_") + std::to_string(FLAGS_KW) +
      std::string("_KH_") + std::to_string(FLAGS_KH);
  if (useFlags && FLAGS_validate_proto) {
    validateProto(
        FLAGS_save_tuner_proto_prefix +
            std::string("/group_convolution_cache") + suffix,
        tc,
        "group_convolution",
        inputs,
        checkFun);
  } else {
    std::vector<at::Tensor> outputs;
    Check(tc, "group_convolution", options, inputs, outputs, checkFun);
    if (useFlags) {
      autotune(
          FLAGS_save_tuner_proto_prefix +
              std::string("/group_convolution_cache") + suffix,
          FLAGS_save_tuner_proto_prefix +
              std::string("/group_convolution_best") + suffix,
          tc,
          "group_convolution",
          inputs,
          options,
          {options},
          checkFun);
    }
  }
}

TEST_F(GroupConvolution, GroupConvolution) {
  auto N = FLAGS_N;
  auto G = FLAGS_G;
  auto C = FLAGS_C;
  auto F = FLAGS_F;
  auto H = FLAGS_H;
  auto W = FLAGS_W;
  auto KH = FLAGS_KH;
  auto KW = FLAGS_KW;
  // If num threads is too small just get some better default
  auto threads = (W >= 10) ? std::vector<size_t>{W / 4, H / 2}
                           : std::vector<size_t>{4, 8, 4};
  auto options = tc::CudaMappingOptions::makeNaiveMappingOptions()
                     .tile(1, 1, 1)
                     .mapToThreads(threads)
                     .mapToBlocks({32, 32})
                     .useSharedMemory(true)
                     .usePrivateMemory(false)
                     .unroll(1);

  runGroupConvolution(N, G, C, F, H, W, KH, KW, options, true);
}

TEST_F(
    GroupConvolution,
    GroupConvolution_P100_autotuned_N_32_G_32_C_16_F_16_W_14_H_14_KW_3_KH_3) {
  uint32_t N = 32;
  uint32_t G = 32;
  uint32_t C = 16;
  uint32_t F = 16;
  uint32_t W = 14;
  uint32_t H = 14;
  uint32_t KW = 3;
  uint32_t KH = 3;
  auto options =
      tc::CudaMappingOptions::makeNaiveMappingOptions()
          .useSharedMemory(true)
          .usePrivateMemory(true)
          .unrollCopyShared(true)
          .outerScheduleFusionStrategy(tc::FusionStrategy::Preserve3Coincident)
          .fixParametersBeforeScheduling(false)
          .tile(1, 1)
          .tileImperfectlyNested(false)
          .mapToBlocks(3, 32)
          .mapToThreads(8, 16, 1)
          .unroll(32);
  runGroupConvolution(N, G, C, F, H, W, KH, KW, options, true);
}

TEST_F(
    GroupConvolution,
    GroupConvolution_P100_autotuned_N_32_G_32_C_32_F_32_W_7_H_7_KW_3_KH_3) {
  uint32_t N = 32;
  uint32_t G = 32;
  uint32_t C = 32;
  uint32_t F = 32;
  uint32_t W = 7;
  uint32_t H = 7;
  uint32_t KW = 3;
  uint32_t KH = 3;
  auto options =
      tc::CudaMappingOptions::makeNaiveMappingOptions()
          .outerScheduleFusionStrategy(tc::FusionStrategy::Preserve3Coincident)
          .outerScheduleAllowSkewing(false)
          .outerSchedulePositiveOrthant(true)
          .intraTileScheduleFusionStrategy(
              tc::FusionStrategy::Preserve3Coincident)
          .intraTileScheduleAllowSkewing(false)
          .intraTileSchedulePositiveOrthant(true)
          .tile(1, 2, 3)
          .mapToThreads(8, 7, 4)
          .mapToBlocks(128, 16, 64)
          .unroll(16)
          .tileImperfectlyNested(false)
          .useSharedMemory(true)
          .usePrivateMemory(true)
          .unrollCopyShared(true)
          .matchLibraryCalls(true);
  runGroupConvolution(N, G, C, F, H, W, KH, KW, options, true);
}

TEST_F(
    GroupConvolution,
    GroupConvolution_P100_autotuned_N_32_G_32_C_4_F_4_W_56_H_56_KW_3_KH_3) {
  uint32_t N = 32;
  uint32_t G = 32;
  uint32_t C = 4;
  uint32_t F = 4;
  uint32_t W = 56;
  uint32_t H = 56;
  uint32_t KW = 3;
  uint32_t KH = 3;
  auto options =
      tc::CudaMappingOptions::makeNaiveMappingOptions()
          .outerScheduleFusionStrategy(tc::FusionStrategy::Preserve3Coincident)
          .outerScheduleAllowSkewing(false)
          .outerSchedulePositiveOrthant(true)
          .intraTileScheduleFusionStrategy(
              tc::FusionStrategy::Preserve3Coincident)
          .intraTileScheduleAllowSkewing(false)
          .intraTileSchedulePositiveOrthant(true)
          .tile(1, 1, 7, 7)
          .mapToThreads(56, 7)
          .mapToBlocks(16, 64, 1)
          .unroll(2)
          .tileImperfectlyNested(false)
          .useSharedMemory(true)
          .usePrivateMemory(false)
          .unrollCopyShared(false)
          .matchLibraryCalls(true);
  runGroupConvolution(N, G, C, F, H, W, KH, KW, options, true);
}

TEST_F(
    GroupConvolution,
    GroupConvolution_P100_autotuned_N_32_G_32_C_8_F_8_W_28_H_28_KW_3_KH_3) {
  uint32_t N = 32;
  uint32_t G = 32;
  uint32_t C = 8;
  uint32_t F = 8;
  uint32_t W = 28;
  uint32_t H = 28;
  uint32_t KW = 3;
  uint32_t KH = 3;
  auto options = tc::CudaMappingOptions::makeNaiveMappingOptions()
                     .outerScheduleFusionStrategy(tc::FusionStrategy::Max)
                     .outerScheduleAllowSkewing(false)
                     .outerSchedulePositiveOrthant(true)
                     .intraTileScheduleFusionStrategy(
                         tc::FusionStrategy::Preserve3Coincident)
                     .intraTileScheduleAllowSkewing(false)
                     .intraTileSchedulePositiveOrthant(true)
                     .tile(1, 1, 256, 14, 16)
                     .mapToThreads(16, 14)
                     .mapToBlocks(7, 16)
                     .unroll(16)
                     .tileImperfectlyNested(false)
                     .useSharedMemory(true)
                     .usePrivateMemory(false)
                     .unrollCopyShared(true)
                     .matchLibraryCalls(true);
  runGroupConvolution(N, G, C, F, H, W, KH, KW, options, true);
}

// So slow we consider this unimplemented
TEST_F(GroupConvolution, ATenGroupConvolutionReference) {
#if 0
  // this is a bad, too slow implementation, in fact it's NYI atm
  auto N = FLAGS_N;
  auto G = FLAGS_G;
  auto C = FLAGS_C;
  auto F = FLAGS_F;
  auto W = FLAGS_W;
  auto H = FLAGS_H;
  auto KW = FLAGS_KW;
  auto KH = FLAGS_KH;

  Reference(
      [&]() {
        at::Tensor I = at::CUDA(at::kFloat).rand({N, G * C, W, H});
        at::Tensor W = at::CUDA(at::kFloat).rand({G * F, C, KW, KH});
        at::Tensor B = at::CUDA(at::kFloat).rand({G * F});
        return std::vector<at::Tensor>{I, W, B};
      },
      [&](std::vector<at::Tensor>& inputs) {
        // in order to perform the group conv, we will be looping
        auto I1 = inputs[0].contiguous();
        auto W2 = inputs[1];
        auto B1 = inputs[2];
        std::vector<at::Tensor> outputs(G);
        for (int g = 0; g < G; ++g) {
          // for each group, first partition out the input tensors
          auto input_g = subtensor(I1, 1, G, g);
          auto weight_g = subtensor(W2, 0, G, g);
          auto bias_g = subtensor(B1, 0, G, g);
          outputs[g] = at::conv2d(input_g, weight_g, bias_g);
        }
        // now its time to concatenate the output tensors
        auto output = outputs[0].type().tensor();
        at::cat_out(output, outputs, 1);
        return output;
      });
#endif

  std::cout << "No ATenGroupConvolutionReference available\n";
}

TEST_F(GroupConvolution, C2GroupConvolutionReference) {
  auto N = FLAGS_N;
  auto G = FLAGS_G;
  auto C = FLAGS_C;
  auto F = FLAGS_F;
  auto W = FLAGS_W;
  auto H = FLAGS_H;
  auto KW = FLAGS_KW;
  auto KH = FLAGS_KH;

  Workspace w;
  auto AddInput =
      TestHarness::AddDeterministicallyRandomInput<float, CUDAContext>;
  AddInput(w, vector<TIndex>{N, G * C, W, H}, "I");
  AddInput(w, vector<TIndex>{G * F, C, KW, KH}, "W");
  AddInput(w, {G * F}, "B");
  Argument groupArg = MakeArgument<int>("group", G);
  Argument kernelHArg = MakeArgument<int>("kernel_h", KH);
  Argument kernelWArg = MakeArgument<int>("kernel_w", KW);
  OperatorDef ndef = TestHarness::ConfigureCUDA(
      "Conv", {"I", "W", "B"}, {"O"}, {groupArg, kernelHArg, kernelWArg});
  std::unique_ptr<OperatorBase> net(CreateOperator(ndef, &w));

  Reference([&]() { return true; }, [&](bool flag) { net->Run(); });
}

int main(int argc, char** argv) {
  ::testing::InitGoogleTest(&argc, argv);
  ::gflags::ParseCommandLineFlags(&argc, &argv, true);
  ::google::InitGoogleLogging(argv[0]);
  setAtenSeed(tc::initRandomSeed(), at::Backend::CUDA);
  return RUN_ALL_TESTS();
}
