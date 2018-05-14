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
#include "group_convolution.h"

#include <iostream>
#include <string>
#include <vector>

#include <gflags/gflags.h>
#include <glog/logging.h>
#include <gtest/gtest.h>

#include "tc/aten/aten.h"

#include "tc/aten/aten_compiler.h"
#include "tc/core/cuda/cuda_mapping_options.h"

#include "../test/caffe2/cuda/test_harness.h"
#include "../test/caffe2/test_harness.h"
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
 protected:
  uint32_t N, G, C, F, H, W, KH, KW;

 public:
  void Init(
      uint32_t n,
      uint32_t g,
      uint32_t c,
      uint32_t f,
      uint32_t h,
      uint32_t w,
      uint32_t kh,
      uint32_t kw) {
    N = n;
    G = g;
    C = c;
    F = f;
    H = h;
    W = w;
    KH = kh;
    KW = kw;
  }
  void runGroupConvolution(const tc::CudaMappingOptions& options);
};

void GroupConvolution::runGroupConvolution(
    const tc::CudaMappingOptions& options) {
  Workspace w;
  auto AddInput = AddDeterministicallyRandomInput<caffe2::CUDABackend, float>;
  AddInput(w, vector<TIndex>{N, G * C, H, W}, "I");
  AddInput(w, vector<TIndex>{G * F, C, KH, KW}, "W");
  AddInput(w, {G * F}, "B");

  Argument group_arg = MakeArgument<int>("group", G);
  Argument kernel_h_arg = MakeArgument<int>("kernel_h", KH);
  Argument kernel_w_arg = MakeArgument<int>("kernel_w", KW);
  OperatorDef op_def = MakeOperatorDef<caffe2::CUDABackend>(
      "Conv", {"I", "W", "B"}, {"O"}, {group_arg, kernel_h_arg, kernel_w_arg});

  std::unique_ptr<OperatorBase> net(CreateOperator(op_def, &w));
  ASSERT_TRUE(net.get());
  net->Run();
  caffe2::Tensor<caffe2::CUDAContext> expected_blob(
      w.GetBlob("O")->Get<caffe2::TensorCUDA>());

  at::Tensor ref_output =
      MakeAtenTensor(expected_blob, at::Backend::CUDA, at::kFloat)
          .resize_({N, G, F, H - KH + 1, W - KW + 1});

  auto check_fun = [&, ref_output](
                       const std::vector<at::Tensor>& inputs,
                       const std::vector<at::Tensor>& outputs) {
    TC_CUDA_RUNTIMEAPI_ENFORCE(cudaDeviceSynchronize());
    double prec = 1e-6; // relax precision to account for CUDNN Winograd kernels
    std::cout << "Checking expected output relative precision @" << prec;
    checkRtol(outputs[0].sub(ref_output), inputs, C * KH * KW, prec);
    return true;
  };

  // Use the underlying C2 tensors CUDA pointers
  at::Tensor t_i = MakeAtenTensor(
                       w.GetBlob("I")->Get<caffe2::TensorCUDA>(),
                       at::Backend::CUDA,
                       at::kFloat)
                       .resize_({N, G, C, H, W});
  at::Tensor t_w = MakeAtenTensor(
                       w.GetBlob("W")->Get<caffe2::TensorCUDA>(),
                       at::Backend::CUDA,
                       at::kFloat)
                       .resize_({G, F, C, KH, KW});
  at::Tensor t_b = MakeAtenTensor(
                       w.GetBlob("B")->Get<caffe2::TensorCUDA>(),
                       at::Backend::CUDA,
                       at::kFloat)
                       .resize_({G, F});
  std::vector<at::Tensor> inputs = {t_i, t_w, t_b};
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
  std::vector<tc::CudaMappingOptions> bestOptions{options};
  if (FLAGS_autotune) {
    bestOptions = autotune(
        FLAGS_save_tuner_proto_prefix +
            std::string("/group_convolution_cache") + suffix,
        FLAGS_save_tuner_proto_prefix + std::string("/group_convolution_best") +
            suffix,
        tc,
        "group_convolution",
        inputs,
        options,
        check_fun);
  }
  Check(tc, "group_convolution", options, inputs, check_fun);
}

TEST_F(GroupConvolution, GroupConvolution) {
  Init(
      FLAGS_N, FLAGS_G, FLAGS_C, FLAGS_F, FLAGS_H, FLAGS_W, FLAGS_KH, FLAGS_KW);
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
  runGroupConvolution(options);
}

TEST_F(
    GroupConvolution,
    GroupConvolution_P100_autotuned_N_32_G_32_C_16_F_16_W_14_H_14_KW_3_KH_3) {
  Init(32, 32, 16, 16, 14, 14, 3, 3);
  runGroupConvolution(
      tc::options_GroupConvolution_P100_autotuned_N_32_G_32_C_16_F_16_W_14_H_14_KW_3_KH_3);
}

TEST_F(
    GroupConvolution,
    GroupConvolution_P100_autotuned_N_32_G_32_C_32_F_32_W_7_H_7_KW_3_KH_3) {
  Init(32, 32, 32, 32, 7, 7, 3, 3);
  runGroupConvolution(
      tc::options_GroupConvolution_P100_autotuned_N_32_G_32_C_32_F_32_W_7_H_7_KW_3_KH_3);
}

TEST_F(
    GroupConvolution,
    GroupConvolution_P100_autotuned_N_32_G_32_C_4_F_4_W_56_H_56_KW_3_KH_3) {
  Init(32, 32, 4, 4, 56, 56, 3, 3);
  runGroupConvolution(
      tc::options_GroupConvolution_P100_autotuned_N_32_G_32_C_4_F_4_W_56_H_56_KW_3_KH_3);
}

TEST_F(
    GroupConvolution,
    GroupConvolution_P100_autotuned_N_32_G_32_C_8_F_8_W_28_H_28_KW_3_KH_3) {
  Init(32, 32, 8, 8, 28, 28, 3, 3);
  runGroupConvolution(
      tc::options_GroupConvolution_P100_autotuned_N_32_G_32_C_8_F_8_W_28_H_28_KW_3_KH_3);
}

// So slow we consider this unimplemented
TEST_F(GroupConvolution, ATenGroupConvolutionReference) {
  Init(
      FLAGS_N, FLAGS_G, FLAGS_C, FLAGS_F, FLAGS_H, FLAGS_W, FLAGS_KH, FLAGS_KW);
  Reference(
      [&]() {
        at::Tensor I = at::CUDA(at::kFloat).rand({N, G * C, W, H});
        at::Tensor W = at::CUDA(at::kFloat).rand({G * F, C, KW, KH});
        at::Tensor B = at::CUDA(at::kFloat).rand({G * F});
        return std::vector<at::Tensor>{I, W, B};
      },
      [&](std::vector<at::Tensor>& inputs) {
        auto I = inputs[0];
        auto W = inputs[1];
        auto B = inputs[2];
        return at::cudnn_convolution(
            I, W, B, {0, 0}, {1, 1}, {1, 1}, FLAGS_G, true, false);
      });
}

TEST_F(GroupConvolution, C2GroupConvolutionReference) {
  Init(
      FLAGS_N, FLAGS_G, FLAGS_C, FLAGS_F, FLAGS_H, FLAGS_W, FLAGS_KH, FLAGS_KW);
  Workspace w;
  auto AddInput = AddDeterministicallyRandomInput<caffe2::CUDABackend, float>;
  AddInput(w, vector<TIndex>{N, G * C, W, H}, "I");
  AddInput(w, vector<TIndex>{G * F, C, KW, KH}, "W");
  AddInput(w, {G * F}, "B");
  Argument group_arg = MakeArgument<int>("group", G);
  Argument kernel_h_arg = MakeArgument<int>("kernel_h", KH);
  Argument kernel_w_arg = MakeArgument<int>("kernel_w", KW);
  OperatorDef ndef = MakeOperatorDef<caffe2::CUDABackend>(
      "Conv", {"I", "W", "B"}, {"O"}, {group_arg, kernel_h_arg, kernel_w_arg});
  std::unique_ptr<OperatorBase> net(CreateOperator(ndef, &w));
  Reference([&]() { return true; }, [&](bool flag) { net->Run(); });
}

int main(int argc, char** argv) {
  ::testing::InitGoogleTest(&argc, argv);
  ::gflags::ParseCommandLineFlags(&argc, &argv, true);
  ::google::InitGoogleLogging(argv[0]);
  tc::aten::setAtenSeed(tc::initRandomSeed(), at::Backend::CUDA);
  return RUN_ALL_TESTS();
}
