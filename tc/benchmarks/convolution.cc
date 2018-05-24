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
DEFINE_uint32(C, 4, "Input channels (NCHW notation)");
DEFINE_uint32(F, 4, "Output filters (NCHW notation)");
DEFINE_uint32(H, 56, "Image width (NCHW notation)");
DEFINE_uint32(W, 56, "Image height (NCHW notation)");
DEFINE_uint32(KH, 3, "Kernel width (NCHW notation)");
DEFINE_uint32(KW, 3, "Kernel height (NCHW notation)");

class Convolution : public Benchmark {
 protected:
  uint32_t N, C, F, H, W, KH, KW;

 public:
  void Init(
      uint32_t n,
      uint32_t c,
      uint32_t f,
      uint32_t h,
      uint32_t w,
      uint32_t kh,
      uint32_t kw) {
    N = n;
    C = c;
    F = f;
    H = h;
    W = w;
    KH = kh;
    KW = kw;
  }
  void runConvolution(const tc::CudaMappingOptions& options);
  void runATenConvolution();
  void runCaffe2Convolution();
};

void Convolution::runConvolution(const tc::CudaMappingOptions& options) {
  Workspace w;
  auto AddInput = AddDeterministicallyRandomInput<caffe2::CUDABackend, float>;
  AddInput(w, vector<TIndex>{N, C, H, W}, "I");
  AddInput(w, vector<TIndex>{F, C, KH, KW}, "W");
  AddInput(w, {F}, "B");

  Argument kernel_h_arg = MakeArgument<int>("kernel_h", KH);
  Argument kernel_w_arg = MakeArgument<int>("kernel_w", KW);
  Argument group_arg = MakeArgument<int>("group", 1);
  OperatorDef op_def = MakeOperatorDef<caffe2::CUDABackend>(
      "Conv", {"I", "W", "B"}, {"O"}, {group_arg, kernel_h_arg, kernel_w_arg});

  std::unique_ptr<OperatorBase> net(CreateOperator(op_def, &w));
  ASSERT_TRUE(net.get());
  net->Run();
  caffe2::Tensor<caffe2::CUDAContext> expected_blob(
      w.GetBlob("O")->Get<caffe2::TensorCUDA>());

  at::Tensor ref_output =
      MakeAtenTensor(expected_blob, at::Backend::CUDA, at::kFloat)
          .resize_({N, F, H - KH + 1, W - KW + 1});

  auto check_fun = [&, ref_output](
                       const std::vector<at::Tensor>& inputs,
                       const std::vector<at::Tensor>& outputs) {
    TC_CUDA_RUNTIMEAPI_ENFORCE(cudaDeviceSynchronize());
    double prec = 1e-5; // relax precision to account for CUDNN Winograd kernels
    std::cout << "Checking expected output relative precision @" << prec;
    checkRtol(outputs[0].sub(ref_output), inputs, C * KH * KW, prec);
    return true;
  };

  // Use the underlying C2 tensors CUDA pointers
  auto tI = GetNamedTensor<CUDABackend>(w, "I");
  at::Tensor t_i = MakeAtenTensor(tI,
                       at::Backend::CUDA,
                       at::kFloat)
                       .resize_({N, C, H, W});
  auto tW = GetNamedTensor<CUDABackend>(w, "W");
  at::Tensor t_w = MakeAtenTensor(tW,
                       at::Backend::CUDA,
                       at::kFloat)
                       .resize_({F, C, KH, KW});
  auto tB = GetNamedTensor<CUDABackend>(w, "B");
  at::Tensor t_b = MakeAtenTensor(tB,
                       at::Backend::CUDA,
                       at::kFloat)
                       .resize_({F});
  std::vector<at::Tensor> inputs = {t_i, t_w, t_b};
  std::string tc = R"(
def convolution(float(N,C,H,W) I, float(F,C,KH,KW) W1, float(F) B)
-> (O)
{
    O(n, f, h, w) +=!
        I(n, r_c, h + r_kh, w + r_kw) * W1(f, r_c, r_kh, r_kw)
    O(n, f, h, w)  = O(n, f, h, w) + B(f)
}
)";

  std::string suffix = std::string("_N_") + std::to_string(FLAGS_N) +
      std::string("_C_") + std::to_string(FLAGS_C) + std::string("_F_") +
      std::to_string(FLAGS_F) + std::string("_W_") + std::to_string(FLAGS_W) +
      std::string("_H_") + std::to_string(FLAGS_H) + std::string("_KW_") +
      std::to_string(FLAGS_KW) + std::string("_KH_") + std::to_string(FLAGS_KH);
  std::vector<tc::CudaMappingOptions> bestOptions{options};
  if (FLAGS_autotune) {
    autotune(
        FLAGS_save_tuner_proto_prefix + std::string("/convolution_cache") +
            suffix,
        FLAGS_save_tuner_proto_prefix + std::string("/convolution_best") +
            suffix,
        tc,
        "convolution",
        inputs,
        options,
        check_fun);
  }
  Check(tc, "convolution", options, inputs, check_fun);
}

void Convolution::runATenConvolution() {
  Reference(
      [&]() {
        at::Tensor I = at::CUDA(at::kFloat).rand({N, C, W, H});
        at::Tensor W = at::CUDA(at::kFloat).rand({F, C, KW, KH});
        at::Tensor B = at::CUDA(at::kFloat).rand({F});
        return std::vector<at::Tensor>{I, W, B};
      },
      [&](std::vector<at::Tensor>& inputs) {
        auto I = inputs[0];
        auto W = inputs[1];
        auto B = inputs[2];
        return at::cudnn_convolution(
            I, W, B, {0, 0}, {1, 1}, {1, 1}, 1, true, false);
      });
}

void Convolution::runCaffe2Convolution() {
  Workspace w;
  auto AddInput = AddDeterministicallyRandomInput<caffe2::CUDABackend, float>;
  AddInput(w, vector<TIndex>{N, C, W, H}, "I");
  AddInput(w, vector<TIndex>{F, C, KW, KH}, "W");
  AddInput(w, {F}, "B");
  Argument kernel_h_arg = MakeArgument<int>("kernel_h", KH);
  Argument kernel_w_arg = MakeArgument<int>("kernel_w", KW);
  Argument group_arg = MakeArgument<int>("group", 1);
  OperatorDef ndef = MakeOperatorDef<caffe2::CUDABackend>(
      "Conv", {"I", "W", "B"}, {"O"}, {group_arg, kernel_h_arg, kernel_w_arg});
  std::unique_ptr<OperatorBase> net(CreateOperator(ndef, &w));
  Reference([&]() { return true; }, [&](bool flag) { net->Run(); });
}

// Generic
TEST_F(Convolution, Convolution) {
  Init(FLAGS_N, FLAGS_C, FLAGS_F, FLAGS_H, FLAGS_W, FLAGS_KH, FLAGS_KW);
  runConvolution(tc::CudaMappingOptions::makeNaiveMappingOptions());
}

TEST_F(Convolution, Convolution_Caffe2) {
  Init(FLAGS_N, FLAGS_C, FLAGS_F, FLAGS_H, FLAGS_W, FLAGS_KH, FLAGS_KW);
  runCaffe2Convolution();
}

TEST_F(Convolution, Convolution_ATen) {
  Init(FLAGS_N, FLAGS_C, FLAGS_F, FLAGS_H, FLAGS_W, FLAGS_KH, FLAGS_KW);
  runATenConvolution();
}

TEST_F(
    Convolution,
    Convolution_autotuned_P6000_N_32_C_256_F_256_H_14_W_14_KH_3_KW_3) {
  Init(32, 256, 256, 14, 14, 3, 3);
  auto options =
      tc::CudaMappingOptions::makeNaiveMappingOptions()
          .outerScheduleFusionStrategy(tc::FusionStrategy::Preserve3Coincident)
          .outerScheduleAllowSkewing(false)
          .outerSchedulePositiveOrthant(true)
          .intraTileScheduleFusionStrategy(tc::FusionStrategy::Min)
          .intraTileScheduleAllowSkewing(false)
          .intraTileSchedulePositiveOrthant(true)
          .fixParametersBeforeScheduling(true)
          .tile(3, 1, 256)
          .unroll(32)
          .tileImperfectlyNested(false)
          .matchLibraryCalls(true)
          .mapToThreads(12, 12)
          .mapToBlocks(256, 256)
          .useSharedMemory(true)
          .usePrivateMemory(false)
          .unrollCopyShared(true);
  //.useReadOnlyCache(true);

  runConvolution(options);
}

int main(int argc, char** argv) {
  ::testing::InitGoogleTest(&argc, argv);
  ::gflags::ParseCommandLineFlags(&argc, &argv, true);
  ::google::InitGoogleLogging(argv[0]);
  tc::aten::setAtenSeed(tc::initRandomSeed(), at::Backend::CUDA);
  return RUN_ALL_TESTS();
}
