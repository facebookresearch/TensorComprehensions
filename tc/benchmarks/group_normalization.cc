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
#include "group_normalization.h"

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

DEFINE_uint32(N, 32, "N batch size");
DEFINE_uint32(C, 512, "Number of channels (that will get divided into groups)");
DEFINE_uint32(G, 32, "Number of groups");
DEFINE_uint32(H, 48, "Height");
DEFINE_uint32(W, 48, "Width");

class GroupNormalization : public Benchmark {
 protected:
  uint32_t N, C, G, D, H, W;

 public:
  void Init(uint32_t n, uint32_t c, uint32_t g, uint32_t h, uint32_t w) {
    N = n;
    C = c;
    G = g;
    D = C / G;
    H = h;
    W = w;
  }
  std::vector<at::Tensor> runGroupNormalization(
      const tc::CudaMappingOptions& options,
      const tc::CudaMappingOptions& optionsMoments =
          tc::CudaMappingOptions::makeNaiveMappingOptions());
  std::vector<at::Tensor> runGroupNormalizationSingleKernel(
      const tc::CudaMappingOptions& options);
  void runCaffe2GroupNormalization();
  void runATenGroupNormalization();
};

std::vector<at::Tensor> GroupNormalization::runGroupNormalization(
    const tc::CudaMappingOptions& options,
    const tc::CudaMappingOptions& optionsMoments) {
  at::Tensor I = at::CUDA(at::kFloat).rand({N, G, D, H, W});
  at::Tensor gamma = at::CUDA(at::kFloat).rand({G, D});
  at::Tensor beta = at::CUDA(at::kFloat).rand({G, D});
  auto view = I.view({N, G, -1});
  auto mean = view.mean(-1, true);
  auto var = view.var(-1, true).view({N, G, 1});

  auto check_fun = [&](const std::vector<at::Tensor>& inputs,
                       const std::vector<at::Tensor>& outputs) {
    TC_CUDA_RUNTIMEAPI_ENFORCE(cudaDeviceSynchronize());
    auto x = ((view - mean) / (var + 1e-5f).sqrt());
    auto y = gamma.view({1, G, D, 1, 1}) * x.view({N, G, D, H, W}) +
        beta.view({1, G, D, 1, 1});
    TC_CUDA_RUNTIMEAPI_ENFORCE(cudaDeviceSynchronize());
    checkRtol(outputs[0] - y, {I}, D * H * W, 1e-6);
    return true;
  };

  auto inputs = std::vector<at::Tensor>{
      I, gamma, beta, mean.view({N, G}), var.view({N, G})};
  std::string suffix = std::string("_N_") + std::to_string(N) +
      std::string("_C_") + std::to_string(C) + std::string("_G_") +
      std::to_string(G) + std::string("_H_") + std::to_string(H) +
      std::string("_W_") + std::to_string(W);
  std::vector<tc::CudaMappingOptions> bestOptions{options};
  if (FLAGS_autotune) {
    bestOptions = autotune(
        tc::TC_GroupNormalization,
        tc::TC_GroupNormalization_NAME,
        inputs,
        options);
    TC_CHECK_GE(bestOptions.size(), 1u);
  }

  auto pExecutorMoments = tc::aten::compile<tc::CudaBackend>(
      tc::TC_GroupNormalization,
      tc::TC_Moments2_2D_1D_NAME,
      {I.view({N * G, D * H * W})},
      optionsMoments);
  std::vector<at::Tensor> outputs = tc::aten::prepareOutputs(
      tc::TC_GroupNormalization,
      tc::TC_Moments2_2D_1D_NAME,
      {I.view({N * G, D * H * W})});
  tc::aten::run(*pExecutorMoments, {I.view({N * G, D * H * W})}, outputs);
  auto computeMoments = [&I, &outputs, &pExecutorMoments, this]() {
    return tc::aten::profile(
        *pExecutorMoments, {I.view({N * G, D * H * W})}, outputs);
  };
  return Check(
      tc::TC_GroupNormalization,
      tc::TC_GroupNormalization_NAME,
      bestOptions[0],
      inputs,
      check_fun,
      computeMoments);
}

std::vector<at::Tensor> GroupNormalization::runGroupNormalizationSingleKernel(
    const tc::CudaMappingOptions& options) {
  at::Tensor I = at::CUDA(at::kFloat).rand({N, G, D, H, W});
  at::Tensor gamma = at::CUDA(at::kFloat).rand({G, D});
  at::Tensor beta = at::CUDA(at::kFloat).rand({G, D});

  auto check_fun = [&](const std::vector<at::Tensor>& inputs,
                       const std::vector<at::Tensor>& outputs) {
    TC_CUDA_RUNTIMEAPI_ENFORCE(cudaDeviceSynchronize());
    auto view = I.view({N, G, -1});
    auto mean = view.mean(-1, true);
    auto var = view.var(-1, true).view({N, G, 1});
    auto x = ((view - mean) / (var + 1e-5f).sqrt());
    auto y = gamma.view({1, G, D, 1, 1}) * x.view({N, G, D, H, W}) +
        beta.view({1, G, D, 1, 1});
    TC_CUDA_RUNTIMEAPI_ENFORCE(cudaDeviceSynchronize());
    checkRtol(outputs[0] - y, {I}, D * H * W, 1e-6);
    return true;
  };

  auto inputs = std::vector<at::Tensor>{I, gamma, beta};
  std::string suffix = std::string("_N_") + std::to_string(N) +
      std::string("_C_") + std::to_string(C) + std::string("_G_") +
      std::to_string(G) + std::string("_H_") + std::to_string(H) +
      std::string("_W_") + std::to_string(W);
  std::vector<tc::CudaMappingOptions> bestOptions{options};
  if (FLAGS_autotune) {
    bestOptions = autotune(
        tc::TC_GroupNormalization,
        tc::TC_GroupNormalizationSingleKernel_NAME,
        inputs,
        options);
    TC_CHECK_GE(bestOptions.size(), 1u);
  }
  return Check(
      tc::TC_GroupNormalization,
      tc::TC_GroupNormalizationSingleKernel_NAME,
      bestOptions[0],
      inputs,
      check_fun);
}

void GroupNormalization::runCaffe2GroupNormalization() {
  Workspace w;
  auto AddInput = AddDeterministicallyRandomInput<caffe2::CUDABackend, float>;
  AddInput(w, {N, C, H, W}, "I");
  AddInput(w, {G, D}, "gamma");
  AddInput(w, {G, D}, "beta");
  OperatorDef def = MakeOperatorDef<caffe2::CUDABackend>(
      "GroupNorm", {"I", "gamma", "beta"}, {"O", "mean", "var"});
  unique_ptr<OperatorBase> op(CreateOperator(def, &w));
  Reference([&]() { return true; }, [&op](bool flag) { op->Run(); });
}

void GroupNormalization::runATenGroupNormalization() {
  at::Tensor I = at::CUDA(at::kFloat).rand({N, G, D, H, W});
  at::Tensor gamma = at::CUDA(at::kFloat).rand({G, D});
  at::Tensor beta = at::CUDA(at::kFloat).rand({G, D});
  Reference(
      [&]() { return true; },
      [&I, &gamma, &beta, this](bool flag) {
        auto v = I.view({N, G, -1});
        auto mean = v.mean(-1, true);
        auto var = v.var(-1, true).view({N, G, 1});
        auto x = ((v - mean) / (var + 1e-5f).sqrt());
        auto y = gamma.view({1, G, D, 1, 1}) * x.view({N, G, D, H, W}) +
            beta.view({1, G, D, 1, 1});
        ;
      });
}

/// Generic
TEST_F(GroupNormalization, GroupNormalization) {
  Init(FLAGS_N, FLAGS_C, FLAGS_G, FLAGS_H, FLAGS_W);
  runGroupNormalization(tc::CudaMappingOptions::makeNaiveMappingOptions());
}

// P100 TC
TEST_F(
    GroupNormalization,
    GroupNormalization_P100_autotuned_N_4_C_512_G_32_H_12_W_12) {
  Init(4, 512, 32, 12, 12);
  runGroupNormalization(
      tc::options_GroupNormalization_P100_autotuned_N_4_C_512_G_32_H_12_W_12,
      tc::options_Moments2_2D_1D_P100_autotuned_N_128_K_2304);
}

TEST_F(
    GroupNormalization,
    GroupNormalization_P100_autotuned_N_32_C_512_G_32_H_48_W_48) {
  Init(32, 512, 32, 48, 48);
  runGroupNormalization(
      tc::options_GroupNormalization_P100_autotuned_N_32_C_512_G_32_H_48_W_48,
      tc::options_Moments2_2D_1D_P100_autotuned_N_1024_K_36864);
}

// P100 Caffe2
TEST_F(
    GroupNormalization,
    GroupNormalization_Caffe2_P100_N_4_C_512_G_32_H_12_W_12) {
  Init(4, 512, 32, 12, 12);
  runCaffe2GroupNormalization();
}

TEST_F(
    GroupNormalization,
    GroupNormalization_Caffe2_P100_N_32_C_512_G_32_H_48_W_48) {
  Init(32, 512, 32, 48, 48);
  runCaffe2GroupNormalization();
}

// P100 ATen
TEST_F(
    GroupNormalization,
    GroupNormalization_ATen_P100_N_4_C_512_G_32_H_12_W_12) {
  Init(4, 512, 32, 12, 12);
  runATenGroupNormalization();
}

TEST_F(
    GroupNormalization,
    GroupNormalization_ATen_P100_N_32_C_512_G_32_H_48_W_48) {
  Init(32, 512, 32, 48, 48);
  runATenGroupNormalization();
}

// V100 TC
TEST_F(
    GroupNormalization,
    GroupNormalization_V100_autotuned_N_4_C_512_G_32_H_12_W_12) {
  Init(4, 512, 32, 12, 12);
  runGroupNormalization(
      tc::options_GroupNormalization_V100_autotuned_N_4_C_512_G_32_H_12_W_12,
      tc::options_Moments2_2D_1D_V100_autotuned_N_128_K_2304);
}

TEST_F(
    GroupNormalization,
    GroupNormalization_V100_autotuned_N_32_C_512_G_32_H_48_W_48) {
  Init(32, 512, 32, 48, 48);
  runGroupNormalization(
      tc::options_GroupNormalization_V100_autotuned_N_32_C_512_G_32_H_48_W_48,
      tc::options_Moments2_2D_1D_V100_autotuned_N_1024_K_36864);
}

// V100 Caffe2
TEST_F(
    GroupNormalization,
    GroupNormalization_Caffe2_V100_N_4_C_512_G_32_H_12_W_12) {
  Init(4, 512, 32, 12, 12);
  runCaffe2GroupNormalization();
}

TEST_F(
    GroupNormalization,
    GroupNormalization_Caffe2_V100_N_32_C_512_G_32_H_48_W_48) {
  Init(32, 512, 32, 48, 48);
  runCaffe2GroupNormalization();
}

// V100 ATen
TEST_F(
    GroupNormalization,
    GroupNormalization_ATen_V100_N_4_C_512_G_32_H_12_W_12) {
  Init(4, 512, 32, 12, 12);
  runATenGroupNormalization();
}

TEST_F(
    GroupNormalization,
    GroupNormalization_ATen_V100_N_32_C_512_G_32_H_48_W_48) {
  Init(32, 512, 32, 48, 48);
  runATenGroupNormalization();
}

// Generic
TEST_F(GroupNormalization, GroupNormalizationSingleKernel) {
  Init(FLAGS_N, FLAGS_C, FLAGS_G, FLAGS_H, FLAGS_W);
  runGroupNormalizationSingleKernel(
      tc::CudaMappingOptions::makeNaiveMappingOptions());
}

// P100 TC
TEST_F(
    GroupNormalization,
    GroupNormalizationSingleKernel_P100_autotuned_N_4_C_512_G_32_H_12_W_12) {
  Init(4, 512, 32, 12, 12);
  runGroupNormalizationSingleKernel(
      tc::options_GroupNormalizationSingleKernel_P100_autotuned_N_4_C_512_G_32_H_12_W_12);
}

TEST_F(
    GroupNormalization,
    GroupNormalizationSingleKernel_P100_autotuned_N_32_C_512_G_32_H_48_W_48) {
  Init(32, 512, 32, 48, 48);
  runGroupNormalizationSingleKernel(
      tc::options_GroupNormalizationSingleKernel_P100_autotuned_N_32_C_512_G_32_H_48_W_48);
}

// V100 TC
TEST_F(
    GroupNormalization,
    GroupNormalizationSingleKernel_V100_autotuned_N_4_C_512_G_32_H_12_W_12) {
  Init(4, 512, 32, 12, 12);
  runGroupNormalizationSingleKernel(
      tc::options_GroupNormalizationSingleKernel_V100_autotuned_N_4_C_512_G_32_H_12_W_12);
}

TEST_F(
    GroupNormalization,
    GroupNormalizationSingleKernel_V100_autotuned_N_32_C_512_G_32_H_48_W_48) {
  Init(32, 512, 32, 48, 48);
  runGroupNormalizationSingleKernel(
      tc::options_GroupNormalizationSingleKernel_V100_autotuned_N_32_C_512_G_32_H_48_W_48);
}

int main(int argc, char** argv) {
  ::testing::InitGoogleTest(&argc, argv);
  ::gflags::ParseCommandLineFlags(&argc, &argv, true);
  ::google::InitGoogleLogging(argv[0]);
  tc::aten::setAtenSeed(tc::initRandomSeed(), at::Backend::CUDA);
  return RUN_ALL_TESTS();
}
