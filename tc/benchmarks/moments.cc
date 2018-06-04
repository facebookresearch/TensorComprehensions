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
#include "moments.h"

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

DEFINE_uint32(N, 1024, "N batch size (32 * 32 from group_norm equivalent)");
DEFINE_uint32(K, 36864, "K row size (16 * 48 * 48 from group_norm equivalent)");

class Moments2_2D_1D : public Benchmark {
 protected:
  uint32_t N, K;
  at::Tensor I, sum, mean, sumSquares, var;

 public:
  void Init(uint32_t n, uint32_t k) {
    N = n;
    K = k;
    I = at::CUDA(at::kFloat).rand({N, K}).uniform_(0.0f, 1.0f);
    at::Tensor v = I.view({N, -1});
    sum = v.sum(1);
    mean = v.mean(-1, true).view({N});
    sumSquares = v.pow(2.0f).sum(1);
    var = v.var(-1, true).view({N});
  }
  void runSum_2D_1D(const tc::CudaMappingOptions& options);
  void runMean_2D_1D(const tc::CudaMappingOptions& options);
  void runSumSquares_2D_1D(const tc::CudaMappingOptions& options);
  void runVar_2D_1D(const tc::CudaMappingOptions& options);
  void runSumAndSquares_2D_1D(const tc::CudaMappingOptions& options);
  void runMoments2_2D_1D(const tc::CudaMappingOptions& options);

 private:
  void autotuneAndCheck(
      const std::string& entryPoint,
      const std::vector<at::Tensor>& inputs,
      const tc::CudaMappingOptions& options,
      std::function<bool(
          const std::vector<at::Tensor>& inputs,
          const std::vector<at::Tensor>& outputs)> checkFun);
};

void Moments2_2D_1D::autotuneAndCheck(
    const std::string& entryPoint,
    const std::vector<at::Tensor>& inputs,
    const tc::CudaMappingOptions& options,
    std::function<bool(
        const std::vector<at::Tensor>& inputs,
        const std::vector<at::Tensor>& outputs)> checkFun) {
  std::string suffix = std::string("_N_") + std::to_string(N) +
      std::string("_K_") + std::to_string(K);
  std::vector<tc::CudaMappingOptions> bestOptions{options};
  if (FLAGS_autotune) {
    bestOptions = autotune(
        FLAGS_save_tuner_proto_prefix + std::string("/moments_cache") + suffix,
        FLAGS_save_tuner_proto_prefix + std::string("/moments_best") + suffix,
        tc::TC_Moments,
        entryPoint,
        inputs,
        options);
    CHECK_GE(bestOptions.size(), 1u);
  }
  Check(tc::TC_Moments, entryPoint, bestOptions[0], inputs, checkFun);
}

void Moments2_2D_1D::runSum_2D_1D(const tc::CudaMappingOptions& options) {
  std::vector<at::Tensor> inputs{I};
  auto check_fun = [&](const std::vector<at::Tensor>& inputs,
                       const std::vector<at::Tensor>& outputs) {
    checkRtol(outputs[0] - sum, inputs, K, 1e-5);
    return true;
  };
  autotuneAndCheck(tc::TC_Sum_2D_1D_NAME, inputs, options, check_fun);
}

void Moments2_2D_1D::runMean_2D_1D(const tc::CudaMappingOptions& options) {
  std::vector<at::Tensor> inputs{I};
  auto check_fun = [&](const std::vector<at::Tensor>& inputs,
                       const std::vector<at::Tensor>& outputs) {
    checkRtol(outputs[0] - mean, inputs, K, 1e-5);
    return true;
  };
  autotuneAndCheck(tc::TC_Mean_2D_1D_NAME, inputs, options, check_fun);
}

void Moments2_2D_1D::runSumSquares_2D_1D(
    const tc::CudaMappingOptions& options) {
  std::vector<at::Tensor> inputs{I};
  auto check_fun = [&](const std::vector<at::Tensor>& inputs,
                       const std::vector<at::Tensor>& outputs) {
    checkRtol(outputs[0] - sumSquares, inputs, 2 * K, 1e5);
    return true;
  };
  autotuneAndCheck(tc::TC_Sum_Squares_2D_1D_NAME, inputs, options, check_fun);
}

void Moments2_2D_1D::runVar_2D_1D(const tc::CudaMappingOptions& options) {
  std::vector<at::Tensor> inputs{I, mean};
  auto check_fun = [&](const std::vector<at::Tensor>& inputs,
                       const std::vector<at::Tensor>& outputs) {
    checkRtol(outputs[0] - var, inputs, K, 1e-5);
    return true;
  };
  autotuneAndCheck(tc::TC_Var_2D_1D_NAME, inputs, options, check_fun);
}

void Moments2_2D_1D::runSumAndSquares_2D_1D(
    const tc::CudaMappingOptions& options) {
  std::vector<at::Tensor> inputs{I};
  auto check_fun = [&](const std::vector<at::Tensor>& inputs,
                       const std::vector<at::Tensor>& outputs) {
    checkRtol(outputs[0] - sum, inputs, 2 * K, 1e-5);
    checkRtol(outputs[1] - sumSquares, inputs, 2 * K, 1e-5);
    return true;
  };
  autotuneAndCheck(
      tc::TC_Sum_And_Squares_2D_1D_NAME, inputs, options, check_fun);
}

void Moments2_2D_1D::runMoments2_2D_1D(const tc::CudaMappingOptions& options) {
  std::vector<at::Tensor> inputs{I};
  auto check_fun = [&](const std::vector<at::Tensor>& inputs,
                       const std::vector<at::Tensor>& outputs) {
    checkRtol(outputs[0] - mean, inputs, K, 1e-5);
    checkRtol(outputs[1] - var, inputs, 2 * K, 1e-5);
    return true;
  };
  autotuneAndCheck(tc::TC_Moments2_2D_1D_NAME, inputs, options, check_fun);
}

/// Sum
// Generic
TEST_F(Moments2_2D_1D, Sum_2D_1D) {
  Init(FLAGS_N, FLAGS_K);
  runSum_2D_1D(tc::CudaMappingOptions::makeNaiveMappingOptions());
}

// P100
TEST_F(Moments2_2D_1D, Sum_2D_1D_P100_autotuned_N_128_K_2304) {
  Init(128, 2304);
  runSum_2D_1D(tc::options_Sum_2D_1D_P100_autotuned_N_128_K_2304);
}

TEST_F(Moments2_2D_1D, Sum_2D_1D_P100_autotuned_N_1024_K_36864) {
  Init(1024, 36864);
  runSum_2D_1D(tc::options_Sum_2D_1D_P100_autotuned_N_1024_K_36864);
}

// V100
TEST_F(Moments2_2D_1D, Sum_2D_1D_V100_autotuned_N_128_K_2304) {
  Init(128, 2304);
  runSum_2D_1D(tc::options_Sum_2D_1D_V100_autotuned_N_128_K_2304);
}

TEST_F(Moments2_2D_1D, Sum_2D_1D_V100_autotuned_N_1024_K_36864) {
  Init(1024, 36864);
  runSum_2D_1D(tc::options_Sum_2D_1D_V100_autotuned_N_1024_K_36864);
}

// Autotunes and benchmarks mean
TEST_F(Moments2_2D_1D, Mean_2D_1D) {
  Init(FLAGS_N, FLAGS_K);
  runMean_2D_1D(tc::CudaMappingOptions::makeNaiveMappingOptions());
}

// P100
TEST_F(Moments2_2D_1D, Mean_2D_1D_P100_autotuned_N_128_K_2304) {
  Init(128, 2304);
  runMean_2D_1D(tc::options_Mean_2D_1D_P100_autotuned_N_128_K_2304);
}

TEST_F(Moments2_2D_1D, Mean_2D_1D_P100_autotuned_N_1024_K_36864) {
  Init(1024, 36864);
  runMean_2D_1D(tc::options_Mean_2D_1D_P100_autotuned_N_1024_K_36864);
}

// V100
TEST_F(Moments2_2D_1D, Mean_2D_1D_V100_autotuned_N_128_K_2304) {
  Init(128, 2304);
  runMean_2D_1D(tc::options_Mean_2D_1D_V100_autotuned_N_128_K_2304);
}

TEST_F(Moments2_2D_1D, Mean_2D_1D_V100_autotuned_N_1024_K_36864) {
  Init(1024, 36864);
  runMean_2D_1D(tc::options_Mean_2D_1D_V100_autotuned_N_1024_K_36864);
}

// Autotunes and benchmarks sum_squares
TEST_F(Moments2_2D_1D, Sum_Squares_2D_1D) {
  Init(FLAGS_N, FLAGS_K);
  runSumSquares_2D_1D(tc::CudaMappingOptions::makeNaiveMappingOptions());
}

// P100
TEST_F(Moments2_2D_1D, Sum_Squares_2D_1D_P100_autotuned_N_128_K_2304) {
  Init(128, 2304);
  runSumSquares_2D_1D(
      tc::options_Sum_Squares_2D_1D_P100_autotuned_N_128_K_2304);
}

TEST_F(Moments2_2D_1D, Sum_Squares_2D_1D_P100_autotuned_N_1024_K_36864) {
  Init(1024, 36864);
  runSumSquares_2D_1D(
      tc::options_Sum_Squares_2D_1D_P100_autotuned_N_1024_K_36864);
}

// V100
TEST_F(Moments2_2D_1D, Sum_Squares_2D_1D_V100_autotuned_N_128_K_2304) {
  Init(128, 2304);
  runSumSquares_2D_1D(
      tc::options_Sum_Squares_2D_1D_V100_autotuned_N_128_K_2304);
}

TEST_F(Moments2_2D_1D, Sum_Squares_2D_1D_V100_autotuned_N_1024_K_36864) {
  Init(1024, 36864);
  runSumSquares_2D_1D(
      tc::options_Sum_Squares_2D_1D_V100_autotuned_N_1024_K_36864);
}

// Autotunes and benchmarks var
TEST_F(Moments2_2D_1D, Var_2D_1D) {
  Init(FLAGS_N, FLAGS_K);
  runVar_2D_1D(tc::CudaMappingOptions::makeNaiveMappingOptions());
}

// P100
TEST_F(Moments2_2D_1D, Var_2D_1D_P100_autotuned_N_128_K_2304) {
  Init(128, 2304);
  runVar_2D_1D(tc::options_Var_2D_1D_P100_autotuned_N_128_K_2304);
}

TEST_F(Moments2_2D_1D, Var_2D_1D_P100_autotuned_N_1024_K_36864) {
  Init(1024, 36864);
  runVar_2D_1D(tc::options_Var_2D_1D_P100_autotuned_N_1024_K_36864);
}

// V100
TEST_F(Moments2_2D_1D, Var_2D_1D_V100_autotuned_N_128_K_2304) {
  Init(128, 2304);
  runVar_2D_1D(tc::options_Var_2D_1D_V100_autotuned_N_128_K_2304);
}

TEST_F(Moments2_2D_1D, Var_2D_1D_V100_autotuned_N_1024_K_36864) {
  Init(1024, 36864);
  runVar_2D_1D(tc::options_Var_2D_1D_V100_autotuned_N_1024_K_36864);
}

// Autotunes and benchmarks sum_and_squares
TEST_F(Moments2_2D_1D, Sum_And_Squares_2D_1D) {
  Init(FLAGS_N, FLAGS_K);
  runSumAndSquares_2D_1D(tc::CudaMappingOptions::makeNaiveMappingOptions());
}

// P100
TEST_F(Moments2_2D_1D, Sum_And_Squares_2D_1D_P100_autotuned_N_128_K_2304) {
  Init(128, 2304);
  runSumAndSquares_2D_1D(
      tc::options_Sum_And_Squares_2D_1D_P100_autotuned_N_128_K_2304);
}

TEST_F(Moments2_2D_1D, Sum_And_Squares_2D_1D_P100_autotuned_N_1024_K_36864) {
  Init(1024, 36864);
  runSumAndSquares_2D_1D(
      tc::options_Sum_And_Squares_2D_1D_P100_autotuned_N_1024_K_36864);
}

// V100
TEST_F(Moments2_2D_1D, Sum_And_Squares_2D_1D_V100_autotuned_N_128_K_2304) {
  Init(128, 2304);
  runSumAndSquares_2D_1D(
      tc::options_Sum_And_Squares_2D_1D_V100_autotuned_N_128_K_2304);
}

TEST_F(Moments2_2D_1D, Sum_And_Squares_2D_1D_V100_autotuned_N_1024_K_36864) {
  Init(1024, 36864);
  runSumAndSquares_2D_1D(
      tc::options_Sum_And_Squares_2D_1D_V100_autotuned_N_1024_K_36864);
}

// Benchmarks 2 moments (mean and var)
TEST_F(Moments2_2D_1D, Moments2_2D_1D) {
  Init(FLAGS_N, FLAGS_K);
  runMoments2_2D_1D(tc::CudaMappingOptions::makeNaiveMappingOptions());
}

// P100
TEST_F(Moments2_2D_1D, Moments2_2D_1D_P100_autotuned_N_128_K_2304) {
  Init(128, 2304);
  runMoments2_2D_1D(tc::options_Moments2_2D_1D_P100_autotuned_N_128_K_2304);
}

TEST_F(Moments2_2D_1D, Moments2_2D_1D_P100_autotuned_N_1024_K_36864) {
  Init(1024, 36864);
  runMoments2_2D_1D(tc::options_Moments2_2D_1D_P100_autotuned_N_1024_K_36864);
}

// V100
TEST_F(Moments2_2D_1D, Moments2_2D_1D_V100_autotuned_N_128_K_2304) {
  Init(128, 2304);
  runMoments2_2D_1D(tc::options_Moments2_2D_1D_V100_autotuned_N_128_K_2304);
}

TEST_F(Moments2_2D_1D, Moments2_2D_1D_V100_autotuned_N_1024_K_36864) {
  Init(1024, 36864);
  runMoments2_2D_1D(tc::options_Moments2_2D_1D_V100_autotuned_N_1024_K_36864);
}

int main(int argc, char** argv) {
  ::testing::InitGoogleTest(&argc, argv);
  ::gflags::ParseCommandLineFlags(&argc, &argv, true);
  ::google::InitGoogleLogging(argv[0]);
  tc::aten::setAtenSeed(tc::initRandomSeed(), at::Backend::CUDA);
  return RUN_ALL_TESTS();
}
