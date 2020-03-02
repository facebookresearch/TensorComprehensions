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
#include "kronecker.h"

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

DEFINE_uint32(M, 256, "batch size, rows of input");
DEFINE_uint32(D0, 32, "rows of W0 (D = D0 * D1 * D2)");
DEFINE_uint32(D1, 32, "rows of W1 (D = D0 * D1 * D2)");
DEFINE_uint32(D2, 32, "rows of W2 (D = D0 * D1 * D2)");
DEFINE_uint32(N0, 16, "cols of W0 (N = N0 * N1 * N2)");
DEFINE_uint32(N1, 16, "cols of W1 (N = N0 * N1 * N2)");
DEFINE_uint32(N2, 16, "cols of W2 (N = N0 * N1 * N2)");
DEFINE_uint32(max_factors, 3, "Don't change this atm");

class Kronecker : public Benchmark {
 protected:
  uint32_t M, D0, D1, D2, N0, N1, N2;

 public:
  void init(
      uint32_t m,
      uint32_t d0,
      uint32_t d1,
      uint32_t d2,
      uint32_t n0,
      uint32_t n1,
      uint32_t n2) {
    M = m;
    D0 = d0;
    D1 = d1;
    D2 = d2;
    N0 = n0;
    N1 = n1;
    N2 = n2;
  }
  std::vector<at::Tensor> runKronecker3_1(
      const tc::CudaMappingOptions& options,
      const at::Tensor* pW2 = nullptr,
      const at::Tensor* pX = nullptr);
  std::vector<at::Tensor> runKronecker3_2(
      const tc::CudaMappingOptions& options,
      const at::Tensor* pW1 = nullptr,
      const at::Tensor* pXW2 = nullptr);
  std::vector<at::Tensor> runKronecker3_3(
      const tc::CudaMappingOptions& options,
      const at::Tensor* pW0 = nullptr,
      const at::Tensor* pXW2W1 = nullptr);
  void checkKronecker3Full(
      const tc::CudaMappingOptions& options1,
      const tc::CudaMappingOptions& options2,
      const tc::CudaMappingOptions& options3);
  void runATenKroneckerAsMatMul();
  void runCaffe2KroneckerAsMatMul();
};

std::vector<at::Tensor> Kronecker::runKronecker3_1(
    const tc::CudaMappingOptions& options,
    const at::Tensor* pW2,
    const at::Tensor* pX) {
  at::Tensor W2 = pW2 ? *pW2 : at::CUDA(at::kFloat).rand({D2, N2});
  at::Tensor X = pX ? *pX : at::CUDA(at::kFloat).rand({M, N0, N1, N2});

  std::vector<at::Tensor> inputs = {W2, X};
  std::string suffix = std::string("_M_") + std::to_string(M) +
      std::string("_D0_") + std::to_string(D0) + std::string("_D1_") +
      std::to_string(D1) + std::string("_D2_") + std::to_string(D2) +
      std::string("_N0_") + std::to_string(N0) + std::string("_N1_") +
      std::to_string(N1) + std::string("_N2_") + std::to_string(N2);
  std::vector<tc::CudaMappingOptions> bestOptions{options};
  if (FLAGS_autotune) {
    bestOptions = autotune(
        tc::TC_Kronecker3_1, tc::TC_Kronecker3_1_NAME, inputs, options);
    TC_CHECK_GE(bestOptions.size(), 1u);
  }
  return Check(
      tc::TC_Kronecker3_1, tc::TC_Kronecker3_1_NAME, bestOptions[0], inputs);
}

std::vector<at::Tensor> Kronecker::runKronecker3_2(
    const tc::CudaMappingOptions& options,
    const at::Tensor* pW1,
    const at::Tensor* pXW2) {
  at::Tensor W1 = pW1 ? *pW1 : at::CUDA(at::kFloat).rand({D1, N1});
  at::Tensor XW2 = pXW2 ? *pXW2 : at::CUDA(at::kFloat).rand({M, N0, N1, D2});

  std::vector<at::Tensor> inputs = {W1, XW2};
  std::string suffix = std::string("_M_") + std::to_string(M) +
      std::string("_D0_") + std::to_string(D0) + std::string("_D1_") +
      std::to_string(D1) + std::string("_D2_") + std::to_string(D2) +
      std::string("_N0_") + std::to_string(N0) + std::string("_N1_") +
      std::to_string(N1) + std::string("_N2_") + std::to_string(N2);
  std::vector<tc::CudaMappingOptions> bestOptions{options};
  if (FLAGS_autotune) {
    bestOptions = autotune(
        tc::TC_Kronecker3_2, tc::TC_Kronecker3_2_NAME, inputs, options);
    TC_CHECK_GE(bestOptions.size(), 1u);
  }
  return Check(
      tc::TC_Kronecker3_2, tc::TC_Kronecker3_2_NAME, bestOptions[0], inputs);
}

std::vector<at::Tensor> Kronecker::runKronecker3_3(
    const tc::CudaMappingOptions& options,
    const at::Tensor* pW0,
    const at::Tensor* pXW2W1) {
  at::Tensor W0 = pW0 ? *pW0 : at::CUDA(at::kFloat).rand({D0, N0});
  at::Tensor XW2W1 =
      pXW2W1 ? *pXW2W1 : at::CUDA(at::kFloat).rand({M, N0, D1, D2});

  std::vector<at::Tensor> inputs = {W0, XW2W1};
  std::string suffix = std::string("_M_") + std::to_string(M) +
      std::string("_D0_") + std::to_string(D0) + std::string("_D1_") +
      std::to_string(D1) + std::string("_D2_") + std::to_string(D2) +
      std::string("_N0_") + std::to_string(N0) + std::string("_N1_") +
      std::to_string(N1) + std::string("_N2_") + std::to_string(N2);
  std::vector<tc::CudaMappingOptions> bestOptions{options};
  if (FLAGS_autotune) {
    bestOptions = autotune(
        tc::TC_Kronecker3_3, tc::TC_Kronecker3_3_NAME, inputs, options);
    TC_CHECK_GE(bestOptions.size(), 1u);
  }
  return Check(
      tc::TC_Kronecker3_3, tc::TC_Kronecker3_3_NAME, bestOptions[0], inputs);
}

void Kronecker::runATenKroneckerAsMatMul() {
  at::Tensor A = at::CUDA(at::kFloat).rand({M, N0 * N1 * N2});
  at::Tensor B = at::CUDA(at::kFloat).rand({N0 * N1 * N2, D0 * D1 * D2});
  Reference(
      [&]() { return at::mm(A, B); },
      [&](at::Tensor& res) { at::mm_out(res, A, B); });
}

void Kronecker::runCaffe2KroneckerAsMatMul() {
  auto ws_init_func = [&](Workspace& w) {
    auto AddInput = AddDeterministicallyRandomInput<caffe2::CUDABackend, float>;
    AddInput(w, {M, N0 * N1 * N2}, "A");
    AddInput(w, {N0 * N1 * N2, D0 * D1 * D2}, "B");
  };
  OperatorDef op_def =
      MakeOperatorDef<caffe2::CUDABackend>("TcMatMulOp", {"A", "B"}, {"C"});
  float precision = 0.0;
  std::unique_ptr<OpTester> reference(new OpTester(op_def, precision));
  reference->InitializeReference(ws_init_func);
  Reference(
      [&]() { return true; }, [&](bool flag) { reference->RunReference(); });
}

void correctnessCheck(
    at::Tensor& expected,
    const at::Tensor& output,
    double precision) {
  long offsetInExpected = expected.numel() - output.numel();
  auto actual = output.toBackend(at::Backend::CPU);
  float relativePrecision = precision * FLAGS_D0 * FLAGS_D1 * FLAGS_D2;
  auto output1d = actual.resize_({actual.numel()});
  auto expected_a = expected.accessor<float, 1>();
  auto actual_a = output1d.accessor<float, 1>();
  for (uint32_t i = 0; i < expected_a.size(0) - offsetInExpected; ++i) {
    ASSERT_NEAR(
        expected_a[i + offsetInExpected],
        actual_a[i],
        relativePrecision * expected_a[i + offsetInExpected])
        << " for output1d at position " << i;
  }
}

std::function<bool(
    const std::vector<at::Tensor>& inputs,
    const std::vector<at::Tensor>& outputs)>
makeKronecker3CheckFunction(
    uint32_t M,
    uint32_t D0,
    uint32_t D1,
    uint32_t D2,
    uint32_t N0,
    uint32_t N1,
    uint32_t N2) {
  return [=](const std::vector<at::Tensor>& inputs,
             const std::vector<at::Tensor>& outputs) {
    at::Tensor W0 = inputs[0];
    at::Tensor W1 = inputs[1];
    at::Tensor W2 = inputs[2];
    at::Tensor X = inputs[3];

    TC_CUDA_RUNTIMEAPI_ENFORCE(cudaDeviceSynchronize());
    double prec = 3e-7;
    std::cout << "Checking expected output relative precision @" << prec
              << "\n";
    at::Tensor cW0 = W0.toBackend(at::Backend::CPU);
    at::Tensor cW1 = W1.toBackend(at::Backend::CPU);
    at::Tensor cW2 = W2.toBackend(at::Backend::CPU);
    at::Tensor cX = X.toBackend(at::Backend::CPU);
    std::vector<const float*> Ws({static_cast<const float*>(cW0.data_ptr()),
                                  static_cast<const float*>(cW1.data_ptr()),
                                  static_cast<const float*>(cW2.data_ptr())});

    uint32_t max_factors = 3;
    std::vector<uint32_t> Wsize({D0, N0, D1, N1, D2, N2});
    std::vector<uint32_t> Ysize(max_factors, -1);
    auto totalYSize = kronecker::kronecker_output_memory(
        M, N0 * N1 * N2, max_factors, Wsize, Ysize);
    std::vector<int64_t> shape({totalYSize});

    // now, create an output ATen tensor with the shape that we expect
    at::Tensor cY = at::getType(at::Backend::CPU, at::ScalarType::Float)
                        .tensor(at::IntList(shape.data(), shape.size()));
    cY.zero_();
    kronecker::cpu_kronecker_real_forward(
        M,
        N0 * N1 * N2,
        static_cast<const float*>(cX.data_ptr()),
        max_factors,
        Wsize,
        Ws,
        Ysize,
        static_cast<float*>(cY.data_ptr()));
    correctnessCheck(cY, outputs[0], prec);
    return true;
  };
}

void Kronecker::checkKronecker3Full(
    const tc::CudaMappingOptions& options1,
    const tc::CudaMappingOptions& options2,
    const tc::CudaMappingOptions& options3) {
  at::Tensor W0 = at::CUDA(at::kFloat).rand({D0, N0});
  at::Tensor W1 = at::CUDA(at::kFloat).rand({D1, N1});
  at::Tensor W2 = at::CUDA(at::kFloat).rand({D2, N2});
  at::Tensor X = at::CUDA(at::kFloat).rand({M, N0, N1, N2});

  std::vector<at::Tensor> inputs = {W0, W1, W2, X};
  std::string suffix = std::string("_M_") + std::to_string(M) +
      std::string("_D0_") + std::to_string(D0) + std::string("_D1_") +
      std::to_string(D1) + std::string("_D2_") + std::to_string(D2) +
      std::string("_N0_") + std::to_string(N0) + std::string("_N1_") +
      std::to_string(N1) + std::string("_N2_") + std::to_string(N2);

  auto r1 = runKronecker3_1(options1, &W2, &X);
  auto r2 = runKronecker3_2(options2, &W1, &r1[0]);
  auto r3 = runKronecker3_3(options3, &W0, &r2[0]);

  auto checkFun = makeKronecker3CheckFunction(M, D0, D1, D2, N0, N1, N2);
  TC_CHECK(checkFun({W0, W1, W2, X}, r3));
}

// Generic
TEST_F(Kronecker, Kronecker3_1) {
  init(FLAGS_M, FLAGS_D0, FLAGS_D1, FLAGS_D2, FLAGS_N0, FLAGS_N1, FLAGS_N2);
  runKronecker3_1(tc::CudaMappingOptions::makeNaiveMappingOptions());
}

TEST_F(Kronecker, Kronecker3_2) {
  init(FLAGS_M, FLAGS_D0, FLAGS_D1, FLAGS_D2, FLAGS_N0, FLAGS_N1, FLAGS_N2);
  runKronecker3_2(tc::CudaMappingOptions::makeNaiveMappingOptions());
}

TEST_F(Kronecker, Kronecker3_3) {
  init(FLAGS_M, FLAGS_D0, FLAGS_D1, FLAGS_D2, FLAGS_N0, FLAGS_N1, FLAGS_N2);
  runKronecker3_3(tc::CudaMappingOptions::makeNaiveMappingOptions());
}

// P100 TC
TEST_F(
    Kronecker,
    Kronecker3_1_P100_autotuned_M_256_D0_16_D1_16_D2_16_N0_32_N1_32_N2_32) {
  init(256, 16, 16, 16, 32, 32, 32);
  runKronecker3_1(
      tc::options_Kronecker3_1_P100_autotuned_M_256_D0_16_D1_16_D2_16_N0_32_N1_32_N2_32);
}

TEST_F(
    Kronecker,
    Kronecker3_1_P100_autotuned_M_256_D0_16_D1_16_D2_16_N0_64_N1_64_N2_64) {
  init(256, 16, 16, 16, 64, 64, 64);
  runKronecker3_1(
      tc::options_Kronecker3_1_P100_autotuned_M_256_D0_16_D1_16_D2_16_N0_64_N1_64_N2_64);
}

TEST_F(
    Kronecker,
    Kronecker3_1_P100_autotuned_M_256_D0_16_D1_16_D2_16_N0_64_N1_128_N2_128) {
  init(256, 16, 16, 16, 64, 128, 128);
  runKronecker3_1(
      tc::options_Kronecker3_1_P100_autotuned_M_256_D0_16_D1_16_D2_16_N0_64_N1_128_N2_128);
}

TEST_F(
    Kronecker,
    Kronecker3_2_P100_autotuned_M_256_D0_16_D1_16_D2_16_N0_32_N1_32_N2_32) {
  init(256, 16, 16, 16, 32, 32, 32);
  runKronecker3_2(
      tc::options_Kronecker3_2_P100_autotuned_M_256_D0_16_D1_16_D2_16_N0_32_N1_32_N2_32);
}

TEST_F(
    Kronecker,
    Kronecker3_2_P100_autotuned_M_256_D0_16_D1_16_D2_16_N0_64_N1_64_N2_64) {
  init(256, 16, 16, 16, 64, 64, 64);
  runKronecker3_2(
      tc::options_Kronecker3_2_P100_autotuned_M_256_D0_16_D1_16_D2_16_N0_64_N1_64_N2_64);
}

TEST_F(
    Kronecker,
    Kronecker3_2_P100_autotuned_M_256_D0_16_D1_16_D2_16_N0_64_N1_128_N2_128) {
  init(256, 16, 16, 16, 64, 128, 128);
  runKronecker3_2(
      tc::options_Kronecker3_2_P100_autotuned_M_256_D0_16_D1_16_D2_16_N0_64_N1_128_N2_128);
}

TEST_F(
    Kronecker,
    Kronecker3_3_P100_autotuned_M_256_D0_16_D1_16_D2_16_N0_32_N1_32_N2_32) {
  init(256, 16, 16, 16, 32, 32, 32);
  runKronecker3_3(
      tc::options_Kronecker3_3_P100_autotuned_M_256_D0_16_D1_16_D2_16_N0_32_N1_32_N2_32);
}

TEST_F(
    Kronecker,
    Kronecker3_3_P100_autotuned_M_256_D0_16_D1_16_D2_16_N0_64_N1_64_N2_64) {
  init(256, 16, 16, 16, 64, 64, 64);
  runKronecker3_3(
      tc::options_Kronecker3_3_P100_autotuned_M_256_D0_16_D1_16_D2_16_N0_64_N1_64_N2_64);
}

TEST_F(
    Kronecker,
    Kronecker3_3_P100_autotuned_M_256_D0_16_D1_16_D2_16_N0_64_N1_128_N2_128) {
  init(256, 16, 16, 16, 64, 128, 128);
  runKronecker3_3(
      tc::options_Kronecker3_3_P100_autotuned_M_256_D0_16_D1_16_D2_16_N0_64_N1_128_N2_128);
}

// P100 ATen
TEST_F(
    Kronecker,
    Kronecker3_ATenAsMatMul_P100_M_256_D0_16_D1_16_D2_16_N0_32_N1_32_N2_32) {
  init(256, 16, 16, 16, 32, 32, 32);
  runATenKroneckerAsMatMul();
}

TEST_F(
    Kronecker,
    Kronecker3_ATenAsMatMul_P100_M_256_D0_16_D1_16_D2_16_N0_64_N1_64_N2_64) {
  init(256, 16, 16, 16, 64, 64, 64);
  runATenKroneckerAsMatMul();
}

TEST_F(
    Kronecker,
    // This OOMs
    DISABLED_Kronecker3_ATenAsMatMul_P100_M_256_D0_16_D1_16_D2_16_N0_64_N1_128_N2_128) {
  init(256, 16, 16, 16, 64, 128, 128);
  runATenKroneckerAsMatMul();
}

// P100 Caffe2
TEST_F(
    Kronecker,
    Kronecker3_Caffe2AsMatMul_P100_M_256_D0_16_D1_16_D2_16_N0_32_N1_32_N2_32) {
  init(256, 16, 16, 16, 32, 32, 32);
  runCaffe2KroneckerAsMatMul();
}

TEST_F(
    Kronecker,
    Kronecker3_Caffe2AsMatMul_P100_M_256_D0_16_D1_16_D2_16_N0_64_N1_64_N2_64) {
  init(256, 16, 16, 16, 64, 64, 64);
  runCaffe2KroneckerAsMatMul();
}

TEST_F(
    Kronecker,
    // This OOMs
    DISABLED_Kronecker3_Caffe2AsMatMul_P100_M_256_D0_16_D1_16_D2_16_N0_64_N1_128_N2_128) {
  init(256, 16, 16, 16, 64, 128, 128);
  runCaffe2KroneckerAsMatMul();
}

// V100 TC
TEST_F(
    Kronecker,
    Kronecker3_1_V100_autotuned_M_256_D0_16_D1_16_D2_16_N0_32_N1_32_N2_32) {
  init(256, 16, 16, 16, 32, 32, 32);
  runKronecker3_1(
      tc::options_Kronecker3_1_V100_autotuned_M_256_D0_16_D1_16_D2_16_N0_32_N1_32_N2_32);
}

TEST_F(
    Kronecker,
    Kronecker3_1_V100_autotuned_M_256_D0_16_D1_16_D2_16_N0_64_N1_64_N2_64) {
  init(256, 16, 16, 16, 64, 64, 64);
  runKronecker3_1(
      tc::options_Kronecker3_1_V100_autotuned_M_256_D0_16_D1_16_D2_16_N0_64_N1_64_N2_64);
}

TEST_F(
    Kronecker,
    Kronecker3_1_V100_autotuned_M_256_D0_16_D1_16_D2_16_N0_64_N1_128_N2_128) {
  init(256, 16, 16, 16, 64, 128, 128);
  runKronecker3_1(
      tc::options_Kronecker3_1_V100_autotuned_M_256_D0_16_D1_16_D2_16_N0_64_N1_128_N2_128);
}

TEST_F(
    Kronecker,
    Kronecker3_2_V100_autotuned_M_256_D0_16_D1_16_D2_16_N0_32_N1_32_N2_32) {
  init(256, 16, 16, 16, 32, 32, 32);
  runKronecker3_2(
      tc::options_Kronecker3_2_V100_autotuned_M_256_D0_16_D1_16_D2_16_N0_32_N1_32_N2_32);
}

TEST_F(
    Kronecker,
    Kronecker3_2_V100_autotuned_M_256_D0_16_D1_16_D2_16_N0_64_N1_64_N2_64) {
  init(256, 16, 16, 16, 64, 64, 64);
  runKronecker3_2(
      tc::options_Kronecker3_2_V100_autotuned_M_256_D0_16_D1_16_D2_16_N0_64_N1_64_N2_64);
}

TEST_F(
    Kronecker,
    Kronecker3_2_V100_autotuned_M_256_D0_16_D1_16_D2_16_N0_64_N1_128_N2_128) {
  init(256, 16, 16, 16, 64, 128, 128);
  runKronecker3_2(
      tc::options_Kronecker3_2_V100_autotuned_M_256_D0_16_D1_16_D2_16_N0_64_N1_128_N2_128);
}

TEST_F(
    Kronecker,
    Kronecker3_3_V100_autotuned_M_256_D0_16_D1_16_D2_16_N0_32_N1_32_N2_32) {
  init(256, 16, 16, 16, 32, 32, 32);
  runKronecker3_3(
      tc::options_Kronecker3_3_V100_autotuned_M_256_D0_16_D1_16_D2_16_N0_32_N1_32_N2_32);
}

TEST_F(
    Kronecker,
    Kronecker3_3_V100_autotuned_M_256_D0_16_D1_16_D2_16_N0_64_N1_64_N2_64) {
  init(256, 16, 16, 16, 64, 64, 64);
  runKronecker3_3(
      tc::options_Kronecker3_3_V100_autotuned_M_256_D0_16_D1_16_D2_16_N0_64_N1_64_N2_64);
}

TEST_F(
    Kronecker,
    Kronecker3_3_V100_autotuned_M_256_D0_16_D1_16_D2_16_N0_64_N1_128_N2_128) {
  init(256, 16, 16, 16, 64, 128, 128);
  runKronecker3_3(
      tc::options_Kronecker3_3_V100_autotuned_M_256_D0_16_D1_16_D2_16_N0_64_N1_128_N2_128);
}

// V100 ATen
TEST_F(
    Kronecker,
    Kronecker3_ATenAsMatMul_V100_M_256_D0_16_D1_16_D2_16_N0_32_N1_32_N2_32) {
  init(256, 16, 16, 16, 32, 32, 32);
  runATenKroneckerAsMatMul();
}

TEST_F(
    Kronecker,
    Kronecker3_ATenAsMatMul_V100_M_256_D0_16_D1_16_D2_16_N0_64_N1_64_N2_64) {
  init(256, 16, 16, 16, 64, 64, 64);
  runATenKroneckerAsMatMul();
}

TEST_F(
    Kronecker,
    // This OOMs
    DISABLED_Kronecker3_ATenAsMatMul_V100_M_256_D0_16_D1_16_D2_16_N0_64_N1_128_N2_128) {
  init(256, 16, 16, 16, 64, 128, 128);
  runATenKroneckerAsMatMul();
}

// V100 Caffe2
TEST_F(
    Kronecker,
    Kronecker3_Caffe2AsMatMul_V100_M_256_D0_16_D1_16_D2_16_N0_32_N1_32_N2_32) {
  init(256, 16, 16, 16, 32, 32, 32);
  runCaffe2KroneckerAsMatMul();
}

TEST_F(
    Kronecker,
    Kronecker3_Caffe2AsMatMul_V100_M_256_D0_16_D1_16_D2_16_N0_64_N1_64_N2_64) {
  init(256, 16, 16, 16, 64, 64, 64);
  runCaffe2KroneckerAsMatMul();
}

TEST_F(
    Kronecker,
    // This OOMs
    DISABLED_Kronecker3_Caffe2AsMatMul_V100_M_256_D0_16_D1_16_D2_16_N0_64_N1_128_N2_128) {
  init(256, 16, 16, 16, 64, 128, 128);
  runCaffe2KroneckerAsMatMul();
}

// Sanity checks
// P100
TEST_F(Kronecker, CheckKronecker3Full_Pascal_autotuned_small) {
  init(256, 16, 16, 16, 32, 32, 32);
  checkKronecker3Full(
      tc::options_Kronecker3_1_P100_autotuned_M_256_D0_16_D1_16_D2_16_N0_32_N1_32_N2_32,
      tc::options_Kronecker3_2_P100_autotuned_M_256_D0_16_D1_16_D2_16_N0_32_N1_32_N2_32,
      tc::options_Kronecker3_3_P100_autotuned_M_256_D0_16_D1_16_D2_16_N0_32_N1_32_N2_32);
}

TEST_F(Kronecker, CheckKronecker3Full_Pascal_autotuned_medium) {
  init(256, 16, 16, 16, 32, 32, 32);
  checkKronecker3Full(
      tc::options_Kronecker3_1_P100_autotuned_M_256_D0_16_D1_16_D2_16_N0_64_N1_64_N2_64,
      tc::options_Kronecker3_2_P100_autotuned_M_256_D0_16_D1_16_D2_16_N0_64_N1_64_N2_64,
      tc::options_Kronecker3_3_P100_autotuned_M_256_D0_16_D1_16_D2_16_N0_64_N1_64_N2_64);
}

TEST_F(Kronecker, CheckKronecker3Full_Pascal_autotuned_large) {
  init(256, 16, 16, 16, 32, 32, 32);
  checkKronecker3Full(
      tc::options_Kronecker3_1_P100_autotuned_M_256_D0_16_D1_16_D2_16_N0_64_N1_128_N2_128,
      tc::options_Kronecker3_2_P100_autotuned_M_256_D0_16_D1_16_D2_16_N0_64_N1_128_N2_128,
      tc::options_Kronecker3_3_P100_autotuned_M_256_D0_16_D1_16_D2_16_N0_64_N1_128_N2_128);
}

// V100
TEST_F(Kronecker, CheckKronecker3Full_Volta_autotuned_small) {
  init(256, 16, 16, 16, 32, 32, 32);
  checkKronecker3Full(
      tc::options_Kronecker3_1_V100_autotuned_M_256_D0_16_D1_16_D2_16_N0_32_N1_32_N2_32,
      tc::options_Kronecker3_2_V100_autotuned_M_256_D0_16_D1_16_D2_16_N0_32_N1_32_N2_32,
      tc::options_Kronecker3_3_V100_autotuned_M_256_D0_16_D1_16_D2_16_N0_32_N1_32_N2_32);
}

TEST_F(Kronecker, CheckKronecker3Full_Volta_autotuned_medium) {
  init(256, 16, 16, 16, 32, 32, 32);
  checkKronecker3Full(
      tc::options_Kronecker3_1_V100_autotuned_M_256_D0_16_D1_16_D2_16_N0_64_N1_64_N2_64,
      tc::options_Kronecker3_2_V100_autotuned_M_256_D0_16_D1_16_D2_16_N0_64_N1_64_N2_64,
      tc::options_Kronecker3_3_V100_autotuned_M_256_D0_16_D1_16_D2_16_N0_64_N1_64_N2_64);
}

TEST_F(Kronecker, CheckKronecker3Full_Volta_autotuned_large) {
  init(256, 16, 16, 16, 32, 32, 32);
  checkKronecker3Full(
      tc::options_Kronecker3_1_V100_autotuned_M_256_D0_16_D1_16_D2_16_N0_64_N1_128_N2_128,
      tc::options_Kronecker3_2_V100_autotuned_M_256_D0_16_D1_16_D2_16_N0_64_N1_128_N2_128,
      tc::options_Kronecker3_3_V100_autotuned_M_256_D0_16_D1_16_D2_16_N0_64_N1_128_N2_128);
}

int main(int argc, char** argv) {
  ::testing::InitGoogleTest(&argc, argv);
  ::gflags::ParseCommandLineFlags(&argc, &argv, true);
  ::google::InitGoogleLogging(argv[0]);
  tc::aten::setAtenSeed(tc::initRandomSeed(), at::Backend::CUDA);
  return RUN_ALL_TESTS();
}
