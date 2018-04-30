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
#include "tc/core/cuda/cuda_tc_executor.h"
#include "tc/library/common.h"

#include "test_harness_aten_cuda.h"

struct CompilationTest : public ::testing::Test {
  static constexpr uint32_t N = 8, C = 16, O = 6, H = 24, W = 27;
  static constexpr uint32_t KH = 3, KW = 3, SH = 1, SW = 1;
  std::vector<at::Tensor> Check(
      const std::string& tc,
      const std::string& name,
      const tc::CudaMappingOptions& mappingOptions,
      const std::vector<at::Tensor> inputs,
      const std::vector<at::Tensor>& preInitializedOutputs =
          std::vector<at::Tensor>()) {
    auto pExecutor =
        tc::aten::compile<tc::CudaBackend>(tc, name, inputs, mappingOptions);
    std::vector<at::Tensor> outputs;
    if (preInitializedOutputs.size() == 0) {
      outputs = tc::aten::prepareOutputs(tc, name, inputs);
    } else {
      outputs = preInitializedOutputs;
    }
    tc::aten::run(*pExecutor, inputs, outputs);
    return outputs;
  }
};

TEST_F(CompilationTest, DISABLED_SoftmaxA) {
  at::Tensor a = at::CUDA(at::kFloat).rand({32, 16});
  std::vector<at::Tensor> inputs = {a};

  // Tensor dependencies should form a DAG
  std::vector<at::Tensor> outputs = Check(
      R"(
def softmax(float(N, D) I) -> (O, tmp) {
    tmp(n) max=     I(n, d)
      O(n, d) = exp(I(n, d) - tmp(n))
    tmp(n)   +=!    O(n, d)
      O(n, d) =     O(n, d) / tmp(n)
}
    )",
      "softmax",
      tc::CudaMappingOptions::makeNaiveMappingOptions(),
      inputs,
      outputs);
}

TEST_F(CompilationTest, DISABLED_SoftmaxB) {
  at::Tensor a = at::CUDA(at::kFloat).rand({32, 16});
  std::vector<at::Tensor> inputs = {a};

  // Tensor dependencies should form a DAG
  std::vector<at::Tensor> outputs = Check(
      R"(
def softmax(float(N, D) I) -> (O, tmp) {
    tmp(n) max=     I(n, d)
      O(n, d) = exp(I(n, d) - tmp(n))
    tmp(n)   +=!    O(n, d)
      O(n, d) =     O(n, d) / tmp(n)
}
    )",
      "softmax",
      tc::CudaMappingOptions::makeNaiveMappingOptions(),
      inputs,
      outputs);
}

TEST_F(CompilationTest, SoftmaxC) {
  at::Tensor a = at::CUDA(at::kFloat).rand({32, 16});
  std::vector<at::Tensor> inputs = {a};

  std::vector<at::Tensor> outputs = Check(
      R"(
def softmax(float(N, D) I) -> (O, expsum, maxVal) {
    maxVal(n) max=!     I(n, d)
    expsum(n)   +=! exp(I(n, d) - maxVal(n))
         O(n, d) =  exp(I(n, d) - maxVal(n)) / expsum(n)
}
    )",
      "softmax",
      tc::CudaMappingOptions::makeNaiveMappingOptions(),
      inputs,
      outputs);
}

TEST_F(CompilationTest, SoftmaxD) {
  at::Tensor a = at::CUDA(at::kFloat).rand({32, 16});
  std::vector<at::Tensor> inputs = {a};

  std::vector<at::Tensor> outputs = Check(
      R"(
def softmax(float(N, D) I) -> (O, maxVal, expDistance, expSum) {
         maxVal(n) max=!     I(n, d)
    expDistance(n, d) =  exp(I(n, d) - maxVal(n))
         expSum(n)   +=! expDistance(n, d)
              O(n, d) =  expDistance(n, d) / expSum(n)
}
    )",
      "softmax",
      tc::CudaMappingOptions::makeNaiveMappingOptions(),
      inputs,
      outputs);
}

TEST_F(CompilationTest, Concat) {
  at::Tensor a = at::CUDA(at::kFloat).rand({32, 16});
  at::Tensor b = at::CUDA(at::kFloat).rand({32, 16});
  std::vector<at::Tensor> inputs = {a, b};

  std::vector<at::Tensor> outputs = Check(
      R"(
def concat(float(M, N) A, float(M, N) B) -> (O1) {
    O1(n, i, m) = i == 0 ? A(m, n) : B(m, n) where i in 0:2
}
    )",
      "concat",
      tc::CudaMappingOptions::makeNaiveMappingOptions(),
      inputs,
      outputs);
}

TEST_F(CompilationTest, Indexing) {
  at::Tensor a = at::CUDA(at::kFloat).rand({3, 4});
  at::Tensor b = at::CUDA(at::kInt).ones({2});
  std::vector<at::Tensor> inputs = {a, b};

  std::vector<at::Tensor> outputs = Check(
      R"(
def indexing(float(H, W) input, int32(L) index) -> (output) {
    output(l, w) = input(index(l), w) where l in 0:2
}
    )",
      "indexing",
      tc::CudaMappingOptions::makeNaiveMappingOptions(),
      inputs,
      outputs);
}

TEST_F(CompilationTest, MatMul) {
  at::Tensor a = at::CUDA(at::kFloat).rand({3, 4});
  at::Tensor b = at::CUDA(at::kFloat).rand({4, 5});
  std::vector<at::Tensor> inputs = {a, b};

  std::vector<at::Tensor> outputs = Check(
      R"(
def matmul(float(M,N) A, float(N,K) B) -> (output) {
    output(m, k) +=! A(m, r_n) * B(r_n, k)
}
    )",
      "matmul",
      tc::CudaMappingOptions::makeMlpMappingOptions(),
      inputs,
      outputs);

  at::Tensor diff = outputs[0].sub(a.mm(b));
  checkRtol(diff, inputs, N);
}

TEST_F(CompilationTest, MatMulInplace) {
  at::Tensor a = at::CUDA(at::kFloat).rand({3, 4});
  at::Tensor b = at::CUDA(at::kFloat).rand({4, 5});
  std::vector<at::Tensor> inputs = {a, b};
  at::Tensor c = at::CUDA(at::kFloat).rand({3, 5});

  std::vector<at::Tensor> outputs = Check(
      R"(
def matmul(float(M,N) A, float(N,K) B) -> (output) {
    output(m, k) += A(m, r_n) * B(r_n, k)
}
    )",
      "matmul",
      tc::CudaMappingOptions::makeMlpMappingOptions(),
      inputs,
      {c.clone()});

  at::Tensor diff = outputs[0].sub(a.mm(b) + c);
  checkRtol(diff, inputs, N);
}

TEST_F(CompilationTest, Convolution2d) {
  at::Tensor I = at::CUDA(at::kFloat).rand({N, C, H, W});
  at::Tensor W1 = at::CUDA(at::kFloat).rand({O, C, KH, KW});
  at::Tensor B = at::CUDA(at::kFloat).rand({O});
  std::vector<at::Tensor> inputs = {I, W1, B};

  std::vector<at::Tensor> outputs = Check(
      R"(
def convolution(float(N,C,H,W) I, float(O,C,KH,KW) W1, float(O) B)
-> (tmp, O1) {
    tmp(n, o, h, w) +=!   I(n, r_c, h + r_kh, w + r_kw) * W1(o, r_c, r_kh, r_kw)
    # this can be equivalently written with =,
    # but this line tests that we correctly handle
    # degenerate +=! that have no reduction dimensions
     O1(n, o, h, w) +=! tmp(n, o, h, w) + B(o)
}
    )",
      "convolution",
      tc::CudaMappingOptions::makeConvolutionMappingOptions(),
      inputs,
      outputs);

  at::Tensor expected = at::conv2d(I, W1, B);
  at::Tensor diff = outputs[1].sub(expected);
  checkRtol(diff, inputs, C * KW * KH, 5e-7);
}

TEST_F(CompilationTest, Convolution2dStrided) {
  at::Tensor I = at::CUDA(at::kFloat).rand({N, C, H, W});
  at::Tensor W1 = at::CUDA(at::kFloat).rand({O, C, KH, KW});
  at::Tensor B = at::CUDA(at::kFloat).rand({O});
  std::vector<at::Tensor> inputs = {I, W1, B};

  constexpr static auto convolutionStrided = R"TC(
def convolutionStrided(float(N,C,H,W) I, float(O,C,KH,KW) W1, float(O) B)
-> (O1) {
    O1(n, o, h, w) +=! I(n, r_c, <sh> * h + r_kh, <sw> * w + r_kw) * W1(o, r_c, r_kh, r_kw)
    O1(n, o, h, w)  = O1(n, o, h, w) + B(o)
}
    )TC";

  std::string tcStr;
  tcStr = convolutionStrided;
  tcStr = tc::replaceString(tcStr, "<sh>", std::to_string(SH));
  tcStr = tc::replaceString(tcStr, "<sw>", std::to_string(SW));

  std::vector<at::Tensor> outputs = Check(
      tcStr,
      "convolutionStrided",
      tc::CudaMappingOptions::makeConvolutionMappingOptions(),
      inputs,
      outputs);

  at::Tensor expected = at::conv2d(I, W1, B);
  at::Tensor diff = outputs[0].sub(expected);
  // Approximate striding effect below, relax precision by factor 2
  checkRtol(diff, inputs, 2 * (C * KW * KH) / (SW * SH), 5e-7);
}

TEST_F(CompilationTest, Casts) {
  at::Tensor a = at::CUDA(at::kFloat).ones({2, 4});
  at::Tensor b = at::CUDA(at::kInt).tensor({}).fill_(4);
  a = a / 2.0 + 1;
  at::Tensor c = at::CUDA(at::kFloat).rand({3, 5});

  std::vector<at::Tensor> outputs = Check(
      R"(
def cast(float(M,N) A, int32 four) -> (int32(M,N) output) {
    output(m,n) = int32(A(m,n) + four)
}
    )",
      "cast",
      tc::CudaMappingOptions::makeNaiveMappingOptions(),
      {a, b},
      outputs);
  auto r = outputs[0].sub(at::CUDA(at::kInt).ones({2, 4}) + 4).max().toCFloat();
  CHECK_EQ(r, 0);
}

int main(int argc, char** argv) {
  ::testing::InitGoogleTest(&argc, argv);
  ::gflags::ParseCommandLineFlags(&argc, &argv, true);
  ::google::InitGoogleLogging(argv[0]);
  setAtenSeed(tc::initRandomSeed(), at::Backend::CUDA);
  return RUN_ALL_TESTS();
}
