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
#include "tc/core/mapping_options.h"
#include "tc/library/common.h"

#include "test_harness_aten.h"

struct ATenCompilationUnitTest : public ::testing::Test {
  static constexpr uint32_t N = 8, C = 16, O = 6, H = 24, W = 27;
  static constexpr uint32_t KH = 3, KW = 3, SH = 1, SW = 1;
  void Check(
      const std::string& tc,
      const std::string& name,
      const tc::MappingOptions& mappingOptions,
      const std::vector<at::Tensor> inputs,
      std::vector<at::Tensor>& outputs) {
    tc::ATenCompilationUnit atCompl;
    atCompl.define(tc);
    auto handle = atCompl.compile(name, inputs, mappingOptions);
    atCompl.run(name, inputs, outputs, handle);
  }
};

TEST_F(ATenCompilationUnitTest, DISABLED_SoftmaxA) {
  at::Tensor a = at::CUDA(at::kFloat).rand({32, 16});
  std::vector<at::Tensor> inputs = {a};
  std::vector<at::Tensor> outputs;

  // Tensor dependencies should strictly be DAG
  Check(
      R"(
      def softmax(float(N, D) I) -> (O, tmp) {
        tmp(n) max= I(n, d)
        O(n, d) = exp(I(n, d) - tmp(n))
        tmp(n) +=! O(n, d)
        O(n, d) = O(n, d) / tmp(n)
      }
    )",
      "softmax",
      tc::MappingOptions::makeNaiveMappingOptions(),
      inputs,
      outputs);
}

TEST_F(ATenCompilationUnitTest, DISABLED_SoftmaxB) {
  at::Tensor a = at::CUDA(at::kFloat).rand({32, 16});
  std::vector<at::Tensor> inputs = {a};
  std::vector<at::Tensor> outputs;

  // Tensor dependencies should strictly be DAG
  Check(
      R"(
      def softmax(float(N, D) I) -> (O, tmp, tmp1) {
        tmp(n) max=! I(n, d)
        O(n, d) = exp(I(n, d) - tmp(n))
        tmp1(n) +=! O(n, d)
        O(n, d) = O(n, d) / tmp1(n)
      }
    )",
      "softmax",
      tc::MappingOptions::makeNaiveMappingOptions(),
      inputs,
      outputs);
}

TEST_F(ATenCompilationUnitTest, SoftmaxC) {
  at::Tensor a = at::CUDA(at::kFloat).rand({32, 16});
  std::vector<at::Tensor> inputs = {a};
  std::vector<at::Tensor> outputs;

  Check(
      R"(
      def softmax(float(N, D) I) -> (O, expsum, maxVal) {
        maxVal(n) max=! I(n, d)
        expsum(n) +=! exp(I(n, d) - maxVal(n))
        O(n, d) = exp(I(n, d) - maxVal(n)) / expsum(n)
      }
    )",
      "softmax",
      tc::MappingOptions::makeNaiveMappingOptions(),
      inputs,
      outputs);
}

TEST_F(ATenCompilationUnitTest, SoftmaxD) {
  at::Tensor a = at::CUDA(at::kFloat).rand({32, 16});
  std::vector<at::Tensor> inputs = {a};
  std::vector<at::Tensor> outputs;

  Check(
      R"(
      def softmax(float(N, D) I) -> (O, maxVal, expDistance, expSum) {
        maxVal(n) max=! I(n, d)
        expDistance(n, d) = exp(I(n, d) - maxVal(n))
        expSum(n) +=! expDistance(n, d)
        O(n, d) = expDistance(n, d) / expSum(n)
      }
    )",
      "softmax",
      tc::MappingOptions::makeNaiveMappingOptions(),
      inputs,
      outputs);
}

TEST_F(ATenCompilationUnitTest, Concat) {
  at::Tensor a = at::CUDA(at::kFloat).rand({32, 16});
  at::Tensor b = at::CUDA(at::kFloat).rand({32, 16});
  std::vector<at::Tensor> inputs = {a, b};
  std::vector<at::Tensor> outputs;

  Check(
      R"(
      def concat(float(M, N) A, float(M, N) B) -> (O1) {
        O1(n, i, m) = i == 0 ? A(m, n) : B(m, n) where i in 0:2
      }
    )",
      "concat",
      tc::MappingOptions::makeNaiveMappingOptions(),
      inputs,
      outputs);
}

TEST_F(ATenCompilationUnitTest, Indexing) {
  at::Tensor a = at::CUDA(at::kFloat).rand({3, 4});
  at::Tensor b = at::CUDA(at::kInt).ones({2});
  std::vector<at::Tensor> inputs = {a, b};
  std::vector<at::Tensor> outputs;

  Check(
      R"(
      def indexing(float(H, W) input, int32(L) index) -> (output) {
          output(l, w) = input(index(l), w) where l in 0:2
      }
    )",
      "indexing",
      tc::MappingOptions::makeNaiveMappingOptions(),
      inputs,
      outputs);
}

TEST_F(ATenCompilationUnitTest, MatMul) {
  at::Tensor a = at::CUDA(at::kFloat).rand({3, 4});
  at::Tensor b = at::CUDA(at::kFloat).rand({4, 5});
  std::vector<at::Tensor> inputs = {a, b};
  std::vector<at::Tensor> outputs;

  Check(
      R"(
      def matmul(float(M,N) A, float(N,K) B) -> (output) {
        output(m, k) +=! A(m, nn) * B(nn, k)
      }
    )",
      "matmul",
      tc::MappingOptions::makeMlpMappingOptions(),
      inputs,
      outputs);

  at::Tensor diff = outputs[0].sub(a.mm(b));
  checkRtol(diff, inputs, N);
}

TEST_F(ATenCompilationUnitTest, MatMulInplace) {
  at::Tensor a = at::CUDA(at::kFloat).rand({3, 4});
  at::Tensor b = at::CUDA(at::kFloat).rand({4, 5});
  std::vector<at::Tensor> inputs = {a, b};
  at::Tensor c = at::CUDA(at::kFloat).rand({3, 5});

  std::vector<at::Tensor> outputs = {c.clone()};

  Check(
      R"(
      def matmul(float(M,N) A, float(N,K) B) -> (output) {
        output(m, k) += A(m, nn) * B(nn, k)
      }
    )",
      "matmul",
      tc::MappingOptions::makeMlpMappingOptions(),
      inputs,
      outputs);

  at::Tensor diff = outputs[0].sub(a.mm(b) + c);
  checkRtol(diff, inputs, N);
}

TEST_F(ATenCompilationUnitTest, Convolution2d) {
  at::Tensor I = at::CUDA(at::kFloat).rand({N, C, H, W});
  at::Tensor W1 = at::CUDA(at::kFloat).rand({O, C, KH, KW});
  at::Tensor B = at::CUDA(at::kFloat).rand({O});
  std::vector<at::Tensor> inputs = {I, W1, B};
  std::vector<at::Tensor> outputs;

  Check(
      R"(
      def convolution(float(N,C,H,W) I, float(O,C,KH,KW) W1, float(O) B)
      -> (tmp, O1) {
        tmp(n, o, h, w) +=! I(n, c, h + kh, w + kw) * W1(o, c, kh, kw)
        # this can be equivalently written with =,
        # but this line tests that we correctly handle
        # degenerate +=! that have no reduction dimensions
        O1(n, o, h, w) +=! tmp(n, o, h, w) + B(o)
      }
    )",
      "convolution",
      tc::MappingOptions::makeConvolutionMappingOptions(),
      inputs,
      outputs);

  at::Tensor expected = at::conv2d(I, W1, at::IntList({KH, KW}), B);
  at::Tensor diff = outputs[1].sub(expected);
  checkRtol(diff, inputs, C * KW * KH, 5e-7);
}

TEST_F(ATenCompilationUnitTest, Convolution2dStrided) {
  at::Tensor I = at::CUDA(at::kFloat).rand({N, C, H, W});
  at::Tensor W1 = at::CUDA(at::kFloat).rand({O, C, KH, KW});
  at::Tensor B = at::CUDA(at::kFloat).rand({O});
  std::vector<at::Tensor> inputs = {I, W1, B};
  std::vector<at::Tensor> outputs;

  constexpr static auto convolutionStrided = R"TC(
      def convolutionStrided(float(N,C,H,W) I, float(O,C,KH,KW) W1, float(O) B)
      -> (O1) {
        O1(n, o, h, w) +=! I(n, c, <sh> * h + kh, <sw> * w + kw) * W1(o, c, kh, kw)
        O1(n, o, h, w) = O1(n, o, h, w) + B(o)
      }
    )TC";

  std::string tcStr;
  tcStr = convolutionStrided;
  tcStr = tc::replaceString(tcStr, "<sh>", std::to_string(SH));
  tcStr = tc::replaceString(tcStr, "<sw>", std::to_string(SW));

  Check(
      tcStr,
      "convolutionStrided",
      tc::MappingOptions::makeConvolutionMappingOptions(),
      inputs,
      outputs);

  at::Tensor expected = at::conv2d(I, W1, at::IntList({KH, KW}), B);
  at::Tensor diff = outputs[0].sub(expected);
  // Approximate striding effect below, relax precision by factor 2
  checkRtol(diff, inputs, 2 * (C * KW * KH) / (SW * SH), 5e-7);
}

TEST_F(ATenCompilationUnitTest, Casts) {
  at::Tensor a = at::CUDA(at::kFloat).ones({2, 4});
  at::Tensor b = at::CUDA(at::kInt).tensor({}).assign_(4);
  a = a / 2.0 + 1;
  at::Tensor c = at::CUDA(at::kFloat).rand({3, 5});

  std::vector<at::Tensor> outputs;

  Check(
      R"(
      def cast(float(M,N) A, int32 four) -> (int32(M,N) output) {
        output(i,j) = int32(A(i,j) + four)
      }
    )",
      "cast",
      tc::MappingOptions::makeNaiveMappingOptions(),
      {a, b},
      outputs);
  auto r = outputs[0].sub(at::CUDA(at::kInt).ones({2, 4}) + 4).max().toLong();
  CHECK_EQ(r, 0);
}

int main(int argc, char** argv) {
  ::testing::InitGoogleTest(&argc, argv);
  ::gflags::ParseCommandLineFlags(&argc, &argv, true);
  ::google::InitGoogleLogging(argv[0]);
  setAtenSeed(tc::initRandomSeed(), at::Backend::CUDA);
  return RUN_ALL_TESTS();
}
