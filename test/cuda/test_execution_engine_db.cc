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
#include "tc/core/cuda/cuda_tc_executor.h"

#include "test_harness_aten_cuda.h"

TEST(ATenCompilationDbTest, MultiTc) {
  static constexpr uint32_t N = 8, C = 16, O = 6, H = 24, W = 27, KH = 3,
                            KW = 3;
  tc::ATenCompilationUnit<tc::CudaTcExecutor> atCompl;
  auto tc = R"(
def matmul(float(M,K) A, float(K,N) B) -> (output) {
    output(m, n) +=! A(m, r_k) * B(r_k, n)
}
def convolution(float(N,C,H,W) I, float(O,C,KH,KW) W1, float(O) B)
-> (tmp, O1) {
    tmp(n, o, h, w) +=!  I(n, r_c, h + r_kh, w + r_kw) * W1(o, r_c, r_kh, r_kw)
    O1(n, o, h, w)   = tmp(n, o, h, w) + B(o)
}
  )";

  atCompl.define(tc);

  // test matmul
  LOG(INFO) << "Testing matmul";
  at::Tensor a = at::CUDA(at::kFloat).rand({3, 4});
  at::Tensor b = at::CUDA(at::kFloat).rand({4, 5});
  std::vector<at::Tensor> inputs = {a, b};
  std::vector<at::Tensor> outputs;
  auto mappingOptions = tc::CudaMappingOptions::makeMlpMappingOptions();
  auto handle = atCompl.compile("matmul", inputs, mappingOptions);
  atCompl.run("matmul", inputs, outputs, handle);
  at::Tensor diff = outputs[0].sub(a.mm(b));
  checkRtol(diff, inputs, 4);

  // test convolution - non-strided
  LOG(INFO) << "Testing convolution2d";
  at::Tensor I = at::CUDA(at::kFloat).rand({N, C, H, W});
  at::Tensor W1 = at::CUDA(at::kFloat).rand({O, C, KH, KW});
  at::Tensor B = at::CUDA(at::kFloat).rand({O});
  std::vector<at::Tensor> inputs1 = {I, W1, B};
  std::vector<at::Tensor> outputs1;
  mappingOptions = tc::CudaMappingOptions::makeGroupConvolutionMappingOptions();
  handle = atCompl.compile("convolution", inputs1, mappingOptions);
  atCompl.run("convolution", inputs1, outputs1, handle);
  at::Tensor expected = at::conv2d(I, W1, B);
  at::Tensor diff1 = outputs1[1].sub(expected);
  checkRtol(diff1, inputs1, C * KH * KW, 1e-6);
}

int main(int argc, char** argv) {
  ::testing::InitGoogleTest(&argc, argv);
  ::gflags::ParseCommandLineFlags(&argc, &argv, true);
  ::google::InitGoogleLogging(argv[0]);
  return RUN_ALL_TESTS();
}
