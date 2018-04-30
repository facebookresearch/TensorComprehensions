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
#include "tc/core/cuda/cuda_mapping_options_cpp_printer.h"
#include "tc/core/cuda/cuda_tc_executor.h"

#include "test_harness_aten_cuda.h"

TEST(ATenCompilationCacheTest, Matmul) {
  auto tc = R"(
def matmul(float(M,K) A, float(K,N) B) -> (output) {
    output(m, n) +=! A(m, r_k) * B(r_k, n)
}
  )";

  // test matmul
  LOG(INFO) << "Testing 1st matmul";
  at::Tensor a = at::CUDA(at::kFloat).rand({3, 4});
  at::Tensor b = at::CUDA(at::kFloat).rand({4, 5});
  std::vector<at::Tensor> inputs = {a, b};

  auto mappingOptions = tc::CudaMappingOptions::makeMlpMappingOptions();
  auto pExecutor =
      tc::aten::compile<tc::CudaBackend>(tc, "matmul", inputs, mappingOptions);
  auto outputs = tc::aten::prepareOutputs(tc, "matmul", inputs);
  tc::aten::run(*pExecutor, inputs, outputs);
  at::Tensor diff = outputs[0].sub(a.mm(b));
  checkRtol(diff, inputs, 4);

  // running matmul again to hit cache
  LOG(INFO) << "Testing 1st matmul again";
  auto outputs1 = tc::aten::prepareOutputs(tc, "matmul", inputs);
  pExecutor =
      tc::aten::compile<tc::CudaBackend>(tc, "matmul", inputs, mappingOptions);
  tc::aten::run(*pExecutor, inputs, outputs1);
  diff = outputs1[0].sub(a.mm(b));
  checkRtol(diff, inputs, 4); // reduction size is dimension of n

  // test matmul on different inputs
  LOG(INFO) << "Testing 2nd matmul with different inputs";
  at::Tensor a2 = at::CUDA(at::kFloat).rand({4, 8});
  at::Tensor b2 = at::CUDA(at::kFloat).rand({8, 7});
  inputs = {a2, b2};
  auto outputs2 = tc::aten::prepareOutputs(tc, "matmul", inputs);
  pExecutor =
      tc::aten::compile<tc::CudaBackend>(tc, "matmul", inputs, mappingOptions);
  tc::aten::run(*pExecutor, inputs, outputs2);
  diff = outputs2[0].sub(a2.mm(b2));
  checkRtol(diff, inputs, 8); // reduction size is dimension of n

  // run the first cached matmul again
  LOG(INFO) << "Testing 1st cached matmul again";
  inputs = {a, b};
  auto outputs3 = tc::aten::prepareOutputs(tc, "matmul", inputs);
  pExecutor =
      tc::aten::compile<tc::CudaBackend>(tc, "matmul", inputs, mappingOptions);
  tc::aten::run(*pExecutor, inputs, outputs3);
  diff = outputs3[0].sub(a.mm(b));
  checkRtol(diff, inputs, 4); // reduction size is dimension of n
}

int main(int argc, char** argv) {
  ::testing::InitGoogleTest(&argc, argv);
  ::gflags::ParseCommandLineFlags(&argc, &argv, true);
  ::google::InitGoogleLogging(argv[0]);
  return RUN_ALL_TESTS();
}
