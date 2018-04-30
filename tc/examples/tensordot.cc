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
#include "tc/autotuner/genetic_autotuner_aten.h"
#include "tc/core/cuda/cuda_mapping_options.h"
#include "tc/core/flags.h"

#include "../test/test_harness_aten_cuda.h"

DEFINE_string(proto_path, "", "Filename to load and store proto cache ");

TEST(TensorDot, SimpleAutotune) {
  // 1. Define and setup the TC compilation unit with CUDA memory
  // management backed by ATen tensors.
  std::string tc = R"TC(
def tensordot(float(N, C1, C2, H, W) I0,
              float(N, C2, C3, H, W) I1)  -> (O)
{
    O(n, c1, c3, h, w) +=! I0(n, c1, r_c2, h, w) * I1(n, r_c2, c3, h, w)
}
  )TC";
  tc::ATenCompilationUnit<tc::CudaTcExecutor> atCompl;
  atCompl.define(tc);

  // 2. Allocate tensors with random data.
  at::Tensor I0 = at::CUDA(at::kFloat).rand({16, 8, 16, 17, 25});
  at::Tensor I1 = at::CUDA(at::kFloat).rand({16, 16, 2, 17, 25});

  // 3. Run autotuning with evolutionary search starting from a naive option.
  auto naiveOptions = tc::CudaMappingOptions::makeNaiveMappingOptions();
  tc::autotune::GeneticAutotunerATen geneticAutotuneATen(tc);
  auto bestOption = geneticAutotuneATen.tune(
      FLAGS_proto_path, "tensordot", {I0, I1}, naiveOptions);

  // 4. Compile and run the TC with the best option.
  // Outputs get allocated; could also be pre-allocated and passed.
  auto handle = atCompl.compile("tensordot", {I0, I1}, bestOption.getValue());
  std::vector<at::Tensor> outputs;
  auto duration = atCompl.run("tensordot", {I0, I1}, outputs, handle, true);
  std::cout
      << "tensordot size I0: " << I0.sizes() << ", "
      << "size I1: " << I1.sizes() << " ran in: "
      << std::chrono::duration_cast<std::chrono::microseconds>(duration).count()
      << "us\n";

  // 5. Optionally, perform precision checks against a ref. implementation.
  // TODO.

  // 6. Reuse bestOptions from autotuning on another kernel
  for (auto sizes : std::vector<std::pair<at::IntList, at::IntList>>{
           {{4, 9, 7, 16, 14}, {4, 7, 3, 16, 14}},
           {{8, 5, 11, 10, 10}, {8, 11, 16, 10, 10}},
       }) {
    at::Tensor I0 = at::CUDA(at::kFloat).rand(sizes.first);
    at::Tensor I1 = at::CUDA(at::kFloat).rand(sizes.second);
    auto handle = atCompl.compile("tensordot", {I0, I1}, bestOption.getValue());
    std::vector<at::Tensor> outputs;
    auto duration = atCompl.run("tensordot", {I0, I1}, outputs, handle, true);
    std::cout << "tensordot size I0: " << I0.sizes() << ", "
              << "size I1: " << I1.sizes() << " ran in: "
              << std::chrono::duration_cast<std::chrono::microseconds>(duration)
                     .count()
              << "us\n";
  }
}

// From root, run with:
//   ./build/examples/tensordot --tuner_threads=10 --tuner_gen_pop_size=10
//   --tuner_gen_generations=3 --tuner_gen_number_elites=4
//   --proto_path="/tmp/tensordot"
int main(int argc, char** argv) {
  ::testing::InitGoogleTest(&argc, argv);
  ::gflags::ParseCommandLineFlags(&argc, &argv, true);
  ::google::InitGoogleLogging(argv[0]);
  setAtenSeed(tc::initRandomSeed(), at::Backend::CUDA);
  return RUN_ALL_TESTS();
}
