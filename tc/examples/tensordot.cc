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
#include "tc/aten/aten_autotuner.h"
#include "tc/aten/aten_compiler.h"
#include "tc/autotuner/genetic_search.h"
#include "tc/core/cpu/cpu_mapping_options.h"
#include "tc/core/cpu/cpu_tc_executor.h"
#include "tc/core/cuda/cuda_mapping_options.h"
#include "tc/core/cuda/cuda_tc_executor.h"
#include "tc/core/flags.h"

DEFINE_string(proto_path, "", "Filename to load and store proto cache ");

template <typename Backend>
at::Tensor makeATenTensor(at::ArrayRef<long int> sizes);

template <>
at::Tensor makeATenTensor<tc::CudaBackend>(at::ArrayRef<long int> sizes) {
  return at::CUDA(at::kFloat).rand(sizes);
}

template <>
at::Tensor makeATenTensor<tc::CpuBackend>(at::ArrayRef<long int> sizes) {
  return at::CPU(at::kFloat).rand(sizes);
}

template <typename Backend>
void doit() {
  // 1. Define and setup the TC compilation unit with CUDA memory
  // management backed by ATen tensors.
  std::string tc = R"TC(
def tensordot(float(N, C1, C2, H, W) I0,
              float(N, C2, C3, H, W) I1)  -> (O)
{
    O(n, c1, c3, h, w) +=! I0(n, c1, r_c2, h, w) * I1(n, r_c2, c3, h, w)
}
  )TC";

  // 2. Allocate tensors with random data.
  at::Tensor I0 = makeATenTensor<Backend>({16, 8, 16, 17, 25});
  at::Tensor I1 = makeATenTensor<Backend>({16, 16, 2, 17, 25});

  // 3. Run autotuning with evolutionary search starting from a naive option.
  auto naiveOptions = Backend::MappingOptionsType::makeNaiveMappingOptions();
  tc::aten::ATenAutotuner<Backend, tc::autotune::GeneticSearch>
      geneticAutotuneATen(tc);
  auto bestOption = geneticAutotuneATen.tune(
      "tensordot", {I0, I1}, naiveOptions, FLAGS_proto_path);
  CHECK_GT(bestOption.size(), 0);

  // 4. Compile and run the TC with the best option.
  // Outputs get allocated; could also be pre-allocated and passed.
  auto pExecutor =
      tc::aten::compile<Backend>(tc, "tensordot", {I0, I1}, bestOption[0]);
  auto outputs = tc::aten::prepareOutputs(tc, "tensordot", {I0, I1});
  auto timings = tc::aten::profile(*pExecutor, {I0, I1}, outputs);
  std::cout << "tensordot size I0: " << I0.sizes() << ", "
            << "size I1: " << I1.sizes() << " ran in: "
            << std::chrono::duration_cast<std::chrono::microseconds>(
                   timings.kernelRuntime)
                   .count()
            << "us\n";

  // 5. Optionally, perform precision checks against a ref. implementation.
  // TODO.

  // 6. Reuse bestOptions from autotuning on another kernel
  for (auto sizes : std::vector<std::pair<at::IntList, at::IntList>>{
           {{4, 9, 7, 16, 14}, {4, 7, 3, 16, 14}},
           {{8, 5, 11, 10, 10}, {8, 11, 16, 10, 10}},
       }) {
    at::Tensor I0 = makeATenTensor<Backend>(sizes.first);
    at::Tensor I1 = makeATenTensor<Backend>(sizes.second);
    auto pExecutor =
        tc::aten::compile<Backend>(tc, "tensordot", {I0, I1}, bestOption[0]);
    auto outputs = tc::aten::prepareOutputs(tc, "tensordot", {I0, I1});
    auto timings = tc::aten::profile(*pExecutor, {I0, I1}, outputs);
    std::cout << "tensordot size I0: " << I0.sizes() << ", "
              << "size I1: " << I1.sizes() << " ran in: "
              << std::chrono::duration_cast<std::chrono::microseconds>(
                     timings.kernelRuntime)
                     .count()
              << "us\n";
  }
}

TEST(TensorDotCPU, SimpleAutotune) {
  doit<tc::CpuBackend>();
}

TEST(TensorDotGPU, SimpleAutotune) {
  doit<tc::CudaBackend>();
}

// From root, run with:
//   ./build/examples/tensordot --tuner_threads=10 --tuner_gen_pop_size=10
//   --tuner_gen_generations=3 --tuner_gen_number_elites=4
//   --proto_path="/tmp/tensordot"
int main(int argc, char** argv) {
  ::testing::InitGoogleTest(&argc, argv);
  ::gflags::ParseCommandLineFlags(&argc, &argv, true);
  ::google::InitGoogleLogging(argv[0]);
  return RUN_ALL_TESTS();
}
