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
#include "tc/core/cuda/cuda.h"
#include "tc/core/cuda/cuda_tc_executor.h"
#include "tc/core/flags.h"
#include "tc/core/mapping_options.h"

DEFINE_string(proto_path, "", "Filename to load and store proto cache ");

TEST(BlockDiagPerm, SimpleAutotune) {
  // 1. Define and setup the TC compilation unit with CUDA memory
  // management backed by ATen tensors.
  std::string tc = R"TC(
# The following TCs (blockdiagperm2d and blockdiagperm2dinlined) illustrate
# how we would likely want to write blockdiagperm to synthesize a single
# kernel. However both versions currently fail to emit a good single cuda kernel.
#   1. blockdiagperm2d requires additional information to relax dependencies and
#     allow fusion
#   2. blockdiagperm2dinlined requires general LHS indexing
# A third version blockdiagperm2dfissioned_1/2 is a workaround by using 2
# independent TCs.
# This TC probably requires extra information to perform fusion which we do
# not know how to propagate at this point
# def blockdiagperm2d(float(B, K, NBYK) I, float(K, NBYK, NBYK) W, float(K, NBYK) IdxR, float(K, NBYK) IdxC)
#     -> (O1, O2) {
#     O1(b, k, nbyk1) +=!  I(b, k, r_nbyk0) * W(k, r_nbyk0, nbyk1)
#     O2(b, k, nbyk)   =  O1(b, Idxr(k, nbyk), Idxc(k, nbyk))
# }
# This TC requires LHS indexing which is a WIP + extra information that all
# accesses are parallel (i.e. (IdxR, IdxC) form a permutation)
# def blockdiagperm2dinlined(float(B, K, NBYK) I, float(K, NBYK, NBYK) W, float(K, NBYK) IdxR, float(K, NBYK) IdxC)
#     -> (O1) {
#     O1(b, IdxR(k, nbyk0), IdxC(k, nbyk0)) +=! I(b, k, r_nbyk0) * W(k, r_nbyk0, nbyk1)
# }

# This is the poor man's way of making things work today with a reshape
# operation in between (in framework land).
def blockdiagperm2dfissioned_1(float(B, K, NBYK) I, float(K, NBYK, NBYK) W) -> (O)
{
    O(b, k, nbyk1) +=! I(b, k, r_nbyk0) * W(k, r_nbyk0, nbyk1)
}
def blockdiagperm2dfissioned_2(float(B, N) I, int32(N) Idx) -> (O) {
    O(b, n) = I(b, Idx(n)) where n in 0:N
}
  )TC";
  tc::ATenCompilationUnit<tc::CudaTcExecutor> atCompl;
  atCompl.define(tc);

  // 1. Allocate and autotune
  at::Tensor I = at::CUDA(at::kFloat).rand({128, 10, 50});
  at::Tensor W = at::CUDA(at::kFloat).rand({10, 50, 50});
  auto options = tc::CudaMappingOptions::makeNaiveMappingOptions();
  tc::autotune::GeneticAutotunerATen geneticAutotuneATen(tc);
  auto bestOption = geneticAutotuneATen.tune(
      FLAGS_proto_path, "blockdiagperm2dfissioned_1", {I, W}, options);
  auto handle = atCompl.compile(
      "blockdiagperm2dfissioned_1", {I, W}, bestOption.getValue());
  std::vector<at::Tensor> outputs;
  auto duration =
      atCompl.run("blockdiagperm2dfissioned_1", {I, W}, outputs, handle, true);

  // 2. Allocate and autotune
  at::Tensor O = outputs[0].clone().resize_({128, 500});
  at::Tensor Idx = at::CPU(at::kInt).randperm({500}).toBackend(at::kCUDA);
  tc::autotune::GeneticAutotunerATen geneticAutotuneATen2(tc);
  auto bestOption2 = geneticAutotuneATen.tune(
      FLAGS_proto_path, "blockdiagperm2dfissioned_2", {O, Idx}, options);
  auto handle2 = atCompl.compile(
      "blockdiagperm2dfissioned_2", {O, Idx}, bestOption2.getValue());
  std::vector<at::Tensor> outputs2;
  auto duration2 = atCompl.run(
      "blockdiagperm2dfissioned_2", {O, Idx}, outputs2, handle2, true);

  // 3. Report best standalone times
  std::cout
      << "blockdiagperm2dfissioned_1 size I: " << I.sizes() << ", "
      << "size W: " << W.sizes() << " ran in: "
      << std::chrono::duration_cast<std::chrono::microseconds>(duration).count()
      << "us\n";
  std::cout << "blockdiagperm2dfissioned_2 size O: " << O.sizes() << ", "
            << "size Idx: " << Idx.sizes() << " ran in: "
            << std::chrono::duration_cast<std::chrono::microseconds>(duration2)
                   .count()
            << "us\n";

  // 4. Run unchecked one last time, use with:
  //   nvprof --profile-from-start off executable --use_nvprof=1
  {
    tc::CudaProfiler cp;
    atCompl.uncheckedRun({I, W}, outputs, handle);
    atCompl.uncheckedRun({O, Idx}, outputs2, handle2);
  }
}

// From root, run with:
//   ./build/examples/blockdiagperm --tuner_threads=10 --tuner_gen_pop_size=10
//   --tuner_gen_generations=3 --tuner_gen_number_elites=4
//   --proto_path="/tmp/blockdiagperm"
int main(int argc, char** argv) {
  ::testing::InitGoogleTest(&argc, argv);
  ::gflags::ParseCommandLineFlags(&argc, &argv, true);
  ::google::InitGoogleLogging(argv[0]);
  return RUN_ALL_TESTS();
}
