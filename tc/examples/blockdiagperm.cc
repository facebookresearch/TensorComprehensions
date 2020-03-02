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
#include "tc/autotuner/options_cache.h"
#include "tc/core/check.h"
#include "tc/core/cuda/cuda_mapping_options.h"
#include "tc/core/cuda/cuda_tc_executor.h"
#include "tc/core/flags.h"
#include "tc/core/tensor.h"

DEFINE_string(
    options_cache,
    "",
    "Filename to load and store best options cache ");

using ATenGeneticCudaTuner =
    tc::aten::ATenAutotuner<tc::CudaBackend, tc::autotune::GeneticSearch>;

using tc::autotune::OptionsCache;

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

  // 1. Allocate and autotune blockdiagperm2dfissioned_1 starting from best
  // options stored in cache (if any).
  at::Tensor I = at::CUDA(at::kFloat).rand({128, 10, 50});
  at::Tensor W = at::CUDA(at::kFloat).rand({10, 50, 50});
  std::vector<tc::CudaMappingOptions> options =
      tc::autotune::loadTopKFromCacheFile<tc::CudaBackend>(
          tc,
          "blockdiagperm2dfissioned_1",
          FLAGS_options_cache,
          extractRawPtrs(tc::aten::makeDLConstTensors({I, W})),
          1);
  if (options.size() == 0) {
    options = {tc::CudaMappingOptions::makeNaiveMappingOptions()};
  }

  ATenGeneticCudaTuner geneticAutotuneATen(tc);
  auto bestOption =
      geneticAutotuneATen.tune("blockdiagperm2dfissioned_1", {I, W}, options);
  TC_CHECK_GT(bestOption.size(), 0u);

  auto pExecutor = tc::aten::compile<tc::CudaBackend>(
      tc, "blockdiagperm2dfissioned_1", {I, W}, bestOption[0]);
  auto outputs =
      tc::aten::prepareOutputs(tc, "blockdiagperm2dfissioned_1", {I, W});
  auto timings = tc::aten::profile(*pExecutor, {I, W}, outputs);

  // 2. Allocate and autotune blockdiagperm2dfissioned_2 starting from best
  // options stored in cache (if any).
  at::Tensor O = outputs[0].clone().resize_({128, 500});
  at::Tensor Idx = at::CPU(at::kInt).randperm({500}).toBackend(at::kCUDA);
  auto options2 = tc::autotune::loadTopKFromCacheFile<tc::CudaBackend>(
      tc,
      "blockdiagperm2dfissioned_2",
      FLAGS_options_cache,
      extractRawPtrs(tc::aten::makeDLConstTensors({O, Idx})),
      1);
  if (options2.size() == 0) {
    options2 = {tc::CudaMappingOptions::makeNaiveMappingOptions()};
  }
  auto bestOption2 = geneticAutotuneATen.tune(
      "blockdiagperm2dfissioned_2", {O, Idx}, options2);
  TC_CHECK_GT(bestOption2.size(), 0u);

  auto pExecutor2 = tc::aten::compile<tc::CudaBackend>(
      tc, "blockdiagperm2dfissioned_2", {O, Idx}, bestOption2[0]);
  auto outputs2 =
      tc::aten::prepareOutputs(tc, "blockdiagperm2dfissioned_2", {O, Idx});
  auto timings2 = tc::aten::profile(*pExecutor2, {O, Idx}, outputs2);

  // 3. Report best standalone times
  std::cout << "blockdiagperm2dfissioned_1 size I: " << I.sizes() << ", "
            << "size W: " << W.sizes()
            << " ran in: " << timings.kernelRuntime.toMicroSeconds() << "us\n";
  std::cout << "blockdiagperm2dfissioned_2 size O: " << O.sizes() << ", "
            << "size Idx: " << Idx.sizes()
            << " ran in: " << timings2.kernelRuntime.toMicroSeconds() << "us\n";

  // 4. Run unchecked one last time, use with:
  //   nvprof --profile-from-start off executable --use_nvprof=1
  {
    tc::CudaProfiler cp;
    tc::aten::uncheckedRun(*pExecutor, {I, W}, outputs);
    tc::aten::uncheckedRun(*pExecutor2, {O, Idx}, outputs2);
  }

  // 5. Explicitly store the best found options in cache so we can reinforce
  // later if required
  if (not FLAGS_options_cache.empty()) {
    tc::autotune::appendTopKToCacheFile(
        *geneticAutotuneATen.optionsCache,
        FLAGS_options_cache,
        tc::FLAGS_tuner_save_best_candidates_count);
  }
}

// Run iteratively with the following options to reinforce short tuning runs:
//   ./build/examples/blockdiagperm --tuner_threads=10 --tuner_gen_pop_size=10
//   --tuner_gen_generations=3 --tuner_gen_number_elites=4
//   --options_cache="/tmp/blockdiagperm.options"
int main(int argc, char** argv) {
  ::testing::InitGoogleTest(&argc, argv);
  ::gflags::ParseCommandLineFlags(&argc, &argv, true);
  ::google::InitGoogleLogging(argv[0]);
  tc::aten::setAtenSeed(tc::initRandomSeed(), at::Backend::CUDA);
  return RUN_ALL_TESTS();
}
