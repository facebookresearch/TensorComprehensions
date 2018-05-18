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

#include "common.h"

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
DEFINE_bool(
    use_best_options,
    false,
    "Start from hardcoded best options; if false start from naive options ");

// These options were found by a longer tuning run on a Pascal card.
// More specifically: Quadro GP100
auto previouslyTunedBestOptions =
    tc::CudaMappingOptions::makeNaiveMappingOptions()
        .outerScheduleFusionStrategy(tc::FusionStrategy::Max)
        .outerScheduleAllowSkewing(false)
        .outerSchedulePositiveOrthant(true)
        .intraTileScheduleFusionStrategy(
            tc::FusionStrategy::Preserve3Coincident)
        .intraTileScheduleAllowSkewing(false)
        .intraTileSchedulePositiveOrthant(true)
        .tile(6, 1, 24)
        .unroll(16)
        .tileImperfectlyNested(false)
        .matchLibraryCalls(false)
        .mapToThreads(48, 6)
        .mapToBlocks(256, 32)
        .useSharedMemory(true)
        .usePrivateMemory(true)
        .unrollCopyShared(false);

template <typename Backend>
void testOnBackend() {
  // 1. Define and setup the TC compilation unit with CUDA memory
  // management backed by ATen tensors.
  std::string tc = R"TC(
def group_normalization(
    float(N, G, D, H, W) I, float(G, D) gamma, float(G, D) beta)
    -> (O, mean, var)
{
# This first implementation uses the formula var = E((x - mean)^2).
# On P100, the autotuner finds a 2.6ms best version
#   mean(n, g) +=! I(n, g, r_d, r_h, r_w)
#   mean(n, g)  = mean(n, g) / (D * H * W)
#    var(n, g) +=! (I(n, g, r_d, r_h, r_w) - mean(n, g))
#                * (I(n, g, r_d, r_h, r_w) - mean(n, g))
#    var(n, g)  =  var(n, g) / (D * H * W)
#   O(n, g, d, h, w) =
#       gamma(g, d) * (I(n, g, d, h, w) - mean(n, g)) * rsqrt(var(n, g) + 1e-5) + beta(g, d)

# This second implementation uses the formula var = E(x^2) - mean^2.
# This time, on a P100, the autotuner finds a 1.6ms best version.
    mean(n, g) +=! I(n, g, r_d, r_h, r_w)
    mean(n,g)   = mean(n,g) / (D * H * W)
     var(n, g) +=! I(n, g, r_d, r_h, r_w) * I(n, g, r_d, r_h, r_w)
     var(n, g)  =  var(n, g) / (D * H * W) - mean(n,g) * mean(n,g)
    O(n, g, d, h, w) = gamma(g, d)
      * ( I(n, g, d, h, w) - mean(n, g) )
      * rsqrt( var(n, g) + 1e-5 )
      + beta(g, d)
}
  )TC";

  // 2. Allocate tensors with random data.
  uint32_t N = 32, C = 512, G = 32, D = C / G, H = 48, W = 48;
  at::Tensor I = makeATenTensor<Backend>({N, G, D, H, W});
  at::Tensor gamma = makeATenTensor<Backend>({G, D});
  at::Tensor beta = makeATenTensor<Backend>({G, D});

  // 3. Run autotuning with evolutionary search starting from a naive option.
  auto baseOptions = FLAGS_use_best_options
      ? previouslyTunedBestOptions
      : Backend::MappingOptionsType::makeNaiveMappingOptions();
  tc::aten::ATenAutotuner<Backend, tc::autotune::GeneticSearch>
      geneticAutotuneATen(tc);
  auto bestOption = geneticAutotuneATen.tune(
      "group_normalization", {I, gamma, beta}, baseOptions, FLAGS_proto_path);
  CHECK_GT(bestOption.size(), 0u);

  // 4. Compile and run the TC with the best option.
  // Outputs get allocated; could also be pre-allocated and passed.
  auto pExecutor = tc::aten::compile<Backend>(
      tc, "group_normalization", {I, gamma, beta}, bestOption[0]);
  auto outputs =
      tc::aten::prepareOutputs(tc, "group_normalization", {I, gamma, beta});
  auto timings = tc::aten::profile(*pExecutor, {I, gamma, beta}, outputs);
  std::cout << "group_normalization size I: " << I.sizes() << ", "
            << " ran in: " << timings.kernelRuntime.toMicroSeconds() << "us\n";
  LOG(INFO) << "best option: " << bestOption << "\n";
}

TEST(GroupNormalizationGPU, SimpleAutotune) {
  testOnBackend<tc::CudaBackend>();
}

/*
  Short run: from build dir, run with:
    ./tc/examples/group_normalization --tuner_threads=10 \
    --tuner_gen_pop_size=10 --tuner_gen_generations=3 \
    --tuner_gen_number_elites=4 \
    --proto_path="/tmp/group_normalization"

  Long run: from build dir, run with:
    ./tc/examples/group_normalization --tuner_threads=10 \
    --proto_path="/tmp/group_normalization"
*/
int main(int argc, char** argv) {
  ::testing::InitGoogleTest(&argc, argv);
  ::gflags::ParseCommandLineFlags(&argc, &argv, true);
  ::google::InitGoogleLogging(argv[0]);
  tc::aten::setAtenSeed(tc::initRandomSeed(), at::Backend::CUDA);
  return RUN_ALL_TESTS();
}
