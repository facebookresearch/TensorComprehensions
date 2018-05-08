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

// These options were copied from GroupNormalization
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
def upsample(
    float(N, C, H, W) X, float(1) rheight, float(1) rwidth, float(1) height, float(1) width)
    -> (output, h1, w1, h1r, w1r, h1p, w1p, h1lambda, h0lambda, w1lambda, w0lambda)
{
    h1r(i) = rheight(0) * i where i in 0:H
    h1(i) = int32(h1r(i)) where i in 0:H
    h1p(i) = (h1(i) < (height(0) - 1)) ? 1 : 0 where i in 0:H
    h1lambda(i) = h1r(i) - h1(i) where i in 0:H
    h0lambda(i) = 1.0 - h1lambda(i) where i in 0:H

    w1r(j) = rwidth(0) * j where j in 0:W
    w1(j) = int32(w1r(j)) where j in 0:W
    w1p(j) = (w1(j) < (width(0) - 1)) ? 1 : 0 where j in 0:W
    w1lambda(j) = w1r(j) - w1(j) where j in 0:W
    w0lambda(j) = 1.0 - w1lambda(j) where j in 0:W

    # Maybe: split kernels here if fusion does not occur

    output(n, c, i, j) +=! h0lambda(i) * (w0lambda(i) * X(n, c, h1(i), w1(j)) +
        w1lambda(j) * X(n, c, h1(i), w1(j) + w1p(j))) +
        h1lambda(i) * (w0lambda(j) * X(n, c, h1(i) + h1p(i), w1(j)) +
        w1lambda(j) * X(n, c, h1(i) + h1p(i), w1(j) + w1p(j)))
      where i in 0:H, j in 0:W
}
  )TC";

  // 2. Allocate tensors with random data.
  auto N = 8, C = 4, H = 4, W = 8;
  auto widthScale = 2.0, heightScale = 2.0;

  auto outH = H * heightScale;
  auto outW = W * widthScale;
  auto rh = (outH > 1) ? (float)(H - 1) / (outH - 1) : 0.f;
  auto rw = (outW > 1) ? (float)(W - 1) / (outW - 1) : 0.f;

  at::Tensor X = makeATenTensor<Backend>({N, C, H, W});
  at::Tensor inputHeight = makeATenTensor<Backend>({1});
  at::Tensor inputWidth = makeATenTensor<Backend>({1});
  at::Tensor rheight = makeATenTensor<Backend>({1});
  at::Tensor rwidth = makeATenTensor<Backend>({1});
  at::Tensor h1 = makeATenTensor<Backend>({1});
  at::Tensor w1 = makeATenTensor<Backend>({1});
  at::Tensor h1r = makeATenTensor<Backend>({1});
  at::Tensor w1r = makeATenTensor<Backend>({1});
  at::Tensor h1p = makeATenTensor<Backend>({1});
  at::Tensor w1p = makeATenTensor<Backend>({1});
  at::Tensor h1lamada = makeATenTensor<Backend>({1});
  at::Tensor h0lamada = makeATenTensor<Backend>({1});
  at::Tensor w1lamada = makeATenTensor<Backend>({1});
  at::Tensor w0lamada = makeATenTensor<Backend>({1});

  inputHeight.fill_(H);
  inputWidth.fill_(W);
  rheight.fill_(rh);
  rwidth.fill_(rw);

  // 3. Run autotuning with evolutionary search starting from a naive option.
  auto baseOptions = FLAGS_use_best_options
      ? previouslyTunedBestOptions
      : Backend::MappingOptionsType::makeNaiveMappingOptions();
  tc::aten::ATenAutotuner<Backend, tc::autotune::GeneticSearch>
      geneticAutotuneATen(tc);
  auto bestOption = geneticAutotuneATen.tune(
      "upsample", {X, rheight, rwidth, inputHeight, inputWidth}, baseOptions, FLAGS_proto_path);
  CHECK_GT(bestOption.size(), 0u);

  // 4. Compile and run the TC with the best option.
  auto pExecutor = tc::aten::compile<Backend>(
      tc, "upsample", {X, rheight, rwidth, inputHeight, inputWidth}, bestOption[0]);
  auto outputs =
      tc::aten::prepareOutputs(tc, "upsample", {X, rheight, rwidth, inputHeight, inputWidth});
  auto timings = tc::aten::profile(*pExecutor, {X, rheight, rwidth, inputHeight, inputWidth}, outputs);
  std::cout << "upsample size X: " << X.sizes() << ", "
            << " ran in: " << timings.kernelRuntime.toMicroSeconds() << "us\n";
  LOG(INFO) << "best option: " << bestOption << "\n";
}

TEST(UpSampleGPU, SimpleAutotune) {
  testOnBackend<tc::CudaBackend>();
}

/*
  Short run: from build dir, run with:
    ./tc/examples/upsample --tuner_threads=10 \
    --tuner_gen_pop_size=10 --tuner_gen_generations=3 \
    --tuner_gen_number_elites=4 \
    --proto_path="/tmp/upsample"

  Long run: from build dir, run with:
    ./tc/examples/upsample --tuner_threads=10 \
    --proto_path="/tmp/upsample"
*/
int main(int argc, char** argv) {
  ::testing::InitGoogleTest(&argc, argv);
  ::gflags::ParseCommandLineFlags(&argc, &argv, true);
  ::google::InitGoogleLogging(argv[0]);
  tc::aten::setAtenSeed(tc::initRandomSeed(), at::Backend::CUDA);
  return RUN_ALL_TESTS();
}
