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
#include "wavenet.h"

#include <iostream>
#include <string>
#include <vector>

#include <gflags/gflags.h>
#include <glog/logging.h>
#include <gtest/gtest.h>

#include "tc/aten/aten.h"

#include "tc/aten/aten_compiler.h"
#include "tc/core/cuda/cuda_mapping_options.h"

#include "../test/caffe2/cuda/test_harness.h"
#include "../test/caffe2/test_harness.h"
#include "../test/test_harness_aten_cuda.h"
#include "benchmark_fixture.h"

#include "tc/c2/context.h"
#include "tc/core/cuda/cuda.h"
#include "tc/core/flags.h"

using namespace caffe2;

DEFINE_uint32(B, 1, "Batch size");
DEFINE_uint32(
    RESIDUAL_C,
    32,
    "Residual channels (i.e. WaveNet block input channels)");
DEFINE_uint32(
    DILATION_C,
    32,
    "Dilation channels (i.e. WaveNet block channels after dilated convolution)");
DEFINE_uint32(
    SKIP_C,
    32,
    "Skip channels (i.e. WaveNet block channels in the skip tensor)");
DEFINE_uint32(
    RECEPTIVE_FIELD,
    4000,
    "https://arxiv.org/pdf/1609.03499.pdf paper mentions 16K samples per second"
    "and a receptive field of 240ms so we approx. set the default to 4000)");
DEFINE_uint32(DILATION_FACTOR, 1, "Powers of 2 from 1 to 512 in the paper");

// https://arxiv.org/pdf/1609.03499.pdf paper mentions 16K samples per second
// and a receptive field of 240ms so about 4K RECEPTIVE_FIELD
class WaveNet : public Benchmark {
 protected:
  uint32_t B;
  uint32_t RESIDUAL_C;
  uint32_t DILATION_C;
  uint32_t SKIP_C;
  uint32_t RECEPTIVE_FIELD;
  uint32_t DILATION_FACTOR; // 2^layer where layer in 0:10

 public:
  void Init(
      uint32_t b,
      uint32_t residual_c,
      uint32_t dilation_c,
      uint32_t skip_c,
      uint32_t receptive_field,
      uint32_t dilation_factor) {
    B = b;
    RESIDUAL_C = residual_c;
    DILATION_C = dilation_c;
    SKIP_C = skip_c;
    RECEPTIVE_FIELD = receptive_field;
    DILATION_FACTOR = dilation_factor;
  }
  void runWaveNet1(const tc::CudaMappingOptions& options);
};

void WaveNet::runWaveNet1(const tc::CudaMappingOptions& options) {
  at::Tensor data = at::CUDA(at::kFloat).rand({B, RESIDUAL_C, RECEPTIVE_FIELD});
  at::Tensor filterWeight =
      at::CUDA(at::kFloat).rand({DILATION_C, RESIDUAL_C, 2});
  at::Tensor filterBias = at::CUDA(at::kFloat).rand({DILATION_C});
  at::Tensor gateWeight =
      at::CUDA(at::kFloat).rand({DILATION_C, RESIDUAL_C, 2});
  at::Tensor gateBias = at::CUDA(at::kFloat).rand({DILATION_C});
  at::Tensor resWeight = at::CUDA(at::kFloat).rand({RESIDUAL_C, DILATION_C});
  at::Tensor resBias = at::CUDA(at::kFloat).rand({RESIDUAL_C});
  at::Tensor skipWeight = at::CUDA(at::kFloat).rand({SKIP_C, DILATION_C});
  at::Tensor skipBias = at::CUDA(at::kFloat).rand({SKIP_C});
  at::Tensor dilation = at::CUDA(at::kFloat).rand({DILATION_FACTOR});

  std::vector<at::Tensor> inputs = {data,
                                    filterWeight,
                                    filterBias,
                                    gateWeight,
                                    gateBias,
                                    resWeight,
                                    resBias,
                                    skipWeight,
                                    skipBias,
                                    dilation};

  std::vector<tc::CudaMappingOptions> bestOptions{options};
  if (FLAGS_autotune) {
    bestOptions = autotune(
        FLAGS_save_tuner_proto_prefix + std::string("/wavenet_1_cache"),
        FLAGS_save_tuner_proto_prefix + std::string("/wavenet_1_best"),
        tc::TC_WAVENET,
        tc::TC_WAVENET1_NAME,
        inputs,
        options);
    TC_CHECK_GE(bestOptions.size(), 1u);
  }
  Check(tc::TC_WAVENET, tc::TC_WAVENET1_NAME, bestOptions[0], inputs);
}

/// WaveNet 1 block
// Generic
TEST_F(WaveNet, WaveNet1) {
  Init(
      FLAGS_B,
      FLAGS_RESIDUAL_C,
      FLAGS_DILATION_C,
      FLAGS_SKIP_C,
      FLAGS_RECEPTIVE_FIELD,
      FLAGS_DILATION_FACTOR);
  runWaveNet1(tc::CudaMappingOptions::makeNaiveMappingOptions());
}

// P100
TEST_F(
    WaveNet,
    WaveNet1_P100_autotuned_B_1_RES_32_DIL_32_SKIP_256_REC_4000_F_1) {
  Init(1, 32, 32, 256, 4000, 1);
  runWaveNet1(
      tc::options_WaveNet1_P100_autotuned_B_1_RES_32_DIL_32_SKIP_256_REC_4000_F_1);
}

TEST_F(
    WaveNet,
    WaveNet1_P100_autotuned_B_1_RES_32_DIL_32_SKIP_256_REC_4000_F_32) {
  Init(1, 32, 32, 256, 4000, 32);
  runWaveNet1(
      tc::options_WaveNet1_P100_autotuned_B_1_RES_32_DIL_32_SKIP_256_REC_4000_F_32);
}

// V100
TEST_F(
    WaveNet,
    WaveNet1_V100_autotuned_B_1_RES_32_DIL_32_SKIP_256_REC_4000_F_1) {
  Init(1, 32, 32, 256, 4000, 1);
  runWaveNet1(
      tc::options_WaveNet1_V100_autotuned_B_1_RES_32_DIL_32_SKIP_256_REC_4000_F_1);
}

TEST_F(
    WaveNet,
    WaveNet1_V100_autotuned_B_1_RES_32_DIL_32_SKIP_256_REC_4000_F_32) {
  Init(1, 32, 32, 256, 4000, 32);
  runWaveNet1(
      tc::options_WaveNet1_V100_autotuned_B_1_RES_32_DIL_32_SKIP_256_REC_4000_F_32);
}

int main(int argc, char** argv) {
  ::testing::InitGoogleTest(&argc, argv);
  ::gflags::ParseCommandLineFlags(&argc, &argv, true);
  ::google::InitGoogleLogging(argv[0]);
  tc::aten::setAtenSeed(tc::initRandomSeed(), at::Backend::CUDA);
  return RUN_ALL_TESTS();
}
