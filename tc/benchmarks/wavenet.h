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
#pragma once

#include "tc/aten/aten.h"
#include "tc/core/cuda/cuda_mapping_options.h"

namespace tc {
constexpr static auto TC_WAVENET1_NAME = "wavenet1";
constexpr static auto TC_WAVENET = R"TC(
# Original data is float(B, C, RECEPTIVE_FIELD) and undergoes a \
# Conv1d to become float(B, RESIDUAL_C, RECEPTIVE_FIELD)

def wavenet1(
    float(B, RESIDUAL_C, RECEPTIVE_FIELD) Data,
    float(DILATION_C, RESIDUAL_C, 2) FilterWeight,
    float(DILATION_C) FilterBias,
    float(DILATION_C, RESIDUAL_C, 2) GateWeight,
    float(DILATION_C) GateBias,
    float(RESIDUAL_C, DILATION_C) ResWeight,
    float(RESIDUAL_C) ResBias,
    float(SKIP_C, DILATION_C) SkipWeight,
    float(SKIP_C) SkipBias,
    float(DILATION_FACTOR) Dilation)
    -> (FilterOut, GateOut, NonLin, Res, Skip)
{
    FilterOut(b, dilation_c, rf)   = FilterBias(dilation_c)
        where b in 0:B, dilation_c in 0:DILATION_C, rf in 0:RECEPTIVE_FIELD
    FilterOut(b, dilation_c, rf)  += Data(b, r_residual_c, rf) * FilterWeight(dilation_c, r_residual_c, 1) +
        (
          (rf - DILATION_FACTOR >= 0) ?
            Data(b, r_residual_c, rf - DILATION_FACTOR) * FilterWeight(dilation_c, r_residual_c, 0) :
            float(0)
        )
        where rf in 0:RECEPTIVE_FIELD

    GateOut(b, dilation_c, rf)   = GateBias(dilation_c)
        where b in 0:B, dilation_c in 0:DILATION_C, rf in 0:RECEPTIVE_FIELD
    GateOut(b, dilation_c, rf)  += Data(b, r_residual_c, rf) * GateWeight(dilation_c, r_residual_c, 1) +
        (
          (rf - DILATION_FACTOR >= 0) ?
            Data(b, r_residual_c, rf - DILATION_FACTOR) * GateWeight(dilation_c, r_residual_c, 0) :
            float(0)
        )
        where rf in 0:RECEPTIVE_FIELD

    NonLin(b, dilation_c, rf)   =         tanh(FilterOut(b, dilation_c, rf))
        where rf in 0:RECEPTIVE_FIELD
    NonLin(b, dilation_c, rf)  *= 1 / (1 + exp( -GateOut(b, dilation_c, rf)))
        where rf in 0:RECEPTIVE_FIELD

       Res(b, residual_c, rf)   =   Data(b,  residual_c, rf) + ResBias(residual_c)
       Res(b, residual_c, rf)  += NonLin(b, r_dilation_c, rf) * ResWeight(residual_c, r_dilation_c)

      Skip(b, skip, rf) +=! NonLin(b, r_dilation_c, rf) * SkipWeight(skip, r_dilation_c)
        where rf in 0:RECEPTIVE_FIELD
      Skip(b, skip, rf)  = Skip(b, skip, rf) + SkipBias(skip)
        where rf in 0:RECEPTIVE_FIELD
}
  )TC";

auto options_WaveNet1_P100_autotuned_B_1_RES_32_DIL_32_SKIP_256_REC_4000_F_1 =
    tc::CudaMappingOptions::makeNaiveMappingOptions()
        .outerScheduleFusionStrategy(tc::FusionStrategy::Max)
        .outerScheduleAllowSkewing(false)
        .outerSchedulePositiveOrthant(true)
        .intraTileScheduleFusionStrategy(tc::FusionStrategy::Min)
        .intraTileScheduleAllowSkewing(false)
        .intraTileSchedulePositiveOrthant(true)
        .fixParametersBeforeScheduling(true)
        .tile(63)
        .unroll(32)
        .tileImperfectlyNested(false)
        .matchLibraryCalls(false)
        .mapToThreads(32, 4, 1)
        .mapToBlocks(256, 4, 63)
        .useSharedMemory(true)
        .usePrivateMemory(true)
        .unrollCopyShared(false)
        .useReadOnlyCache(false);

auto options_WaveNet1_P100_autotuned_B_1_RES_32_DIL_32_SKIP_256_REC_4000_F_32 =
    tc::CudaMappingOptions::makeNaiveMappingOptions()
        .outerScheduleFusionStrategy(tc::FusionStrategy::Max)
        .outerScheduleAllowSkewing(false)
        .outerSchedulePositiveOrthant(true)
        .intraTileScheduleFusionStrategy(
            tc::FusionStrategy::Preserve3Coincident)
        .intraTileScheduleAllowSkewing(false)
        .intraTileSchedulePositiveOrthant(true)
        .fixParametersBeforeScheduling(true)
        .tile(128, 4096, 1000, 64)
        .unroll(1)
        .tileImperfectlyNested(false)
        .matchLibraryCalls(true)
        .mapToThreads(128)
        .mapToBlocks(63)
        .useSharedMemory(true)
        .usePrivateMemory(true)
        .unrollCopyShared(false)
        .useReadOnlyCache(false);

auto options_WaveNet1_V100_autotuned_B_1_RES_32_DIL_32_SKIP_256_REC_4000_F_1 =
    tc::CudaMappingOptions::makeNaiveMappingOptions()
        .outerScheduleFusionStrategy(tc::FusionStrategy::Max)
        .outerScheduleAllowSkewing(false)
        .outerSchedulePositiveOrthant(true)
        .intraTileScheduleFusionStrategy(
            tc::FusionStrategy::Preserve3Coincident)
        .intraTileScheduleAllowSkewing(false)
        .intraTileSchedulePositiveOrthant(true)
        .fixParametersBeforeScheduling(false)
        .tile(1000, 128, 500)
        .unroll(2)
        .tileImperfectlyNested(false)
        .matchLibraryCalls(false)
        .mapToThreads(256)
        .mapToBlocks(4000, 128)
        .useSharedMemory(true)
        .usePrivateMemory(true)
        .unrollCopyShared(true)
        .useReadOnlyCache(false);

auto options_WaveNet1_V100_autotuned_B_1_RES_32_DIL_32_SKIP_256_REC_4000_F_32 =
    tc::CudaMappingOptions::makeNaiveMappingOptions()
        .outerScheduleFusionStrategy(tc::FusionStrategy::Max)
        .outerScheduleAllowSkewing(false)
        .outerSchedulePositiveOrthant(true)
        .intraTileScheduleFusionStrategy(tc::FusionStrategy::Min)
        .intraTileScheduleAllowSkewing(false)
        .intraTileSchedulePositiveOrthant(true)
        .fixParametersBeforeScheduling(true)
        .tile(8, 125, 512, 500)
        .unroll(32)
        .tileImperfectlyNested(false)
        .matchLibraryCalls(false)
        .mapToThreads(16, 16)
        .mapToBlocks(4000, 2048, 4096)
        .useSharedMemory(true)
        .usePrivateMemory(true)
        .unrollCopyShared(true)
        .useReadOnlyCache(false);

} // namespace tc
