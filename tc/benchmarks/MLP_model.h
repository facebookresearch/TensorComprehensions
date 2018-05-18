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

auto options_1LUT_P100_autotuned_B_128_D_64_L1_50_E1_10000000 =
    tc::CudaMappingOptions::makeNaiveMappingOptions()
        .outerScheduleFusionStrategy(tc::FusionStrategy::Preserve3Coincident)
        .fixParametersBeforeScheduling(true)
        .tile(1)
        .tileImperfectlyNested(false)
        .mapToBlocks(524288)
        .mapToThreads(153)
        .unroll(8);

auto
    options_2LUT_P100_autotuned_B_128_D_64_L1_50_E1_10000000_L2_50_E2_10000000 =
        tc::CudaMappingOptions::makeNaiveMappingOptions()
            .outerScheduleFusionStrategy(
                tc::FusionStrategy::Preserve3Coincident)
            .fixParametersBeforeScheduling(false)
            .tile(1, 256, 1250000)
            .tileImperfectlyNested(false)
            .mapToBlocks(5000000)
            .mapToThreads(306)
            .unroll(64);

auto options_C3_P100_autotuned_B_128_WX_1000_WY_1024 =
    tc::CudaMappingOptions::makeNaiveMappingOptions()
        .outerScheduleFusionStrategy(tc::FusionStrategy::Max)
        .outerScheduleAllowSkewing(false)
        .outerSchedulePositiveOrthant(true)
        .intraTileScheduleFusionStrategy(tc::FusionStrategy::Min)
        .intraTileScheduleAllowSkewing(false)
        .intraTileSchedulePositiveOrthant(true)
        .tile(1024, 8, 125)
        .mapToThreads(4, 32, 1)
        .mapToBlocks(128, 128, 250)
        .unroll(128)
        .tileImperfectlyNested(false)
        .useSharedMemory(true)
        .usePrivateMemory(true)
        .unrollCopyShared(true)
        .matchLibraryCalls(true);

auto options_MLP1_P100_autotuned_B_128_M_2000_N_128 =
    tc::CudaMappingOptions::makeNaiveMappingOptions()
        .outerScheduleFusionStrategy(tc::FusionStrategy::Preserve3Coincident)
        .outerScheduleAllowSkewing(false)
        .outerSchedulePositiveOrthant(true)
        .intraTileScheduleFusionStrategy(tc::FusionStrategy::Max)
        .intraTileScheduleAllowSkewing(false)
        .intraTileSchedulePositiveOrthant(true)
        .tile(4, 250)
        .mapToThreads(64, 8)
        .mapToBlocks(2000, 16)
        .unroll(32)
        .tileImperfectlyNested(false)
        .useSharedMemory(true)
        .usePrivateMemory(true)
        .unrollCopyShared(false)
        .matchLibraryCalls(true);

auto options_MLP3_P100_autotuned_B_128_N_128_O_64_P_32_Q_2 =
    tc::CudaMappingOptions::makeNaiveMappingOptions()
        .outerScheduleFusionStrategy(tc::FusionStrategy::Max)
        .outerScheduleAllowSkewing(false)
        .outerSchedulePositiveOrthant(true)
        .intraTileScheduleFusionStrategy(
            tc::FusionStrategy::Preserve3Coincident)
        .intraTileScheduleAllowSkewing(false)
        .intraTileSchedulePositiveOrthant(true)
        .tile(4, 8)
        .mapToThreads(128, 4)
        .mapToBlocks(128)
        .unroll(2)
        .tileImperfectlyNested(false)
        .useSharedMemory(true)
        .usePrivateMemory(false)
        .unrollCopyShared(true)
        .matchLibraryCalls(false);

} // namespace tc
