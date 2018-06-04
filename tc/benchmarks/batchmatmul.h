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
auto options_TransposedBatchMatMul_P100_autotuned_B_500_K_26_M_72_N_26 =
    tc::CudaMappingOptions::makeNaiveMappingOptions()
        .outerScheduleFusionStrategy(tc::FusionStrategy::Max)
        .outerScheduleAllowSkewing(false)
        .outerSchedulePositiveOrthant(true)
        .intraTileScheduleFusionStrategy(tc::FusionStrategy::Min)
        .intraTileScheduleAllowSkewing(false)
        .intraTileSchedulePositiveOrthant(true)
        .tile(3)
        .mapToThreads(4, 36, 3)
        .mapToBlocks(512)
        .unroll(64)
        .tileImperfectlyNested(false)
        .useSharedMemory(true)
        .usePrivateMemory(false)
        .unrollCopyShared(true)
        .matchLibraryCalls(true);

auto options_TransposedBatchMatMul_V100_autotuned_B_500_K_26_M_72_N_26 =
    tc::CudaMappingOptions::makeNaiveMappingOptions()
        .outerScheduleFusionStrategy(tc::FusionStrategy::Preserve3Coincident)
        .outerScheduleAllowSkewing(false)
        .outerSchedulePositiveOrthant(true)
        .intraTileScheduleFusionStrategy(
            tc::FusionStrategy::Preserve3Coincident)
        .intraTileScheduleAllowSkewing(false)
        .intraTileSchedulePositiveOrthant(true)
        .fixParametersBeforeScheduling(true)
        .tile(1, 1, 32, 32)
        .unroll(1)
        .tileImperfectlyNested(false)
        .matchLibraryCalls(false)
        .mapToThreads(72)
        .mapToBlocks(500)
        .useSharedMemory(true)
        .usePrivateMemory(true)
        .unrollCopyShared(false)
        .useReadOnlyCache(false);
} // namespace tc
