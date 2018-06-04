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

auto
    options_GroupConvolution_P100_autotuned_N_32_G_32_C_4_F_4_W_56_H_56_KW_3_KH_3 =
        tc::CudaMappingOptions::makeNaiveMappingOptions()
            .outerScheduleFusionStrategy(
                tc::FusionStrategy::Preserve3Coincident)
            .outerScheduleAllowSkewing(false)
            .outerSchedulePositiveOrthant(true)
            .intraTileScheduleFusionStrategy(
                tc::FusionStrategy::Preserve3Coincident)
            .intraTileScheduleAllowSkewing(false)
            .intraTileSchedulePositiveOrthant(true)
            .tile(1, 1, 7, 7)
            .mapToThreads(56, 7)
            .mapToBlocks(16, 64, 1)
            .unroll(2)
            .tileImperfectlyNested(false)
            .useSharedMemory(true)
            .usePrivateMemory(false)
            .unrollCopyShared(false)
            .matchLibraryCalls(true);

auto
    options_GroupConvolution_P100_autotuned_N_32_G_32_C_8_F_8_W_28_H_28_KW_3_KH_3 =
        tc::CudaMappingOptions::makeNaiveMappingOptions()
            .outerScheduleFusionStrategy(tc::FusionStrategy::Max)
            .outerScheduleAllowSkewing(false)
            .outerSchedulePositiveOrthant(true)
            .intraTileScheduleFusionStrategy(
                tc::FusionStrategy::Preserve3Coincident)
            .intraTileScheduleAllowSkewing(false)
            .intraTileSchedulePositiveOrthant(true)
            .tile(1, 1, 256, 14, 16)
            .mapToThreads(16, 14)
            .mapToBlocks(7, 16)
            .unroll(16)
            .tileImperfectlyNested(false)
            .useSharedMemory(true)
            .usePrivateMemory(false)
            .unrollCopyShared(true)
            .matchLibraryCalls(true);

auto
    options_GroupConvolution_P100_autotuned_N_32_G_32_C_16_F_16_W_14_H_14_KW_3_KH_3 =
        tc::CudaMappingOptions::makeNaiveMappingOptions()
            .outerScheduleFusionStrategy(tc::FusionStrategy::Max)
            .outerScheduleAllowSkewing(false)
            .outerSchedulePositiveOrthant(true)
            .intraTileScheduleFusionStrategy(
                tc::FusionStrategy::Preserve3Coincident)
            .intraTileScheduleAllowSkewing(false)
            .intraTileSchedulePositiveOrthant(true)
            .fixParametersBeforeScheduling(true)
            .tile(14, 1)
            .unroll(8)
            .tileImperfectlyNested(false)
            .matchLibraryCalls(true)
            .mapToThreads(16, 32)
            .mapToBlocks(32, 64)
            .useSharedMemory(true)
            .usePrivateMemory(true)
            .unrollCopyShared(true)
            .useReadOnlyCache(true);

auto
    options_GroupConvolution_P100_autotuned_N_32_G_32_C_32_F_32_W_7_H_7_KW_3_KH_3 =
        tc::CudaMappingOptions::makeNaiveMappingOptions()
            .outerScheduleFusionStrategy(
                tc::FusionStrategy::Preserve3Coincident)
            .outerScheduleAllowSkewing(false)
            .outerSchedulePositiveOrthant(true)
            .intraTileScheduleFusionStrategy(
                tc::FusionStrategy::Preserve3Coincident)
            .intraTileScheduleAllowSkewing(false)
            .intraTileSchedulePositiveOrthant(true)
            .tile(1, 2, 3)
            .mapToThreads(8, 7, 4)
            .mapToBlocks(128, 16, 64)
            .unroll(16)
            .tileImperfectlyNested(false)
            .useSharedMemory(true)
            .usePrivateMemory(true)
            .unrollCopyShared(true)
            .matchLibraryCalls(true);

auto
    options_GroupConvolution_V100_autotuned_N_32_G_32_C_4_F_4_W_56_H_56_KW_3_KH_3 =
        tc::CudaMappingOptions::makeNaiveMappingOptions()
            .outerScheduleFusionStrategy(
                tc::FusionStrategy::Preserve3Coincident)
            .outerScheduleAllowSkewing(false)
            .outerSchedulePositiveOrthant(true)
            .intraTileScheduleFusionStrategy(tc::FusionStrategy::Max)
            .intraTileScheduleAllowSkewing(false)
            .intraTileSchedulePositiveOrthant(true)
            .fixParametersBeforeScheduling(false)
            .tile(1, 2, 56, 28)
            .unroll(1)
            .tileImperfectlyNested(false)
            .matchLibraryCalls(false)
            .mapToThreads(14, 28)
            .mapToBlocks(32, 32)
            .useSharedMemory(true)
            .usePrivateMemory(true)
            .unrollCopyShared(false)
            .useReadOnlyCache(true);

auto
    options_GroupConvolution_V100_autotuned_N_32_G_32_C_8_F_8_W_28_H_28_KW_3_KH_3 =
        tc::CudaMappingOptions::makeNaiveMappingOptions()
            .outerScheduleFusionStrategy(tc::FusionStrategy::Max)
            .outerScheduleAllowSkewing(false)
            .outerSchedulePositiveOrthant(true)
            .intraTileScheduleFusionStrategy(
                tc::FusionStrategy::Preserve3Coincident)
            .intraTileScheduleAllowSkewing(false)
            .intraTileSchedulePositiveOrthant(true)
            .fixParametersBeforeScheduling(false)
            .tile(1, 1, 256, 1)
            .unroll(1)
            .tileImperfectlyNested(false)
            .matchLibraryCalls(false)
            .mapToThreads(32, 14)
            .mapToBlocks(128, 32)
            .useSharedMemory(true)
            .usePrivateMemory(true)
            .unrollCopyShared(true)
            .useReadOnlyCache(false);

auto
    options_GroupConvolution_V100_autotuned_N_32_G_32_C_16_F_16_W_14_H_14_KW_3_KH_3 =
        tc::CudaMappingOptions::makeNaiveMappingOptions()
            .outerScheduleFusionStrategy(tc::FusionStrategy::Max)
            .outerScheduleAllowSkewing(false)
            .outerSchedulePositiveOrthant(true)
            .intraTileScheduleFusionStrategy(
                tc::FusionStrategy::Preserve3Coincident)
            .intraTileScheduleAllowSkewing(false)
            .intraTileSchedulePositiveOrthant(true)
            .fixParametersBeforeScheduling(true)
            .tile(2, 1, 128)
            .unroll(2)
            .tileImperfectlyNested(false)
            .matchLibraryCalls(false)
            .mapToThreads(32, 14, 1)
            .mapToBlocks(32, 32, 16)
            .useSharedMemory(true)
            .usePrivateMemory(false)
            .unrollCopyShared(true)
            .useReadOnlyCache(false);

auto
    options_GroupConvolution_V100_autotuned_N_32_G_32_C_32_F_32_W_7_H_7_KW_3_KH_3 =
        tc::CudaMappingOptions::makeNaiveMappingOptions()
            .outerScheduleFusionStrategy(
                tc::FusionStrategy::Preserve3Coincident)
            .outerScheduleAllowSkewing(false)
            .outerSchedulePositiveOrthant(true)
            .intraTileScheduleFusionStrategy(
                tc::FusionStrategy::Preserve3Coincident)
            .intraTileScheduleAllowSkewing(false)
            .intraTileSchedulePositiveOrthant(true)
            .fixParametersBeforeScheduling(true)
            .tile(16, 1, 1)
            .unroll(4)
            .tileImperfectlyNested(false)
            .matchLibraryCalls(true)
            .mapToThreads(7, 8)
            .mapToBlocks(256, 32, 32)
            .useSharedMemory(true)
            .usePrivateMemory(true)
            .unrollCopyShared(false)
            .useReadOnlyCache(false);
} // namespace tc
