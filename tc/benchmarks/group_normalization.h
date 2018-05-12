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
constexpr static auto TC_Moments2_2D_1D_NAME = "moments2_2D_1D";
constexpr static auto TC_GroupNormalization_NAME = "group_normalization";
constexpr static auto TC_GroupNormalizationSingleKernel_NAME =
    "group_normalization_single_kernel";
constexpr static auto TC_GroupNormalization = R"TC(
def moments2_2D_1D(float(N, K) I) -> (mean, var)
{
# var = E(x^2) - mean^2.
    mean(n) +=! I(n, r_k)
     var(n) +=! I(n, r_k) * I(n, r_k)
    mean(n)  = mean(n) / (K)
     var(n)  =  var(n) / (K) - mean(n) * mean(n)
}

def group_normalization(
    float(N, G, D, H, W) I, float(G, D) gamma, float(G, D) beta,
    float(N, G) mean, float(N, G) var)
    -> (O)
{
    O(n, g, d, h, w) = gamma(g, d)
      * ( I(n, g, d, h, w) - mean(n, g) )
      * rsqrt( var(n, g) + 1e-5 )
      + beta(g, d)
}

def group_normalization_single_kernel(
    float(N, G, D, H, W) I, float(G, D) gamma, float(G, D) beta)
    -> (O, sum, sumSquares)
{
# This implementation uses the formula var = E(x^2) - mean^2 and
# inlining. This gets another 20% on V100.
            sum(n, g) +=! I(n, g, r_d, r_h, r_w)
     sumSquares(n, g) +=! I(n, g, r_d, r_h, r_w) * I(n, g, r_d, r_h, r_w)
    O(n, g, d, h, w) = gamma(g, d)
      * ( I(n, g, d, h, w) - sum(n, g) / (D * H * W))
      * rsqrt( sumSquares(n, g) / (D * H * W)
            - sum(n, g) * sum(n, g)  / (D * H * W)  / (D * H * W)
            + 1e-5 )
      + beta(g, d)
}
)TC";

auto options_Moments2_2D_1D_P100_autotuned_N_128_K_2304 =
    tc::CudaMappingOptions::makeNaiveMappingOptions()
        .outerScheduleFusionStrategy(tc::FusionStrategy::Max)
        .outerScheduleAllowSkewing(false)
        .outerSchedulePositiveOrthant(true)
        .intraTileScheduleFusionStrategy(tc::FusionStrategy::Min)
        .intraTileScheduleAllowSkewing(false)
        .intraTileSchedulePositiveOrthant(true)
        .fixParametersBeforeScheduling(false)
        .tile(1)
        .unroll(4)
        .tileImperfectlyNested(false)
        .matchLibraryCalls(true)
        .mapToThreads(32)
        .mapToBlocks(288, 576)
        .useSharedMemory(true)
        .usePrivateMemory(false)
        .unrollCopyShared(true)
        .useReadOnlyCache(false);

auto options_Moments2_2D_1D_P100_autotuned_N_1024_K_36864 =
    tc::CudaMappingOptions::makeNaiveMappingOptions()
        .outerScheduleFusionStrategy(tc::FusionStrategy::Preserve3Coincident)
        .outerScheduleAllowSkewing(false)
        .outerSchedulePositiveOrthant(true)
        .intraTileScheduleFusionStrategy(tc::FusionStrategy::Min)
        .intraTileScheduleAllowSkewing(false)
        .intraTileSchedulePositiveOrthant(true)
        .fixParametersBeforeScheduling(false)
        .tile(8, 256, 5)
        .unroll(16)
        .tileImperfectlyNested(false)
        .matchLibraryCalls(true)
        .mapToThreads(72, 4)
        .mapToBlocks(288, 8)
        .useSharedMemory(true)
        .usePrivateMemory(false)
        .unrollCopyShared(false)
        .useReadOnlyCache(true);

auto options_GroupNormalization_P100_autotuned_N_4_C_512_G_32_H_12_W_12 =
    tc::CudaMappingOptions::makeNaiveMappingOptions()
        .outerScheduleFusionStrategy(tc::FusionStrategy::Max)
        .outerScheduleAllowSkewing(false)
        .outerSchedulePositiveOrthant(true)
        .intraTileScheduleFusionStrategy(
            tc::FusionStrategy::Preserve3Coincident)
        .intraTileScheduleAllowSkewing(false)
        .intraTileSchedulePositiveOrthant(true)
        .fixParametersBeforeScheduling(false)
        .tile(1, 1, 0, 4, 0)
        .unroll(8)
        .tileImperfectlyNested(false)
        .matchLibraryCalls(true)
        .mapToThreads(16, 16)
        .mapToBlocks(6, 128, 1)
        .useSharedMemory(true)
        .usePrivateMemory(false)
        .unrollCopyShared(true)
        .useReadOnlyCache(true);

auto options_GroupNormalization_P100_autotuned_N_32_C_512_G_32_H_48_W_48 =
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

auto options_Moments2_2D_1D_V100_autotuned_N_128_K_2304 =
    tc::CudaMappingOptions::makeNaiveMappingOptions()
        .outerScheduleFusionStrategy(tc::FusionStrategy::Preserve3Coincident)
        .outerScheduleAllowSkewing(false)
        .outerSchedulePositiveOrthant(true)
        .intraTileScheduleFusionStrategy(tc::FusionStrategy::Min)
        .intraTileScheduleAllowSkewing(false)
        .intraTileSchedulePositiveOrthant(true)
        .fixParametersBeforeScheduling(false)
        .tile(1)
        .unroll(32)
        .tileImperfectlyNested(false)
        .matchLibraryCalls(true)
        .mapToThreads(256)
        .mapToBlocks(512)
        .useSharedMemory(true)
        .usePrivateMemory(true)
        .unrollCopyShared(true)
        .useReadOnlyCache(false);

auto options_Moments2_2D_1D_V100_autotuned_N_1024_K_36864 =
    tc::CudaMappingOptions::makeNaiveMappingOptions()
        .outerScheduleFusionStrategy(tc::FusionStrategy::Preserve3Coincident)
        .outerScheduleAllowSkewing(false)
        .outerSchedulePositiveOrthant(true)
        .intraTileScheduleFusionStrategy(tc::FusionStrategy::Min)
        .intraTileScheduleAllowSkewing(false)
        .intraTileSchedulePositiveOrthant(true)
        .fixParametersBeforeScheduling(true)
        .tile(1, 4608)
        .unroll(32)
        .tileImperfectlyNested(false)
        .matchLibraryCalls(true)
        .mapToThreads(144)
        .mapToBlocks(288, 512)
        .useSharedMemory(true)
        .usePrivateMemory(false)
        .unrollCopyShared(true)
        .useReadOnlyCache(false);

auto options_GroupNormalization_V100_autotuned_N_4_C_512_G_32_H_12_W_12 =
    tc::CudaMappingOptions::makeNaiveMappingOptions()
        .outerScheduleFusionStrategy(tc::FusionStrategy::Preserve3Coincident)
        .outerScheduleAllowSkewing(false)
        .outerSchedulePositiveOrthant(true)
        .intraTileScheduleFusionStrategy(
            tc::FusionStrategy::Preserve3Coincident)
        .intraTileScheduleAllowSkewing(false)
        .intraTileSchedulePositiveOrthant(true)
        .fixParametersBeforeScheduling(false)
        .tile(6, 1, 8)
        .unroll(2)
        .tileImperfectlyNested(false)
        .matchLibraryCalls(false)
        .mapToThreads(12, 12, 2)
        .mapToBlocks(12, 256, 128)
        .useSharedMemory(true)
        .usePrivateMemory(true)
        .unrollCopyShared(true)
        .useReadOnlyCache(true);

auto options_GroupNormalization_V100_autotuned_N_32_C_512_G_32_H_48_W_48 =
    tc::CudaMappingOptions::makeNaiveMappingOptions()
        .outerScheduleFusionStrategy(tc::FusionStrategy::Min)
        .outerScheduleAllowSkewing(false)
        .outerSchedulePositiveOrthant(true)
        .intraTileScheduleFusionStrategy(
            tc::FusionStrategy::Preserve3Coincident)
        .intraTileScheduleAllowSkewing(false)
        .intraTileSchedulePositiveOrthant(true)
        .fixParametersBeforeScheduling(true)
        .tile(1, 1, 1)
        .unroll(4)
        .tileImperfectlyNested(false)
        .matchLibraryCalls(true)
        .mapToThreads(16, 16)
        .mapToBlocks(64, 32, 256)
        .useSharedMemory(false)
        .usePrivateMemory(false)
        .unrollCopyShared(false)
        .useReadOnlyCache(true);

auto
    options_GroupNormalizationSingleKernel_P100_autotuned_N_4_C_512_G_32_H_12_W_12 =
        tc::CudaMappingOptions::makeNaiveMappingOptions()
            .outerScheduleFusionStrategy(tc::FusionStrategy::Max)
            .outerScheduleAllowSkewing(false)
            .outerSchedulePositiveOrthant(true)
            .intraTileScheduleFusionStrategy(tc::FusionStrategy::Min)
            .intraTileScheduleAllowSkewing(false)
            .intraTileSchedulePositiveOrthant(true)
            .fixParametersBeforeScheduling(false)
            .tile(12, 1, 8, 64)
            .unroll(4)
            .tileImperfectlyNested(false)
            .matchLibraryCalls(true)
            .mapToThreads(12, 4, 6)
            .mapToBlocks(1, 256, 128)
            .useSharedMemory(true)
            .usePrivateMemory(false)
            .unrollCopyShared(false)
            .useReadOnlyCache(false);

auto
    options_GroupNormalizationSingleKernel_P100_autotuned_N_32_C_512_G_32_H_48_W_48 =
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

auto
    options_GroupNormalizationSingleKernel_V100_autotuned_N_4_C_512_G_32_H_12_W_12 =
        tc::CudaMappingOptions::makeNaiveMappingOptions()
            .outerScheduleFusionStrategy(tc::FusionStrategy::Max)
            .outerScheduleAllowSkewing(false)
            .outerSchedulePositiveOrthant(true)
            .intraTileScheduleFusionStrategy(tc::FusionStrategy::Min)
            .intraTileScheduleAllowSkewing(false)
            .intraTileSchedulePositiveOrthant(true)
            .fixParametersBeforeScheduling(false)
            .tile(1, 2)
            .unroll(16)
            .tileImperfectlyNested(false)
            .matchLibraryCalls(true)
            .mapToThreads(16, 32)
            .mapToBlocks(32, 256, 8)
            .useSharedMemory(true)
            .usePrivateMemory(false)
            .unrollCopyShared(true)
            .useReadOnlyCache(false);

auto
    options_GroupNormalizationSingleKernel_V100_autotuned_N_32_C_512_G_32_H_48_W_48 =
        tc::CudaMappingOptions::makeNaiveMappingOptions()
            .outerScheduleFusionStrategy(tc::FusionStrategy::Max)
            .outerScheduleAllowSkewing(false)
            .outerSchedulePositiveOrthant(true)
            .intraTileScheduleFusionStrategy(
                tc::FusionStrategy::Preserve3Coincident)
            .intraTileScheduleAllowSkewing(false)
            .intraTileSchedulePositiveOrthant(true)
            .fixParametersBeforeScheduling(true)
            .tile(2, 1, 8, 12, 32)
            .unroll(1)
            .tileImperfectlyNested(false)
            .matchLibraryCalls(false)
            .mapToThreads(8, 12)
            .mapToBlocks(256, 128, 48)
            .useSharedMemory(true)
            .usePrivateMemory(true)
            .unrollCopyShared(false)
            .useReadOnlyCache(false);

} // namespace tc
