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
constexpr static auto TC_Sum_2D_1D_NAME = "sum_2D_1D";
constexpr static auto TC_Mean_2D_1D_NAME = "mean_2D_1D";
constexpr static auto TC_Sum_Squares_2D_1D_NAME = "sum_squares_2D_1D";
constexpr static auto TC_Var_2D_1D_NAME = "var_2D_1D";
constexpr static auto TC_Sum_And_Squares_2D_1D_NAME = "sum_and_squares_2D_1D";
constexpr static auto TC_Moments2_2D_1D_NAME = "moments2_2D_1D";
constexpr static auto TC_Moments2_2D_1D = R"TC(
def sum_2D_1D(float(N, K) I) -> (sum)
{
    sum(n) +=! I(n, r_k)
}
def mean_2D_1D(float(N, K) I) -> (mean)
{
    mean(n) +=! I(n, r_k)
    mean(n)  = mean(n) / (K)
}
def sum_squares_2D_1D(float(N, K) I) -> (sum_squares)
{
     sum_squares(n) +=! I(n, r_k) * I(n, r_k)
}
def var_2D_1D(float(N, K) I, float(N) mean) -> (var)
{
     var(n) +=! I(n, r_k) * I(n, r_k)
     var(n)  =  var(n) / (K) - mean(n) * mean(n)
}
def sum_and_squares_2D_1D(float(N, K) I) -> (sum, sum_squares)
{
             sum(n) +=! I(n, r_k)
     sum_squares(n) +=! I(n, r_k) * I(n, r_k)
}
def moments2_2D_1D(float(N, K) I) -> (mean, var)
{
# var = E(x^2) - mean^2.
    mean(n) +=! I(n, r_k)
     var(n) +=! I(n, r_k) * I(n, r_k)
    mean(n)  = mean(n) / (K)
     var(n)  =  var(n) / (K) - mean(n) * mean(n)
}
)TC";

auto options_Sum_2D_1D_P100_autotuned_N_1024_K_36864 =
    tc::CudaMappingOptions::makeNaiveMappingOptions()
        .outerScheduleFusionStrategy(tc::FusionStrategy::Max)
        .outerScheduleAllowSkewing(false)
        .outerSchedulePositiveOrthant(true)
        .intraTileScheduleFusionStrategy(tc::FusionStrategy::Max)
        .intraTileScheduleAllowSkewing(false)
        .intraTileSchedulePositiveOrthant(true)
        .fixParametersBeforeScheduling(false)
        .tile(2)
        .unroll(4)
        .tileImperfectlyNested(false)
        .matchLibraryCalls(true)
        .mapToThreads(288)
        .mapToBlocks(288)
        .useSharedMemory(false)
        .usePrivateMemory(false)
        .unrollCopyShared(true)
        .useReadOnlyCache(false);

auto options_Mean_2D_1D_P100_autotuned_N_1024_K_36864 =
    tc::CudaMappingOptions::makeNaiveMappingOptions()
        .outerScheduleFusionStrategy(tc::FusionStrategy::Preserve3Coincident)
        .outerScheduleAllowSkewing(false)
        .outerSchedulePositiveOrthant(true)
        .intraTileScheduleFusionStrategy(
            tc::FusionStrategy::Preserve3Coincident)
        .intraTileScheduleAllowSkewing(false)
        .intraTileSchedulePositiveOrthant(true)
        .fixParametersBeforeScheduling(false)
        .tile(5, 9, 18432)
        .unroll(16)
        .tileImperfectlyNested(false)
        .matchLibraryCalls(false)
        .mapToThreads(32)
        .mapToBlocks(512)
        .useSharedMemory(true)
        .usePrivateMemory(true)
        .unrollCopyShared(true)
        .useReadOnlyCache(false);

auto options_Sum_Squares_2D_1D_P100_autotuned_N_1024_K_36864 =
    tc::CudaMappingOptions::makeNaiveMappingOptions()
        .outerScheduleFusionStrategy(tc::FusionStrategy::Max)
        .outerScheduleAllowSkewing(false)
        .outerSchedulePositiveOrthant(true)
        .intraTileScheduleFusionStrategy(tc::FusionStrategy::Max)
        .intraTileScheduleAllowSkewing(false)
        .intraTileSchedulePositiveOrthant(true)
        .fixParametersBeforeScheduling(false)
        .tile(5)
        .unroll(8)
        .tileImperfectlyNested(false)
        .matchLibraryCalls(true)
        .mapToThreads(128)
        .mapToBlocks(1024, 4096, 576)
        .useSharedMemory(false)
        .usePrivateMemory(true)
        .unrollCopyShared(true)
        .useReadOnlyCache(false);

auto options_Var_2D_1D_P100_autotuned_N_1024_K_36864 =
    tc::CudaMappingOptions::makeNaiveMappingOptions()
        .outerScheduleFusionStrategy(tc::FusionStrategy::Preserve3Coincident)
        .outerScheduleAllowSkewing(false)
        .outerSchedulePositiveOrthant(true)
        .intraTileScheduleFusionStrategy(
            tc::FusionStrategy::Preserve3Coincident)
        .intraTileScheduleAllowSkewing(false)
        .intraTileSchedulePositiveOrthant(true)
        .fixParametersBeforeScheduling(false)
        .tile(9, 65536, 3)
        .unroll(16)
        .tileImperfectlyNested(false)
        .matchLibraryCalls(true)
        .mapToThreads(256)
        .mapToBlocks(8192)
        .useSharedMemory(false)
        .usePrivateMemory(true)
        .unrollCopyShared(true)
        .useReadOnlyCache(true);

auto options_Sum_And_Squares_2D_1D_P100_autotuned_N_1024_K_36864 =
    tc::CudaMappingOptions::makeNaiveMappingOptions()
        .outerScheduleFusionStrategy(tc::FusionStrategy::Max)
        .outerScheduleAllowSkewing(false)
        .outerSchedulePositiveOrthant(true)
        .intraTileScheduleFusionStrategy(tc::FusionStrategy::Min)
        .intraTileScheduleAllowSkewing(false)
        .intraTileSchedulePositiveOrthant(true)
        .fixParametersBeforeScheduling(true)
        .tile(1)
        .unroll(1)
        .tileImperfectlyNested(false)
        .matchLibraryCalls(true)
        .mapToThreads(512)
        .mapToBlocks(4608, 144, 3)
        .useSharedMemory(false)
        .usePrivateMemory(true)
        .unrollCopyShared(false)
        .useReadOnlyCache(false);

auto options_Moments2_2D_1D_P100_autotuned_N_1024_K_36864 =
    tc::CudaMappingOptions::makeNaiveMappingOptions()
        .outerScheduleFusionStrategy(tc::FusionStrategy::Preserve3Coincident)
        .outerScheduleAllowSkewing(false)
        .outerSchedulePositiveOrthant(true)
        .intraTileScheduleFusionStrategy(
            tc::FusionStrategy::Preserve3Coincident)
        .intraTileScheduleAllowSkewing(false)
        .intraTileSchedulePositiveOrthant(true)
        .fixParametersBeforeScheduling(false)
        .tile(9)
        .unroll(4)
        .tileImperfectlyNested(false)
        .matchLibraryCalls(true)
        .mapToThreads(128)
        .mapToBlocks(1152)
        .useSharedMemory(false)
        .usePrivateMemory(true)
        .unrollCopyShared(true)
        .useReadOnlyCache(false);

} // namespace tc
