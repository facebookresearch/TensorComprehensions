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
constexpr static auto TC_GroupNormalization_NAME = "group_normalization";
constexpr static auto TC_GroupNormalization = R"TC(
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
#    mean(n, g) +=! I(n, g, r_d, r_h, r_w)
#    mean(n,g)   = mean(n,g) / (D * H * W)
#     var(n, g) +=! I(n, g, r_d, r_h, r_w) * I(n, g, r_d, r_h, r_w)
#     var(n, g)  =  var(n, g) / (D * H * W) - mean(n,g) * mean(n,g)
#    O(n, g, d, h, w) = gamma(g, d)
#      * ( I(n, g, d, h, w) - mean(n, g) )
#      * rsqrt( var(n, g) + 1e-5 )
#      + beta(g, d)

# This implementation uses the formula var = E(x^2) - mean^2 and
# inlining. This gets another 20% on V100.
    mean(n, g) +=! I(n, g, r_d, r_h, r_w)
     var(n, g) +=! I(n, g, r_d, r_h, r_w) * I(n, g, r_d, r_h, r_w)
    O(n, g, d, h, w) = gamma(g, d)
      * ( I(n, g, d, h, w) - mean(n, g) / (D * H * W))
      * rsqrt( var(n, g) / (D * H * W)
            - mean(n, g) * mean(n, g)  / (D * H * W)  / (D * H * W)
            + 1e-5 )
      + beta(g, d)
}
  )TC";

// These options were found by a longer tuning run on a Maxwell card.
// More specifically: Tesla M40
auto options_GroupNormalization_M40_autotuned_M_32_C_512_G_32_H_48_W_48 =
    tc::CudaMappingOptions::makeNaiveMappingOptions()
        .outerScheduleFusionStrategy(tc::FusionStrategy::Max)
        .outerScheduleAllowSkewing(false)
        .outerSchedulePositiveOrthant(true)
        .intraTileScheduleFusionStrategy(
            tc::FusionStrategy::Preserve3Coincident)
        .intraTileScheduleAllowSkewing(false)
        .intraTileSchedulePositiveOrthant(true)
        .fixParametersBeforeScheduling(true)
        .tile(6, 4, 128, 48, 24)
        .unroll(1)
        .tileImperfectlyNested(false)
        .matchLibraryCalls(false)
        .mapToThreads(16, 24)
        .mapToBlocks(32, 12, 128)
        .useSharedMemory(false)
        .usePrivateMemory(true)
        .unrollCopyShared(false)
        .useReadOnlyCache(true);

// These options were found by a longer tuning run on a Pascal card.
// More specifically: Quadro GP100
auto options_GroupNormalization_P100_autotuned_M_32_C_512_G_32_H_48_W_48 =
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

// These options were found by a longer tuning run on a Volta card.
// More specifically: Tesla V100-SXM2-16GB.
auto options_GroupNormalization_V100_autotuned_M_32_C_512_G_32_H_48_W_48 =
    tc::CudaMappingOptions::makeNaiveMappingOptions()
        .outerScheduleFusionStrategy(tc::FusionStrategy::Max)
        .outerScheduleAllowSkewing(false)
        .outerSchedulePositiveOrthant(true)
        .intraTileScheduleFusionStrategy(
            tc::FusionStrategy::Preserve3Coincident)
        .intraTileScheduleAllowSkewing(false)
        .intraTileSchedulePositiveOrthant(true)
        .fixParametersBeforeScheduling(true)
        .tile(2, 3)
        .unroll(2)
        .tileImperfectlyNested(false)
        .matchLibraryCalls(false)
        .mapToThreads(16, 16, 1)
        .mapToBlocks(32, 256, 1)
        .useSharedMemory(true)
        .usePrivateMemory(false)
        .unrollCopyShared(false)
        .useReadOnlyCache(false);
} // namespace tc
