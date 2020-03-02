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

namespace kronecker {
/*
The code computes Y = X . W^T
Input matrix X \in R^{M \times N}
Wsize is an array of size 2K containing the number of rows and columns each
Kronecker factors {P_0, Q_0 ..., P_{K-1}, Q_{K-1}} : \prod_{k=0}^{K-1} P_k = D :
: \prod_{k=0}^{K-1} Q_k = N
Kronecker factors \{W_0, ..., W_{K-1}\} : W_k \in R^{P_k\times Q_k},
Ysize is an array of length K such that \sum_{k = 0}^{K - 1} Ysize[i] = size of
Y, 1.e each entry gives the memory required to
store the product with corresponding Kronecker factors.
size of Y =  O(MDK) = \sum_{k = 0}^{K - 1} Ysize[i]
*/
void cpu_kronecker_real_forward_kernel(
    uint32_t M,
    uint32_t N,
    uint32_t rowk,
    uint32_t colk,
    uint32_t stride,
    const float* W_k,
    const float* X,
    float* Y) {
  uint32_t index = 0;
  for (uint32_t m = 0; m < M; m++) {
    const float* X_m = X + m * N;
    for (uint32_t p = 0; p < rowk; p++) {
      for (uint32_t q = 0; q < stride; q++) {
        Y[index] = 0;
        for (uint32_t r = 0; r < colk; r++) {
          Y[index] += X_m[r * stride + q] * W_k[p * colk + r];
        }
        index++;
      }
    }
  }
}

void cpu_kronecker_real_forward(
    uint32_t M,
    uint32_t N,
    const float* X,
    uint32_t K,
    // const uint32_t* Wsize,
    const std::vector<uint32_t>& Wsize,
    // const float* W,
    std::vector<const float*>& Ws,
    // const uint32_t* Ysize,
    const std::vector<uint32_t>& Ysize,
    float* Y) {
  uint32_t offset = 0, k;
  const float* X_k = X;
  float* Y_k = Y;
  for (k = 0; k < K; k++) {
    uint32_t rowk = Wsize[2 * k];
    uint32_t colk = Wsize[2 * k + 1];
    uint32_t stride = N / colk;
    // const float* W_k = W + offset;
    const float* W_k = Ws[k];
    if (k > 0) {
      // assert(Ysize[k-1] == M * N);
      TC_CHECK_EQ(M * N, Ysize[k - 1])
          << "@k=" << k - 1 << ": " << M * N << " vs " << Ysize[k - 1];
    }
    cpu_kronecker_real_forward_kernel(M, N, rowk, colk, stride, W_k, X_k, Y_k);

    N = stride;
    M = M * rowk;
    offset += rowk * colk;
    X_k = Y_k;
    // assert(Ysize[k] == M * N);
    TC_CHECK_EQ(M * N, Ysize[k])
        << "@k=" << k << ": " << M * N << " vs " << Ysize[k];
    Y_k += Ysize[k];
  }
}

uint32_t kronecker_output_memory(
    uint32_t M,
    uint32_t N,
    uint32_t K,
    const std::vector<uint32_t>& Wsize,
    std::vector<uint32_t>& Ysize) {
  uint32_t size = 0;
  for (uint32_t k = 0; k < K; k++) {
    uint32_t rowk = Wsize[2 * k];
    uint32_t colk = Wsize[2 * k + 1];
    uint32_t stride = N / colk;
    N = stride;
    M = M * rowk;
    Ysize[k] = M * N;
    size += Ysize[k];
  }
  return size;
}
} // namespace kronecker

namespace tc {
constexpr static auto TC_Kronecker3_1_NAME = "Kronecker3_1";
constexpr static auto TC_Kronecker3_2_NAME = "Kronecker3_2";
constexpr static auto TC_Kronecker3_3_NAME = "Kronecker3_3";
constexpr static auto TC_Kronecker3_FULL_NAME = "Kronecker3Full";

constexpr static auto TC_Kronecker3_1 = R"TC(
  def Kronecker3_1(float(D2, N2) W2, float(M, N0, N1, N2) X) -> (XW2) {
     XW2(m, n0, n1, d2)   +=! X(m, n0, n1, r_n2) * W2(d2, r_n2)
  }
)TC";

constexpr static auto TC_Kronecker3_2 = R"TC(
  def Kronecker3_2(float(D1, N1) W1, float(M, N0, N1, D2) XW2) -> (XW2W1) {
     XW2W1(m, n0, d1, d2) +=! XW2(m, n0, r_n1, d2) * W1(d1, r_n1)
  }
)TC";

constexpr static auto TC_Kronecker3_3 = R"TC(
  def Kronecker3_3(float(D0, N0) W0, float(M, N0, D1, D2) XW2W1) -> (Y) {
     Y(m, d0, d1, d2)     +=! XW2W1(m, r_n0, d1, d2) * W0(d0, r_n0)
  }
)TC";

constexpr static auto TC_Kronecker3_FULL = R"TC(
  def Kronecker3Full(float(D0, N0) W0, float(D1, N1) W1,
               float(D2, N2) W2, float(M, N0, N1, N2) X) -> (Y, XW2, XW2W1) {
       XW2(m, n0, n1, d2) +=!     X(m,   n0,   n1, r_n2) * W2(d2, r_n2)
     XW2W1(m, n0, d1, d2) +=!   XW2(m,   n0, r_n1,   d2) * W1(d1, r_n1)
         Y(m, d0, d1, d2) +=! XW2W1(m, r_n0,   d1,   d2) * W0(d0, r_n0)
  }
)TC";

// P100
auto
    options_Kronecker3_1_P100_autotuned_M_256_D0_16_D1_16_D2_16_N0_32_N1_32_N2_32 =
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
            .tile(1, 1, 16)
            .unroll(8)
            .tileImperfectlyNested(false)
            .matchLibraryCalls(false)
            .mapToThreads(16, 4)
            .mapToBlocks(256, 16, 8)
            .useSharedMemory(true)
            .usePrivateMemory(true)
            .unrollCopyShared(true)
            .useReadOnlyCache(true);

auto
    options_Kronecker3_1_P100_autotuned_M_256_D0_16_D1_16_D2_16_N0_64_N1_64_N2_64 =
        tc::CudaMappingOptions::makeNaiveMappingOptions()
            .outerScheduleFusionStrategy(tc::FusionStrategy::Max)
            .outerScheduleAllowSkewing(false)
            .outerSchedulePositiveOrthant(true)
            .intraTileScheduleFusionStrategy(tc::FusionStrategy::Min)
            .intraTileScheduleAllowSkewing(false)
            .intraTileSchedulePositiveOrthant(true)
            .fixParametersBeforeScheduling(false)
            .tile(1, 1, 16, 64, 4)
            .unroll(32)
            .tileImperfectlyNested(false)
            .matchLibraryCalls(false)
            .mapToThreads(8, 16, 1)
            .mapToBlocks(32, 128, 8)
            .useSharedMemory(true)
            .usePrivateMemory(true)
            .unrollCopyShared(true)
            .useReadOnlyCache(true);

auto
    options_Kronecker3_1_P100_autotuned_M_256_D0_16_D1_16_D2_16_N0_64_N1_128_N2_128 =
        tc::CudaMappingOptions::makeNaiveMappingOptions()
            .outerScheduleFusionStrategy(
                tc::FusionStrategy::Preserve3Coincident)
            .outerScheduleAllowSkewing(false)
            .outerSchedulePositiveOrthant(true)
            .intraTileScheduleFusionStrategy(
                tc::FusionStrategy::Preserve3Coincident)
            .intraTileScheduleAllowSkewing(false)
            .intraTileSchedulePositiveOrthant(true)
            .fixParametersBeforeScheduling(false)
            .tile(4, 2)
            .unroll(32)
            .tileImperfectlyNested(false)
            .matchLibraryCalls(false)
            .mapToThreads(1, 32, 1)
            .mapToBlocks(128, 16, 4)
            .useSharedMemory(true)
            .usePrivateMemory(true)
            .unrollCopyShared(false)
            .useReadOnlyCache(true);

auto
    options_Kronecker3_2_P100_autotuned_M_256_D0_16_D1_16_D2_16_N0_32_N1_32_N2_32 =
        tc::CudaMappingOptions::makeNaiveMappingOptions()
            .outerScheduleFusionStrategy(tc::FusionStrategy::Max)
            .outerScheduleAllowSkewing(false)
            .outerSchedulePositiveOrthant(true)
            .intraTileScheduleFusionStrategy(tc::FusionStrategy::Max)
            .intraTileScheduleAllowSkewing(false)
            .intraTileSchedulePositiveOrthant(true)
            .fixParametersBeforeScheduling(false)
            .tile(2)
            .unroll(32)
            .tileImperfectlyNested(false)
            .matchLibraryCalls(false)
            .mapToThreads(16, 2, 8)
            .mapToBlocks(128, 32)
            .useSharedMemory(true)
            .usePrivateMemory(true)
            .unrollCopyShared(true)
            .useReadOnlyCache(true);

auto
    options_Kronecker3_2_P100_autotuned_M_256_D0_16_D1_16_D2_16_N0_64_N1_64_N2_64 =
        tc::CudaMappingOptions::makeNaiveMappingOptions()
            .outerScheduleFusionStrategy(tc::FusionStrategy::Max)
            .outerScheduleAllowSkewing(false)
            .outerSchedulePositiveOrthant(true)
            .intraTileScheduleFusionStrategy(tc::FusionStrategy::Min)
            .intraTileScheduleAllowSkewing(false)
            .intraTileSchedulePositiveOrthant(true)
            .fixParametersBeforeScheduling(false)
            .tile(2, 1, 8, 32, 2)
            .unroll(32)
            .tileImperfectlyNested(false)
            .matchLibraryCalls(false)
            .mapToThreads(8, 16)
            .mapToBlocks(128, 8)
            .useSharedMemory(true)
            .usePrivateMemory(true)
            .unrollCopyShared(true)
            .useReadOnlyCache(true);

auto
    options_Kronecker3_2_P100_autotuned_M_256_D0_16_D1_16_D2_16_N0_64_N1_128_N2_128 =
        tc::CudaMappingOptions::makeNaiveMappingOptions()
            .useSharedMemory(true)
            .usePrivateMemory(true)
            .unrollCopyShared(true)
            .outerScheduleFusionStrategy(
                tc::FusionStrategy::Preserve3Coincident)
            .fixParametersBeforeScheduling(false)
            .tile(1, 1)
            .tileImperfectlyNested(false)
            .mapToBlocks(256, 64)
            .mapToThreads(16, 16)
            .unroll(256);

auto
    options_Kronecker3_3_P100_autotuned_M_256_D0_16_D1_16_D2_16_N0_32_N1_32_N2_32 =
        tc::CudaMappingOptions::makeNaiveMappingOptions()
            .outerScheduleFusionStrategy(tc::FusionStrategy::Max)
            .outerScheduleAllowSkewing(false)
            .outerSchedulePositiveOrthant(true)
            .intraTileScheduleFusionStrategy(
                tc::FusionStrategy::Preserve3Coincident)
            .intraTileScheduleAllowSkewing(false)
            .intraTileSchedulePositiveOrthant(true)
            .fixParametersBeforeScheduling(true)
            .tile(4, 16, 1)
            .unroll(32)
            .tileImperfectlyNested(false)
            .matchLibraryCalls(false)
            .mapToThreads(16, 8)
            .mapToBlocks(64, 32, 64)
            .useSharedMemory(true)
            .usePrivateMemory(true)
            .unrollCopyShared(true)
            .useReadOnlyCache(false);

auto
    options_Kronecker3_3_P100_autotuned_M_256_D0_16_D1_16_D2_16_N0_64_N1_64_N2_64 =
        tc::CudaMappingOptions::makeNaiveMappingOptions()
            .useSharedMemory(true)
            .usePrivateMemory(true)
            .unrollCopyShared(true)
            .outerScheduleFusionStrategy(
                tc::FusionStrategy::Preserve3Coincident)
            .fixParametersBeforeScheduling(true)
            .tile(1, 256)
            .tileImperfectlyNested(false)
            .mapToBlocks(256, 2)
            .mapToThreads(16, 16)
            .unroll(256);

auto
    options_Kronecker3_3_P100_autotuned_M_256_D0_16_D1_16_D2_16_N0_64_N1_128_N2_128 =
        tc::CudaMappingOptions::makeNaiveMappingOptions()
            .useSharedMemory(true)
            .usePrivateMemory(false)
            .unrollCopyShared(true)
            .outerScheduleFusionStrategy(
                tc::FusionStrategy::Preserve3Coincident)
            .fixParametersBeforeScheduling(false)
            .tile(1, 8, 64, 128)
            .tileImperfectlyNested(false)
            .mapToBlocks(256, 256, 16)
            .mapToThreads(16, 16)
            .unroll(128);

// V100
// TODO: RERUN ME
auto
    options_Kronecker3_1_V100_autotuned_M_256_D0_16_D1_16_D2_16_N0_32_N1_32_N2_32 =
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
            .tile(1, 1, 16)
            .unroll(8)
            .tileImperfectlyNested(false)
            .matchLibraryCalls(false)
            .mapToThreads(16, 4)
            .mapToBlocks(256, 16, 8)
            .useSharedMemory(true)
            .usePrivateMemory(true)
            .unrollCopyShared(true)
            .useReadOnlyCache(true);

// TODO: RERUN ME
auto
    options_Kronecker3_1_V100_autotuned_M_256_D0_16_D1_16_D2_16_N0_64_N1_64_N2_64 =
        tc::CudaMappingOptions::makeNaiveMappingOptions()
            .outerScheduleFusionStrategy(tc::FusionStrategy::Max)
            .outerScheduleAllowSkewing(false)
            .outerSchedulePositiveOrthant(true)
            .intraTileScheduleFusionStrategy(tc::FusionStrategy::Min)
            .intraTileScheduleAllowSkewing(false)
            .intraTileSchedulePositiveOrthant(true)
            .fixParametersBeforeScheduling(false)
            .tile(1, 1, 16, 64, 4)
            .unroll(32)
            .tileImperfectlyNested(false)
            .matchLibraryCalls(false)
            .mapToThreads(8, 16, 1)
            .mapToBlocks(32, 128, 8)
            .useSharedMemory(true)
            .usePrivateMemory(true)
            .unrollCopyShared(true)
            .useReadOnlyCache(true);

// TODO: RERUN ME
auto
    options_Kronecker3_1_V100_autotuned_M_256_D0_16_D1_16_D2_16_N0_64_N1_128_N2_128 =
        tc::CudaMappingOptions::makeNaiveMappingOptions()
            .outerScheduleFusionStrategy(
                tc::FusionStrategy::Preserve3Coincident)
            .outerScheduleAllowSkewing(false)
            .outerSchedulePositiveOrthant(true)
            .intraTileScheduleFusionStrategy(
                tc::FusionStrategy::Preserve3Coincident)
            .intraTileScheduleAllowSkewing(false)
            .intraTileSchedulePositiveOrthant(true)
            .fixParametersBeforeScheduling(false)
            .tile(4, 2)
            .unroll(32)
            .tileImperfectlyNested(false)
            .matchLibraryCalls(false)
            .mapToThreads(1, 32, 1)
            .mapToBlocks(128, 16, 4)
            .useSharedMemory(true)
            .usePrivateMemory(true)
            .unrollCopyShared(false)
            .useReadOnlyCache(true);

auto
    options_Kronecker3_2_V100_autotuned_M_256_D0_16_D1_16_D2_16_N0_32_N1_32_N2_32 =
        tc::CudaMappingOptions::makeNaiveMappingOptions()
            .outerScheduleFusionStrategy(tc::FusionStrategy::Max)
            .outerScheduleAllowSkewing(false)
            .outerSchedulePositiveOrthant(true)
            .intraTileScheduleFusionStrategy(
                tc::FusionStrategy::Preserve3Coincident)
            .intraTileScheduleAllowSkewing(false)
            .intraTileSchedulePositiveOrthant(true)
            .fixParametersBeforeScheduling(false)
            .tile(16, 1)
            .unroll(2)
            .tileImperfectlyNested(false)
            .matchLibraryCalls(false)
            .mapToThreads(16, 16)
            .mapToBlocks(16, 32, 256)
            .useSharedMemory(true)
            .usePrivateMemory(true)
            .unrollCopyShared(true)
            .useReadOnlyCache(true);

auto
    options_Kronecker3_2_V100_autotuned_M_256_D0_16_D1_16_D2_16_N0_64_N1_64_N2_64 =
        tc::CudaMappingOptions::makeNaiveMappingOptions()
            .outerScheduleFusionStrategy(
                tc::FusionStrategy::Preserve3Coincident)
            .outerScheduleAllowSkewing(false)
            .outerSchedulePositiveOrthant(true)
            .intraTileScheduleFusionStrategy(tc::FusionStrategy::Max)
            .intraTileScheduleAllowSkewing(false)
            .intraTileSchedulePositiveOrthant(true)
            .fixParametersBeforeScheduling(false)
            .tile(2)
            .unroll(8)
            .tileImperfectlyNested(false)
            .matchLibraryCalls(false)
            .mapToThreads(16, 8, 2)
            .mapToBlocks(256, 128)
            .useSharedMemory(true)
            .usePrivateMemory(true)
            .unrollCopyShared(false)
            .useReadOnlyCache(true);

auto
    options_Kronecker3_2_V100_autotuned_M_256_D0_16_D1_16_D2_16_N0_64_N1_128_N2_128 =
        tc::CudaMappingOptions::makeNaiveMappingOptions()
            .outerScheduleFusionStrategy(
                tc::FusionStrategy::Preserve3Coincident)
            .outerScheduleAllowSkewing(false)
            .outerSchedulePositiveOrthant(true)
            .intraTileScheduleFusionStrategy(tc::FusionStrategy::Min)
            .intraTileScheduleAllowSkewing(false)
            .intraTileSchedulePositiveOrthant(true)
            .fixParametersBeforeScheduling(false)
            .tile(8, 1)
            .unroll(16)
            .tileImperfectlyNested(false)
            .matchLibraryCalls(false)
            .mapToThreads(16, 4)
            .mapToBlocks(128, 64, 128)
            .useSharedMemory(true)
            .usePrivateMemory(false)
            .unrollCopyShared(true)
            .useReadOnlyCache(false);

auto
    options_Kronecker3_3_V100_autotuned_M_256_D0_16_D1_16_D2_16_N0_32_N1_32_N2_32 =
        tc::CudaMappingOptions::makeNaiveMappingOptions()
            .outerScheduleFusionStrategy(tc::FusionStrategy::Max)
            .outerScheduleAllowSkewing(false)
            .outerSchedulePositiveOrthant(true)
            .intraTileScheduleFusionStrategy(
                tc::FusionStrategy::Preserve3Coincident)
            .intraTileScheduleAllowSkewing(false)
            .intraTileSchedulePositiveOrthant(true)
            .fixParametersBeforeScheduling(true)
            .tile(2, 64, 64)
            .unroll(8)
            .tileImperfectlyNested(false)
            .matchLibraryCalls(false)
            .mapToThreads(16, 16, 2)
            .mapToBlocks(128)
            .useSharedMemory(false)
            .usePrivateMemory(true)
            .unrollCopyShared(true)
            .useReadOnlyCache(true);

auto
    options_Kronecker3_3_V100_autotuned_M_256_D0_16_D1_16_D2_16_N0_64_N1_64_N2_64 =
        tc::CudaMappingOptions::makeNaiveMappingOptions()
            .outerScheduleFusionStrategy(tc::FusionStrategy::Max)
            .outerScheduleAllowSkewing(false)
            .outerSchedulePositiveOrthant(true)
            .intraTileScheduleFusionStrategy(
                tc::FusionStrategy::Preserve3Coincident)
            .intraTileScheduleAllowSkewing(false)
            .intraTileSchedulePositiveOrthant(true)
            .fixParametersBeforeScheduling(true)
            .tile(1, 1)
            .unroll(32)
            .tileImperfectlyNested(false)
            .matchLibraryCalls(false)
            .mapToThreads(16, 8)
            .mapToBlocks(32, 16, 16)
            .useSharedMemory(true)
            .usePrivateMemory(true)
            .unrollCopyShared(true)
            .useReadOnlyCache(false);

auto
    options_Kronecker3_3_V100_autotuned_M_256_D0_16_D1_16_D2_16_N0_64_N1_128_N2_128 =
        tc::CudaMappingOptions::makeNaiveMappingOptions()
            .outerScheduleFusionStrategy(
                tc::FusionStrategy::Preserve3Coincident)
            .outerScheduleAllowSkewing(false)
            .outerSchedulePositiveOrthant(true)
            .intraTileScheduleFusionStrategy(tc::FusionStrategy::Max)
            .intraTileScheduleAllowSkewing(false)
            .intraTileSchedulePositiveOrthant(true)
            .fixParametersBeforeScheduling(false)
            .tile(2, 256, 32, 0, 8)
            .unroll(32)
            .tileImperfectlyNested(false)
            .matchLibraryCalls(false)
            .mapToThreads(16, 16, 1)
            .mapToBlocks(128, 8, 16)
            .useSharedMemory(true)
            .usePrivateMemory(true)
            .unrollCopyShared(true)
            .useReadOnlyCache(false);

} // namespace tc
