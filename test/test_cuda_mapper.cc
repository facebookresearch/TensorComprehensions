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
#include <random>
#include <sstream>
#include <string>
#include <unordered_set>
#include <vector>

#include <gflags/gflags.h>
#include <glog/logging.h>
#include <gtest/gtest.h>

#include "tc/core/constants.h"
#include "tc/core/cuda/cuda_mapping_options.h"
#include "tc/core/libraries.h"
#include "tc/core/polyhedral/cuda/codegen.h"
#include "tc/core/polyhedral/cuda/mapped_scop.h"
#include "tc/core/polyhedral/functional.h"
#include "tc/core/polyhedral/mapping_types.h"
#include "tc/core/polyhedral/schedule_isl_conversion.h"
#include "tc/core/polyhedral/schedule_transforms.h"
#include "tc/core/polyhedral/schedule_tree.h"
#include "tc/core/polyhedral/schedule_tree_matcher.h"
#include "tc/core/polyhedral/scop.h"
#include "tc/core/scope_guard.h"
#include "tc/external/isl.h"
#include "tc/library/matmul.h"

using namespace std;

using namespace tc;
using namespace tc::polyhedral;
using namespace tc::polyhedral::detail;

struct PolyhedralMapperTest : public ::testing::Test {
  std::unique_ptr<Scop> Prepare(std::string tc) {
    auto ctx = isl::with_exceptions::globalIslCtx();
    // Build the SCoP corresponding to the Tc
    return Scop::makeScop(ctx, tc);
  }

  std::unique_ptr<Scop> PrepareAndJoinBands(std::string tc) {
    auto scop = Prepare(tc);
    // Join bands for ISL schedule tree to be tilable (set permutable flag
    // to true manually). This is generally incorrect and is only used for the
    // purpose of unit testing.
    joinBandsIterative(scop->scheduleRoot()->child({0}), true);
    return scop;
  }
  std::unique_ptr<Scop> PrepareAndJoinBandsMatMul() {
    auto scop = Prepare(makeMatmulTc());
    scop = Scop::makeScheduled(*scop, SchedulerOptions().view);
    auto root = scop->scheduleRoot();
    bandSplit(root, root->child({0}), 2);
    return scop;
  }

  std::unique_ptr<MappedScop> makeUnmapped(std::string tc) {
    return MappedScop::makeOneBlockOneThread(Prepare(tc));
  }

  static CudaMappingOptions DefaultOptions() {
    return CudaMappingOptions::makeNaiveMappingOptions();
  }

  std::unique_ptr<MappedScop> TileAndMapThreads(
      std::unique_ptr<Scop>&& scop,
      const vector<size_t>& tileSizes,
      const array<size_t, 2>& blockSizes = {32ul, 8ul}) {
    // Keep non-const schedule tree pointer for testing purposes.
    auto root = scop->scheduleRoot();
    bandTile(root->child({0}), tileSizes, TileOptions::ShiftPointLoops);

    // Map to blocks (1 single block here)
    auto mscop = MappedScop::makeMappedScop(
        std::move(scop),
        Grid{1},
        Block{blockSizes[0], blockSizes[1]},
        0,
        false);
    auto band = mscop->mapBlocksForward(root->child({0}), 1);
    bandScale(band, tileSizes);

    auto ns = ScheduleTree::collectDFSPostorder(root, ScheduleTreeType::Band);
    mscop->mapThreadsBackward(ns[1]);
    mscop->insertMappingContext();
    return mscop;
  }

  std::unique_ptr<MappedScop> TileAndMapBlocksAndThreads(
      std::unique_ptr<Scop>&& scop,
      const vector<size_t>& tileSizes,
      const array<size_t, 2>& gridSizes,
      const array<size_t, 2>& blockSizes) {
    // Keep non-const schedue tree pointer for testing purposes.
    auto root = scop->scheduleRoot();
    bandTile(root->child({0}), tileSizes, TileOptions::ShiftPointLoops);
    auto mscop = MappedScop::makeMappedScop(
        std::move(scop),
        Grid{gridSizes[0], gridSizes[1]},
        Block{blockSizes[0], blockSizes[1]},
        0,
        false);

    // Map to blocks
    auto band = mscop->mapBlocksForward(root->child({0}), 2);
    bandScale(band, tileSizes);

    band = mscop->mapThreadsBackward(band->child({0}));
    mscop->insertMappingContext();
    return mscop;
  }

  void TileAndCheckStructuralEquality(
      Scop& scop,
      TileOptions tileOptions,
      const std::vector<unsigned long>& tileSizes) {
    auto schedule = scop.scheduleRoot();
    auto schedule2 = ScheduleTree::makeScheduleTree(*schedule); // make a copy
    bandTile(schedule->child({0}), tileSizes, tileOptions);
    auto scheduleISLPP = fromIslSchedule(toIslSchedule(schedule).reset_user());

    {
      auto ctx = isl::with_exceptions::globalIslCtx();
      applyTileOptions(ctx, tileOptions);
      auto islNode = toIslSchedule(schedule2.get()).get_root().child(0);
      auto mv = isl::makeMultiVal(
          schedule2->child({0})
              ->elemAs<ScheduleTreeElemBand>()
              ->mupa_.get_space(),
          tileSizes);
      islNode = islNode.as<isl::schedule_node_band>().tile(mv);
      auto scheduleISL = fromIslSchedule(islNode.get_schedule().reset_user());

      ASSERT_TRUE(*scheduleISL == *scheduleISLPP) << *scheduleISL << "\nVS\n"
                                                  << *scheduleISLPP;
    }
  }

  std::string codegenMapped(
      std::string tc,
      const CudaMappingOptions& mappingOptions) {
    auto scop = Prepare(tc);
    auto mscop = MappedScop::makeWithOuterBlockInnerThreadStrategy(
        std::move(scop), mappingOptions);
    return std::get<0>(mscop->codegen(specializedName));
  }

  static constexpr auto specializedName = "kernel_anon";
  std::unique_ptr<Scop> scop;
};

// struct used to test the corectness and the optimality of
// MappedScop::findBestSyncConfigInSeq.
struct SyncConfigInSeqOptimalityTest : public ::testing::Test {
  // Check if the configuration config is correct when given bestSync.
  bool checkConfig(
      const std::vector<std::pair<int, int>>& config,
      const std::vector<std::vector<int>>& bestSync,
      size_t nChildren,
      bool hasOuterSequentialMember) {
    // For every pair of children i and i+k, test if they is a
    // needed synchronization between them.
    for (size_t k = 1; k < nChildren; ++k) {
      auto range = hasOuterSequentialMember ? nChildren : nChildren - k;
      for (size_t i = 0; i < range; ++i) {
        if (bestSync[i][k] == 0) {
          continue;
        }
        bool areSeparated = false;
        for (size_t c = 0; c < config.size(); ++c) {
          if ((i + k) % nChildren > i) {
            if (config[c].first >= (int)i && config[c].first < (int)(i + k) &&
                config[c].second >= bestSync[i][k]) {
              areSeparated = true;
              break;
            }
          } else {
            // The child i+k correspond to the child (i+k) % nChildren
            // at the next iteration of the outer sequential member
            if ((config[c].first < (int)((i + k) % nChildren) ||
                 config[c].first >= (int)i) &&
                config[c].second >= bestSync[i][k]) {
              areSeparated = true;
              break;
            }
          }
        }
        if (not areSeparated) {
          return false;
        }
      }
    }

    if (not hasOuterSequentialMember) {
      return true;
    }
    // If there is an outer sequential member, check also if there is
    // the synchronization needed between a child i and the same child but
    // at the next iteration.
    int maxValue = 0;
    for (size_t i = 0; i < nChildren; ++i) {
      maxValue = std::max(maxValue, bestSync[i][0]);
    }
    if (maxValue == 0) {
      return true;
    }
    for (size_t c = 0; c < config.size(); ++c) {
      if (config[c].second >= maxValue) {
        return true;
      }
    }
    return false;
  }

  // Get the number of synchronizations of a configuration
  // The first element is the number of __syncthreads, and
  // the second one is the number of __syncwarp.
  std::pair<int, int> getConfigValue(
      const std::vector<std::pair<int, int>>& config) {
    std::pair<int, int> value = {0, 0};
    for (size_t i = 0; i < config.size(); i++) {
      if (config[i].second == 1) {
        value.second++;
      }
      if (config[i].second == 2) {
        value.first++;
      }
    }
    return value;
  }

  // Find the number of synchronizations of the optimal configuration
  // The optimal configuration is found by brute force.
  pair<int, int> findBestConfigValue(
      const std::vector<std::vector<int>>& bestSync,
      size_t nChildren,
      bool hasOuterSequentialMember) {
    auto range = hasOuterSequentialMember ? nChildren + 1 : nChildren;
    std::vector<std::pair<int, int>> config(range);
    for (size_t i = 0; i < range; i++) {
      config[i] = {i, 0};
    }

    // Test for every config possible.
    pair<int, int> bestValue = {range + 1, range + 1};
    while (true) {
      // Check if the current configuration is correct,
      // and update the best value if needed.
      if (checkConfig(config, bestSync, nChildren, hasOuterSequentialMember)) {
        bestValue = std::min(bestValue, getConfigValue(config));
      }
      // Compute the next configuration
      int current = range - 1;
      while (current != -1 && config[current].second == 2) {
        config[current].second = 0;
        current--;
      }
      if (current == -1) {
        // Every configuration has been tested.
        return bestValue;
      }
      config[current].second++;
    }
  }

  // Test the correctness of the findBestSyncConfigInSeq function
  // on an example.
  bool testCorrectness(
      const std::vector<std::vector<int>>& bestSync,
      size_t nChildren,
      bool hasOuterSequentialMember) {
    std::vector<std::pair<int, int>> config =
        MappedScop::findBestSyncConfigInSeq(
            bestSync, nChildren, hasOuterSequentialMember);
    return checkConfig(config, bestSync, nChildren, hasOuterSequentialMember);
  }

  // Test the optimality of the findBestSyncConfigInSeq function
  // on an example.
  bool testOptimality(
      const std::vector<std::vector<int>>& bestSync,
      size_t nChildren,
      bool hasOuterSequentialMember) {
    std::vector<std::pair<int, int>> config =
        MappedScop::findBestSyncConfigInSeq(
            bestSync, nChildren, hasOuterSequentialMember);
    std::pair<int, int> computedBestValue = getConfigValue(config);
    std::pair<int, int> bestValue =
        findBestConfigValue(bestSync, nChildren, hasOuterSequentialMember);
    return bestValue == computedBestValue;
  }

  //  Generate a random example to be used by findBestSyncConfigInSeq.
  std::vector<std::vector<int>> generateRandomBestSync(
      size_t nChildren,
      bool hasOuterSequentialMember,
      int seed) {
    std::mt19937 generator(seed);
    std::uniform_int_distribution<int> distribution(0, 2);
    std::vector<std::vector<int>> bestSync(
        nChildren, std::vector<int>(nChildren));
    for (size_t i = 0; i < nChildren; ++i) {
      for (size_t k = 0; k < nChildren; ++k) {
        bestSync[i][k] = distribution(generator);
      }
    }

    return bestSync;
  }
};

TEST_F(SyncConfigInSeqOptimalityTest, WithoutOuterSequentialMember) {
  std::vector<std::vector<int>> bestSync;
  for (size_t nChildren = 2; nChildren < 5; ++nChildren) {
    for (int i = 0; i < 50; i++) {
      bestSync = generateRandomBestSync(nChildren, false, i);
      ASSERT_TRUE(testCorrectness(bestSync, nChildren, false))
          << "Synchronization configuration is not correct.";
      ASSERT_TRUE(testOptimality(bestSync, nChildren, false))
          << "Synchronization configuration is not optimal.";
    }
  }
}

TEST_F(SyncConfigInSeqOptimalityTest, WithOuterSequentialMember) {
  std::vector<std::vector<int>> bestSync;
  for (size_t nChildren = 2; nChildren < 5; ++nChildren) {
    for (int i = 0; i < 50; i++) {
      bestSync = generateRandomBestSync(nChildren, true, i);
      ASSERT_TRUE(testCorrectness(bestSync, nChildren, true))
          << "Synchronization configuration is no correct.";
      ASSERT_TRUE(testOptimality(bestSync, nChildren, true))
          << "Synchronization configuration is not optimal.";
    }
  }
}

TEST_F(PolyhedralMapperTest, Basic) {
  string tc = R"TC(
def fun(float(N, M) A, float(N, M) B) -> (C) {
    C(n, m) = A(n, m) + B(n, m)
}
)TC";

  auto scop = PrepareAndJoinBands(tc);
  scop = Scop::makeSpecializedScop<int>(*scop, {{"N", 512}});
  auto mscop = TileAndMapBlocksAndThreads(
      std::move(scop), {16ul, 16ul}, {256ul, 256ul}, {16ul, 16ul});

  auto res = mscop->codegen(specializedName);

  std::string expected(
      R"RES(int b0 = blockIdx.x; int b1 = blockIdx.y; int b2 = blockIdx.z;
  int t0 = threadIdx.x; int t1 = threadIdx.y; int t2 = threadIdx.z;
  float32 (*C)[M] = reinterpret_cast<float32 (*)[M]>(pC);
  const float32 (*A)[M] = reinterpret_cast<const float32 (*)[M]>(pA);
  const float32 (*B)[M] = reinterpret_cast<const float32 (*)[M]>(pB);
  for (int c1 = 16 * b1; c1 < M; c1 += 4096) {
    if (M >= t0 + c1 + 1) {
      C[(t1 + 16 * b0)][(t0 + c1)] = (A[(t1 + 16 * b0)][(t0 + c1)] + B[(t1 + 16 * b0)][(t0 + c1)]);
    }
  }
}
)RES");

  ASSERT_NE(std::string::npos, std::get<0>(res).find(expected))
      << std::get<0>(res);
  ASSERT_EQ(32u, std::get<1>(res).view[0])
      << "Improper dim in: " << std::get<1>(res).view;
}

TEST_F(PolyhedralMapperTest, MultiStmt) {
  string tc = R"TC(
def fun(float(N, N, N, N) A, float(N, N) B, float(N, N) C, float(N, N) D)
-> (O1, O2, O3)
{
    O1(n0, n1) +=! A(n0, n1, r_n2, r_n3) * B(n0, n1)
    O2(n0, n1)  =  C(n0, n1) *  D(n0, n1)
    O3(n0, n1)  = O1(n0, n1) + O2(n0, n1)
}
)TC";

  auto mscop = makeUnmapped(tc);
  // Don't intersect context with the domain and see what happens
  auto res = std::get<0>(mscop->codegen(specializedName));

  std::string expected(
      R"RES(int b0 = blockIdx.x; int b1 = blockIdx.y; int b2 = blockIdx.z;
  int t0 = threadIdx.x; int t1 = threadIdx.y; int t2 = threadIdx.z;
  float32 (*O1)[N] = reinterpret_cast<float32 (*)[N]>(pO1);
  float32 (*O2)[N] = reinterpret_cast<float32 (*)[N]>(pO2);
  float32 (*O3)[N] = reinterpret_cast<float32 (*)[N]>(pO3);
  const float32 (*A)[N][N][N] = reinterpret_cast<const float32 (*)[N][N][N]>(pA);
  const float32 (*B)[N] = reinterpret_cast<const float32 (*)[N]>(pB);
  const float32 (*C)[N] = reinterpret_cast<const float32 (*)[N]>(pC);
  const float32 (*D)[N] = reinterpret_cast<const float32 (*)[N]>(pD);
  for (int c0 = 0; c0 < N; c0 += 1) {
    for (int c1 = 0; c1 < N; c1 += 1) {
      O1[c0][c1] = 0.000000f;
    }
  }
  for (int c0 = 0; c0 < N; c0 += 1) {
    for (int c1 = 0; c1 < N; c1 += 1) {
      for (int c2 = 0; c2 < N; c2 += 1) {
        for (int c3 = 0; c3 < N; c3 += 1) {
          O1[c0][c1] = (O1[c0][c1] + (A[c0][c1][c2][c3]*B[c0][c1]));
        }
      }
    }
  }
  for (int c0 = 0; c0 < N; c0 += 1) {
    for (int c1 = 0; c1 < N; c1 += 1) {
      O2[c0][c1] = (C[c0][c1]*D[c0][c1]);
    }
  }
  for (int c0 = 0; c0 < N; c0 += 1) {
    for (int c1 = 0; c1 < N; c1 += 1) {
      O3[c0][c1] = (O1[c0][c1] + O2[c0][c1]);
    }
  }
}
)RES");

  ASSERT_NE(std::string::npos, res.find(expected)) << res;
}

TEST_F(PolyhedralMapperTest, BareVariables) {
  string tc = R"TC(
def fun(float(N, N) A) -> (O)
{
    O(n0, n1) = A(n0, n1) + n0 + n1 + N
}
)TC";

  auto mscop = makeUnmapped(tc);
  auto res = std::get<0>(mscop->codegen(specializedName));

  string expected(
      R"RES(__global__ void kernel_anon(int32 N, float32* pO, const float32* pA) {
  int b0 = blockIdx.x; int b1 = blockIdx.y; int b2 = blockIdx.z;
  int t0 = threadIdx.x; int t1 = threadIdx.y; int t2 = threadIdx.z;
  float32 (*O)[N] = reinterpret_cast<float32 (*)[N]>(pO);
  const float32 (*A)[N] = reinterpret_cast<const float32 (*)[N]>(pA);
  for (int c0 = 0; c0 < N; c0 += 1) {
    for (int c1 = 0; c1 < N; c1 += 1) {
      O[c0][c1] = (((A[c0][c1] + float32(c0)) + float32(c1)) + float32(N));
    }
  }
}
)RES");

  ASSERT_NE(std::string::npos, res.find(expected)) << res;
}

TEST_F(PolyhedralMapperTest, CudaFunctions) {
  string tc = R"TC(
def fun(float(N, N) A, float(N, N) B, float(N) C) -> (O)
{
    O(n0, n1) = nextafter(C(n0), exp(A(n0, n1))) + log(B(n1, n0))
}
)TC";

  auto mscop = makeUnmapped(tc);
  mscop->fixParameters<int>({{"N", 512}});
  auto res = std::get<0>(mscop->codegen(specializedName));

  string expected =
      R"RES(__global__ void kernel_anon(int32 N, float32* pO, const float32* pA, const float32* pB, const float32* pC) {
  int b0 = blockIdx.x; int b1 = blockIdx.y; int b2 = blockIdx.z;
  int t0 = threadIdx.x; int t1 = threadIdx.y; int t2 = threadIdx.z;
  float32 (*O)[512] = reinterpret_cast<float32 (*)[512]>(pO);
  const float32 (*A)[512] = reinterpret_cast<const float32 (*)[512]>(pA);
  const float32 (*B)[512] = reinterpret_cast<const float32 (*)[512]>(pB);
  const float32 (*C) = reinterpret_cast<const float32 (*)>(pC);
  for (int c0 = 0; c0 <= 511; c0 += 1) {
    for (int c1 = 0; c1 <= 511; c1 += 1) {
      O[c0][c1] = (nextafter(C[c0], exp(A[c0][c1])) + log(B[c1][c0]));
    }
  }
}
)RES";

  ASSERT_NE(std::string::npos, res.find(expected)) << res;
}

constexpr auto kExpectedMatmul_64_64_64 =
    R"CUDA(int b0 = blockIdx.x; int b1 = blockIdx.y; int b2 = blockIdx.z;
  int t0 = threadIdx.x; int t1 = threadIdx.y; int t2 = threadIdx.z;
  float32 (*O)[64] = reinterpret_cast<float32 (*)[64]>(pO);
  const float32 (*A)[64] = reinterpret_cast<const float32 (*)[64]>(pA);
  const float32 (*B)[64] = reinterpret_cast<const float32 (*)[64]>(pB);
  for (int c0 = 0; c0 <= 63; c0 += 16) {
    for (int c1 = 0; c1 <= 63; c1 += 16) {
      for (int c2 = t1; c2 <= 15; c2 += 8) {
        O[(c0 + c2)][(t0 + c1)] = 0.000000f;
        for (int c4 = 0; c4 <= 63; c4 += 1) {
          O[(c0 + c2)][(t0 + c1)] = (O[(c0 + c2)][(t0 + c1)] + (A[(c0 + c2)][c4]*B[c4][(t0 + c1)]));
        }
      }
    }
  }
}
)CUDA";

TEST_F(PolyhedralMapperTest, MergedContexts) {
  auto scop = PrepareAndJoinBandsMatMul();

  // Unit test claims to use the specialized context properly
  scop->fixParameters<int>({{"M", 64}, {"N", 64}, {"K", 64}});
  scop->specializeToContext();

  auto mscop = TileAndMapThreads(std::move(scop), {16, 16}, {32ul, 8ul});
  auto res = std::get<0>(mscop->codegen(specializedName));
  ASSERT_TRUE(std::string::npos != res.find(kExpectedMatmul_64_64_64)) << res;
}

TEST_F(PolyhedralMapperTest, Match1) {
  auto scop = PrepareAndJoinBandsMatMul();
  auto schedule = scop->scheduleRoot();

  auto mscop = TileAndMapThreads(std::move(scop), {16, 16}, {32ul, 8ul});
  auto f = match(
      band(sequence(
          filter([](isl::union_set f) {
            return f.get_space().dim(isl::dim_type::param) == 3;
          }),
          filter())),
      schedule);
  EXPECT_EQ(1u, f.size());
}

TEST_F(PolyhedralMapperTest, CopyTC) {
  string tc = R"TC(
def fun(float(M, N) I) -> (O) {
    O(m, n) = I(m, n)
}
)TC";

  auto scop = PrepareAndJoinBands(tc);
  auto tileOptions = TileOptions::ShiftPointLoops | TileOptions::ScaleTileLoops;
  TileAndCheckStructuralEquality(*scop, tileOptions, {3ul, 5});
}

TEST_F(PolyhedralMapperTest, MatmulTC) {
  auto scop = PrepareAndJoinBandsMatMul();
  auto tileOptions = TileOptions::ShiftPointLoops | TileOptions::ScaleTileLoops;
  TileAndCheckStructuralEquality(*scop, tileOptions, {3ul, 4ul});
}

TEST_F(PolyhedralMapperTest, MatmulShiftScale) {
  auto scop = PrepareAndJoinBandsMatMul();
  auto tileOptions = TileOptions::ShiftPointLoops | TileOptions::ScaleTileLoops;
  TileAndCheckStructuralEquality(*scop, tileOptions, {3ul, 4ul});
}

TEST_F(PolyhedralMapperTest, MatmulShift) {
  auto scop = PrepareAndJoinBandsMatMul();
  auto tileOptions = TileOptions::ShiftPointLoops;
  TileAndCheckStructuralEquality(*scop, tileOptions, {3ul, 4ul});
}

TEST_F(PolyhedralMapperTest, MatmulScale) {
  auto scop = PrepareAndJoinBandsMatMul();
  auto tileOptions = TileOptions::ScaleTileLoops;
  TileAndCheckStructuralEquality(*scop, tileOptions, {3ul, 4ul});
}

TEST_F(PolyhedralMapperTest, MatmulNoshiftNoscale) {
  auto scop = PrepareAndJoinBandsMatMul();
  auto tileOptions = TileOptions();
  TileAndCheckStructuralEquality(*scop, tileOptions, {3ul, 4ul});
}

static const string kTcAdd = R"TC(
def fun(float(N, M) A, float(N, M) B) -> (C) {
    C(n, m) = A(n, m) + B(n, m)
}
)TC";

/*
 * Check that only a single-dimensional unrolling is performed
 * if the unrolling factor is smaller than the number of instances
 * of the innermost two loops.
 * The loop iterator is "c2" and is only present
 * if the second innermost loop has not been unrolled.
 */
TEST_F(PolyhedralMapperTest, Unroll1D) {
  auto mappingOptions = DefaultOptions().tile(64, 64).unroll(15);
  auto scop = PrepareAndJoinBands(kTcAdd);
  scop->fixParameters<int>({{"N", 1024}, {"M", 1024}});
  auto mscop = MappedScop::makeWithOuterBlockInnerThreadStrategy(
      std::move(scop), mappingOptions);
  auto code = std::get<0>(mscop->codegen(specializedName));
  std::string expected("C[(64 * b0 + c2)][(t0 + 64 * b1)]");
  ASSERT_TRUE(code.find(expected) != std::string::npos) << code;
}

/*
 * Check that two-dimensional unrolling is performed if the unrolling factor
 * is greater than or equal to the number of instances of
 * the innermost two loops.
 * In particular, check for one of the unrolled instances
 * where the second innermost loop iterator ("c2") is replaced by t1 + 32 and
 * the innermost loop iterator ("c3") is replaced by t0 + 32.
 */
TEST_F(PolyhedralMapperTest, Unroll2D) {
  auto mappingOptions = DefaultOptions().tile(64, 64).unroll(16);
  auto scop = PrepareAndJoinBands(kTcAdd);
  scop->fixParameters<int>({{"N", 1024}, {"M", 1024}});
  auto mscop = MappedScop::makeWithOuterBlockInnerThreadStrategy(
      std::move(scop), mappingOptions);
  auto code = std::get<0>(mscop->codegen(specializedName));
  std::string expected("C[(t1 + 64 * b0 + 32)][(t0 + 64 * b1 + 32)]");
  ASSERT_TRUE(code.find(expected) != std::string::npos);
}

/*
 * Map 1D code to 2D grid (set up by makeNaiveMappingOptions()) and
 * check that the code is pinned to one particular value of
 * block identifier b1 and thread identifier t1.
 */
TEST_F(PolyhedralMapperTest, Copy1D) {
  auto tc = R"TC(
def fun(float(N) I) -> (O) {
    O(n) = I(n)
}
)TC";
  auto scop = Prepare(tc);
  auto mscop = MappedScop::makeWithOuterBlockInnerThreadStrategy(
      std::move(scop), DefaultOptions());
  auto codeAndLaunchBounds = mscop->codegen(specializedName);
  USING_MAPPING_SHORT_NAMES(BX, BY, BZ, TX, TY, TZ);
  EXPECT_EQ(1u, BY.mappingSize(std::get<1>(codeAndLaunchBounds).view));
  EXPECT_EQ(1u, TY.mappingSize(std::get<1>(codeAndLaunchBounds).view));
}

/*
 * Check that a schedule tree without any bands gets mapped properly,
 * i.e., that a band is inserted at the leaf.
 * Also check that scalars are dereferenced properly.
 */
TEST_F(PolyhedralMapperTest, DISABLED_0D) {
  auto tc = R"TC(
def fun() -> (O) {
    O = 0
}
)TC";
  auto code = codegenMapped(tc, DefaultOptions());
  EXPECT_TRUE(code.find("*O = 0;") != std::string::npos);
}

/*
 * Check that a schedule tree without a single outer band gets mapped
 * properly, i.e., that a band is inserted above the branching.
 * Use the minimal fusion strategy to ensure the scheduler produces
 * an outer sequence.
 * Check that no synchronizations are inserted, since there is no
 * dependences between threads.
 */
TEST_F(PolyhedralMapperTest, Copy2) {
  auto tc = R"TC(
def fun(float(N) I) -> (O1, O2) {
    O1(n) =  I(n)
    O2(n) = O1(n)
}
)TC";
  auto mappingOptions = DefaultOptions();
  mappingOptions.scheduleFusionStrategy(FusionStrategy::Min);
  auto code = codegenMapped(tc, mappingOptions);
  auto loop = "for (int c0 = t0; c0 < N; c0 += 32)";
  auto blockSync = "__syncthreads();";
  auto pos1 = code.find(loop);
  auto pos2 = code.find(loop, pos1 + 1);
  auto pos3 = code.find(blockSync);
  EXPECT_TRUE(pos1 != std::string::npos);
  EXPECT_TRUE(pos2 != std::string::npos);
  EXPECT_TRUE(pos3 == std::string::npos);
}

/*
 * Check that children of a sequence that are (initially) mapped to fewer
 * thread identifiers than other children of the same sequence
 * eventually get mapped to a single instance of the remaining
 * thread identifier(s).
 * Use the minimal fusion strategy to ensure the scheduler produces
 * a sequence.
 */
TEST_F(PolyhedralMapperTest, CopyUnbalanced) {
  auto tc = R"TC(
def fun(float(N) I1, float(N, N) I2) -> (O1, O2) {
    O1(n)      = I1(n)
    O2(n0, n1) = I2(n0, n1)
}
)TC";
  auto mappingOptions = DefaultOptions();
  mappingOptions.scheduleFusionStrategy(FusionStrategy::Min);
  auto code = codegenMapped(tc, mappingOptions);
  ASSERT_TRUE(code.find("t1 == 0") != std::string::npos);
}

/*
 * Check that point loops rescheduling allows for a different fusion strategy.
 * In particular, we want tile loops to be fused and point loops to be
 * fissioned.
 */
TEST_F(PolyhedralMapperTest, ReschedulingMaxMinFuse) {
  std::string tc = R"TC(
def fun(float(N, M) A, float(N, M) B) -> (C,D) {
    C(n, m) = A(n, m)
    D(n, m) = B(n, m)
})TC";

  auto originalScop = Prepare(tc);

  auto tiling = Tiling({32, 32});

  auto minFusionSchedulerOptions = SchedulerOptions();
  minFusionSchedulerOptions.view.proto.set_fusion_strategy(FusionStrategy::Min);

  auto maxFusionSchedulerOptions = SchedulerOptions();
  maxFusionSchedulerOptions.view.proto.set_fusion_strategy(FusionStrategy::Max);

  originalScop->computeAllDependences();
  // Schedule with maximal fusion, then tile and reschedule point loops for
  // minimal fusion.
  auto scop =
      Scop::makeScheduled(*originalScop, maxFusionSchedulerOptions.view);
  auto outerBand = scop->tileOuterBand(tiling.view);
  scop->reschedule(outerBand->child({0}), minFusionSchedulerOptions.view);
  auto maxMinOuterBand = ScheduleTree::makeScheduleTree(*outerBand);

  // Schedule with maximal fusion, then tile and reschedule point loops for
  // minimal fusion again.  Schedule is expected to be identical to
  // the one without rescheduling.
  scop = Scop::makeScheduled(*originalScop, maxFusionSchedulerOptions.view);
  outerBand = scop->tileOuterBand(tiling.view);
  scop->reschedule(outerBand->child({0}), maxFusionSchedulerOptions.view);
  auto maxMaxOuterBand = ScheduleTree::makeScheduleTree(*outerBand);

  // Schedule with maximal fusion and tile.
  scop = Scop::makeScheduled(*originalScop, maxFusionSchedulerOptions.view);
  auto tiledBand =
      ScheduleTree::makeScheduleTree(*scop->tileOuterBand(tiling.view));

  ASSERT_TRUE(maxMinOuterBand->elemAs<ScheduleTreeElemBand>());
  ASSERT_TRUE(maxMaxOuterBand->elemAs<ScheduleTreeElemBand>());
  ASSERT_TRUE(tiledBand->elemAs<ScheduleTreeElemBand>());

  auto maxMinStructure = // expected structure when rescheduling with MinFuse
      band( // tile band
          sequence(
              filter( // S1
                  band() // point band
                  ),
              filter( // S2
                  band() // point band
                  )));

  auto maxMaxStructure = // expected structure when rescheduling with MaxFuse
      band( // tile band
          band( // point band
              sequence(
                  filter(), // S1
                  filter()))); // S2

  auto maxMinMatches = match(maxMinStructure, maxMinOuterBand.get());
  auto maxMaxMatches = match(maxMaxStructure, maxMaxOuterBand.get());
  auto tiledMatches = match(maxMaxStructure, tiledBand.get());

  // The right structure should be matched.
  ASSERT_EQ(maxMinMatches.size(), 1u);
  ASSERT_EQ(maxMaxMatches.size(), 1u);
  ASSERT_EQ(tiledMatches.size(), 1u);

  // The wrong structure should not be matched.
  EXPECT_EQ(match(maxMaxStructure, maxMinOuterBand.get()).size(), 0u);
  EXPECT_EQ(match(maxMinStructure, maxMaxOuterBand.get()).size(), 0u);
  EXPECT_EQ(match(maxMinStructure, tiledBand.get()).size(), 0u);

  // The tile band subtree should be the top node of the match.
  EXPECT_EQ(maxMinMatches[0], maxMinOuterBand.get());
  EXPECT_EQ(maxMaxMatches[0], maxMaxOuterBand.get());
  EXPECT_EQ(tiledMatches[0], tiledBand.get());
}

/*
 * Check that the point loop is rescheduled when
 * the outer and intra-tile fusion strategies are different.
 * In particular, check that minimal fusion on the point band
 * results in four inner loops.
 * Since the AST generator peels off an iteration from a tile loop,
 * the output code actually contains six inner loops.
 * The check for these extra two inner loops can be removed
 * when the input to the AST generator is changed to prevent this peeling.
 */
TEST_F(PolyhedralMapperTest, Rescheduling2MM) {
  std::string tc = R"TC(
def fun(float(M, K) A, float(K, N) B, float(K, N) C) -> (D, E) {
    D(m, n) +=! A(m, r_k) * B(r_k, n)
    E(m, n) +=! A(m, r_k) * C(r_k, n)
})TC";

  auto mappingOptions = DefaultOptions();
  mappingOptions.outerScheduleFusionStrategy(FusionStrategy::Max);
  mappingOptions.intraTileScheduleFusionStrategy(FusionStrategy::Min);
  auto code = codegenMapped(tc, mappingOptions);
  auto innerLoopIncrement = "c3 += 8";
  auto pos1 = code.find(innerLoopIncrement);
  auto pos2 = code.find(innerLoopIncrement, pos1 + 1);
  auto pos3 = code.find(innerLoopIncrement, pos2 + 1);
  auto pos4 = code.find(innerLoopIncrement, pos3 + 1);
  auto pos5 = code.find(innerLoopIncrement, pos4 + 1);
  EXPECT_TRUE(pos1 != std::string::npos);
  EXPECT_TRUE(pos2 != std::string::npos);
  EXPECT_TRUE(pos3 != std::string::npos);
  EXPECT_TRUE(pos4 != std::string::npos);
  EXPECT_TRUE(pos5 != std::string::npos);
}

/*
 * Check that a 1D reduction is properly separated into full and partial blocks
 * and that the full blocks get mapped to a library call.
 * In practice, check that the library call appears in the code and
 * that there is also an update in the code that did not get mapped
 * to a library call (the partial blocks).
 */
TEST_F(PolyhedralMapperTest, Reduction1D) {
  string tc = R"TC(
def fun(float(N) I) -> (O) {
    O +=! I(r_n)
}
)TC";
  auto mappingOptions = DefaultOptions();
  mappingOptions.matchLibraryCalls(true);
  mappingOptions.mapToThreads({32});
  auto code = codegenMapped(tc, mappingOptions);
  using tc::code::cuda::kCUBReductionName;
  EXPECT_TRUE(code.find(kCUBReductionName) != std::string::npos);
  EXPECT_TRUE(code.find("O[0] = (O") != std::string::npos);
}

struct ReductionTest : public PolyhedralMapperTest {
  static CudaMappingOptions reductionTestMappingOptions() {
    return DefaultOptions()
        .outerScheduleFusionStrategy(tc::FusionStrategy::Preserve3Coincident)
        .outerScheduleAllowSkewing(false)
        .outerSchedulePositiveOrthant(true)
        .intraTileScheduleFusionStrategy(
            tc::FusionStrategy::Preserve3Coincident)
        .intraTileScheduleAllowSkewing(false)
        .intraTileSchedulePositiveOrthant(true)
        .fixParametersBeforeScheduling(false)
        .tile(18, 32)
        .unroll(16)
        .tileImperfectlyNested(false)
        .matchLibraryCalls(true)
        .mapToThreads({512})
        .mapToBlocks({16384})
        .useSharedMemory(true)
        .usePrivateMemory(false)
        .unrollCopyShared(true);
  }

  void Check(const string& tc) {
    auto code = codegenMapped(tc, reductionTestMappingOptions());
    using tc::code::cuda::kCUBReductionName;
    EXPECT_TRUE(code.find(kCUBReductionName) != std::string::npos);
  }
};

/*
 * Check that a reduction library call is produced when the reduction
 * instruction is before an instruction modifying the same tensor.
 */
TEST_F(ReductionTest, BeforeInstruction) {
  Check(R"TC(
def fun(float(N, K) I) -> (O) {
    O(n) +=! I(n, r_n)
    O(n) = O(n) / (K)
}
)TC");
}

/*
 * Check that a reduction library call is produced when the reduction
 * instruction is after an instruction modifying the same tensor.
 */
TEST_F(ReductionTest, AfterInstruction) {
  Check(R"TC(
def fun(float(N, K) I, float(N) O0) -> (O) {
    O(n) = 0.0 where n in 0:N
    O(n) += O0(n)
    O(n) += I(n, r_n)
}
)TC");
}

/*
 * Check that a reduction library call is produced when the reduction
 * instruction is placed after an instruction modifying the same tensor and
 * before an instruction modifying the same tensor.
 */
TEST_F(ReductionTest, BetweenInstructions) {
  Check(R"TC(
def fun(float(N, K) I, float(N) O0) -> (O) {
    O(n) = 0.0 where n in 0:N
    O(n) += O0(n)
    O(n) += I(n, r_n)
    O(n) = O(n) / (K)
}
)TC");
}

/*
 * Check that a 2D mean with these parameters does not produce a library call.
 * The call is not produced because the band is tiled by 32 and 512 threads are
 * mapped to the band.
 * In practice, check that the library call does not appear in the code.
 */
TEST_F(PolyhedralMapperTest, Mean2DNonParametric_512threads) {
  string tc = R"TC(
def fun(float(36864, 1024) I) -> (O) {
    O(n) +=! I(n, r_n)
    O(n) = O(n) / (1024)
}
)TC";
  auto mappingOptions =
      DefaultOptions()
          .outerScheduleFusionStrategy(tc::FusionStrategy::Preserve3Coincident)
          .outerScheduleAllowSkewing(false)
          .outerSchedulePositiveOrthant(true)
          .intraTileScheduleFusionStrategy(tc::FusionStrategy::Min)
          .intraTileScheduleAllowSkewing(false)
          .intraTileSchedulePositiveOrthant(true)
          .fixParametersBeforeScheduling(false)
          .tile(18, 32)
          .unroll(16)
          .tileImperfectlyNested(false)
          .matchLibraryCalls(true)
          .mapToThreads({512})
          .mapToBlocks({16384})
          .useSharedMemory(true)
          .usePrivateMemory(false)
          .unrollCopyShared(true);

  auto code = codegenMapped(tc, mappingOptions);
  using tc::code::cuda::kCUBReductionName;
  EXPECT_TRUE(code.find(kCUBReductionName) == std::string::npos);
}

/*
 * Check that a 2D mean with these parameters produce a reduction library call.
 * In practice, check that the library call appears in the code.
 */
TEST_F(PolyhedralMapperTest, Mean2DNonParametric_32threads) {
  string tc = R"TC(
def fun(float(36864, 1024) I) -> (O) {
    O(n) +=! I(n, r_n)
    O(n) = O(n) / (1024)
}
)TC";
  auto mappingOptions =
      DefaultOptions()
          .outerScheduleFusionStrategy(tc::FusionStrategy::Preserve3Coincident)
          .outerScheduleAllowSkewing(false)
          .outerSchedulePositiveOrthant(true)
          .intraTileScheduleFusionStrategy(tc::FusionStrategy::Min)
          .intraTileScheduleAllowSkewing(false)
          .intraTileSchedulePositiveOrthant(true)
          .fixParametersBeforeScheduling(false)
          .tile(18, 32)
          .unroll(16)
          .tileImperfectlyNested(false)
          .matchLibraryCalls(true)
          .mapToThreads({32})
          .mapToBlocks({16384})
          .useSharedMemory(true)
          .usePrivateMemory(false)
          .unrollCopyShared(true);

  auto code = codegenMapped(tc, mappingOptions);
  using tc::code::cuda::kCUBReductionName;
  EXPECT_TRUE(code.find(kCUBReductionName) != std::string::npos);
}

static const string kTcMM = R"TC(
def fun(float(M, K) A, float(K, N) B) -> (C) {
    C(m, n) +=! A(m, r_k) * B(r_k, n)
})TC";

/*
 * Check that a reduction mapped to a single-dimensional block
 * is properly separated into full and partial blocks and
 * that the full blocks get mapped to a library call.
 * In practice, check that the library call appears in the code and
 * that there is also an update in the code that did not get mapped
 * to a library call (the partial blocks).
 */
TEST_F(PolyhedralMapperTest, ReductionMM1D) {
  auto mappingOptions = DefaultOptions();
  mappingOptions.matchLibraryCalls(true);
  mappingOptions.mapToThreads({32});
  auto code = codegenMapped(kTcMM, mappingOptions);
  using tc::code::cuda::kCUBReductionName;
  EXPECT_TRUE(code.find(kCUBReductionName) != std::string::npos);
  EXPECT_TRUE(code.find("C[(c0 + c3)][(t0 + c1)] = (C") != std::string::npos);
}

/*
 * Check that a reduction mapped to a two-dimensional block
 * is properly separated into full and partial blocks and
 * that the full blocks get mapped to a library call.
 * In practice, check that the library call appears in the code and
 * that there is also an update in the code that did not get mapped
 * to a library call (the partial blocks).
 */
TEST_F(PolyhedralMapperTest, ReductionMM2D) {
  auto mappingOptions = DefaultOptions();
  mappingOptions.matchLibraryCalls(true);
  mappingOptions.mapToThreads({32, 32});
  auto code = codegenMapped(kTcMM, mappingOptions);
  using tc::code::cuda::kCUBReductionName;
  EXPECT_TRUE(code.find(kCUBReductionName) != std::string::npos);
  EXPECT_TRUE(code.find("C[(t1 + c0)][(t0 + c1)] = (C") != std::string::npos);
}

/*
 * Check that a subscript with affine and non-affine parts is handled by the
 * Halide to isl conversion, in particular that the conversion does not crash.
 */
TEST_F(PolyhedralMapperTest, NonAffineBoundLHSInBinOp) {
  string tc = R"TC(
def shiftedLut(float(E, D) LUT, int32(B, L) I) -> (O) {
  O(i, j) +=! LUT(I(i, k) + 1, j)
}
)TC";
  // This triggers tc2halide conversion and should not throw.
  Prepare(tc);
}

/*
 * Check that a subscript with affine and non-affine parts is handled by the
 * Halide to isl conversion, in particular that the conversion does not crash.
 */
TEST_F(PolyhedralMapperTest, NonAffineBoundRHSInBinOp) {
  string tc = R"TC(
def shiftedLut(float(E, D) LUT, int32(B, L) I) -> (O) {
  O(i, j) +=! LUT(1 + j + I(i, k), j)
}
)TC";
  // This triggers tc2halide conversion and should not throw.
  Prepare(tc);
}

/*
 * Check that a subscript with affine and non-affine parts is handled by the
 * Halide to isl conversion, in particular that the conversion does not crash.
 */
TEST_F(PolyhedralMapperTest, PerforatedConvolution) {
  string tc = R"TC(
def perforatedConvolution(float(N, C, H, W) input, float(M, C, KH, KW) weights,
                          int32(N, L) index) -> (output) {
  output(n, m, l) +=! input(n, c, index(n, l) + kh, index(n, l) + kw) * weights(m, c, kh, kw) where l in 0:L
 }
)TC";
  // This triggers tc2halide conversion and should not throw.
  Prepare(tc);
}

TEST_F(PolyhedralMapperTest, ReadOnlyCache) {
  auto tc = R"TC(
def fun(float(N) I) -> (O) {
    O(n)      = I(n)
}
)TC";
  auto mappingOptions = DefaultOptions().useReadOnlyCache(true);
  auto code = codegenMapped(tc, mappingOptions);
  using tc::code::cuda::kLdg;
  ASSERT_TRUE(code.find(kLdg + "(&O") == std::string::npos) << code; // no
  ASSERT_TRUE(code.find(kLdg + "(&I") != std::string::npos) << code; // yes
}

/*
 * Check that isolating the update statements does not introduce
 * an empty mapping filter.
 */
TEST_F(PolyhedralMapperTest, EmptyMapping) {
  constexpr static auto tc = R"TC(
  def var_2D_1D(float(N, K) I, float(N) mean) -> (var)
  {
       var(n) +=! I(n, r_k) * I(n, r_k)
       var(n)  =  var(n) / (K) - mean(n) * mean(n)
  }
)TC";
  auto mappingOptions = DefaultOptions()
                            .fixParametersBeforeScheduling(false)
                            .matchLibraryCalls(true)
                            .mapToThreads(256);
  auto scop = Prepare(tc);
  scop->fixParameters<int>({{"N", 1024}, {"K", 36864}});
  auto mscop = MappedScop::makeWithOuterBlockInnerThreadStrategy(
      std::move(scop), mappingOptions);
  mscop->codegen(specializedName);
}

TEST_F(PolyhedralMapperTest, ModulusConstantRHS) {
  string tc = R"TC(
def fun(float(N) a) -> (b) { b(i) = a(i % 3) where i in 0:N }
)TC";
  // This triggers tc2halide conversion and should not throw.
  auto scop = Prepare(tc);
  for (auto r : scop->reads.wrap().get_set_list()) {
    auto read = r.unwrap();
    // skip irrelevant reads, if any
    if (read.range().get_tuple_name() != std::string("a")) {
      continue;
    }
    EXPECT_EQ(r.get_stride(0), 3);
  }
}

TEST_F(PolyhedralMapperTest, ModulusVariableRHS) {
  string tc = R"TC(
def local_sparse_convolution(float(N, C, H, W) I, float(O, KC, KH, KW) W1) -> (O1) {
  O1(n, o, h, w) +=! I(n, kc % c, h + kh, w + kw) * W1(o, kc, kh, kw) where c in 1:C
}
)TC";
  // This triggers tc2halide conversion and should not throw.
  auto scop = Prepare(tc);
  for (auto r : scop->reads.range().get_set_list()) {
    // skip irrelevant reads, if any
    if (r.get_tuple_name() != std::string("I")) {
      continue;
    }
    EXPECT_TRUE(r.plain_is_universe());
  }
}

int main(int argc, char** argv) {
  ::testing::InitGoogleTest(&argc, argv);
  ::gflags::ParseCommandLineFlags(&argc, &argv, true);
  ::google::InitGoogleLogging(argv[0]);
  return RUN_ALL_TESTS();
}
