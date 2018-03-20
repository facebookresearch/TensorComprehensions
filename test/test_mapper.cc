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
#include "tc/core/polyhedral/codegen_cuda.h"
#include "tc/core/polyhedral/cuda/cuda_mapping_types.h"
#include "tc/core/polyhedral/functional.h"
#include "tc/core/polyhedral/mapped_scop.h"
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

using tc::polyhedral::detail::fromIslSchedule;
using tc::polyhedral::detail::toIslSchedule;

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

  std::unique_ptr<MappedScop> makeUnmapped(std::string tc) {
    return MappedScop::makeOneBlockOneThread(Prepare(tc));
  }

  static CudaMappingOptions DefaultOptions() {
    return CudaMappingOptions::makeNaiveCudaMappingOptions();
  }

  std::unique_ptr<MappedScop> TileAndMapThreads(
      std::unique_ptr<Scop>&& scop,
      const vector<size_t>& tileSizes,
      const array<size_t, 2>& blockSizes = {32ul, 8ul}) {
    // Keep non-const schedue tree pointer for testing purposes.
    auto root = scop->scheduleRoot();
    bandTile(root->child({0}), tileSizes, TileOptions::ShiftPointLoops);

    // Map to blocks (1 single block here)
    auto mscop = MappedScop::makeMappedScop(
        std::move(scop), Grid{1}, Block{blockSizes[0], blockSizes[1]}, 0);
    USING_MAPPING_SHORT_NAMES(BX, BY, BZ, TX, TY, TZ);
    auto band = mscop->map(root->child({0}), 0, BX);
    bandScale(band, tileSizes);

    auto ns = detail::ScheduleTree::collectDFSPostorder(
        root, detail::ScheduleTreeType::Band);
    mscop->map(ns[0], 0, TX);
    mscop->map(ns[1], 0, TY);
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
        0);

    // Map to blocks
    USING_MAPPING_SHORT_NAMES(BX, BY, BZ, TX, TY, TZ);
    auto band = mscop->map(root->child({0}), 0, BX);
    band = mscop->map(band, 1, BY);
    bandScale(band, tileSizes);

    band = mscop->map(band->child({0}), 0, TX);
    band = mscop->map(band, 1, TY);
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
              ->elemAs<detail::ScheduleTreeElemBand>()
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

TEST_F(PolyhedralMapperTest, Basic) {
  string tc = R"TC(
def fun(float(N, M) A, float(N, M) B) -> (C) {
  C(i, j) = A(i, j) + B(i, j)
}
)TC";

  auto scop = PrepareAndJoinBands(tc);
  auto scopPtr = scop.get();
  auto context =
      scopPtr->makeContext(std::unordered_map<std::string, int>{{"N", 512}});
  scop = Scop::makeSpecializedScop(
      *scop, context.intersect(scop->globalParameterContext));
  auto mscop = TileAndMapBlocksAndThreads(
      std::move(scop), {16ul, 16ul}, {256ul, 256ul}, {16ul, 16ul});

  auto res = mscop->codegen(specializedName);

  std::string expected(
      R"RES(int b0 = blockIdx.x; int b1 = blockIdx.y; int b2 = blockIdx.z;
  int t0 = threadIdx.x; int t1 = threadIdx.y; int t2 = threadIdx.z;
  float32 (*C)[M] = reinterpret_cast<float32 (*)[M]>(pC);
  float32 (*A)[M] = reinterpret_cast<float32 (*)[M]>(pA);
  float32 (*B)[M] = reinterpret_cast<float32 (*)[M]>(pB);
  for (int c1 = 16 * b1; c1 < M; c1 += 4096) {
    if (M >= t1 + c1 + 1) {
      C[t0 + 16*b0][t1 + c1] = (A[t0 + 16*b0][t1 + c1] + B[t0 + 16*b0][t1 + c1]);
    }
  }
}
)RES");

  ASSERT_NE(std::string::npos, std::get<0>(res).find(expected))
      << std::get<0>(res);
  ASSERT_EQ(32, std::get<1>(res)[0]) << "Improper dim in: " << std::get<1>(res);
}

TEST_F(PolyhedralMapperTest, MultiStmt) {
  string tc = R"TC(
def fun(float(N, N, N, N) A, float(N, N) B, float(N, N) C, float(N, N) D)
-> (O1, O2, O3)
{
  O1(i, j) +=! A(i, j, rk, rl) * B(i, j)
  O2(i, j) = C(i, j) * D(i, j)
  O3(i, j) = O1(i, j) + O2(i, j)
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
  float32 (*A)[N][N][N] = reinterpret_cast<float32 (*)[N][N][N]>(pA);
  float32 (*B)[N] = reinterpret_cast<float32 (*)[N]>(pB);
  float32 (*C)[N] = reinterpret_cast<float32 (*)[N]>(pC);
  float32 (*D)[N] = reinterpret_cast<float32 (*)[N]>(pD);
  for (int c0 = 0; c0 < N; c0 += 1) {
    for (int c1 = 0; c1 < N; c1 += 1) {
      O1[c0][c1] = 0.000000f;
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
  O(i, j) = A(i, j) + i + j + N
}
)TC";

  auto mscop = makeUnmapped(tc);
  auto res = std::get<0>(mscop->codegen(specializedName));

  string expected(
      R"RES(__global__ void kernel_anon(int32 N, float32* pO, float32* pA) {
  int b0 = blockIdx.x; int b1 = blockIdx.y; int b2 = blockIdx.z;
  int t0 = threadIdx.x; int t1 = threadIdx.y; int t2 = threadIdx.z;
  float32 (*O)[N] = reinterpret_cast<float32 (*)[N]>(pO);
  float32 (*A)[N] = reinterpret_cast<float32 (*)[N]>(pA);
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
  O(i, j) = nextafter(C(i), exp(A(i, j))) + log(B(j, i))
}
)TC";

  auto mscop = makeUnmapped(tc);
  mscop->fixParameters<int>({{"N", 512}});
  auto res = std::get<0>(mscop->codegen(specializedName));

  string expected =
      R"RES(__global__ void kernel_anon(int32 N, float32* pO, float32* pA, float32* pB, float32* pC) {
  int b0 = blockIdx.x; int b1 = blockIdx.y; int b2 = blockIdx.z;
  int t0 = threadIdx.x; int t1 = threadIdx.y; int t2 = threadIdx.z;
  float32 (*O)[512] = reinterpret_cast<float32 (*)[512]>(pO);
  float32 (*A)[512] = reinterpret_cast<float32 (*)[512]>(pA);
  float32 (*B)[512] = reinterpret_cast<float32 (*)[512]>(pB);
  float32 (*C) = reinterpret_cast<float32 (*)>(pC);
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
  float32 (*A)[64] = reinterpret_cast<float32 (*)[64]>(pA);
  float32 (*B)[64] = reinterpret_cast<float32 (*)[64]>(pB);
  for (int c0 = 0; c0 <= 63; c0 += 16) {
    for (int c1 = 0; c1 <= 63; c1 += 16) {
      for (int c2 = t1; c2 <= 15; c2 += 8) {
        for (int c3 = 0; c3 <= 15; c3 += 1) {
          O[c0 + c2][c1 + c3] = 0.000000f;
          for (int c4 = t0; c4 <= 63; c4 += 32) {
            O[c0 + c2][c1 + c3] = (O[c0 + c2][c1 + c3] + (A[c0 + c2][c4]*B[c4][c1 + c3]));
          }
        }
      }
    }
  }
}
)CUDA";

TEST_F(PolyhedralMapperTest, MergedContexts) {
  auto scop = PrepareAndJoinBands(makeMatmulTc());

  // Unit test claims to use scop->globalParameterContext properly
  auto context = scop->makeContext(std::vector<int>{64, 64, 64});
  auto& globalParameterContext =
      const_cast<isl::set&>(scop->globalParameterContext);
  globalParameterContext = globalParameterContext.intersect(context);
  scop->domain() = scop->domain().intersect(globalParameterContext);

  auto mscop = TileAndMapThreads(std::move(scop), {16, 16}, {32ul, 8ul});
  auto res = std::get<0>(mscop->codegen(specializedName));
  ASSERT_TRUE(std::string::npos != res.find(kExpectedMatmul_64_64_64));
}

TEST_F(PolyhedralMapperTest, FilterMerge) {
  auto scop = PrepareAndJoinBands(makeMatmulTc());
  auto schedule = scop->scheduleRoot();

  // Unit test claims to use scop->globalParameterContext properly
  auto context = scop->makeContext(std::vector<int>{64, 64, 64});
  auto& globalParameterContext =
      const_cast<isl::set&>(scop->globalParameterContext);
  globalParameterContext = globalParameterContext.intersect(context);
  scop->domain() = scop->domain().intersect(globalParameterContext);

  auto mscop = TileAndMapThreads(std::move(scop), {16, 16}, {32ul, 8ul});
  mergeConsecutiveMappingFilters(schedule, schedule->child({0}));
  mscop->codegen(specializedName);
}

TEST_F(PolyhedralMapperTest, Match1) {
  auto scop = PrepareAndJoinBands(makeMatmulTc());
  auto schedule = scop->scheduleRoot();

  auto mscop = TileAndMapThreads(std::move(scop), {16, 16}, {32ul, 8ul});
  auto f = match(
      sequence(
          filter([](isl::union_set f) {
            return f.get_space().dim(isl::dim_type::param) == 3;
          }),
          filter(mapping_filter(band()))),
      schedule);
  EXPECT_EQ(1, f.size());
}

TEST_F(PolyhedralMapperTest, CopyTC) {
  string tc = R"TC(
def fun(float(M, N) I) -> (O) {
  O(i, j) = I(i, j)
}
)TC";

  auto scop = PrepareAndJoinBands(tc);
  auto tileOptions = TileOptions::ShiftPointLoops | TileOptions::ScaleTileLoops;
  TileAndCheckStructuralEquality(*scop, tileOptions, {3ul, 5});
}

TEST_F(PolyhedralMapperTest, MatmulTC) {
  string tc = R"TC(
def fun(float(M, K) A, float(K, N) B) -> (C) {
  C(i, j) +=! A(i, k) * B(k, j)
}
)TC";

  auto scop = PrepareAndJoinBands(tc);
  auto tileOptions = TileOptions::ShiftPointLoops | TileOptions::ScaleTileLoops;
  TileAndCheckStructuralEquality(*scop, tileOptions, {3ul, 4ul});
}

TEST_F(PolyhedralMapperTest, MatmulShiftScale) {
  auto scop = PrepareAndJoinBands(makeMatmulTc());
  auto tileOptions = TileOptions::ShiftPointLoops | TileOptions::ScaleTileLoops;
  TileAndCheckStructuralEquality(*scop, tileOptions, {3ul, 4ul});
}

TEST_F(PolyhedralMapperTest, MatmulShift) {
  auto scop = PrepareAndJoinBands(makeMatmulTc());
  auto tileOptions = TileOptions::ShiftPointLoops;
  TileAndCheckStructuralEquality(*scop, tileOptions, {3ul, 4ul});
}

TEST_F(PolyhedralMapperTest, MatmulScale) {
  auto scop = PrepareAndJoinBands(makeMatmulTc());
  auto tileOptions = TileOptions::ScaleTileLoops;
  TileAndCheckStructuralEquality(*scop, tileOptions, {3ul, 4ul});
}

TEST_F(PolyhedralMapperTest, MatmulNoshiftNoscale) {
  auto scop = PrepareAndJoinBands(makeMatmulTc());
  auto tileOptions = TileOptions();
  TileAndCheckStructuralEquality(*scop, tileOptions, {3ul, 4ul});
}

static const string kTcAdd = R"TC(
def fun(float(N, M) A, float(N, M) B) -> (C) {
  C(i, j) = A(i, j) + B(i, j)
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
  auto mappingOptions = DefaultOptions().tile({64, 64}).unroll(15);
  auto scop = PrepareAndJoinBands(kTcAdd);
  scop->fixParameters<int>({{"N", 1024}, {"M", 1024}});
  auto mscop = MappedScop::makeWithOuterBlockInnerThreadStrategy(
      std::move(scop), mappingOptions);
  auto code = std::get<0>(mscop->codegen(specializedName));
  std::string expected("C[64*b0 + c2][t0 + 64*b1]");
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
  auto mappingOptions = DefaultOptions().tile({64, 64}).unroll(16);
  auto scop = PrepareAndJoinBands(kTcAdd);
  scop->fixParameters<int>({{"N", 1024}, {"M", 1024}});
  auto mscop = MappedScop::makeWithOuterBlockInnerThreadStrategy(
      std::move(scop), mappingOptions);
  auto code = std::get<0>(mscop->codegen(specializedName));
  std::string expected("C[32 + t1 + 64*b0][32 + t0 + 64*b1]");
  ASSERT_TRUE(code.find(expected) != std::string::npos);
}

/*
 * Map 1D code to 2D grid (set up by makeNaiveCudaMappingOptions()) and
 * check that the code is pinned to one particular value of
 * block identifier b1 and thread identifier t1.
 */
TEST_F(PolyhedralMapperTest, Copy1D) {
  auto tc = R"TC(
def fun(float(N) I) -> (O) {
  O(i) = I(i)
}
)TC";
  auto scop = Prepare(tc);
  auto mscop = MappedScop::makeWithOuterBlockInnerThreadStrategy(
      std::move(scop), DefaultOptions());
  auto codeAndLaunchBounds = mscop->codegen(specializedName);
  USING_MAPPING_SHORT_NAMES(BX, BY, BZ, TX, TY, TZ);
  EXPECT_EQ(1, BY.mappingSize(std::get<1>(codeAndLaunchBounds)));
  EXPECT_EQ(1, TY.mappingSize(std::get<1>(codeAndLaunchBounds)));
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
 * Also check that synchronization is introduced between the two children.
 */
TEST_F(PolyhedralMapperTest, Copy2) {
  auto tc = R"TC(
def fun(float(N) I) -> (O1, O2) {
  O1(i) = I(i)
  O2(i) = O1(i)
}
)TC";
  auto mappingOptions = DefaultOptions();
  mappingOptions.scheduleFusionStrategy(FusionStrategy::Min);
  auto code = codegenMapped(tc, mappingOptions);
  auto sync = "__syncthreads()";
  auto loop = "for (int c0 = t0; c0 < N; c0 += 32)";
  auto pos1 = code.find(loop);
  auto pos2 = code.find(sync, pos1 + 1);
  auto pos3 = code.find(loop, pos2 + 1);
  EXPECT_TRUE(pos1 != std::string::npos);
  EXPECT_TRUE(pos2 != std::string::npos);
  EXPECT_TRUE(pos3 != std::string::npos);
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
  O1(i) = I1(i)
  O2(i, j) = I2(i, j)
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
  C(i, j) = A(i, j)
  D(i, j) = B(i, j)
})TC";

  auto originalScop = Prepare(tc);

  auto tiling = Tiling({32, 32});

  auto minFusionSchedulerOptions = SchedulerOptions();
  minFusionSchedulerOptions.proto.set_fusion_strategy(FusionStrategy::Min);

  auto maxFusionSchedulerOptions = SchedulerOptions();
  maxFusionSchedulerOptions.proto.set_fusion_strategy(FusionStrategy::Max);

  // Schedule with maximal fusion, then tile and reschedule point loops for
  // minimal fusion.
  auto scop = Scop::makeScheduled(*originalScop, maxFusionSchedulerOptions);
  auto outerBand = scop->tileOuterBand(tiling);
  scop->reschedule(outerBand->child({0}), minFusionSchedulerOptions);
  auto maxMinOuterBand = ScheduleTree::makeScheduleTree(*outerBand);

  // Schedule with maximal fusion, then tile and reschedule point loops for
  // minimal fusion again.  Schedule is expected to be identical to
  // the one without rescheduling.
  scop = Scop::makeScheduled(*originalScop, maxFusionSchedulerOptions);
  outerBand = scop->tileOuterBand(tiling);
  scop->reschedule(outerBand->child({0}), maxFusionSchedulerOptions);
  auto maxMaxOuterBand = ScheduleTree::makeScheduleTree(*outerBand);

  // Schedule with maximal fusion and tile.
  scop = Scop::makeScheduled(*originalScop, maxFusionSchedulerOptions);
  auto tiledBand = ScheduleTree::makeScheduleTree(*scop->tileOuterBand(tiling));

  using detail::ScheduleTreeElemBand;
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
  ASSERT_EQ(maxMinMatches.size(), 1);
  ASSERT_EQ(maxMaxMatches.size(), 1);
  ASSERT_EQ(tiledMatches.size(), 1);

  // The wrong structure should not be matched.
  EXPECT_EQ(match(maxMaxStructure, maxMinOuterBand.get()).size(), 0);
  EXPECT_EQ(match(maxMinStructure, maxMaxOuterBand.get()).size(), 0);
  EXPECT_EQ(match(maxMinStructure, tiledBand.get()).size(), 0);

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
  D(i, j) +=! A(i, k) * B(k, j)
  E(i, j) +=! A(i, k) * C(k, j)
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
  auto pos6 = code.find(innerLoopIncrement, pos5 + 1);
  EXPECT_TRUE(pos1 != std::string::npos);
  EXPECT_TRUE(pos2 != std::string::npos);
  EXPECT_TRUE(pos3 != std::string::npos);
  EXPECT_TRUE(pos4 != std::string::npos);
  EXPECT_TRUE(pos5 != std::string::npos);
  EXPECT_TRUE(pos6 != std::string::npos);
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
  O +=! I(i)
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

static const string kTcMM = R"TC(
def fun(float(M, K) A, float(K, N) B) -> (C) {
  C(i, j) +=! A(i, k) * B(k, j)
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
  EXPECT_TRUE(code.find("C[c0 + c3][t0 + c1] = (C") != std::string::npos);
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
  EXPECT_TRUE(code.find("C[t1 + c0][t0 + c1] = (C") != std::string::npos);
}

int main(int argc, char** argv) {
  ::testing::InitGoogleTest(&argc, argv);
  ::gflags::ParseCommandLineFlags(&argc, &argv, true);
  ::google::InitGoogleLogging(argv[0]);
  return RUN_ALL_TESTS();
}
