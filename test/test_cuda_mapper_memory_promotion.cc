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
#include <gflags/gflags.h>
#include <glog/logging.h>
#include <gtest/gtest.h>

#include "tc/core/check.h"
#include "tc/core/polyhedral/cuda/codegen.h"
#include "tc/core/polyhedral/cuda/mapped_scop.h"
#include "tc/core/polyhedral/cuda/memory_promotion_heuristic.h"
#include "tc/core/polyhedral/exceptions.h"
#include "tc/core/polyhedral/memory_promotion.h"
#include "tc/core/polyhedral/scop.h"

using namespace std;

using namespace tc;
using namespace tc::polyhedral;

namespace {
int npoints(isl::set s) {
  int cnt = 0;
  isl::union_set(s).foreach_point([&cnt](isl::point pt) { ++cnt; });
  return cnt;
}
} // namespace

class TestMapper : public ::testing::Test {
 protected:
  std::unique_ptr<MappedScop> makeMappedScop(
      std::string tc,
      const CudaMappingOptions& mappingOptions,
      std::unordered_map<std::string, size_t> problemSizes) {
    auto ctx = isl::with_exceptions::globalIslCtx();
    auto scop = Scop::makeScop(ctx, tc);
    scop = Scop::makeSpecializedScop(*scop, problemSizes);
    scop->specializeToContext();
    return MappedScop::makeWithOuterBlockInnerThreadStrategy(
        std::move(scop), mappingOptions);
  }

  // This mimics behavior of the old TensorReferenceGroup::accessedBySubtree.
  // In particular, it takes the partial schedule until "tree" inclusive,
  // intersecting its domain with all (mapping) filter ancestors and computes
  // accessed tensor reference groups within that schedule.
  // If the schedule happens to contain the block/thread mapping filter, the
  // groups are per-block/thread.  Otherwise, they include all blocks/threads,
  // which is questionable, but corresponds to what the test currently checks.
  TensorGroups accessedBySubtree(
      const polyhedral::detail::ScheduleTree* tree,
      const Scop& scop) {
    auto schedule = partialSchedule(scop.scheduleRoot(), tree);
    return TensorReferenceGroup::accessedWithin(
        schedule, scop.reads, scop.writes);
  }
};

class MapperMemoryPromotion2DHelper : public TestMapper {
 protected:
  std::unique_ptr<MappedScop> prepareScop(
      std::string tc,
      std::unordered_map<std::string, size_t> problemSizes,
      std::vector<size_t> tileSizes) {
    auto mappingOptions =
        CudaMappingOptions::makeNaiveMappingOptions()
            .tile(tileSizes) // passing more tiling values triggers a check...
            .useSharedMemory(false) // do not auto-promote
            .usePrivateMemory(false);
    return makeMappedScop(tc, mappingOptions, problemSizes);
  }
};

class Sum4D : public TestMapper {
 public:
  std::string emitCode(
      std::vector<size_t> problemSizes,
      std::vector<size_t> tileSizes,
      std::vector<size_t> childPos) {
    string tc = R"TC(
def fun(float(N,M,K,L) A, float(N,M,K,L) B) -> (C) {
    C(n,m,k,l) = A(n,m,k,l) + B(n,m,k,l)
}
)TC";

    auto mappingOptions = CudaMappingOptions::makeNaiveMappingOptions()
                              .tile(tileSizes)
                              .useSharedMemory(false) // do not autopromote
                              .usePrivateMemory(false);
    auto mscop = makeMappedScop(
        tc,
        mappingOptions,
        std::unordered_map<string, size_t>{{"N", problemSizes[0]},
                                           {"M", problemSizes[1]},
                                           {"K", problemSizes[2]},
                                           {"L", problemSizes[3]}});
    auto& scop = const_cast<Scop&>(mscop->scop());
    scop.promoteEverythingAt(childPos);
    return std::get<0>(mscop->codegen("fun"));
  }
};

TEST_F(Sum4D, CodeOuterBand) {
  auto declarations = {"__shared__ float32 _A_0[16][16][16][16];",
                       "__shared__ float32 _B_0[16][16][16][16];",
                       "__shared__ float32 _C_0[16][16][16][16];"};

  auto copyA =
      "_A_0[c4][c5][c6][c7] = A[16 * b0 + c4][16 * b1 + c5][c2 + c6][c3 + c7];";
  auto copyB =
      "_B_0[c4][c5][c6][c7] = B[16 * b0 + c4][16 * b1 + c5][c2 + c6][c3 + c7];";
  auto compute =
      "_C_0[c4][c5][c6][t0] = (_A_0[c4][c5][c6][t0] + _B_0[c4][c5][c6][t0]);";
  auto copyC =
      "C[16 * b0 + c4][16 * b1 + c5][c2 + c6][c3 + c7] = _C_0[c4][c5][c6][c7];";
  auto sync = "__syncthreads()";

  auto code = emitCode({256, 128, 192, 224}, {16, 16, 16, 16}, {0, 0, 0});
  // Order of copies may be arbitrary, but syncs must be inserted before and
  // after
  for (auto d : declarations) {
    EXPECT_TRUE(code.find(d) != std::string::npos);
  }
  auto posA = code.find(copyA);
  auto posB = code.find(copyB);
  auto posCompute = code.find(compute);
  auto posC = code.find(copyC);
  auto posSync1 = code.find(sync);
  auto posSync2 = code.rfind(sync, posCompute);
  auto posSync3 = code.find(sync, posCompute);
  auto posSync4 = code.rfind(sync);
  EXPECT_NE(posA, std::string::npos);
  EXPECT_NE(posB, std::string::npos);
  EXPECT_NE(posC, std::string::npos);
  EXPECT_NE(posCompute, std::string::npos);
  EXPECT_NE(posSync1, std::string::npos);
  EXPECT_NE(posSync2, std::string::npos);
  EXPECT_NE(posSync3, std::string::npos);
  EXPECT_NE(posSync4, std::string::npos);
  EXPECT_LT(posSync1, std::min(posA, posB));
  EXPECT_GT(posSync2, std::max(posA, posB));
  EXPECT_LT(posSync3, posC);
  EXPECT_GT(posSync4, posC);
}

/*
 * Check code when promotion is performed above the mapping to threads.
 * Note that the copying code is not mapped to threads because
 * promoteEverythingAt does not call mapCopiesToThreads.
 */
TEST_F(Sum4D, CodeAboveThreadMapping) {
  auto declarations = {"__shared__ float32 _A_0[16][16][16][16];",
                       "__shared__ float32 _B_0[16][16][16][16];",
                       "__shared__ float32 _C_0[16][16][16][16];"};
  auto copyA =
      "_A_0[c4][c5][c6][c7] = A[16 * b0 + c4][16 * b1 + c5][c2 + c6][c3 + c7]";
  auto copyB =
      "_B_0[c4][c5][c6][c7] = B[16 * b0 + c4][16 * b1 + c5][c2 + c6][c3 + c7]";
  auto compute =
      "_C_0[c4][c5][c6][t0] = (_A_0[c4][c5][c6][t0] + _B_0[c4][c5][c6][t0]);";
  auto copyC =
      "C[16 * b0 + c4][16 * b1 + c5][c2 + c6][c3 + c7] = _C_0[c4][c5][c6][c7];";
  auto sync = "__syncthreads()";

  auto code = emitCode({256, 128, 192, 224}, {16, 16, 16, 16}, {0, 0, 0});

  // Order of copies may be arbitrary, but syncs must be inserted before and
  // after
  for (auto d : declarations) {
    EXPECT_TRUE(code.find(d) != std::string::npos);
  }
  auto posA = code.find(copyA);
  auto posB = code.find(copyB);
  auto posCompute = code.find(compute);
  auto posC = code.find(copyC);
  auto posSync1 = code.find(sync);
  auto posSync2 = code.rfind(sync, posCompute);
  auto posSync3 = code.find(sync, posCompute);
  auto posSync4 = code.rfind(sync);
  EXPECT_NE(posA, std::string::npos);
  EXPECT_NE(posB, std::string::npos);
  EXPECT_NE(posC, std::string::npos);
  EXPECT_NE(posCompute, std::string::npos);
  EXPECT_NE(posSync1, std::string::npos);
  EXPECT_NE(posSync2, std::string::npos);
  EXPECT_NE(posSync3, std::string::npos);
  EXPECT_NE(posSync4, std::string::npos);
  EXPECT_LT(posSync1, std::min(posA, posB));
  EXPECT_GT(posSync2, std::max(posA, posB));
  EXPECT_LT(posSync3, posC);
  EXPECT_GT(posSync4, posC);
}

TEST_F(Sum4D, CodeInnerBand) {
  auto declarations = {"__shared__ float32 _C_0[1][1][1][1];",
                       "__shared__ float32 _A_0[1][1][1][1];",
                       "__shared__ float32 _B_0[1][1][1][1];"};
  auto copyA =
      "_A_0[0][0][0][0] = A[16 * b0 + c4][16 * b1 + c5][c2 + c6][t0 + c3];";
  auto copyB =
      "_B_0[0][0][0][0] = B[16 * b0 + c4][16 * b1 + c5][c2 + c6][t0 + c3];";
  auto compute = "_C_0[0][0][0][0] = (_A_0[0][0][0][0] + _B_0[0][0][0][0]);";
  auto copyC =
      "C[16 * b0 + c4][16 * b1 + c5][c2 + c6][t0 + c3] = _C_0[0][0][0][0];";
  auto sync = "__syncthreads()";

  auto code =
      emitCode({256, 128, 192, 224}, {16, 16, 16, 16}, {0, 0, 0, 0, 0, 0});
  // Order of copies may be arbitrary, but syncs must be inserted before and
  // after
  for (auto d : declarations) {
    EXPECT_NE(std::string::npos, code.find(d));
  }
  auto posA = code.find(copyA);
  auto posB = code.find(copyB);
  auto posCompute = code.find(compute);
  auto posC = code.find(copyC);
  auto posSync1 = code.find(sync);
  auto posSync2 = code.rfind(sync, posCompute);
  auto posSync3 = code.find(sync, posCompute);
  auto posSync4 = code.rfind(sync);
  EXPECT_NE(posA, std::string::npos);
  EXPECT_NE(posB, std::string::npos);
  EXPECT_NE(posC, std::string::npos);
  EXPECT_NE(posCompute, std::string::npos);
  EXPECT_NE(posSync1, std::string::npos);
  EXPECT_NE(posSync2, std::string::npos);
  EXPECT_NE(posSync3, std::string::npos);
  EXPECT_NE(posSync4, std::string::npos);
  EXPECT_LT(posSync1, std::min(posA, posB));
  EXPECT_GT(posSync2, std::max(posA, posB));
  EXPECT_LT(posSync3, posC);
  EXPECT_GT(posSync4, posC);
}

class MapperMemoryPromotionSum2D : public MapperMemoryPromotion2DHelper {
 public:
  const string tc = R"TC(
def fun(float(N, M) A, float(N, M) B) -> (C) {
    C(n, m) = A(n, m) + B(n, m)
}
)TC";

  void checkPromoteAll(
      size_t problemSize1,
      size_t problemSize2,
      size_t tile1,
      size_t tile2,
      std::vector<size_t> childPos) {
    auto mscop = prepareScop(
        tc,
        std::unordered_map<std::string, size_t>{{"N", problemSize1},
                                                {"M", problemSize2}},
        {tile1, tile2});
    auto& scop = const_cast<Scop&>(mscop->scop());
    // Must force domain intersection for overapproximation to work
    scop.specializeToContext();
    auto ctx = scop.domain().get_ctx();
    auto groups = accessedBySubtree(scop.scheduleRoot()->child(childPos), scop);
    LOG(INFO) << "Groups:\n" << groups;

    EXPECT_EQ(groups.size(), 3u);

    isl::space tileSpace = isl::space(ctx, 0).unnamed_set_from_params(2);
    // Work around missing isl_set_from_multi_aff.
    auto tileZero =
        isl::map(isl::multi_aff::zero(tileSpace.from_range())).range();

    // Must have groups for these tensors, in arbitrary order.
    unordered_set<string> names{"A", "B", "C"};
    for (const auto& g : groups) {
      auto name = g.first.get_name();
      EXPECT_EQ(names.count(name), 1u);
      names.erase(name);
      const auto& tensorGroups = g.second;

      ASSERT_EQ(tensorGroups.size(), 1u) << "Expected one group";
      auto oneGroup = tensorGroups[0].get();

      ASSERT_EQ(oneGroup->references.size(), 1u)
          << "Expected one reference in group";
      if (name != "C") {
        EXPECT_TRUE(oneGroup->isReadOnly());
      }

      auto ref = oneGroup->references[0].get();
      ASSERT_EQ(oneGroup->approximation.dim(), 2u)
          << "Could not compute footprint for" << ref->scopedAccess;

      EXPECT_EQ(
          oneGroup->approximation.size(0),
          isl::val(ctx, std::min(tile1, problemSize1)));
      EXPECT_EQ(
          oneGroup->approximation.size(1),
          isl::val(ctx, std::min(tile2, problemSize2)));
      auto footprint = tileZero.apply(oneGroup->approximateFootprint());
      size_t np = npoints(footprint);
      EXPECT_EQ(
          np, std::min(tile1, problemSize1) * std::min(tile2, problemSize2));

      auto schedule = partialSchedule(
          scop.scheduleRoot(), scop.scheduleRoot()->child(childPos));
      auto scopedAccess = oneGroup->originalAccesses().apply_domain(schedule);
      TC_CHECK(scopedAccess.is_equal(oneGroup->scopedAccesses()))
          << "expected original accesses " << oneGroup->originalAccesses()
          << " to be equal to scoped accesses " << oneGroup->scopedAccesses()
          << " after applying the partial schedule " << schedule;
    }
    scop.promoteEverythingAt(childPos);
  }
};

class MapperMemoryPromotionRAW : public MapperMemoryPromotion2DHelper {
 public:
  const string tc = R"TC(
def fun(float(N, M) A) -> (B, C) {
    B(n, m) = A(n, m)
    C(m, n) = B(n, m)
}
)TC";

  void checkPromoteAll(
      size_t problemSize1,
      size_t problemSize2,
      size_t tile1,
      size_t tile2,
      std::vector<size_t> childPos) {
    auto mscop = prepareScop(
        tc,
        std::unordered_map<string, size_t>{{"N", problemSize1},
                                           {"M", problemSize2}},
        {tile1, tile2});
    auto& scop = const_cast<Scop&>(mscop->scop());
    // Must force domain intersection for overapproximation to work
    scop.specializeToContext();
    auto ctx = scop.domain().get_ctx();
    auto groups = accessedBySubtree(scop.scheduleRoot()->child(childPos), scop);
    LOG(INFO) << "Groups:\n" << groups;

    ASSERT_EQ(groups.size(), 3u);
    auto idA = isl::id(ctx, std::string("A"));
    auto idB = isl::id(ctx, std::string("B"));
    auto idC = isl::id(ctx, std::string("C"));
    EXPECT_EQ(groups.count(idA), 1u) << "could not group A references";
    EXPECT_EQ(groups.count(idB), 1u) << "could not group B references";
    EXPECT_EQ(groups.count(idC), 1u) << "could not group C references";

    const auto& groupsB = groups.at(idB);
    ASSERT_EQ(groupsB.size(), 1u) << "expected B refereces to be grouped";
    ASSERT_EQ(groupsB[0]->approximation.dim(), 2u) << "B should be a 2D array";
    EXPECT_EQ(
        groupsB[0]->approximation.size(0),
        isl::val(ctx, std::min(tile1, problemSize1)));
    EXPECT_EQ(
        groupsB[0]->approximation.size(1),
        isl::val(ctx, std::min(tile2, problemSize2)));

    auto t = scop.scheduleRoot()->child(childPos);
    LOG(INFO) << "Create copy for group:\n" << groupsB << "\t@" << *t;

    EXPECT_EQ(1u, groupsB.size());
    LOG(INFO) << "Read: " << groupsB[0]->readFootprint();
    LOG(INFO) << "Write: " << groupsB[0]->writeFootprint();

    auto active = activeDomainPoints(scop.scheduleRoot(), t);
    LOG(INFO) << "Active: " << active;

    auto schedule = partialSchedule(scop.scheduleRoot(), t);
    auto scopedAccess = groupsB[0]->originalAccesses().apply_domain(schedule);
    TC_CHECK(scopedAccess.is_equal(groupsB[0]->scopedAccesses()))
        << "expected original accesses " << groupsB[0]->originalAccesses()
        << " to be equal to scoped accesses " << groupsB[0]->scopedAccesses()
        << " after applying the partial schedule " << schedule;
  }

  std::unique_ptr<MappedScop> makeWithSharedGreedy(
      size_t problemSize1,
      size_t problemSize2,
      size_t tileSize1,
      size_t tileSize2,
      size_t depth,
      size_t maxSharedMemory) {
    auto mscop = prepareScop(
        tc, {{"N", problemSize1}, {"M", problemSize2}}, {tileSize1, tileSize2});
    promoteGreedilyAtDepth(*mscop, depth, maxSharedMemory, false);
    return mscop;
  }
};

TEST_F(MapperMemoryPromotionSum2D, array42x256_tile32x32_firstBand) {
  checkPromoteAll(42, 256, 32, 32, {0, 0, 0});
}

TEST_F(MapperMemoryPromotionSum2D, array42x256_tile32x64_firstBand) {
  checkPromoteAll(42, 256, 32, 64, {0, 0, 0});
}

TEST_F(MapperMemoryPromotionSum2D, array42x40_tile64x64_firstBand) {
  checkPromoteAll(42, 40, 64, 64, {0, 0, 0});
}

TEST_F(MapperMemoryPromotionRAW, array42x256_tile32x32_firstBand) {
  checkPromoteAll(42, 256, 32, 32, {0, 0, 0});
}

TEST_F(MapperMemoryPromotionRAW, array42x40_tile64x64_firstBand) {
  checkPromoteAll(42, 40, 64, 64, {0, 0, 0});
}

TEST_F(MapperMemoryPromotionRAW, fitAtOuterDepths) {
  auto mscop1 = makeWithSharedGreedy(42, 40, 64, 64, 1, 8192);
  EXPECT_EQ(mscop1->scop().promotedDecls().size(), 1u)
      << "expected one reference group to be promoted";

  auto mscop2 = makeWithSharedGreedy(42, 40, 64, 64, 2, 8192);
  EXPECT_EQ(mscop2->scop().promotedDecls().size(), 1u)
      << "expected one reference group to be promoted";

  // Note that due to bank conflict heuristic, we will allocate 32x33 arrays in
  // shared memory which require 32x33x2x4=8448 bytes.
  auto mscop3 = makeWithSharedGreedy(42, 40, 32, 32, 2, 8448);
  EXPECT_EQ(mscop3->scop().promotedDecls().size(), 2u)
      << "expected two reference groups to fit";

  auto mscop4 = makeWithSharedGreedy(42, 40, 32, 32, 2, 8447);
  EXPECT_EQ(mscop4->scop().promotedDecls().size(), 1u)
      << "expected one reference group to be promoted";
}

TEST_F(MapperMemoryPromotionRAW, throwIfCopiesBelowThreads) {
  EXPECT_THROW(
      makeWithSharedGreedy(42, 40, 64, 64, 3, 8192),
      promotion::PromotionBelowThreadsException);

  EXPECT_THROW(
      makeWithSharedGreedy(42, 40, 64, 64, 4, 8192),
      promotion::PromotionBelowThreadsException);
}

class MatMulBias : public TestMapper {
 protected:
  std::unique_ptr<MappedScop> prepare(
      const std::unordered_map<std::string, size_t>& parameters,
      const CudaMappingOptions& mappingOptions) {
    std::string tc = R"TC(
def fun(float(N,K) A, float(K,M) B, float(N,M) C) -> (O) {
  O(i,j) +=! A(i,k) * B(k,j)
  O(i,j) = O(i,j) + C(i,j)
}
)TC";

    return makeMappedScop(tc, mappingOptions, parameters);
  }

  std::string emitCode(const std::unique_ptr<MappedScop>& mscop) {
    return std::get<0>(mscop->codegen("fun"));
  }

  std::string emitCode(
      const std::unordered_map<std::string, size_t>& parameters,
      const CudaMappingOptions& mappingOptions) {
    return emitCode(prepare(parameters, mappingOptions));
  }

  void expectNoABCPromotion(const std::string& code) {
    auto aDeclPos = code.find("  float32 _A_0");
    auto bDeclPos = code.find("  float32 _B_0");
    auto cDeclPos = code.find("  float32 _C_0");
    EXPECT_TRUE(aDeclPos == std::string::npos)
        << "tensor A promoted to register but has elements accessed "
        << "by multiple threads";
    EXPECT_TRUE(bDeclPos == std::string::npos)
        << "tensor B promoted to register but has elements accessed "
        << "by multiple threads";
    EXPECT_TRUE(cDeclPos == std::string::npos)
        << "tensor C promoted to register but has no reuse";
  }

  void expectNoSymbolicSubscript(const std::string& code) {
    // We don't know the exact name of the iterator, but it starts with c.
    auto oWithIteratorPos = code.find("_O_0[c");
    auto oWithThreadPos = code.find("_O_0[t1");

    EXPECT_TRUE(oWithIteratorPos == std::string::npos)
        << "accessing local arrays with iterators in subscripts makes "
        << "these arrays placed in local memory instead of registers";
    EXPECT_TRUE(oWithThreadPos == std::string::npos)
        << "expected per-thread groups to be computed, i.e. thread "
        << "identifiers should not appear in the subscripts";
  }
};

TEST_F(MatMulBias, RegisterPromotion) {
  auto mappingOptions = CudaMappingOptions::makeNaiveMappingOptions()
                            .tile(32, 32, 32)
                            .useSharedMemory(false)
                            .usePrivateMemory(true);

  auto code = emitCode({{"N", 42}, {"M", 56}, {"K", 37}}, mappingOptions);
  auto declPos = code.find("float32 _O_0");
  auto copyToPos =
      code.find("_O_0[0][0] = O[32 * b0 + c3][t0 + 32 * b1]", declPos + 1);
  auto copyFromPos =
      code.find("O[32 * b0 + c3][t0 + 32 * b1] = _O_0[0][0]", copyToPos + 1);

  auto originalAccPos =
      code.find("O[32 * b0 + c3][t0 + 32 * b1]", copyToPos + 1);

  EXPECT_TRUE(declPos != std::string::npos) << "no declaration of the register";
  EXPECT_TRUE(copyToPos != std::string::npos) << "expected copy to register";
  EXPECT_TRUE(copyFromPos != std::string::npos)
      << "expected copy from register";

  EXPECT_NE(originalAccPos, copyFromPos)
      << "global array reference is used in main computation";

  expectNoABCPromotion(code);
}

TEST_F(MatMulBias, RegisterPromotionSharedPreference) {
  auto mappingOptions = CudaMappingOptions::makeNaiveMappingOptions()
                            .tile(32, 32, 32)
                            .maxSharedMemory(32768)
                            .useSharedMemory(true)
                            .usePrivateMemory(true);

  auto code = emitCode({{"N", 42}, {"M", 56}, {"K", 37}}, mappingOptions);

  auto declPos = code.find("float32 _O_0[1][1]");
  EXPECT_TRUE(declPos == std::string::npos)
      << "not expected promotion to register because promoted to shared";

  expectNoABCPromotion(code);
}

TEST_F(MatMulBias, RegistersAtRoot) {
  // Disable automatic promotion to registers because we are going to call it
  // manually.  Require sufficient unrolling to actually hit registers.
  auto mappingOptions = CudaMappingOptions::makeNaiveMappingOptions()
                            .unroll(512)
                            .useSharedMemory(false)
                            .usePrivateMemory(false);

  auto mscop = prepare({{"N", 42}, {"M", 56}, {"K", 37}}, mappingOptions);
  promoteToRegistersBelow(*mscop, mscop->scop().scheduleRoot());
  auto code = emitCode(mscop);

  // Expecting 4 elements because we map the loop i in O[i][j] to 8 threads
  // after tiling by 32.
  auto oDeclPos = code.find("float32 _O_0[4][1];");
  EXPECT_TRUE(oDeclPos != std::string::npos)
      << "expected O to be promoted to registers";

  expectNoABCPromotion(code);
  expectNoSymbolicSubscript(code);

  auto o00Pos = code.find("_O_0[0][0]");
  auto o10Pos = code.find("_O_0[1][0]");
  auto o20Pos = code.find("_O_0[2][0]");
  auto o30Pos = code.find("_O_0[3][0]");

  EXPECT_TRUE(o00Pos != std::string::npos)
      << "expected constant subscripts in _O_0";
  EXPECT_TRUE(o10Pos != std::string::npos)
      << "expected constant subscripts in _O_0";
  EXPECT_TRUE(o20Pos != std::string::npos)
      << "expected constant subscripts in _O_0";
  EXPECT_TRUE(o30Pos != std::string::npos)
      << "expected constant subscripts in _O_0";
}

TEST_F(MatMulBias, RegistersAtRootNotEnoughUnroll) {
  // Disable automatic promotion to registers because we are going to call it
  // manually.  Require no unrolling so as to make promotion to registers
  // invalid.
  auto mappingOptions = CudaMappingOptions::makeNaiveMappingOptions()
                            .unroll(1)
                            .useSharedMemory(false)
                            .usePrivateMemory(false);

  auto mscop = prepare({{"N", 42}, {"M", 56}, {"K", 37}}, mappingOptions);
  promoteToRegistersBelow(*mscop, mscop->scop().scheduleRoot());
  auto code = emitCode(mscop);
  auto oDeclPos = code.find("float32 _O_0;");

  EXPECT_TRUE(oDeclPos == std::string::npos)
      << "not expected O to be promoted to registers";

  expectNoABCPromotion(code);
  expectNoSymbolicSubscript(code);
}

TEST_F(MatMulBias, RegistersBelowFirstBand) {
  using namespace polyhedral::detail;

  // Disable automatic promotion to registers because we are going to call it
  // manually.
  auto mappingOptions = CudaMappingOptions::makeNaiveMappingOptions()
                            .useSharedMemory(false)
                            .usePrivateMemory(false);
  auto mscop = prepare({{"N", 42}, {"M", 56}, {"K", 37}}, mappingOptions);

  auto nodes = ScheduleTree::collectDFSPostorder(
      mscop->scop().scheduleRoot(), ScheduleTreeType::Band);
  ASSERT_GT(nodes.size(), 0u);
  auto node = nodes[0];
  promoteToRegistersBelow(*mscop, node);
  auto code = emitCode(mscop);

  auto oDeclPos = code.find("float32 _O_0[1][1];");
  EXPECT_TRUE(oDeclPos != std::string::npos)
      << "expected O to be promoted to registers";
  expectNoABCPromotion(code);
  expectNoSymbolicSubscript(code);
}

class Strided : public TestMapper {
 public:
  std::unique_ptr<MappedScop> makeScopAndCheck(
      const std::string& tc,
      const std::unordered_map<std::string, size_t>& sizes,
      long groupISize,
      long groupIStride,
      long groupIConstOffset) {
    auto options = CudaMappingOptions::makeNaiveMappingOptions()
                       .tile(32, 32)
                       .mapToThreads(21, 21)
                       .useSharedMemory(false);
    auto mscop = makeMappedScop(tc, options, sizes);
    auto& scop = mscop->scop();
    auto ctx = scop.domain().get_ctx();

    auto groups =
        accessedBySubtree(scop.scheduleRoot()->child({0, 0, 0}), scop);
    EXPECT_EQ(groups.size(), 2u) << "expected groups for both tensors";

    for (const auto& g : groups) {
      auto name = g.first.get_name();
      if (name != "I") {
        continue;
      }

      const auto& perTensorGroups = g.second;
      // One cannot use ASSERT_EQ in a function that returns something, because
      // it would trigger an immediate return without value.  Use EXPECT_EQ and
      // return nullptr manually.
      EXPECT_EQ(perTensorGroups.size(), 1u) << "expected one group for I";
      if (perTensorGroups.size() != 1u) {
        return nullptr;
      }
      const auto& oneGroup = perTensorGroups[0];

      EXPECT_EQ(oneGroup->references.size(), 1u)
          << "expected one reference in the group for I";
      if (oneGroup->references.size() != 1u) {
        return nullptr;
      }
      const auto& ref = oneGroup->references[0];

      EXPECT_EQ(oneGroup->approximation.dim(), 2u)
          << "could not compute approximation for " << ref->scopedAccess;

      EXPECT_EQ(oneGroup->approximation.size(1), isl::val(ctx, groupISize))
          << "expected strides to be removed";

      isl::val stride = isl::val(ctx, groupIStride);
      EXPECT_EQ(oneGroup->approximation.stride(1), stride);

      auto expectedOffset =
          isl::aff::zero_on_domain(ref->scopedAccess.domain().get_space()) +
          groupIConstOffset;
      // Convert to pw_aff because it has is_equal whereas a simple aff only has
      // is_plain_equal that fails here.
      EXPECT_TRUE(
          isl::pw_aff(oneGroup->approximation.strideOffset(1).mod(stride))
              .is_equal(isl::pw_aff(expectedOffset.mod(stride))))
          << oneGroup->approximation.strideOffset(1) << "\n"
          << expectedOffset;
    }
    return mscop;
  }
};

// Check that strides are effectively handled in memory promotion.  In
// particular, check that array elements that are jumped over
// by the main computation are not copied into shared memory.
TEST_F(Strided, Stride2) {
  std::string tc = R"TC(
def strided(float(N,M) I) -> (O) {
  O(i, j) = I(j, 2 * i + 1)
}
)TC";

  // Expect the promoted size to be 32x32, with stride 2 and offset -1 along
  // the second dimension.
  auto mscop = makeScopAndCheck(tc, {{"N", 42}, {"M", 420}}, 32, 2, -1);
  ASSERT_TRUE(mscop.get() != nullptr);
  auto& scop = mscop->scop();

  // Additionally check that copies look fine.
  scop.promoteEverythingAt({0, 0, 0});
  auto code = std::get<0>(mscop->codegen("strided"));
  EXPECT_TRUE(
      code.find("_I_0[c2][c3] = I[32 * b1 + c2][64 * b0 + 2 * c3 + 1]") !=
      std::string::npos)
      << "expected strided accesses to global array in copies";
  EXPECT_TRUE(code.find("= _I_0[c3][c2]") != std::string::npos)
      << "expected non-strided access to promoted array in main computation";
  EXPECT_TRUE(code.find("= _I_0[c3][2 * c2") == std::string::npos)
      << "did not expect strided access to promoted array in main computation";
}

TEST_F(Strided, Stride5) {
  std::string tc = R"TC(
def strided(float(N,M) I) -> (O) {
  O(i, j) = I(j, 5 * i)
}
)TC";

  // Expect the promoted size to be 32x32, with stride 5 and offset 0 along
  // the second dimension.
  makeScopAndCheck(tc, {{"N", 42}, {"M", 420}}, 32, 5, 0);
}

int main(int argc, char** argv) {
  ::testing::InitGoogleTest(&argc, argv);
  ::gflags::ParseCommandLineFlags(&argc, &argv, true);
  ::google::InitGoogleLogging(argv[0]);
  return RUN_ALL_TESTS();
}
