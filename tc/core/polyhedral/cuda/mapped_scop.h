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

#include <map>
#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

#include "tc/core/cuda/cuda_mapping_options.h"
#include "tc/core/polyhedral/cuda/mapping_types.h"
#include "tc/core/polyhedral/cuda/memory_promotion_heuristic.h"
#include "tc/core/polyhedral/domain_types.h"
#include "tc/core/polyhedral/scop.h"
#include "tc/core/tensor.h"
#include "tc/external/isl.h"

namespace tc {
namespace polyhedral {
namespace detail {
class ScheduleTree;
} // namespace detail

namespace cuda {

// Scop associated with fixed block and grid dimensions.
//
// Different branches of the schedule tree may be mapped to GPU blocks or
// threads.  The role of this class is to ensure that the number of required
// blocks and threads is consistent for the entire Scop.  It does so by
// requiring to provide grid and block configuration when constructing its
// instance.  Different parts of the schedule tree may be mapped to blocks and
// threads but the values remain those specified at construction.  If less
// blocks or threads is necessary to execute certain parts of the Scop, the
// blocks or threads dimensions will be further restricted locally in a
// specific branch of schedule tree.
//
// Two invariants must be preserved:
// 1. All paths from schedule tree root to its leaves must have exactly the
//    same number of block and thread mappings.  Code generation will fail if
//    it is not the case (TODO: automatically map to 1 thread and 1 block
//    instead).
// 2. Mapping to each block and thread must appear exactly once on each path
//    from schedule tree root to its leaves.  Mapping will fail if this
//    invariant is violated.
//
// Only const and copy accessors to the members of the original Scop are
// exposed since mapping to blocks and threads introduces schedule tree
// elements incompatible with other Scop modifications.
class MappedScop {
 private:
  MappedScop(
      std::unique_ptr<Scop>&& scop,
      ::tc::Grid grid,
      ::tc::Block block,
      uint64_t unroll_,
      bool useReadOnlyCache_)
      : scop_(std::move(scop)),
        numBlocks(grid),
        numThreads(block),
        unroll(unroll_),
        useReadOnlyCache(useReadOnlyCache_) {}

 public:
  static inline std::unique_ptr<MappedScop> makeOneBlockOneThread(
      std::unique_ptr<Scop>&& scop) {
    auto mscop = std::unique_ptr<MappedScop>(new MappedScop(
        std::move(scop), ::tc::Grid{1, 1, 1}, ::tc::Block{1, 1, 1}, 1, false));
    auto band = mscop->scop_->obtainOuterBand();
    mscop->mapBlocksForward(band, 0);
    mscop->mapThreadsBackward(band);
    return mscop;
  }
  // The MappedScop returned by this method does not satisfy the invariant
  // of having a mapping to blocks and threads.  It is up to the caller
  // to insert these mappings.
  static inline std::unique_ptr<MappedScop> makeMappedScop(
      std::unique_ptr<Scop>&& scop,
      ::tc::Grid grid,
      ::tc::Block block,
      uint64_t unroll,
      bool useReadOnlyCache) {
    return std::unique_ptr<MappedScop>(
        new MappedScop(std::move(scop), grid, block, unroll, useReadOnlyCache));
  }

  // Apply the hand-written OuterBlockInnerThread mapping strategy.
  static std::unique_ptr<MappedScop> makeWithOuterBlockInnerThreadStrategy(
      std::unique_ptr<Scop>&& scopUPtr,
      const CudaMappingOptions& mappingOptions);

  // Map the initial (up to "nToMap") band members of "band"
  // to successive block identifiers.
  // This function can only be called once on the entire tree.
  detail::ScheduleTree* mapBlocksForward(
      detail::ScheduleTree* band,
      size_t nToMap);
  // Map the final band members of "band"
  // to successive thread identifiers, with the last member mapped
  // to thread identifier X.
  // This function can only be called once in any branch of the tree.
  detail::ScheduleTree* mapThreadsBackward(detail::ScheduleTree* band);

  // Fix the values of the specified parameters in the context
  // to the corresponding specified values.
  template <typename T>
  void fixParameters(const std::unordered_map<std::string, T>& sizes) {
    scop_->fixParameters(sizes);
  }

  // Insert a context node for the block and thread identifiers.
  void insertMappingContext();

  // Generate CUDA code at the current state of transformation provided a
  // name for the generated function.
  std::tuple<std::string, tc::Grid, tc::Block> codegen(
      const std::string& specializedName) const;

  // Accessors..
  // Const accessor to schedule of underlying Scop.
  inline const detail::ScheduleTree* schedule() const {
    return scop_->scheduleRoot();
  }
  // Reference to underlying scop, no ownership transfer intended.
  inline const Scop& scop() const {
    return *scop_;
  }
  inline Scop& scop() {
    return *scop_;
  }

 private:
  // Map the elements in "list" to successive blocks or thread identifiers,
  // with the first element mapped to identifier X.
  // Return a pointer to the updated node (below the inserted filter)
  // for call chaining purposes.
  template <typename MappingTypeId>
  detail::ScheduleTree* map(
      detail::ScheduleTree* tree,
      isl::union_pw_aff_list list);
  // Map "band" to block identifiers and then scale
  // the band members by "tileSizes".
  void mapToBlocksAndScaleBand(
      detail::ScheduleTree* band,
      std::vector<size_t> tileSizes);
  // Look for innermost reduction bands.
  // Store them in reductionBandUpdates_.
  // Return true if any were found.
  // A band is considered to be a reduction band if it only involves
  // instances of a single reduction update statement (modulo other
  // statements that can be moved out of the way) and if the outer
  // coincident members in the band, together with the prefix schedule,
  // determine individual reductions.  In particular, each instance
  // of this combined outer schedule only writes to a single tensor element.
  bool detectReductions(detail::ScheduleTree* band);
  // Does separateReduction need to be called on this node?
  bool needReductionSeparation(const detail::ScheduleTree* st);
  // Return the schedule that will be used by mapInnermostBandsToThreads
  // for mapping to thread identifiers, with the last function
  // corresponding to thread identifier x.
  isl::MultiUnionPwAff<Statement, ReductionSchedule> reductionMapSchedule(
      const detail::ScheduleTree* st);
  // Separate out reductions that can be mapped to an entire block.
  // The remaining parts, if any, are no longer considered for replacement
  // by a library call.
  detail::ScheduleTree* separateReduction(detail::ScheduleTree* band);

  // Find best thread sync between st1 and st2 when st2 is scheduled after
  // st1.
  // This function assumes that it is called before block mapping
  // and that st1 and st2 are already mapped to threads.
  // "domainToThread" and "domainToWarp" map the domain elements
  // of st1 and st2 to thread and warp identifiers, respectively.
  Scop::SyncLevel findBestSync(
      detail::ScheduleTree* st1,
      detail::ScheduleTree* st2,
      isl::multi_union_pw_aff domainToThread,
      isl::multi_union_pw_aff domainToWarp);

 public:
  // Find best configuration of synchronizations in a sequence, minimizing
  // the number of __syncthreads, and then the number of __syncwarp
  // bestSync[i][k] == l means that there must be a synchronization at level at
  // least l between child i and child i + k.
  // if i + k > nChildren, this means that it corresponds to synchronizations
  // between child i and child (i + k) % nChildren at two different iteration
  // of the outer sequential member if hasOuterSequentialMember is true.
  // However, these cells should still exist if hasOuterSequentialMember is
  // false.
  static std::vector<std::pair<int, int>> findBestSyncConfigInSeq(
      std::vector<std::vector<int>> bestSync,
      size_t nChildren,
      bool hasOuterSequentialMember);

  // Extract a mapping from the domain elements active at "tree"
  // to the thread identifiers, where all branches in "tree"
  // are assumed to have been mapped to thread identifiers.
  // The result lives in a space of the form block[x, ...].
  isl::MultiUnionPwAff<Statement, Thread> threadMappingSchedule(
      const detail::ScheduleTree* tree) const;

  // Extract a mapping from the domain elements active at "tree"
  // to the block identifiers, where all branches in "tree"
  // are assumed to have been mapped to block identifiers.
  // The result lives in a space of the form grid[x, ...].
  isl::MultiUnionPwAff<Statement, Block> blockMappingSchedule(
      const detail::ScheduleTree* tree) const;

 private:
  // Insert the optimal combination of synchronizations in the sequence
  void insertBestSyncInSeq(detail::ScheduleTree* seq);
  // Split out a single reduction tile (in the directions other than
  // the reduction) and insert reduction synchronizations.
  // Return a pointer to the split off tile.
  detail::ScheduleTree* splitOutReductionTileAndInsertSyncs(
      detail::ScheduleTree* band);
  // Map "band" to thread identifiers using as many blockSizes values as outer
  // coincident dimensions (plus reduction dimension, if any),
  // insert synchronization in case of a reduction, and
  // return the number of mapped thread identifiers.
  // A marker is added to mark the part of the tree that is thread specific
  // (right underneath the innermost band member mapped to a thread identifier).
  size_t mapToThreads(detail::ScheduleTree* band);
  // Map innermost bands to thread identifiers,
  // inserting synchronization in case of a reduction, and
  // return the number of mapped thread identifiers.
  size_t mapInnermostBandsToThreads(detail::ScheduleTree* st);

 private:
  std::unique_ptr<Scop> scop_;

 public:
  const ::tc::Grid numBlocks;
  const ::tc::Block numThreads;
  const uint64_t unroll;
  const bool useReadOnlyCache;

 private:
  // Information about a detected reduction that can potentially
  // be mapped to a library call.
  struct Reduction {
    Reduction(std::vector<isl::id> ids) : ids(ids), separated(false) {}
    // The statement identifiers of the reduction update statements.
    std::vector<isl::id> ids;
    // Has the reduction been separated out as a full block?
    bool separated;
  };
  // Map isolated innermost reduction band members to information
  // about the detected reduction.
  std::map<const detail::ScheduleTree*, Reduction> reductionBandUpdates_;
};

} // namespace cuda
} // namespace polyhedral
} // namespace tc
