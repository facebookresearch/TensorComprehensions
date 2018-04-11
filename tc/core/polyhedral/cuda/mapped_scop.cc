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
#include "tc/core/polyhedral/cuda/mapped_scop.h"

#include <algorithm>
#include <array>
#include <iostream>
#include <numeric>
#include <sstream>
#include <stdexcept>
#include <unordered_set>

#include "tc/core/flags.h"
#include "tc/core/gpu.h"
#include "tc/core/libraries.h"
#include "tc/core/polyhedral/cuda/codegen.h"
#include "tc/core/polyhedral/cuda/mapping_types.h"
#include "tc/core/polyhedral/cuda/memory_promotion_heuristic.h"
#include "tc/core/polyhedral/cuda/tighten_launch_bounds.h"
#include "tc/core/polyhedral/exceptions.h"
#include "tc/core/polyhedral/functional.h"
#include "tc/core/polyhedral/schedule_transforms.h"
#include "tc/core/polyhedral/schedule_tree_matcher.h"
#include "tc/core/polyhedral/scop.h"
#include "tc/core/polyhedral/separation.h"
#include "tc/core/polyhedral/unroll.h"
#include "tc/core/scope_guard.h"

#include <glog/logging.h>

namespace tc {
namespace polyhedral {

namespace {

template <typename ExceptionType>
inline void throwIfHasPattern(
    ScheduleTreeMatcher matcher,
    const detail::ScheduleTree* root) {
  auto candidates = match(matcher, root);
  if (candidates.size() > 0) {
    std::stringstream ss;
    ss << "Found bad pattern:\n" << *candidates[0] << "\nin:\n" << *root;
    LOG(ERROR) << ss.str();
    throw ExceptionType(ss.str());
  }
}

void validate(const detail::ScheduleTree* root) {
  throwIfHasPattern<EmptyFilterException>(
      filter(
          [](isl::union_set uset) { return !uset || uset.is_empty(); }, any()),
      root);
  throwIfHasPattern<EmptyMappingFilterException>(
      mapping_filter(
          [](isl::union_set uset) { return !uset || uset.is_empty(); }, any()),
      root);
}

/*
 * Create a filter that maps the identifiers of type "MappingTypeId"
 * in the range [begin, end) to zero for all elements in "domain".
 */
template <typename MappingTypeId>
isl::union_set makeFixRemainingZeroFilter(
    isl::union_set activeDomain,
    std::unordered_set<MappingTypeId, typename MappingTypeId::Hash> ids) {
  std::unordered_map<MappingTypeId, size_t, typename MappingTypeId::Hash>
      idExtents;
  for (auto id : ids) {
    idExtents.insert(std::make_pair(id, 1ul));
  }
  auto space = activeDomain.get_space();
  auto filter = makeParameterContext(space, idExtents.begin(), idExtents.end());
  return filter & activeDomain.universe();
}

bool anyNonCoincidentMember(const detail::ScheduleTreeElemBand* band) {
  return band->nOuterCoincident() < band->nMember();
}
} // namespace

template <typename MappingTypeId>
void MappedScop::mapRemaining(
    detail::ScheduleTree* tree,
    size_t nMapped,
    size_t nToMap) {
  if (nMapped >= nToMap) {
    return;
  }

  std::unordered_set<MappingTypeId, typename MappingTypeId::Hash> ids;
  for (size_t i = nMapped; i < nToMap; ++i) {
    ids.insert(MappingTypeId::makeId(i));
  }
  auto root = scop_->scheduleRoot();
  auto domain = activeDomainPoints(root, tree);
  auto filter = makeFixRemainingZeroFilter(domain, ids);
  insertMappingFilterAbove(root, tree, filter, ids);

  for (size_t i = nMapped; i < nToMap; ++i) {
    if (MappingTypeId::makeId(i) == mapping::ThreadId::x()) {
      threadIdxXScheduleDepthState.emplace_back(std::make_pair(
          activeDomainPoints(schedule(), tree),
          tree->scheduleDepth(schedule())));
    }
  }
}

// Uses as many blockSizes elements as outer coincident dimensions in the
// outermost band
void MappedScop::mapToBlocksAndScaleBand(
    detail::ScheduleTree* band,
    std::vector<size_t> tileSizes) {
  using namespace tc::polyhedral::detail;

  auto bandNode = band->elemAs<ScheduleTreeElemBand>();
  CHECK(bandNode->permutable_) << "cannot map non-permutable band to blocks";

  auto nBlocksToMap = bandNode->nOuterCoincident();
  // Can map at most 3 dimensions
  nBlocksToMap = std::min(nBlocksToMap, 3ul);
  // and no more than block dimensions to be mapped
  nBlocksToMap = std::min(nBlocksToMap, numBlocks.view.size());

  for (size_t i = 0; i < nBlocksToMap; ++i) {
    band = map(band, i, mapping::BlockId::makeId(i));
  }
  mapRemaining<mapping::BlockId>(band, nBlocksToMap, numBlocks.view.size());
  bandScale(band, tileSizes);
}

/*
 * Given a filter node in the schedule tree of a mapped scop,
 * insert another filter underneath (if needed) that fixes
 * the thread identifiers in the range [begin, end) to zero.
 */
void fixThreadsBelowFilter(
    MappedScop& mscop,
    detail::ScheduleTree* filterTree,
    size_t begin,
    size_t end) {
  if (begin == end) {
    return;
  }

  std::unordered_set<mapping::ThreadId, mapping::ThreadId::Hash> ids;
  for (size_t i = begin; i < end; ++i) {
    ids.insert(mapping::ThreadId::makeId(i));
  }
  auto root = mscop.schedule();
  auto domain = activeDomainPoints(root, filterTree);
  auto mappingFilter = makeFixRemainingZeroFilter(domain, ids);
  auto filter = filterTree->elemAs<detail::ScheduleTreeElemFilter>();
  CHECK(filter) << "Not a filter: " << *filter;
  // Active domain points will contain spaces for different statements
  // When inserting below a leaf filter, this would break the tightening
  // invariant that leaf mapping filters have a single space.
  // So we intersect with the universe set of the filter to only keep the
  // space for the legitimate statement.
  insertMappingFilterBelow(
      filterTree, mappingFilter & filter->filter_.universe(), ids);

  for (size_t i = begin; i < end; ++i) {
    if (mapping::ThreadId::makeId(i) == mapping::ThreadId::x()) {
      // Mapping happend below filterTree, so we need points active for its
      // children.  After insertion, filterTree is guaranteed to have at least
      // one child.
      mscop.threadIdxXScheduleDepthState.emplace_back(std::make_pair(
          activeDomainPoints(mscop.schedule(), filterTree->child({0})),
          filterTree->scheduleDepth(mscop.schedule())));
    }
  }
}

bool MappedScop::detectReductions(detail::ScheduleTree* tree) {
  bool found = false;
  for (auto c : tree->children()) {
    found |= detectReductions(c);
  }
  auto band = tree->elemAs<detail::ScheduleTreeElemBand>();
  // Nested reductions are not currently supported.
  if (!band || found) {
    return found;
  }

  // For now, only support reductions with a sufficient number
  // of coincident outer band members for the remaining thread identifiers.
  auto nCoincident = band->nOuterCoincident();
  if (nCoincident < numThreads.view.size() - 1) {
    return found;
  }

  // Look for a reduction band member, but only if it involves
  // a single reduction for now.
  // Support for multiple reductions would require a check
  // that these reductions do not interfere with each other.
  auto initsUpdates = reductionInitsUpdates(band->mupa_.domain(), scop());
  auto inits = initsUpdates.first;
  auto updates = initsUpdates.second;
  if (updates.n_set() != 1) {
    return false;
  }
  std::vector<isl::id> updateIds;
  updates.foreach_set([&updateIds](isl::set set) {
    updateIds.emplace_back(set.get_tuple_id());
  });
  // The reduction member needs to appear right underneath
  // the coincident members.
  auto reductionDim = nCoincident;
  auto member = band->mupa_.get_union_pw_aff(reductionDim);
  if (!isReductionMember(member, updates, scop())) {
    return false;
  }
  auto reductionTree = bandSplitOut(scop_->scheduleRoot(), tree, reductionDim);
  // Order the init statements (if any) before the update statements
  // to ensure the band from which the reduction band has been split off
  // only contains update statements.
  // Note that this relies on the outer members being coincident.
  if (!inits.is_empty()) {
    orderBefore(scop_->scheduleRoot(), tree, inits);
  }
  reductionFromParent_.emplace(tree, reductionTree);
  reductionBandUpdates_.emplace(reductionTree, updateIds);
  return true;
}

bool MappedScop::needReductionSeparation(const detail::ScheduleTree* st) {
  // It is the parent band of the reduction band that needs to be separated.
  if (reductionFromParent_.count(st) != 1) {
    return false;
  }
  st = reductionFromParent_.at(st);
  CHECK(reductionBandUpdates_.count(st) == 1);
  // No need to separate if already separated.
  return !reductionBandUpdates_.at(st).separated;
}

isl::multi_union_pw_aff MappedScop::reductionMapSchedule(
    const detail::ScheduleTree* st) {
  CHECK(reductionFromParent_.count(st) == 1);
  auto parent = st;
  st = reductionFromParent_.at(st);
  CHECK(reductionBandUpdates_.count(st) == 1);

  auto reductionBand = st->elemAs<detail::ScheduleTreeElemBand>();
  CHECK(reductionBand);
  // Start with the schedule of the reduction band (in last position).
  auto reductionSchedule = reductionBand->mupa_;

  // Total size of returned schedule needs to be equal
  // to the number of thread identifiers.
  if (numThreads.view.size() > 1) {
    CHECK(parent != st);
  }
  // Prepend last members of parent band (if any).
  if (parent != st) {
    auto parentBand = parent->elemAs<detail::ScheduleTreeElemBand>();
    CHECK(parentBand);
    auto parentSchedule = parentBand->mupa_;
    auto nMember = parentBand->nMember();
    CHECK_GE(nMember, numThreads.view.size() - 1);
    parentSchedule = parentSchedule.drop_dims(
        isl::dim_type::set, 0, nMember - (numThreads.view.size() - 1));
    reductionSchedule = parentSchedule.flat_range_product(reductionSchedule);
  }

  return reductionSchedule;
}

detail::ScheduleTree* MappedScop::separateReduction(detail::ScheduleTree* st) {
  CHECK(reductionFromParent_.count(st) == 1);
  auto reduction = reductionFromParent_.at(st);
  // This function either separates full blocks (if needed) or
  // disables the reduction handling.
  reductionBandUpdates_.at(reduction).separated = true;

  auto root = scop_->scheduleRoot();
  auto domain = activeDomainPoints(root, st);
  auto prefixSchedule = prefixScheduleMupa(root, st);
  auto reductionSchedule = reductionMapSchedule(st);
  auto space = reductionSchedule.get_space();
  auto size = isl::multi_val::zero(space);
  for (size_t i = 0; i < numThreads.view.size(); ++i) {
    auto pos = numThreads.view.size() - 1 - i;
    size = size.set_val(pos, isl::val(st->ctx_, numThreads.view[i]));
  }
  // Domain elements that map to partial tiles in the reduction schedule
  // for any fixed value of the prefix schedule.
  auto partial =
      partialTargetTiles(domain, prefixSchedule, reductionSchedule, size);

  if (partial.is_empty()) {
    LOG_IF(INFO, FLAGS_debug_tc_mapper)
        << "Full/partial tile separation not needed "
        << "because all tiles are full.\n";
    return st;
  }
  // Try a bit harder to simplify the conditions describing
  // the partial tiles.  Ideally, the second gist should not have
  // any effect, but in some cases it does.
  partial = partial.gist(domain).gist(domain);
  auto full = domain.subtract(partial);
  if (full.is_empty()) {
    LOG_IF(INFO, FLAGS_debug_tc_mapper)
        << "No mapping to reductions because there are no full block tiles.\n";
    reductionBandUpdates_.erase(reduction);
    return st;
  }
  if (isl::set::from_union_set(full).n_basic_set() != 1) {
    LOG_IF(INFO, FLAGS_debug_tc_mapper)
        << "Full/partial tile separation skipped\n"
        << "because full block tile condition is too complicated.\n";
    reductionBandUpdates_.erase(reduction);
    return st;
  }
  orderAfter(root, st, partial);
  return st->ancestor(root, 2);
}

size_t MappedScop::mapToThreads(detail::ScheduleTree* band, size_t nInner) {
  using namespace tc::polyhedral::detail;

  if (nInner >= numThreads.view.size()) {
    return nInner;
  }
  if (reductionBandUpdates_.count(band) == 1) {
    // A reduction is assumed to get mapped to threadIdx.x
    if (nInner != 0) {
      reductionBandUpdates_.erase(band);
      return nInner;
    }
    CHECK(reductionBandUpdates_.at(band).separated);
    threadIdxXScheduleDepthState.emplace_back(std::make_pair(
        activeDomainPoints(schedule(), band),
        band->scheduleDepth(schedule()) + 0));
    band = map(band, 0, mapping::ThreadId::x());
    markUnroll(scop_->scheduleRoot(), band, unroll);
    return 1;
  }
  auto bandNode = band->elemAs<ScheduleTreeElemBand>();
  // If any inner node was mapped to threads and
  // the current node has a non-coincident member,
  // then synchronization needs to be introduced.
  // This also implies that the mapping needs to be completed first.
  if (anyNonCoincidentMember(bandNode) && nInner > 0) {
    // Since some thread identifiers were mapped already (nInner > 0),
    // the band should have descendants.  Double check.
    CHECK_EQ(band->numChildren(), 1);
    mapRemaining<mapping::ThreadId>(
        band->child({0}), nInner, numThreads.view.size());
    scop_->insertSyncAfter(band->child({0}));
    return numThreads.view.size();
  }
  // With current isl scheduler, if coincident dimensions exist in a band,
  // they are outermost.
  // If a band has more than 3 coincident dimensions, this will choose
  // outermost, but we may also want innermost.
  auto nOuterCoincident = bandNode->nOuterCoincident();
  if (!bandNode->permutable_ || nOuterCoincident < 1) {
    return nInner;
  }

  auto nMappedThreads = std::min(
      numThreads.view.size() - nInner, static_cast<size_t>(nOuterCoincident));
  CHECK_GT(nMappedThreads, 0) << "not mapping to threads";
  CHECK_LE(nMappedThreads, 3 - nInner) << "mapping to too many threads";

  // Map the coincident dimensions to threads starting from the innermost and
  // from thread x.
  for (int i = 0, dim = nOuterCoincident - 1; i < nMappedThreads && dim >= 0;
       ++i, --dim) {
    auto id = mapping::ThreadId::makeId(nInner + i);
    if (id == mapping::ThreadId::x()) {
      threadIdxXScheduleDepthState.emplace_back(std::make_pair(
          activeDomainPoints(schedule(), band),
          band->scheduleDepth(schedule()) + dim));
    }
    band = map(band, dim, id);
  }

  if (nInner == 0) {
    markUnroll(scop_->scheduleRoot(), band, unroll);
  }

  return nInner + nMappedThreads;
}

namespace {

/*
 * Does "st" have a sequential outer band member in the tree
 * with the given root within the same branch
 * of the innermost outer sequence node (if any)?
 * That is, assuming "st" is a sequence node, does the last child
 * need to be protected from the next iteration of the first child?
 */
bool hasOuterSequentialMember(
    const detail::ScheduleTree* root,
    detail::ScheduleTree* st) {
  auto ancestors = st->ancestors(root);
  std::reverse(ancestors.begin(), ancestors.end());
  for (auto a : ancestors) {
    auto band = a->elemAs<detail::ScheduleTreeElemBand>();
    if (band && band->nMember() > band->nOuterCoincident()) {
      return true;
    }
    if (a->elemAs<detail::ScheduleTreeElemSequence>()) {
      return false;
    }
  }
  return false;
}
} // namespace

// Maps bands to threads in DFS postorder, keeping track of
// the (maximal) number of threads already mapped by descendants.
//
// If any separation is needed for mapping reductions to full blocks,
// then do so first.
//
// If "st" has multiple children, then make sure they are mapped
// to the same number of thread identifiers by fixing those
// that are originally mapped to fewer identifiers to value zero
// for the remaining thread identifiers.
// If, moreover, "st" is a sequence node and at least one of its
// children is mapped to threads, then introduce synchronization
// before and after children that are mapped to threads.
// Also add synchronization between the last child and
// the next iteration of the first child if there may be such
// a next iteration that is not already covered by synchronization
// on an outer node.
// If any synchronization is introduced, then the mapping
// to threads needs to be completed to all thread ids
// because the synchronization needs to be introduced outside
// any mapping to threads.
size_t MappedScop::mapInnermostBandsToThreads(detail::ScheduleTree* st) {
  if (needReductionSeparation(st)) {
    st = separateReduction(st);
  }
  auto children = st->children();
  auto nChildren = children.size();
  std::vector<size_t> nInner(nChildren);
  for (size_t i = 0; i < nChildren; ++i) {
    nInner[i] = mapInnermostBandsToThreads(children[i]);
  }
  auto n = nChildren > 0 ? *std::max_element(nInner.begin(), nInner.end()) : 0;
  if (nChildren > 1) {
    auto needSync = st->elemAs<detail::ScheduleTreeElemSequence>() && n > 0;
    if (needSync) {
      n = numThreads.view.size();
    }
    for (size_t i = 0; i < nChildren; ++i) {
      fixThreadsBelowFilter(*this, children[i], nInner[i], n);
    }
    if (needSync) {
      auto outer = hasOuterSequentialMember(scop_->scheduleRoot(), st);
      if (outer && (nInner[0] > 0 || nInner[nChildren - 1] > 0)) {
        scop_->insertSync(st, nChildren);
      }
      for (size_t i = nChildren - 1; i > 0; --i) {
        if (nInner[i] > 0 || nInner[i - 1] > 0) {
          scop_->insertSync(st, i);
        }
      }
    }
  }

  if (st->elemAs<detail::ScheduleTreeElemBand>()) {
    n = mapToThreads(st, n);
  }

  return n;
}

/*
 * Creates a context set for the block and thread identifiers passed in the
 * mappingIds map.
 *
 * The context is of the form:
 *   [..., id_i] -> { [] : 0 <= id_i < extent_i }
 * where i ranges over all block and thread identifiers.
 * To avoid special corner cases we create the context for all existing
 * mapping dimensions even if the scop is unmapped along some of those.
 * Note that context nodes in a schedule tree live in set spaces with as many
 * set dimensions as outer schedule dimensions (here zero).
 */
void MappedScop::insertMappingContext() {
  Scop& scop = *scop_;
  const Grid& grid = numBlocks;
  const Block& block = numThreads;
  USING_MAPPING_SHORT_NAMES(BX, BY, BZ, TX, TY, TZ);
  std::unordered_map<isl::id, size_t, isl::IslIdIslHash> mappingIdsWithSizes{
      {BX, BX.mappingSize(grid)},
      {BY, BY.mappingSize(grid)},
      {BZ, BZ.mappingSize(grid)},
      {TX, TX.mappingSize(block)},
      {TY, TY.mappingSize(block)},
      {TZ, TZ.mappingSize(block)}};
  auto space = scop.domain().universe().get_space();
  auto mappingContext = makeParameterContext(
      space, mappingIdsWithSizes.begin(), mappingIdsWithSizes.end());
  updateTopLevelContext(scop.scheduleRoot(), mappingContext.from_params());
}

namespace {
// Specialize a MappedScop with respect to its context.
// The context is explicitly inserted as a specialization context in
// the cloned underlying scop.
// After underlying scop specialization, mapping parameter tightening is
// performed to ensure launch bounds are consistent with problem size and
// tiling (i.e. that we do not launch empty threads/blocks).
// Note this should happen pretty late in the process because:
//   1. we want to specialize on as many parameters as are available so as
//      to perform the tightest launch bound specialization possible,
//   2. the less specialized we are the more chances
//      ensureLibrariesCalledInAllThreads will throw (at least in currently
//      envisioned first impl),
//   3. we want to allow more threads for copies to/from shared memory than
//      for compute, so we want to take copies into account when tightening
//      launch bounds.
std::unique_ptr<MappedScop> makeSpecializedMappedScop(
    const MappedScop& mappedScop) {
  auto scop = Scop::makeScop(mappedScop.scop());

  // In this particular specialized Scop, we can add a context just below root.
  // Context nodes in the schedule tree use _set_ spaces rather than _parameter_
  // spaces because they may depend on outer schedule dimensions.  In this
  // particular case, the "root is domain" invariant guarantees there are no
  // outer schedule dimensions, so the space of a parameter context code is that
  // of a zero-dimensional space.
  auto root = scop->scheduleRoot();
  updateTopLevelContext(root, scop->globalParameterContext.from_params());

  tc::Grid grid = mappedScop.numBlocks;
  tc::Block block = mappedScop.numThreads;
  std::tie(grid, block) = tightenLaunchBounds(*scop, grid, block);
  auto res = MappedScop::makeMappedScop(
      std::move(scop), grid, block, mappedScop.unroll);
  res->insertMappingContext();

  LOG_IF(INFO, FLAGS_debug_tc_mapper)
      << "Codegen with tightened bounds [blocks:" << grid
      << ", threads:" << block << "] for tree:\n"
      << *res->schedule();

  return res;
}
} // namespace

// Before generating code, make a copy of the scop and insert
// the globalParameterContext of the original scop as top-level
// context node in schedule tree.
std::tuple<std::string, tc::Grid, tc::Block> MappedScop::codegen(
    const std::string& specializedName) const {
  validate(schedule());

  auto mappedScopForCodegen = makeSpecializedMappedScop(*this);

  std::stringstream code;
  code << code::cpp::boundsAsTemplate << code::c::types << code::c::defines
       << std::endl;
  if (mappedScopForCodegen->scop().treeSyncUpdateMap.size() != 0) {
    code << code::cuda::common;
    code << code::cuda::cubBlockReduce;
  }
  code << "extern \"C\" {" << std::endl
       << emitCudaKernel(specializedName, *mappedScopForCodegen) << "}"
       << std::endl;

  return std::make_tuple(
      code.str(),
      mappedScopForCodegen->numBlocks,
      mappedScopForCodegen->numThreads);
}

std::unique_ptr<MappedScop> MappedScop::makeWithOuterBlockInnerThreadStrategy(
    std::unique_ptr<Scop>&& scopUPtr,
    const CudaMappingOptions& cudaOptions) {
  using namespace polyhedral::detail;

  const auto& generic = cudaOptions.generic;
  auto mappedScop = std::unique_ptr<MappedScop>(new MappedScop(
      std::move(scopUPtr),
      ::tc::Grid(cudaOptions.grid),
      ::tc::Block(cudaOptions.block),
      generic.proto.unroll()));
  auto& scop = mappedScop->scop_;

  // 1a. Optionally specialize before scheduling...
  if (generic.proto.fix_parameters_before_scheduling()) {
    scop->specializeToContext();
  }

  // 2. Schedule
  scop = Scop::makeScheduled(*scop, generic.outerScheduleOptions);

  // 3. Tile
  CHECK_LT(0, generic.tiling.size())
      << "Must pass tile vector with >= 1 tile sizes";
  auto outerBand = scop->tileOuterBand(generic.tiling);

  // 4. Optionally reschedule if point loops need a different strategy than
  // tile loops
  if (generic.outerScheduleOptions != generic.intraTileScheduleOptions) {
    scop->reschedule(outerBand->child({0}), generic.intraTileScheduleOptions);
    LOG_IF(INFO, FLAGS_debug_tc_mapper)
        << "After intra-tile rescheduling:" << std::endl
        << *mappedScop->schedule();
  }

  // 1b. ...or after rescheduling
  if (!generic.proto.fix_parameters_before_scheduling()) {
    scop->specializeToContext();
  }

  // 5. Map to threads
  if (outerBand->numChildren() > 0) {
    CHECK_EQ(1, outerBand->numChildren());
    // 5.1. Optionally detect reductions while mapping to threads
    if (generic.proto.match_library_calls()) {
      mappedScop->detectReductions(outerBand->child({0}));
    }
    auto child = outerBand->child({0});
    size_t numMappedInnerThreads =
        mappedScop->mapInnermostBandsToThreads(child);
    mappedScop->mapRemaining<mapping::ThreadId>(
        child, numMappedInnerThreads, mappedScop->numThreads.view.size());
    LOG_IF(INFO, FLAGS_debug_tc_mapper)
        << "After mapping to threads:" << std::endl
        << *mappedScop->schedule();
  }

  // 6. Map to blocks
  mappedScop->mapToBlocksAndScaleBand(
      outerBand, generic.tiling.extractVector());
  LOG_IF(INFO, FLAGS_debug_tc_mapper) << "After mapping to blocks:" << std::endl
                                      << *mappedScop->schedule();

  // 7. Promote to shared memory below the loops mapped to blocks.
  // This may split the outer band, so find the new outer band after promotion.
  if (cudaOptions.proto().use_shared_memory()) {
    size_t sharedMemorySize = cudaOptions.proto().has_max_shared_memory()
        ? cudaOptions.proto().max_shared_memory()
        : querySharedMemorySize();
    // If reductions found, their synchronization requires an opaque cache in
    // shared memory.  Subtract 4k from available shared memory for each
    // reduction found, this is hack based on each thread of max 1024 in the
    // block needing one float in shared memory in the worst case.
    // FIXME: introduce an actual model of shared memory requirements for
    // reductions.
    size_t reductionMemoryRequirement =
        4096 * mappedScop->reductionBandUpdates_.size();
    if (reductionMemoryRequirement > sharedMemorySize) {
      sharedMemorySize = 0;
    } else {
      sharedMemorySize -= reductionMemoryRequirement;
    }

    auto band = outerBand->elemAs<ScheduleTreeElemBand>();
    LOG_IF(WARNING, FLAGS_debug_tc_mapper && band->nMember() == 0)
        << "Aborting memory promotion because outer band has 0 members (NYI)";
    if (band->nMember() > 0 && sharedMemorySize > 0) {
      LOG_IF(
          WARNING,
          cudaOptions.proto().unroll_copy_shared() &&
              !generic.proto.has_unroll())
          << "requested to unroll copies to shared memory without providing the unroll size";

      promoteGreedilyAtDepth(
          *mappedScop,
          mappedScop->threadIdxXScheduleDepthState,
          std::min(band->nOuterCoincident(), mappedScop->numBlocks.view.size()),
          sharedMemorySize,
          cudaOptions.proto().unroll_copy_shared() &&
              generic.proto.has_unroll());

      auto bands = ScheduleTree::collectDFSPreorder(
          scop->scheduleRoot(), ScheduleTreeType::Band);
      if (bands.size() == 0) { // Sanity check.
        throw NoBandsException("no bands after promotion");
      }
      outerBand = bands[0];
    }
  }

  // 8. Promote to registers below the loops mapped to threads.
  if (cudaOptions.proto().use_private_memory()) {
    promoteToRegistersBelowThreads(
        mappedScop->scop(), mappedScop->threadIdxXScheduleDepthState, -1ull);
  }

  // 9. Insert mapping context
  mappedScop->insertMappingContext();

  // 10. Optionally insert reduction synchronizations
  for (auto bandUpdate : mappedScop->reductionBandUpdates_) {
    for (auto updateId : bandUpdate.second.ids) {
      scop->insertReductionSync1D(
          const_cast<ScheduleTree*>(bandUpdate.first), updateId);
    }
  }
  LOG_IF(INFO, FLAGS_debug_tc_mapper)
      << "After inserting reduction synchronization:" << std::endl
      << *mappedScop->schedule();

  return mappedScop;
}

} // namespace polyhedral
} // namespace tc
