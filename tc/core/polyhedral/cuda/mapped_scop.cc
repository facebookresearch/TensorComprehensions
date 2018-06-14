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

#include "tc/core/check.h"
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
  throwIfHasPattern<EmptyMappingException>(
      mapping_filter(
          [](isl::union_set uset) { return !uset || uset.is_empty(); }, any()),
      root);
}

bool anyNonCoincidentMember(const detail::ScheduleTreeElemBand* band) {
  return band->nOuterCoincident() < band->nMember();
}

/*
 * Return a reference to the mapping sizes
 * for the mapping of type "MappingTypeId".
 */
template <typename MappingTypeId>
const CudaDim& mappingSize(const MappedScop* mscop);
template <>
const CudaDim& mappingSize<mapping::BlockId>(const MappedScop* mscop) {
  return mscop->numBlocks;
}
template <>
const CudaDim& mappingSize<mapping::ThreadId>(const MappedScop* mscop) {
  return mscop->numThreads;
}
} // namespace

// Map the elements in "list" to successive blocks or thread identifiers,
// with the first element mapped to identifier X.  The extents are obtained
// from the initial elements of numBlocks or numThreads.  The identifiers
// must not be present in the space of the partial schedules in "list" and
// extents must be non-zero.  The mapping corresponds to inserting a filter
// node with condition 'list % extent = ids'.
// The mapping is inserted above "tree".
//
// Return a pointer to the updated node (below the inserted filter)
// for call chaining purposes.
template <typename MappingTypeId>
detail::ScheduleTree* MappedScop::map(
    detail::ScheduleTree* tree,
    isl::union_pw_aff_list list) {
  size_t nToMap = list.n();
  const auto& extent = mappingSize<MappingTypeId>(this).view;
  TC_CHECK_LE(nToMap, extent.size()) << "dimension overflow";

  auto root = scop_->scheduleRoot();
  auto domain = activeDomainPoints(root, tree).universe();

  std::vector<MappingTypeId> idList;
  auto affList = isl::union_pw_aff_list(list.get_ctx(), 0);
  for (size_t i = 0; i < nToMap; ++i) {
    auto id = MappingTypeId::makeId(i);
    auto upa = list.get(i);
    TC_CHECK_NE(extent[i], 0u) << "NYI: mapping to 0";
    upa = upa.mod_val(isl::val(tree->ctx_, extent[i]));
    affList = affList.add(upa);
    idList.emplace_back(id);
  }

  for (size_t i = nToMap; i < extent.size(); ++i) {
    auto id = MappingTypeId::makeId(i);
    affList = affList.add(
        isl::union_pw_aff(domain, isl::val::zero(domain.get_ctx())));
    idList.emplace_back(id);
  }

  auto mapping = detail::ScheduleTree::makeMapping(idList, affList);
  tree = insertNodeAbove(root, tree, std::move(mapping))->child({0});

  return tree;
}

detail::ScheduleTree* MappedScop::mapBlocksForward(
    detail::ScheduleTree* band,
    size_t nToMap) {
  auto bandNode = band->elemAs<detail::ScheduleTreeElemBand>();
  TC_CHECK(bandNode) << "expected a band, got " << *band;

  auto list = bandNode->mupa_.get_union_pw_aff_list();
  list = list.drop(nToMap, list.n() - nToMap);
  return map<mapping::BlockId>(band, list);
}

// Uses as many blockSizes elements as outer coincident dimensions in the
// outermost band
void MappedScop::mapToBlocksAndScaleBand(
    detail::ScheduleTree* band,
    std::vector<size_t> tileSizes) {
  using namespace tc::polyhedral::detail;

  auto bandNode = band->elemAs<ScheduleTreeElemBand>();
  TC_CHECK(bandNode->permutable_) << "cannot map non-permutable band to blocks";

  auto nBlocksToMap = bandNode->nOuterCoincident();
  // Can map at most 3 dimensions
  nBlocksToMap = std::min(nBlocksToMap, 3ul);
  // and no more than block dimensions to be mapped
  nBlocksToMap = std::min(nBlocksToMap, numBlocks.view.size());

  mapBlocksForward(band, nBlocksToMap);
  bandScale(band, tileSizes);
}

namespace {

/*
 * Given a node in the schedule tree of a mapped scop,
 * insert a mapping filter underneath (if needed) that fixes
 * the remaining thread identifiers starting at "begin" to zero.
 * Add a marker underneath that marks the subtree that is thread specific.
 */
void fixThreadsBelow(
    MappedScop& mscop,
    detail::ScheduleTree* tree,
    size_t begin) {
  size_t end = mscop.numThreads.view.size();
  if (begin == end) {
    return;
  }

  auto band = detail::ScheduleTree::makeEmptyBand(mscop.scop().scheduleRoot());
  auto bandTree = insertNodeBelow(tree, std::move(band));
  mscop.mapThreadsBackward(bandTree);
}

/*
 * Try and order the other active statements at "tree" (if any)
 * away from the "updates" statements, returning true is the operation succeeds.
 * In particular, only do this if it doesn't violate any dependences.
 * Anything that depends on an update statement is ordered after
 * the update statements.  Anything else is ordered before.
 */
bool separatedOut(
    Scop& scop,
    detail::ScheduleTree* tree,
    isl::union_set updates) {
  auto domain = activeDomainPoints(scop.scheduleRoot(), tree);
  auto other = domain.subtract(updates);
  if (other.is_empty()) {
    return true;
  }
  auto dependences = scop.activeDependences(tree);
  auto after =
      dependences.intersect_domain(updates).intersect_range(other).range();
  auto before = other.subtract(after);
  if (!canOrderBefore(scop.scheduleRoot(), tree, before, dependences) ||
      !canOrderAfter(scop.scheduleRoot(), tree, after, dependences)) {
    return false;
  }
  if (!before.is_empty()) {
    orderBefore(scop.scheduleRoot(), tree, before);
  }
  if (!after.is_empty()) {
    orderAfter(scop.scheduleRoot(), tree, after);
  }
  return true;
}

} // namespace

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

  // Only reductions that appear in permutable bands are mapped to threads.
  if (!band->permutable_) {
    return false;
  }

  // For now, only support reductions with a sufficient number
  // of coincident outer band members for the remaining thread identifiers and
  // at least one non-coincident member.
  auto nCoincident = band->nOuterCoincident();
  auto nMember = band->nMember();
  if (nCoincident < numThreads.view.size() - 1 || nCoincident >= nMember) {
    return found;
  }

  // Look for a reduction band member, but only if it involves
  // a single reduction for now.
  // Support for multiple reductions would require a check
  // that these reductions do not interfere with each other.
  auto domain = band->mupa_.domain();
  auto updates = reductionUpdates(domain, scop());
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
  // Order the other statements (if any) before the update statements
  // to ensure the band from which the reduction band has been split off
  // only contains update statements.
  if (!separatedOut(scop(), tree, updates)) {
    return false;
  }
  reductionBandUpdates_.emplace(tree, Reduction(updateIds));
  return true;
}

bool MappedScop::needReductionSeparation(const detail::ScheduleTree* st) {
  if (reductionBandUpdates_.count(st) != 1) {
    return false;
  }
  // No need to separate if already separated.
  return !reductionBandUpdates_.at(st).separated;
}

isl::multi_union_pw_aff MappedScop::reductionMapSchedule(
    const detail::ScheduleTree* st) {
  TC_CHECK(reductionBandUpdates_.count(st) == 1);
  auto reductionBand = st->elemAs<detail::ScheduleTreeElemBand>();
  TC_CHECK(reductionBand);

  // Drop band members following the reduction dimension and preceding those
  // mapped to threads.
  auto reductionSchedule = reductionBand->mupa_;
  auto nMember = reductionBand->nMember();
  auto reductionDim = reductionBand->nOuterCoincident();
  auto nMappedThreads = std::min(numThreads.view.size(), reductionDim + 1);
  TC_CHECK_GE(nMember, reductionDim);
  reductionSchedule = reductionSchedule.drop_dims(
      isl::dim_type::set, reductionDim + 1, nMember - (reductionDim + 1));
  reductionSchedule = reductionSchedule.drop_dims(
      isl::dim_type::set, 0, reductionDim - nMappedThreads + 1);

  return reductionSchedule;
}

detail::ScheduleTree* MappedScop::separateReduction(detail::ScheduleTree* st) {
  auto reduction = st;
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

detail::ScheduleTree* MappedScop::mapThreadsBackward(
    detail::ScheduleTree* band) {
  auto bandNode = band->elemAs<detail::ScheduleTreeElemBand>();
  TC_CHECK(bandNode);
  auto nMember = bandNode->nMember();
  auto nToMap = std::min(nMember, numThreads.view.size());
  TC_CHECK_LE(nToMap, 3u) << "mapping to too many threads";

  auto ctx = band->ctx_;
  insertNodeBelow(band, detail::ScheduleTree::makeThreadSpecificMarker(ctx));

  auto list = bandNode->mupa_.get_union_pw_aff_list().reverse();
  list = list.drop(nToMap, list.n() - nToMap);
  return map<mapping::ThreadId>(band, list);
}

size_t MappedScop::mapToThreads(detail::ScheduleTree* band) {
  using namespace tc::polyhedral::detail;

  auto bandNode = band->elemAs<ScheduleTreeElemBand>();
  // Cannot map non-permutable bands.
  if (!bandNode->permutable_) {
    return 0;
  }

  // With current isl scheduler, if coincident dimensions exist in a band,
  // they are outermost.
  // If a band has more than 3 coincident dimensions,
  // then the innermost of those will be used.
  auto nCanMap = bandNode->nOuterCoincident();

  auto isReduction = reductionBandUpdates_.count(band) == 1;
  // If the band has a detected reduction, then the first member
  // after the coincident members is the reduction member and
  // this member has to be mapped as well.
  // In particular, it will get mapped to threadIdx.x
  if (isReduction) {
    TC_CHECK(reductionBandUpdates_.at(band).separated);
    nCanMap++;
  }

  if (nCanMap < 1) {
    return 0;
  }

  auto nMappedThreads = nCanMap;
  if (nMappedThreads > numThreads.view.size()) {
    // Split band such that mapping filters get inserted
    // right above the first member mapped to a thread identifier.
    nMappedThreads = numThreads.view.size();
    bandSplit(scop_->scheduleRoot(), band, nCanMap - nMappedThreads);
    auto child = band->child({0});
    if (isReduction) {
      // Update reductionBandUpdates_ such that
      // splitOutReductionTileAndInsertSyncs
      // can find the information it needs.
      reductionBandUpdates_.emplace(child, reductionBandUpdates_.at(band));
      reductionBandUpdates_.erase(band);
    }
    band = child;
    bandNode = band->elemAs<ScheduleTreeElemBand>();
  }

  if (nMappedThreads < bandNode->nMember()) {
    bandSplit(scop_->scheduleRoot(), band, nMappedThreads);
  }

  TC_CHECK_GT(nMappedThreads, 0u) << "not mapping to threads";

  if (isReduction) {
    band = splitOutReductionTileAndInsertSyncs(band);
  }

  mapThreadsBackward(band);

  return numThreads.view.size();
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

// Name of the space of blocks inside the grid
constexpr auto kGrid = "grid";
// Name of the space of threads inside a block
constexpr auto kBlock = "block";
// Name of the space of warps
constexpr auto kWarp = "warp";

/*
 * Construct a mapping
 *
 *  block[x] -> warp[floor((x)/warpSize)]
 *  block[x, y] -> warp[floor((x + s_x * (y))/warpSize)]
 *  block[x, y, z] -> warp[floor((x + s_x * (y + s_y * (z)))/warpSize)]
 *
 * uniquely mapping thread identifiers that belong to the same warp
 * (of size "warpSize") to a warp identifier,
 * based on the thread sizes s_x, s_y up to s_z in "block".
 */
isl::multi_aff constructThreadToWarp(
    isl::ctx ctx,
    const unsigned warpSize,
    const Block& block) {
  auto space = isl::space(ctx, 0);
  auto id = isl::id(ctx, kBlock);
  auto blockSpace = space.named_set_from_params_id(id, block.view.size());
  auto warpSpace = space.named_set_from_params_id(isl::id(ctx, kWarp), 1);
  auto aff = isl::aff::zero_on_domain(blockSpace);

  auto nThread = block.view.size();
  auto identity = isl::multi_aff::identity(blockSpace.map_from_set());
  for (int i = nThread - 1; i >= 0; --i) {
    aff = aff.scale(isl::val(ctx, block.view[i]));
    aff = aff.add(identity.get_aff(i));
  }

  aff = aff.scale_down(isl::val(ctx, warpSize)).floor();
  auto mapSpace = blockSpace.product(warpSpace).unwrap();
  return isl::multi_aff(mapSpace, isl::aff_list(aff));
}
} // namespace

isl::multi_union_pw_aff MappedScop::threadMappingSchedule(
    const detail::ScheduleTree* tree) const {
  std::vector<mapping::MappingId> ids;
  for (size_t i = 0; i < numThreads.view.size(); ++i) {
    ids.emplace_back(mapping::ThreadId::makeId(i));
  }
  auto tupleId = isl::id(tree->ctx_, kBlock);
  return extractDomainToIds(scop_->scheduleRoot(), tree, ids, tupleId);
}

isl::multi_union_pw_aff MappedScop::blockMappingSchedule(
    const detail::ScheduleTree* tree) const {
  std::vector<mapping::MappingId> ids;
  for (size_t i = 0; i < numBlocks.view.size(); ++i) {
    ids.emplace_back(mapping::BlockId::makeId(i));
  }
  auto tupleId = isl::id(tree->ctx_, kGrid);
  return extractDomainToIds(scop_->scheduleRoot(), tree, ids, tupleId);
}

Scop::SyncLevel MappedScop::findBestSync(
    detail::ScheduleTree* st1,
    detail::ScheduleTree* st2,
    isl::multi_union_pw_aff domainToThread,
    isl::multi_union_pw_aff domainToWarp) {
  // Active points in the two schedule trees
  auto stRoot = scop_->scheduleRoot();
  auto activePoints1 = activeDomainPointsBelow(stRoot, st1);
  auto activePoints2 = activeDomainPointsBelow(stRoot, st2);

  // The dependences between the two schedule trees
  auto dependences = scop_->dependences;
  dependences = dependences.intersect_domain(activePoints1);
  dependences = dependences.intersect_range(activePoints2);
  if (dependences.is_empty()) {
    return Scop::SyncLevel::None;
  }

  TC_CHECK_LE(1u, scop_->scheduleRoot()->children().size());
  auto contextSt = scop_->scheduleRoot()->children()[0];
  auto contextElem = contextSt->elemAs<detail::ScheduleTreeElemContext>();
  TC_CHECK(nullptr != contextElem);
  dependences = dependences.intersect_params(contextElem->context_);

  if (dependences.is_subset(dependences.eq_at(domainToThread))) {
    return Scop::SyncLevel::None;
  }
  if (dependences.is_subset(dependences.eq_at(domainToWarp))) {
    return Scop::SyncLevel::Warp;
  }
  return Scop::SyncLevel::Block;
}

std::vector<std::pair<int, int>> MappedScop::findBestSyncConfigInSeq(
    std::vector<std::vector<int>> bestSync,
    size_t nChildren,
    bool hasOuterSequentialMember) {
  // Get the least strict synchronization level that is needed in the sequence
  // children[i], ..., children[i+k] to be correct and optimal. If the level
  // is l, this mean that a synchronization of level l has to be inserted
  // in this sequence to be correct, and that no synchronizations of level
  // greater than l is needed.
  // if i + k is greater than nChildren, it represents the child
  // (i + k) % nChildren at the next iteration of the outer sequential member if
  // it exists.
  std::vector<std::vector<int>> bestSyncInRange(
      nChildren, std::vector<int>(nChildren));
  for (size_t i = 0; i < nChildren; ++i) {
    bestSyncInRange[i][0] = 0;
  }
  for (size_t k = 1; k < nChildren; ++k) {
    for (size_t i = 0; i < nChildren; ++i) {
      bestSyncInRange[i][k] = bestSync[i][k];
      bestSyncInRange[i][k] =
          std::max(bestSyncInRange[i][k - 1], bestSyncInRange[i][k]);
      bestSyncInRange[i][k] = std::max(
          bestSyncInRange[(i + 1) % nChildren][k - 1], bestSyncInRange[i][k]);
    }
  }

  // The optimal number of block sync and thread sync needed to
  // have the sequence children[i], ..., children[i + k] correctly
  // synchronized
  std::vector<std::vector<std::pair<int, int>>> optimalValue(
      nChildren, std::vector<std::pair<int, int>>(nChildren, {-1, -1}));

  // An optimal position for doing a synchronization between children[i]
  // and children[i + k]. The first member indicates after which child the
  // synchronization should be inserted, and the second member indicates which
  // synchronization should be inserted. This should be used recursively to
  // get the optimal synchronization. If the second member is equal to 0,
  // this means that no synchronization is needed.
  std::vector<std::vector<std::pair<int, int>>> optimalSyncPosition(
      nChildren, std::vector<std::pair<int, int>>(nChildren, {-1, -1}));

  // The dynamic programming algorithm to compute the optimal synchronizations
  // To compute the optimal synchronizations for the sequence
  // children[i] ... children[i + k],
  // it splits the sequence into children[i], ..., children[i + s] and
  // children[i + s + 1], ..., children[i + k] for all possible s, and
  // insert between children[i + s] and children[i + s + 1] the least
  // strict synchronization needed.
  for (size_t i = 0; i < nChildren; ++i) {
    optimalValue[i][0] = {0, 0};
    optimalSyncPosition[i][0] = {0, 0};
  }
  for (size_t k = 1; k < nChildren; ++k) {
    for (size_t i = 0; i < nChildren; ++i) {
      if (bestSyncInRange[i][k] == 0) {
        optimalValue[i][k] = {0, 0};
        optimalSyncPosition[i][k] = {0, 0};
      } else if (bestSyncInRange[i][k] == 1) {
        optimalValue[i][k] = {nChildren, nChildren};
        // Separation in [i, i+s] [i+s+1, i+k]
        for (size_t s = 0; s < k; ++s) {
          // OptimalValue.first is always equal to 0 here,
          // since there is no need to do a block synchronization
          // between children[i] and children[i+k]
          std::pair<int, int> costOfSeparation = {
              0,
              1 + optimalValue[i][s].second +
                  optimalValue[(i + s + 1) % nChildren][k - s - 1].second};
          if (costOfSeparation < optimalValue[i][k]) {
            optimalValue[i][k] = costOfSeparation;
            optimalSyncPosition[i][k] = {s, 1};
          }
        }
      } else { // bestSyncInRange[i][k] == 2
        optimalValue[i][k] = {nChildren, nChildren};
        // Separation in [i, i+s] [i+s+1, i+k]
        for (size_t s = 0; s < k; ++s) {
          std::pair<int, int> costOfSeparation;
          costOfSeparation.first = 1 + optimalValue[i][s].first +
              optimalValue[(i + s + 1) % nChildren][k - s - 1].first;
          costOfSeparation.second = optimalValue[i][s].second +
              optimalValue[(i + s + 1) % nChildren][k - s - 1].second;
          if (costOfSeparation < optimalValue[i][k]) {
            optimalValue[i][k] = costOfSeparation;
            optimalSyncPosition[i][k] = {s, 2};
          }
        }
      }
    }
  }

  // Construct the list of all the synchronizations in the optimal configuation
  // for the range [begining, beginging + nChildren - 1].
  auto constructSynchronizationsList = [&](int begining) {
    // The stack recurse in the optimalSyncPosition table
    std::vector<std::pair<int, int>> stack = {{begining, nChildren - 1}};
    std::vector<std::pair<int, int>> synchronizations;
    while (!stack.empty()) {
      auto range = stack.back();
      auto i = range.first;
      auto k = range.second;
      stack.pop_back();
      auto syncLevel = optimalSyncPosition[i][k].second;
      if (syncLevel != 0) {
        auto separation = optimalSyncPosition[i][k].first;
        stack.push_back({i, separation});
        stack.push_back({(i + separation + 1) % nChildren, k - separation - 1});
        synchronizations.push_back({(separation + i) % nChildren, syncLevel});
      }
    }
    return synchronizations;
  };

  // If there is no outer sequential member,
  // the problem is simple and only the range [0, nChildren - 1] should
  // be considered
  if (not hasOuterSequentialMember) {
    return constructSynchronizationsList(0);
  }

  // If there is an outer sequential member, there might have dependences
  // between a child i and a child j in another iteration of the outer
  // sequential member.
  // To solve that, we first find the least strict synchronization needed,
  // and try to insert it in all possible position s, and then get the
  // solution of the problem computed for the range [s + 1, s + nChildren].
  int maxValue = 0;
  for (size_t i = 0; i < nChildren; ++i) {
    for (size_t k = 0; k < nChildren; ++k) {
      maxValue = std::max(maxValue, bestSync[i][k]);
    }
  }
  if (maxValue == 0) {
    return {};
  }
  int bestBegining = 0;
  for (size_t begining = 1; begining < nChildren; ++begining) {
    if (optimalValue[begining][nChildren - 1] <
        optimalValue[bestBegining][nChildren - 1]) {
      bestBegining = begining;
    }
  }
  auto solutionWithBestBegining = constructSynchronizationsList(bestBegining);
  solutionWithBestBegining.push_back(
      {(bestBegining + nChildren - 1) % nChildren, maxValue});
  return solutionWithBestBegining;
}

void MappedScop::insertBestSyncInSeq(detail::ScheduleTree* seq) {
  TC_CHECK(seq->elemAs<detail::ScheduleTreeElemSequence>());

  auto children = seq->children();
  auto nChildren = children.size();

  auto outer = hasOuterSequentialMember(scop_->scheduleRoot(), seq);

  auto domainToThread = threadMappingSchedule(seq);
  auto threadToWarp = constructThreadToWarp(seq->ctx_, 32, numThreads);
  auto domainToWarp = domainToThread.apply(threadToWarp);

  std::vector<std::vector<int>> bestSync(
      nChildren, std::vector<int>(nChildren + 1));
  // Get the synchronization needed between children[i] and children[i+k]
  // without considering the children in between
  // if k == 0, the synchronization needed between children[i] and children[i+k]
  // correspond to the synchronization needed them at different iterations of
  // the outer sequential member. Thus, when there is no outer sequential
  // member, bestSync[i][0] == (int)SyncLevel::None
  for (size_t i = 0; i < nChildren; ++i) {
    for (size_t k = 0; k < nChildren; ++k) {
      auto ik = (i + k) % nChildren;
      bestSync[i][k] = (int)findBestSync(
          children[i], children[ik], domainToThread, domainToWarp);
    }
  }

  // Get the optimal synchronizations configuration
  std::vector<std::pair<int, int>> synchronizations =
      findBestSyncConfigInSeq(bestSync, nChildren, outer);

  // Insert all the synchronizations
  std::sort(
      synchronizations.begin(),
      synchronizations.end(),
      std::greater<std::pair<int, int>>());
  for (size_t i = 0; i < synchronizations.size(); i++) {
    auto level = static_cast<Scop::SyncLevel>(synchronizations[i].second);
    scop_->insertSync(seq, synchronizations[i].first + 1, level);
  }
}

// Maps bands to threads in DFS postorder.
// Mapping is only allowed if descendants are not already mapped to threads.
// Mapping nested bands to threads is invalid because members of those bands
// are not necessarily permutable, and there is no guaranteed nesting between
// thread dimensions (e.g., there is no guarantee that all threads with
// threadIdx.y=0 will be executed before any thread with threadIdx.y=1).
//
// If any separation is needed for mapping reductions to full blocks,
// then do so first.
//
// If "st" has multiple children and if any of those children
// is mapped to threads, then make sure the other children
// are also mapped to threads, by fixing the thread identifiers to value zero.
// If, moreover, "st" is a sequence node and at least one of its
// children is mapped to threads, then introduce synchronization
// before and after children that are mapped to threads.
// Also add synchronization between the last child and
// the next iteration of the first child if there may be such
// a next iteration that is not already covered by synchronization
// on an outer node.
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
    if (n > 0) {
      for (size_t i = 0; i < nChildren; ++i) {
        fixThreadsBelow(*this, children[i], nInner[i]);
      }
    }
    if (needSync) {
      insertBestSyncInSeq(st);
    }
  }

  if (auto band = st->elemAs<detail::ScheduleTreeElemBand>()) {
    if (n == 0) {
      // If children were not mapped to threads, the current band can be mapped.
      // First, map the coincidence and reduction dimension to threads.
      // Then, if some threads were mapped, fix unused thread dimensions to 0
      // because we cannot map parent bands anyway.
      auto nMapped = mapToThreads(st);
      if (nMapped > 0) {
        markUnroll(scop_->scheduleRoot(), st, unroll);
        return nMapped;
      }
    } else if (anyNonCoincidentMember(band)) {
      // If children were mapped to threads, and this band has a non-coincident
      // member, insert a synchronization after its last child.
      // The node must have children if some of them were mapped to threads,
      // double-check.  Note that a band node has at most one child.
      TC_CHECK_EQ(st->numChildren(), 1u);
      // The mapping should be always complete, double-check.
      TC_CHECK_EQ(n, numThreads.view.size());
      scop_->insertSyncAfter(st->child({0}));
    }
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
  updateTopLevelContext(root, scop->context().from_params());

  tc::Grid grid = mappedScop.numBlocks;
  tc::Block block = mappedScop.numThreads;
  std::tie(grid, block) = tightenLaunchBounds(*scop, grid, block);
  auto res = MappedScop::makeMappedScop(
      std::move(scop),
      grid,
      block,
      mappedScop.unroll,
      mappedScop.useReadOnlyCache);
  res->insertMappingContext();

  LOG_IF(INFO, FLAGS_debug_tc_mapper)
      << "Codegen with tightened bounds [blocks:" << grid
      << ", threads:" << block << "] for tree:\n"
      << *res->schedule();

  return res;
}
} // namespace

// Before generating code, make a copy of the scop and insert
// the context of the original scop as top-level
// context node in schedule tree.
std::tuple<std::string, tc::Grid, tc::Block> MappedScop::codegen(
    const std::string& specializedName) const {
  validate(schedule());

  auto mappedScopForCodegen = makeSpecializedMappedScop(*this);

  std::stringstream code;
  code << code::cpp::boundsAsTemplate << code::c::types << code::c::defines;
  code << code::c::warpSyncFunctions;
  code << std::endl;
  if (useReadOnlyCache) {
    code << code::cuda::ldg;
  }
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

// Split out a single reduction tile (in the directions other than
// the reduction) and insert reduction synchronizations outside this tile.
// Return a pointer to the split off tile.
detail::ScheduleTree* MappedScop::splitOutReductionTileAndInsertSyncs(
    detail::ScheduleTree* band) {
  using namespace polyhedral::detail;
  size_t n = numThreads.view.size();

  // The current band contains only full blocks.
  // Split off a band that iterates over these blocks,
  // such that only a single block gets mapped to thread identifiers.
  // The mapping to thread identifier X is allowed to iterate
  // over multiple blocks, so this direction is not tiled.
  std::vector<size_t> sizes(n);
  for (size_t i = 1; i < n; ++i) {
    sizes[n - 1 - i] = numThreads.view[i];
  }
  sizes[n - 1] = 0;
  bandTile(band, sizes, TileOptions::ScaleTileLoops);

  // Insert synchronization outside the single block.
  auto child = band->child({0});
  for (auto updateId : reductionBandUpdates_.at(band).ids) {
    scop_->insertReductionSync1D(child, updateId);
  }
  return child;
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
      generic.proto.unroll(),
      cudaOptions.proto().use_readonly_cache()));
  auto& scop = mappedScop->scop_;

  // 1a. Optionally specialize before scheduling...
  if (generic.proto.fix_parameters_before_scheduling()) {
    scop->specializeToContext();
  }

  // 2. Schedule
  scop = Scop::makeScheduled(*scop, generic.outerScheduleOptions);

  // 3. Tile
  TC_CHECK_LT(0u, generic.tiling.size())
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

  // 5. Insert mapping context
  mappedScop->insertMappingContext();

  // 6. Map to threads
  if (outerBand->numChildren() > 0) {
    TC_CHECK_EQ(1u, outerBand->numChildren());
    // 6.1. Optionally detect reductions while mapping to threads

    if (generic.proto.match_library_calls()) {
      mappedScop->detectReductions(outerBand->child({0}));
    }
    auto child = outerBand->child({0});
    size_t numMappedInnerThreads =
        mappedScop->mapInnermostBandsToThreads(child);
    fixThreadsBelow(*mappedScop, outerBand, numMappedInnerThreads);
    LOG_IF(INFO, FLAGS_debug_tc_mapper)
        << "After mapping to threads:" << std::endl
        << *mappedScop->schedule();
  }

  // 7. Map to blocks
  mappedScop->mapToBlocksAndScaleBand(
      outerBand, generic.tiling.extractVector());
  LOG_IF(INFO, FLAGS_debug_tc_mapper) << "After mapping to blocks:" << std::endl
                                      << *mappedScop->schedule();

  // 8. Promote to shared memory below the loops mapped to blocks.
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

  // 9. Promote to registers below the loops mapped to threads.
  if (cudaOptions.proto().use_private_memory()) {
    promoteToRegistersBelowThreads(*mappedScop, -1ull);
  }

  LOG_IF(INFO, FLAGS_debug_tc_mapper)
      << "After outerBlockInnerThread strategy:" << std::endl
      << *mappedScop->schedule();

  return mappedScop;
}

} // namespace polyhedral
} // namespace tc
