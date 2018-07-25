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
#include "tc/core/polyhedral/cuda/memory_promotion_heuristic.h"

#include <glog/logging.h>

#include "tc/core/polyhedral/cuda/mapped_scop.h"
#include "tc/core/polyhedral/cuda/mapping_types.h"
#include "tc/core/polyhedral/exceptions.h"
#include "tc/core/polyhedral/memory_promotion.h"
#include "tc/core/polyhedral/schedule_tree.h"
#include "tc/core/polyhedral/schedule_tree_matcher.h"
#include "tc/core/polyhedral/schedule_utils.h"
#include "tc/core/polyhedral/unroll.h"

#include <algorithm>
#include <numeric>
#include <sstream>
#include <type_traits>

namespace tc {
namespace polyhedral {
namespace {

// Map global<->shared copy bands to threads, starting from the innermost
// loop as it iterates over the last subscript and will result in coalescing.
void mapCopiesToThreads(MappedScop& mscop, bool unroll) {
  using namespace detail;

  // Find all filters with reads from or writes to global memory.
  auto matcher = filter(
      [](isl::union_set uset) {
        auto sets = isl::UnionAsVector<isl::union_set>(uset);
        return std::all_of(sets.begin(), sets.end(), [](isl::set s) {
          auto readId = isl::id(s.get_ctx(), std::string(kReadIdName));
          auto writeId = isl::id(s.get_ctx(), std::string(kWriteIdName));
          return s.get_tuple_id() == readId || s.get_tuple_id() == writeId;
        });
      },
      any());

  auto root = mscop.scop().scheduleRoot();
  for (auto constNode : match(matcher, root)) {
    // We need to modify the nodes and have non-const mscop.
    auto node = const_cast<ScheduleTree*>(constNode);
    if (match(filter(band()), node).size() != 1) {
      std::stringstream ss;
      ss << "read/write filter not followed by a single band" << std::endl
         << *node;
      throw promotion::PromotionLogicError(ss.str());
    }

    auto bandNode = node->child({0});
    auto band = bandNode->as<ScheduleTreeBand>();
    if (!band) {
      throw promotion::PromotionLogicError("no copy band");
    }

    // Check that we are not mapping to threads below other thread mappings.
    std::unordered_set<mapping::ThreadId, mapping::ThreadId::Hash> usedThreads;
    for (auto n : node->ancestors(root)) {
      if (isMappingTo<mapping::ThreadId>(n)) {
        throw promotion::PromotionBelowThreadsException(
            "attempted to map memory copies to threads below "
            "another thread mapping");
      }
    }

    mscop.mapThreadsBackward(bandNode);

    // Unroll if requested.
    if (unroll) {
      markUnroll(root, bandNode, mscop.unroll);
    }
  }
}

/*
 * Starting from the root, find all thread specific markers.  Use
 * DFSPreorder to make sure order is specified and consistent for tests.
 */
template <typename T>
std::vector<T> findThreadSpecificMarkers(T root) {
  using namespace tc::polyhedral::detail;
  static_assert(
      std::is_convertible<T, const ScheduleTree*>::value,
      "expecting ScheduleTree");

  return ScheduleTree::collectDFSPreorder(
      root, ScheduleTreeType::ThreadSpecificMarker);
}

/*
 * Return the thread specific markers in the tree rooted at "root"
 * that are relevant for "node".
 *
 * Every branch in the tree has exactly one thread marker.
 * If "node" appears underneath a thread marker, then return
 * that single thread marker.
 * Otherwise, return the (possibly multiple) thread markers
 * in the subtree rooted at "node".
 */
template <typename T>
std::vector<T> collectBranchMarkers(T root, T node) {
  using namespace detail;
  static_assert(
      std::is_convertible<T, const ScheduleTree*>::value,
      "expecting ScheduleTree");

  auto filterMarker = [](T tree) {
    return tree->type_ == ScheduleTreeType::ThreadSpecificMarker;
  };

  auto ancestors = node->ancestors(root);
  ancestors = functional::Filter(filterMarker, ancestors);
  if (ancestors.size() > 0) {
    return ancestors;
  }
  return findThreadSpecificMarkers(node);
}

/*
 * Check if a reference group features reuse within the "outer" schedule.
 * In particular, check that for some given point in the outer schedule and
 * some given group element, there is more than one statement instance
 * accessing the element within the point in the outer schedule.
 * In other words, check that the mapping from statement instances
 * to pairs of outer schedule points and group elements is not injective.
 */
bool hasReuseWithin(
    const TensorReferenceGroup& group,
    isl::multi_union_pw_aff outer) {
  auto map = isl::union_map::from(outer);
  map = map.range_product(group.originalAccesses());
  return !map.is_injective();
}

/*
 * Create a map that increments the "dim"-th dimension and keeps all other
 * dimensions unchanged.
 */
isl::map makeNextElementMap(isl::space setSpace, unsigned dim) {
  auto mapSpace = setSpace.map_from_set();
  auto identityMA = isl::multi_aff::identity(mapSpace);

  size_t size = identityMA.size();
  if (dim < 0 || dim >= size) {
    std::stringstream ss;
    ss << dim << "  is out of [0, " << size << ") range";
    throw promotion::OutOfRangeException(ss.str());
  }

  auto aff = identityMA.get_aff(dim);
  identityMA = identityMA.set_aff(dim, aff + 1);
  return isl::map(identityMA);
}

/*
 * Return the outermost thread mapping filter among the ancestors of "node",
 * assuming that there is at least one.
 */
const detail::ScheduleTree* findThreadMappingAncestor(
    const detail::ScheduleTree* root,
    const detail::ScheduleTree* node) {
  using namespace tc::polyhedral::detail;

  auto ancestors = node->ancestors(root);
  ancestors = functional::Filter(isMappingTo<mapping::ThreadId>, ancestors);
  if (ancestors.size() < 1) {
    throw promotion::PromotionLogicError("missing Mapping");
  }
  return ancestors[0];
}

/*
 * Should this reference group be promoted for the purpose of coalescing?
 *
 * If the reference group is not already accessed in a coalesced way,
 * then the group should be promoted.
 * If a branch is mapped to a single thread, then the accesses
 * in that branch are not considered to contribute to the usefulness
 * of promoting.
 *
 * The check for coalesced accesses is performed as follows.
 * Check if incrementing the schedule dimension mapped to
 * Thread::x results in the last tensor index being incremented as well.
 * Since accesses in the group may belong to different statements, which may
 * have different loops mapped to Thread::x, perform the check for each thread
 * mapping on the statements active at "node" (either a single ancestor,
 * or one or more descendants).
 * The iteration over the spaces is used to handle the case where
 * one of the subbranches does not access the tensor and
 * the scheduled accesses are empty.  The group is
 * accessed in a coalesced way if all references in this group are accessed in
 * a coalesced way.
 */
bool promotionImprovesCoalescing(
    const detail::ScheduleTree* root,
    const detail::ScheduleTree* node,
    const TensorReferenceGroup& group) {
  auto originalAccesses = group.originalAccesses();

  auto tensorDim = group.approximation.dim();
  auto markers = collectBranchMarkers(root, node);
  for (auto marker : markers) {
    auto mapping = findThreadMappingAncestor(root, marker);
    size_t nMappedThreads = marker->scheduleDepth(mapping);
    if (nMappedThreads == 0) {
      continue;
    }
    auto depth = marker->scheduleDepth(root);
    auto activePoints = activeDomainPoints(root, mapping);
    auto localAccesses = originalAccesses.intersect_domain(activePoints);
    auto schedule = prefixSchedule(root, marker);
    auto scheduledAccesses = localAccesses.apply_domain(schedule);
    for (auto access : isl::UnionAsVector<isl::union_map>(scheduledAccesses)) {
      auto scheduleSpace = access.get_space().domain();
      auto tensorSpace = access.get_space().range();
      auto elementToNext = makeNextElementMap(tensorSpace, tensorDim - 1);
      auto scheduleToNextX = makeNextElementMap(scheduleSpace, depth - 1);
      auto accessedByAdjacentX =
          scheduleToNextX.apply_domain(access).apply_range(access);

      if (not accessedByAdjacentX.is_subset(elementToNext)) {
        return true;
      }
    }
  }
  return false;
}

/*
 * Returns the union of all mapping filters to "MappingType" in "scop".
 */
template <typename MappingType>
isl::union_set collectMappingsTo(const Scop& scop) {
  auto root = scop.scheduleRoot();
  auto domain = scop.domain();
  auto mappingFilters =
      detail::ScheduleTree::collect(root, detail::ScheduleTreeType::Mapping);
  mappingFilters = functional::Filter(isMappingTo<MappingType>, mappingFilters);
  auto mapping = isl::union_set::empty(domain.get_space());
  for (auto mf : mappingFilters) {
    auto filterNode = mf->as<detail::ScheduleTreeMapping>();
    auto filter = filterNode->filter_.intersect(activeDomainPoints(root, mf));
    mapping = mapping.unite(filterNode->filter_);
  }
  return mapping;
}

/*
 * Check that only unrolled loops may appear in access subscripts.
 * Because the scoping point can be above a branching tree, descend into each
 * leaf of the subtree below the scoping point.  For each leaf, construct an
 * affine multi-expression containing only those band members between the
 * scoping point and the leaf that are fully unrolled.
 *
 * Within each instance of the scope loops, check that loops that are either
 * unrolled or mapped to threads access a single tensor element in the group
 * (other loop indices will then not appear in the subscripts, making register
 * promotion possible).  In other words, check that the relation between the
 * flat product of prefix, thread-mapped, and unrolled loop indices and
 * accessed elements is single-valued.
 *
 * If band members are mapped to blocks(threads), they may still correspond to
 * loops in the code in cases where the number of blocks(threads) is less than
 * the extent of the band member.  If there is no "unroll" flag on these
 * members, we require that they not appear in the access subscripts similarly
 * to regular loops.  This is slightly more conservative than necessary because
 * the actual generated loop iterators may disappear from the access after
 * mapping to threads in cases where they are used with a modulo that is less
 * than the number of blocks(threads).  Precise analysis requires non-trivial
 * schedule manipulations or explicit tiling by grid(block) sizes before
 * mapping to blocks(threads).
 *
 * TODO: note that if a group is formed from partially overlapping references,
 * one must consider per-reference access relation for single-valuedness as
 * different references may have different values, but all of them remain
 * independent of non-unrolled loop iterators.
 */
bool accessSubscriptsAreUnrolledLoops(
    const TensorReferenceGroup& group,
    const detail::ScheduleTree* root,
    const detail::ScheduleTree* scope,
    isl::multi_union_pw_aff outerSchedule) {
  using namespace detail;

  auto nodes = ScheduleTree::collect(scope);
  auto leaves = functional::Filter(
      [](const ScheduleTree* tree) { return tree->numChildren() == 0; }, nodes);

  auto domainNode = root->as<detail::ScheduleTreeDomain>();
  TC_CHECK(domainNode);
  auto domain = domainNode->domain_;

  // Descend into every leaf.
  for (auto leaf : leaves) {
    auto ancestors = leaf->ancestors(root);
    ancestors.push_back(leaf);
    auto subdomain = activeDomainPointsBelow(root, leaf);

    auto unrolledDims = isl::union_pw_aff_list(leaf->ctx_, 1);
    for (auto node : ancestors) {
      auto band = node->as<detail::ScheduleTreeBand>();
      if (!band) {
        continue;
      }

      isl::multi_union_pw_aff schedule = band->mupa_;
      schedule = schedule.intersect_domain(subdomain);
      for (size_t i = 0, e = band->nMember(); i < e; ++i) {
        if (!band->unroll_[i]) {
          continue;
        }
        unrolledDims = unrolledDims.add(schedule.get_union_pw_aff(i));
      }
    }

    auto space =
        subdomain.get_space().add_unnamed_tuple_ui(unrolledDims.size());
    auto unrolledDimsMupa = isl::multi_union_pw_aff(space, unrolledDims);

    // It is possible that no loops are unrolled, in which case
    // unrolledDimsMupa is zero-dimensional and needs an explicit domain
    // to be convertible to a union_map.
    unrolledDimsMupa =
        unrolledDimsMupa.intersect_domain(group.originalAccesses().domain());

    auto accesses = group.originalAccesses();
    auto schedule = outerSchedule.flat_range_product(unrolledDimsMupa);
    accesses = accesses.apply_domain(isl::union_map::from(schedule));

    if (!accesses.is_single_valued()) {
      return false;
    }
  }

  return true;
}

/*
 * Check if the given "group" can be promoted to registers for the given
 * mapping to thread identifiers and within the given outer schedule.
 *
 * In particular, all tensor subscripts that may appear in the promoted access
 * must be either unrolled loops or thread identifiers and the
 * same tensor element should never be accessed by two different threads
 * within the same iteration of the outer schedule.
 * The second test is performed by checking that there is only a single
 * thread associated to a given pair of tensor element and outer schedule
 * iteration.
 */
bool isPromotableToRegistersBelow(
    const TensorReferenceGroup& group,
    const detail::ScheduleTree* root,
    const detail::ScheduleTree* scope,
    isl::multi_union_pw_aff outer,
    isl::multi_union_pw_aff thread) {
  if (!accessSubscriptsAreUnrolledLoops(
          group, root, scope, outer.flat_range_product(thread))) {
    return false;
  }

  auto originalAccesses = group.originalAccesses();
  auto map = isl::union_map::from(outer);
  map = map.range_product(originalAccesses);
  map = map.apply_domain(isl::union_map::from(thread));

  return map.is_injective();
}

/*
 * Starting from the root, find bands where depth is reached.  If zero depth is
 * requested, insert a zero-dimensional band node below the root (or the
 * context node if present) and return it.  Otherwise, use
 * DFSPreorder to make sure order is specified and consistent for tests.
 */
std::vector<detail::ScheduleTree*> bandsContainingScheduleDepth(
    detail::ScheduleTree* root,
    size_t depth) {
  using namespace tc::polyhedral::detail;

  if (depth == 0) {
    return {insertTopLevelEmptyBand(root)};
  }

  auto bands =
      ScheduleTree::collectDFSPreorder(root, detail::ScheduleTreeType::Band);
  std::function<bool(ScheduleTree * st)> containsDepth = [&](ScheduleTree* st) {
    auto depthBefore = st->scheduleDepth(root);
    auto band = st->as<ScheduleTreeBand>();
    auto depthAfter = depthBefore + band->nMember();
    return depthBefore < depth && depthAfter >= depth;
  };
  return functional::Filter(containsDepth, bands);
}

/*
 * Split bands so that the "depth"-th dimension is always the last in some
 * band.  Return such bands.
 */
std::vector<detail::ScheduleTree*> bandsSplitAfterDepth(
    const std::vector<detail::ScheduleTree*>& bands,
    detail::ScheduleTree* root,
    size_t depth) {
  using namespace tc::polyhedral::detail;

  std::function<ScheduleTree*(ScheduleTree*)> splitAtDepth =
      [&](ScheduleTree* st) {
        auto nMember = st->as<ScheduleTreeBand>()->nMember();
        auto scheduleDepth = st->scheduleDepth(root);
        auto depthAfter = scheduleDepth + nMember;
        return depthAfter == depth ? st
                                   : bandSplit(root, st, depth - scheduleDepth);
      };
  return functional::Map(splitAtDepth, bands);
}

/*
 * Check if "node" or any of its ancestors until "root" are thread mappings.
 */
bool isInThreadMappedScope(
    const detail::ScheduleTree* root,
    const detail::ScheduleTree* node) {
  auto ancestors = node->ancestors(root);
  ancestors.push_back(node);
  for (auto ancestor : ancestors) {
    if (isMappingTo<mapping::ThreadId>(ancestor)) {
      return true;
    }
  }
  return false;
}

/*
 * Promote to shared memory in "scop" below "node".  Use at most
 * "remainingMemory" bytes, and update the variable to reflect the amount of
 * available shared memory remaining after promotion.
 */
void promoteToSharedBelow(
    Scop& scop,
    detail::ScheduleTree* node,
    size_t& remainingMemory) {
  auto root = scop.scheduleRoot();

  // Promotion to shared below threads does not make sense because the computed
  // groups would be specific to threads thus not benefiting from coalescing or
  // inter-thread communication through shared memory (use registers instead).
  if (isInThreadMappedScope(root, node)) {
    throw promotion::IncorrectScope(
        "shared memory promotion below thread mapping");
  }
  // Children of a sequence/set band must be filters, but promotion would
  // insert an extension node.
  if (node->as<detail::ScheduleTreeSequence>() ||
      node->as<detail::ScheduleTreeSet>()) {
    throw promotion::IncorrectScope("cannot promote below a sequence/set node");
  }

  auto partialSched = partialSchedule(root, node);
  auto mapping = collectMappingsTo<mapping::BlockId>(scop);

  auto groupMap = TensorReferenceGroup::accessedWithin(
      partialSched.intersect_domain(mapping), scop.body);
  // Pure affine schedule without (mapping) filters.
  auto partialSchedMupa = partialScheduleMupa(root, node);

  // Prepare groups for sorting, to have specified order necessary for
  // reproducibility and tests.
  using TensorGroupList = std::pair<isl::id, TensorGroupsInfo>;
  std::vector<TensorGroupList> groupLists(
      std::make_move_iterator(groupMap.begin()),
      std::make_move_iterator(groupMap.end()));

  // Computes the total number of references in all groups.
  auto refsCount = [](const TensorGroupsInfo& info) {
    size_t refs = 0;
    for (auto const& group : info) {
      refs += group->referenceIds().size();
    }
    return refs;
  };

  // Sort by the total number of references, then by name.  Because names are
  // guarenteed to be unique, the order is total.
  std::sort(
      groupLists.begin(),
      groupLists.end(),
      [refsCount](const TensorGroupList& l1, const TensorGroupList& l2) {
        auto r1 = refsCount(l1.second);
        auto r2 = refsCount(l2.second);
        return r1 == r2 ? l1.first.get_name() < l2.first.get_name() : r1 < r2;
      });
  for (auto& tensorGroups : groupLists) {
    auto tensorId = tensorGroups.first;
    // Sort the reference groups to prioritize groups with more references as
    // they are more likely to benefit from promotion.
    std::sort(
        tensorGroups.second.begin(),
        tensorGroups.second.end(),
        [refsCount](
            const std::unique_ptr<TensorReferenceGroup>& group1,
            const std::unique_ptr<TensorReferenceGroup>& group2) {
          return group1->referenceIds().size() > group2->referenceIds().size();
        });

    for (auto& group : tensorGroups.second) {
      auto sizes = group->approximationSizes();
      if (sizes.size() == 0) {
        throw promotion::PromotionLogicError("cannot promote a scalar");
      }
      if (sizes.back() % 2 == 0) {
        sizes.back() += 1;
      }
      auto nApproximationElements = std::accumulate(
          sizes.begin(), sizes.end(), 1, std::multiplies<size_t>());
      size_t memoryRequirement =
          nApproximationElements * scop.findArgument(tensorId).type().bytes();
      if (memoryRequirement > remainingMemory) {
        continue;
      }
      // Do not promote if the group features no reuse and is accessed in a
      // coalesced way.
      if (!hasReuseWithin(*group, partialSchedMupa) &&
          !promotionImprovesCoalescing(root, node, *group)) {
        continue;
      }

      scop.promoteGroup(
          Scop::PromotedDecl::Kind::SharedMem,
          tensorId,
          std::move(group),
          node,
          partialSched,
          true);
      remainingMemory -= memoryRequirement;
    }
  }
  scop.insertSyncsAroundCopies(node);
}

/*
 * Check if "tree" is a band node mapped to threads.  In particular, check that
 * "tree" is a band and a thread-specific node appears as its only child.
 */
inline bool isThreadMappedBand(const detail::ScheduleTree* tree) {
  return matchOne(band(threadSpecific(any())), tree) ||
      matchOne(band(threadSpecific()), tree);
}
} // namespace

/*
 * For every place in the schedule tree where schedule depth (i.e., the number
 * of preceding band members) is "depth", promote tensor reference groups to
 * shared memory if there is no thread mapping above this place.  Split bands
 * if necessary to insert promotions.
 *
 * Use at most "maxMemory" bytes.  If a groups does not fit the remaining
 * memory, do not promote it and keep looking for a smaller group.
 *
 * Only promote if the tensor elements referenced by the group are reused or
 * accessed in a non-coalesced way.
 *
 * If "unrollCopies" is set, use the unroll factor from "mscop" to unroll the
 * loops that copy values from global to shared memory and back.
 */
void promoteToSharedAtDepth(
    MappedScop& mscop,
    size_t depth,
    size_t maxMemory,
    bool unrollCopies) {
  using namespace tc::polyhedral::detail;

  auto& scop = mscop.scop();
  auto root = scop.scheduleRoot();

  // 1. Collect all bands with a member located at the given depth in the
  // overall schedule.  Make sure this is the last member of the band by
  // splitting off the subsequent members into a different band.
  auto bands = bandsContainingScheduleDepth(root, depth);
  bands = bandsSplitAfterDepth(bands, root, depth);

  // 2. For each band that ends at "depth", take decisions about promotion
  // immediately below it in the tree.  In particular, promote if the
  // approximated footprint fits into the remaining memory, and the reference
  // group either features reuse or is accessed in a non-coalesced way, or
  // both.  Do not promote if the band node is located below the thread mapping
  // as promotion to shared is not allowed in this context.
  size_t remainingMemory = maxMemory;
  for (auto bandNode : bands) {
    if (isInThreadMappedScope(root, bandNode)) {
      LOG_IF(INFO, FLAGS_debug_tc_mapper)
          << "not promoting subtree to shared because it is below "
          << "a thread mapping node";
      continue;
    }
    promoteToSharedBelow(scop, bandNode, remainingMemory);
  }

  // 3. Map copies to shared.
  mapCopiesToThreads(mscop, unrollCopies);
}

/*
 * Perform promotion to registers below the node "scope" in the schedule tree
 * of "mscop".  Throw if promotion would violate the well-formedness of the
 * schedule tree, in particular in cases of promotion immediately below
 * a set/sequence node or immediately above a thread-specific marker node.
 */
void promoteToRegistersBelow(MappedScop& mscop, detail::ScheduleTree* scope) {
  // Cannot promote below a sequence or a set node.  Promotion may insert an
  // extension node, but sequence/set must be followed by filters.
  if (scope->as<detail::ScheduleTreeSequence>() ||
      scope->as<detail::ScheduleTreeSet>()) {
    throw promotion::IncorrectScope("cannot promote under a sequence/set node");
  }
  // Cannot promote between a thread-mapped band and a thread-specific marker
  // node because the latter is used to identify thread-mapped bands as
  // immediate ancestors.
  if (scope->numChildren() == 1 &&
      scope->child({0})->as<detail::ScheduleTreeThreadSpecificMarker>()) {
    throw promotion::IncorrectScope(
        "cannot promote above a thread-specific marker node");
  }

  auto& scop = mscop.scop();
  auto root = scop.scheduleRoot();

  // Compute groups specific to threads and block by including the mappings
  // into the domain of the partials schedule.
  auto blockMapping = collectMappingsTo<mapping::BlockId>(scop);
  auto mapping =
      collectMappingsTo<mapping::ThreadId>(scop).intersect(blockMapping);
  auto schedule = partialSchedule(scop.scheduleRoot(), scope);
  auto groupMap = TensorReferenceGroup::accessedWithin(
      schedule.intersect_domain(mapping), scop.body);

  auto threadSchedule = mscop.threadMappingSchedule(mscop.schedule());
  auto blockSchedule = mscop.blockMappingSchedule(mscop.schedule());

  // Pure affine schedule without (mapping) filters.
  auto partialSchedMupa = partialScheduleMupa(root, scope);
  // Schedule with block mapping filter.
  auto partialSched =
      isl::union_map::from(partialSchedMupa).intersect_domain(blockMapping);
  // The following promotion validity and profitability checks need to be
  // performed with respect to the block mapping, so append the block schedule.
  // If the partial schedule contains it already, it will just end up with
  // identical dimensions without affecting the result of the checks.
  partialSchedMupa = partialSchedMupa.flat_range_product(blockSchedule);

  for (auto& tensorGroups : groupMap) {
    auto tensorId = tensorGroups.first;

    // TODO: sorting of groups and counting the number of promoted elements

    for (auto& group : tensorGroups.second) {
      auto sizes = group->approximationSizes();
      // No point in promoting a scalar that will go to a register anyway.
      if (sizes.size() == 0) {
        continue;
      }
      if (!isPromotableToRegistersBelow(
              *group, root, scope, partialSchedMupa, threadSchedule)) {
        continue;
      }
      // Check reuse within threads.
      auto schedule = partialSchedMupa.flat_range_product(threadSchedule);
      if (!hasReuseWithin(*group, schedule)) {
        continue;
      }

      // TODO: if something is already in shared, but reuse it within one
      // thread only, there is no point in keeping it in shared _if_ it
      // gets promoted into a register.
      scop.promoteGroup(
          Scop::PromotedDecl::Kind::Register,
          tensorId,
          std::move(group),
          scope,
          partialSched);
    }
  }

  // Return immediately if nothing was promoted.
  if (scope->numChildren() == 0 ||
      !matchOne(extension(sequence(any())), scope->child({0}))) {
    return;
  }

  // If promoting above thread mapping, insert synchronizations.
  // It is possible that promoted array elements are accessed by different
  // threads outside the current scope (either in different iterations of the
  // scope loops, or in sibling subtrees).  For now, always insert
  // synchronizations, similarly to copies to shared memory.
  //
  // TODO: The exact check for sync insertion requires the dependences between
  // the elements in the scope and those before/after the scope and a check if
  // the dependent instances belong to the same thread.
  auto ancestors = scope->ancestors(root);
  if (functional::Filter(isMappingTo<mapping::ThreadId>, ancestors).empty()) {
    scop.insertSyncsAroundSeqChildren(scope->child({0, 0}));
  }
}

/*
 * Promote to registers below "depth" schedule dimensions.  Split bands if
 * necessary to create promotion scopes.  Do not promote if it would require
 * splitting the band mapped to threads as we assume only one band can be
 * mapped.
 */
void promoteToRegistersAtDepth(MappedScop& mscop, size_t depth) {
  using namespace detail;

  auto root = mscop.scop().scheduleRoot();

  // 1. Collect all bands with a member located at the given depth in the
  // overall schedule.  Make sure this is the last member of the band by
  // splitting off the subsequent members into a different band.  Ignore bands
  // mapped to threads if splitting is required as it would break the invariant
  // of a single band being mapped to threads in a subtree.
  // TODO: allow splitting the thread-mapped bands; for example, tile them
  // explicitly with block size, use the point loops for thread mapping
  // but ignore them in depth computation.
  auto bands = bandsContainingScheduleDepth(root, depth);
  bands = functional::Filter(
      [root, depth](ScheduleTree* tree) {
        auto band = tree->as<ScheduleTreeBand>();
        return !isThreadMappedBand(tree) ||
            tree->scheduleDepth(root) + band->nMember() == depth;
      },
      bands);
  bands = bandsSplitAfterDepth(bands, root, depth);

  // 2. We don't want copies inserted between thread-mapped bands and the
  // thread-specific marker, but rather below that marker.  If any of the bands
  // are mapped to threads, take their first children as promotion scope
  // instead of the band itself.
  std::function<ScheduleTree*(ScheduleTree*)> findScope =
      [](ScheduleTree* tree) {
        return isThreadMappedBand(tree) ? tree->child({0}) : tree;
      };
  auto scopes = functional::Map(findScope, bands);

  for (auto scope : scopes) {
    promoteToRegistersBelow(mscop, scope);
  }
}

} // namespace polyhedral
} // namespace tc
