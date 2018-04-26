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
#include "tc/core/polyhedral/exceptions.h"
#include "tc/core/polyhedral/memory_promotion.h"
#include "tc/core/polyhedral/schedule_tree.h"
#include "tc/core/polyhedral/schedule_tree_matcher.h"
#include "tc/core/polyhedral/unroll.h"

#include <algorithm>
#include <numeric>
#include <sstream>
#include <type_traits>

namespace tc {
namespace polyhedral {
namespace {

/*
 * Is "tree" a mapping filter that maps a thread identifier?
 */
bool isThreadMapping(const detail::ScheduleTree* tree) {
  using namespace detail;

  if (auto filterNode = tree->elemAs<ScheduleTreeElemMappingFilter>()) {
    for (auto id : filterNode->mappingIds) {
      if (id.isThreadId()) {
        return true;
      }
    }
  }
  return false;
}

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
    auto band = bandNode->elemAs<ScheduleTreeElemBand>();
    if (!band) {
      throw promotion::PromotionLogicError("no copy band");
    }

    auto ctx = node->ctx_;
    insertNodeBelow(
        bandNode, detail::ScheduleTree::makeThreadSpecificMarker(ctx));

    // Check that we are not mapping to threads below other thread mappings.
    std::unordered_set<mapping::ThreadId, mapping::ThreadId::Hash> usedThreads;
    for (auto n : node->ancestors(root)) {
      if (isThreadMapping(n)) {
        throw promotion::PromotionBelowThreadsException(
            "attempted to map memory copies to threads below "
            "another thread mapping");
      }
    }

    // Map band dimensions to threads, in inverse order since the last member
    // iterates over the last subscript and is likely to result in coalescing.
    // Step over band members that iterate over size-1 arrays subscripts as
    // they would have been executed by a single thread.
    // If not all available thread ids are used, fix remaining to 1 thread.
    auto filter = node->elemAs<ScheduleTreeElemFilter>()->filter_;
    auto filterSets = isl::UnionAsVector<isl::union_set>(filter);
    size_t t = 0;
    for (int i = band->nMember() - 1;
         i >= 0 && t < mscop.numThreads.view.size();
         --i) {
      auto skip = std::all_of(
          filterSets.begin(), filterSets.end(), [&mscop, i](isl::set s) {
            auto groupId =
                s.get_space().unwrap().get_tuple_id(isl::dim_type::out);
            if (mscop.scop().promotedDecls().count(groupId) != 1) {
              std::stringstream ss;
              ss << "promoted group " << groupId << " has no declaration";
              throw promotion::PromotionLogicError(ss.str());
            }
            auto decl = mscop.scop().promotedDecls().at(groupId);
            return static_cast<size_t>(i) >= decl.sizes.size() ||
                decl.sizes[i] == 1;
          });
      if (skip) {
        continue;
      }

      mapToParameterWithExtent(
          root,
          bandNode,
          i,
          mapping::ThreadId::makeId(t),
          mscop.numThreads.view[t]);
      ++t;
    }
    mscop.mapRemaining<mapping::ThreadId>(bandNode, t);

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
 * Transform schedule bands into a union_map.
 * Takes all partial schedules at leaves as MUPAs (without accounting for
 * intermediate non-band nodes), intersects
 * their domain with the filters between the root and the
 * current leaves and transforms them into union maps.
 * Mapping filters are ignored.
 */
isl::union_map fullSchedule(const detail::ScheduleTree* root) {
  using namespace tc::polyhedral::detail;

  if (!root->elemAs<ScheduleTreeElemDomain>()) {
    throw promotion::PromotionLogicError("expected root to be a domain node");
  }

  std::function<bool(const ScheduleTree* tree)> isLeaf =
      [](const ScheduleTree* tree) { return tree->numChildren() == 0; };

  // Find all innermost nodes.
  auto leaves = functional::Filter(isLeaf, ScheduleTree::collect(root));

  // Take a union of partial schedules of the innermost nodes.  Because they
  // are innermost, the partial schedule can no longer be affected by deeper
  // nodes and hence is full.
  auto schedule = isl::union_map::empty(
      root->elemAs<ScheduleTreeElemDomain>()->domain_.get_space());
  for (auto node : leaves) {
    auto domain = root->elemAs<ScheduleTreeElemDomain>()->domain_;
    auto prefixMupa = prefixScheduleMupa(root, node);
    if (auto band = node->elemAs<ScheduleTreeElemBand>()) {
      prefixMupa = prefixMupa.flat_range_product(band->mupa_);
    }

    auto pathToRoot = node->ancestors(root);
    pathToRoot.push_back(node);
    for (auto n : pathToRoot) {
      if (auto filterNode = n->elemAs<ScheduleTreeElemFilter>()) {
        domain = domain.intersect(filterNode->filter_);
      }
    }

    prefixMupa = prefixMupa.intersect_domain(domain);

    schedule = schedule.unite(isl::union_map::from(prefixMupa));
    if (!schedule.is_single_valued()) {
      std::stringstream ss;
      ss << "schedules must be single-valued " << schedule << std::endl
         << *root;
      throw promotion::PromotionLogicError(ss.str());
    }
  }
  return schedule;
}

/*
 * Insert map constraints that equate first "nDims" input dimensions to newly
 * introduced parameters.
 */
isl::map fixOuterInputDimsAsParameters(isl::map map, unsigned nDims) {
  if (nDims < 0 || nDims > map.dim(isl::dim_type::in)) {
    std::stringstream ss;
    ss << nDims << "  is out of [0, " << map.dim(isl::dim_type::in)
       << ") range";
    throw promotion::OutOfRangeException(ss.str());
  }

  auto fixedMap = map;
  auto localSpace = isl::local_space(map.get_space().domain());
  auto nParams = map.dim(isl::dim_type::param);
  localSpace = localSpace.add_dims(isl::dim_type::param, nDims);
  for (unsigned i = 0; i < nDims; ++i) {
    localSpace = localSpace.set_dim_name(
        isl::dim_type::param,
        nParams + i,
        "__tcFixerParam" + std::to_string(i));
  }
  for (unsigned i = 0; i < nDims; ++i) {
    auto left = isl::aff(localSpace, isl::dim_type::param, nParams + i);
    auto right = isl::aff(localSpace, isl::dim_type::set, i);
    auto dom = isl::aff_set(left) == right;
    fixedMap = fixedMap.intersect_domain(dom);
  }
  return fixedMap;
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
  if (dim < 0 || dim >= setSpace.dim(isl::dim_type::set)) {
    std::stringstream ss;
    ss << dim << "  is out of [0, " << setSpace.dim(isl::dim_type::set)
       << ") range";
    throw promotion::OutOfRangeException(ss.str());
  }

  auto mapSpace = setSpace.map_from_set();
  auto identityMA = isl::multi_aff::identity(mapSpace);
  auto aff = identityMA.get_aff(dim);
  identityMA = identityMA.set_aff(dim, aff + 1);
  return isl::map(identityMA);
}

// Obtain the depth of the schedule dimension that was mapped to threadIdx.x
// for the domain elements identified by "s".  Assumes the depth is the same
// for all these elements.
size_t computeThreadIdxXScheduleDepth(
    const ThreadIdxXScheduleDepthState& threadIdxXScheduleDepthState,
    isl::union_set s) {
  std::unordered_set<size_t> depths;
  for (auto p : threadIdxXScheduleDepthState) {
    if (!p.first.intersect(s).is_empty()) {
      depths.insert(p.second);
    }
  }
  if (depths.size() != 1) {
    std::stringstream ss;
    ss << "threadIdx.x depth " << (depths.size() == 0 ? "unknown" : "diverged")
       << " for " << s;
    throw promotion::PromotionLogicError(ss.str());
  }
  return *depths.begin();
}

/*
 * Return the outermost thread mapping filter among the ancestors of "node",
 * assuming that there is at least one.
 */
const detail::ScheduleTree* findThreadMappingAncestor(
    const detail::ScheduleTree* root,
    const detail::ScheduleTree* node) {
  auto ancestors = node->ancestors(root);
  ancestors = functional::Filter(isThreadMapping, ancestors);
  if (ancestors.size() < 1) {
    throw promotion::PromotionLogicError("missing MappingFilter");
  }
  return ancestors[0];
}

/*
 * Should this reference group be promoted for the purpose of coalescing?
 *
 * If the reference group is not already accessed in a coalesced way,
 * then the group should be promoted.
 * The check for coalesced accesses is performed as follows.
 * Check if incrementing the schedule dimension mapped to
 * Thread::x results in the last tensor index being incremented as well.
 * Since accesses in the group may belong to different statements, which may
 * have different loops mapped to Thread::x, perform the check for each basic
 * map in the union of access maps taking into account which dimension is
 * mapped for a particular statement (domain of the basic map).  The group is
 * accessed in a coalesced way if all references in this group are accessed in
 * a coalesced way.
 */
bool promotionImprovesCoalescing(
    const ThreadIdxXScheduleDepthState& threadIdxXScheduleDepthState,
    const TensorReferenceGroup& group,
    isl::union_map schedule,
    isl::union_set activePoints) {
  auto originalAccesses = group.originalAccesses();

  for (auto accessMap : isl::UnionAsVector<isl::union_map>(originalAccesses)) {
    for (auto access : accessMap.get_basic_map_list()) {
      auto tensorSpace = access.get_space().range();
      auto elementToNext = makeNextElementMap(
          tensorSpace, tensorSpace.dim(isl::dim_type::set) - 1);
      auto domainUMap = isl::union_set(isl::set(access.domain()));
      int threadIdxXDepth = computeThreadIdxXScheduleDepth(
          threadIdxXScheduleDepthState, domainUMap.intersect(activePoints));
      auto partialScheduleUMap =
          schedule.intersect_domain(domainUMap.universe());
      if (partialScheduleUMap.n_map() != 1) {
        throw promotion::PromotionLogicError("expected single schedule space");
      }
      auto partialSchedule = isl::map::from_union_map(partialScheduleUMap);
      auto scheduleToNextX = makeNextElementMap(
          partialSchedule.get_space().range(), threadIdxXDepth);
      auto scheduledAccess = isl::map(access).apply_domain(partialSchedule);
      auto accessedByAdjacentX = scheduleToNextX.apply_domain(scheduledAccess)
                                     .apply_range(scheduledAccess);

      if (not accessedByAdjacentX.is_subset(elementToNext)) {
        return true;
      }
    }
  }
  return false;
}

/*
 * Check if the given "group" can be promoted to registers for the given active
 * domain points under full "schedule" where "nThreads" consecutive dimensions
 * at "depth"
 * are mapped to threads (the innermost of them being mapped to thread x).
 *
 * In particular, the group's footprint must contain only one element and the
 * same tensor element should never be accessed by two different threads.
 */
bool isPromotableToRegisterBelowThreads(
    const TensorReferenceGroup& group,
    isl::union_map schedule,
    size_t depth,
    size_t nThreads) {
  auto originalAccesses = group.originalAccesses();

  // Return early if more than one element needs to be stored in registers.
  // TODO: support arrays in registers if they are only accessed with constant
  // subscripts, e.g. if the inner loops are fully unrolled.
  auto sizes = group.approximationSizes();
  auto nElements =
      std::accumulate(sizes.begin(), sizes.end(), 1, std::multiplies<size_t>());
  if (nElements != 1) {
    return false;
  }

  auto scheduledAccesses = originalAccesses.apply_domain(schedule);

  // Scheduled accesses contain maps from schedule dimensions to tensor
  // subscripts.  Compute the relation between the schedule dimensions
  // mapped to threads and tensor subscripts by first removing dimensions
  // following the one mapped to thread x (last one assuming inverse mapping
  // order), then by equating all dimensions not mapped to threads to
  // parameters.  Promotion to registers is only allowed if the resulting
  // relation is injective, i.e. the same tensor element is never accessed by
  // more than one thread.  Note that our current check is overly conservative
  // because different values of schedule dimension may get mapped to the same
  // thread, in which case they could access the same tensor element.
  for (auto sa : isl::UnionAsVector<isl::union_map>(scheduledAccesses)) {
    sa = sa.project_out(
        isl::dim_type::in, depth, sa.dim(isl::dim_type::in) - depth);
    sa = fixOuterInputDimsAsParameters(sa, depth - nThreads);
    if (!sa.is_injective()) {
      return false;
    }
  }

  return true;
}

/*
 * Starting from the root, find bands where depth is reached.  Using
 * DFSPreorder to make sure order is specified and consistent for tests.
 */
std::vector<detail::ScheduleTree*> bandsContainingScheduleDepth(
    detail::ScheduleTree* root,
    size_t depth) {
  using namespace tc::polyhedral::detail;

  auto bands =
      ScheduleTree::collectDFSPreorder(root, detail::ScheduleTreeType::Band);
  std::function<bool(ScheduleTree * st)> containsDepth = [&](ScheduleTree* st) {
    auto depthBefore = st->scheduleDepth(root);
    auto band = st->elemAs<ScheduleTreeElemBand>();
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
        auto nMember = st->elemAs<ScheduleTreeElemBand>()->nMember();
        auto scheduleDepth = st->scheduleDepth(root);
        auto depthAfter = scheduleDepth + nMember;
        return depthAfter == depth ? st
                                   : bandSplit(root, st, depth - scheduleDepth);
      };
  return functional::Map(splitAtDepth, bands);
}

/*
 * For every place in the schedule tree where schedule depth (i.e., the number
 * of preceding band members) is "depth", promote tensor reference groups to
 * shared memory.  Split bands if necessary to insert promotions.
 *
 * Use at most "maxMemory" bytes.  If a groups does not fit the remaining
 * memory, do not promote it and keep looking for a smaller group.
 *
 * Only promote if the tensor elements referenced by the group are reused or
 * accessed in a non-coalesced way.
 */
void promoteToSharedGreedy(
    Scop& scop,
    const ThreadIdxXScheduleDepthState& threadIdxXScheduleDepthState,
    const Block& block,
    size_t depth,
    size_t maxMemory) {
  using namespace tc::polyhedral::detail;

  if (depth == 0) {
    throw promotion::PromotionNYI("promotion before any band");
  }

  auto root = scop.scheduleRoot();

  // 1. Collect all bands with a member located at the given depth in the
  // overall schedule.  Make sure this is the last member of the band by
  // splitting off the subsequent members into a different band.
  auto bands = bandsContainingScheduleDepth(root, depth);
  bands = bandsSplitAfterDepth(bands, root, depth);

  // 2. Compute full schedule without mapping filters.  The filters would make
  // it impossible to test for coalescing by incrementing a member of a band as
  // only the values divisible by grid or block size pass through the filter.
  auto fullSched = fullSchedule(root);

  // 3. For each band that ends at "depth", take decisions about promotion
  // immediately below it in the tree.  In particular, promote if the
  // approximated footprint fits into the remaining memory, and the reference
  // group either features reuse or is accessed in a non-coalesced way, or
  // both.
  size_t remainingMemory = maxMemory;
  for (auto bandNode : bands) {
    auto groupMap = TensorReferenceGroup::accessedBySubtree(bandNode, scop);
    auto partialSched = partialSchedule(root, bandNode);
    // Pure affine schedule without (mapping) filters.
    auto partialSchedMupa = partialScheduleMupa(root, bandNode);
    auto activePoints = activeDomainPoints(root, bandNode);

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
            return group1->referenceIds().size() >
                group2->referenceIds().size();
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
            !promotionImprovesCoalescing(
                threadIdxXScheduleDepthState,
                *group,
                fullSched,
                activePoints)) {
          continue;
        }

        scop.promoteGroup(
            Scop::PromotedDecl::Kind::SharedMem,
            tensorId,
            std::move(group),
            bandNode,
            partialSched,
            true);
        remainingMemory -= memoryRequirement;
      }
    }
    scop.insertSyncsAroundCopies(bandNode);
  }
}
} // namespace

void promoteGreedilyAtDepth(
    MappedScop& mscop,
    const ThreadIdxXScheduleDepthState& threadIdxXScheduleDepthState,
    size_t depth,
    size_t sharedMemorySize,
    bool unrollCopies) {
  // 1. Promote using heuristic.
  promoteToSharedGreedy(
      mscop.scop(),
      threadIdxXScheduleDepthState,
      mscop.numThreads,
      depth,
      sharedMemorySize);

  // 2. Map copies to shared, state by copy
  mapCopiesToThreads(mscop, unrollCopies);
}

// Promote at the positions of the thread specific markers.
void promoteToRegistersBelowThreads(Scop& scop, size_t nRegisters) {
  using namespace tc::polyhedral::detail;

  auto root = scop.scheduleRoot();

  auto fullSched = fullSchedule(root);
  {
    auto markers = findThreadSpecificMarkers(root);

    for (auto marker : markers) {
      auto partialSched = prefixSchedule(root, marker);
      // Pure affine schedule without (mapping) filters.
      auto partialSchedMupa = prefixScheduleMupa(root, marker);

      auto depth = marker->scheduleDepth(root);

      // Thread mapping filters are inserted immediately above the members
      // mapped to threads.  The number of intermediate band members
      // is therefore equal to the number of mapped thread identifiers.
      auto mapping = findThreadMappingAncestor(root, marker);
      size_t nMappedThreads = marker->scheduleDepth(mapping);

      auto groupMap = TensorReferenceGroup::accessedBySubtree(marker, scop);
      for (auto& tensorGroups : groupMap) {
        auto tensorId = tensorGroups.first;

        // TODO: sorting of groups and counting the number of promoted elements

        for (auto& group : tensorGroups.second) {
          auto sizes = group->approximationSizes();
          // No point in promoting a scalar that will go to a register anyway.
          if (sizes.size() == 0) {
            continue;
          }
          if (!isPromotableToRegisterBelowThreads(
                  *group, fullSched, depth, nMappedThreads)) {
            continue;
          }
          if (!hasReuseWithin(*group, partialSchedMupa)) {
            continue;
          }
          // TODO: if something is already in shared, but reuse it within one
          // thread only, there is no point in keeping it in shared _if_ it
          // gets promoted into a register.
          scop.promoteGroup(
              Scop::PromotedDecl::Kind::Register,
              tensorId,
              std::move(group),
              marker,
              partialSched);
        }
      }
    }
  }
}

} // namespace polyhedral
} // namespace tc
