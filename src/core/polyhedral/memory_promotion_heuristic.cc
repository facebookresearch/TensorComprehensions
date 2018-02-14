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
#include "tc/core/polyhedral/memory_promotion_heuristic.h"

#include <glog/logging.h>

#include "tc/core/polyhedral/exceptions.h"
#include "tc/core/polyhedral/mapped_scop.h"
#include "tc/core/polyhedral/memory_promotion.h"
#include "tc/core/polyhedral/schedule_tree.h"
#include "tc/core/polyhedral/schedule_tree_matcher.h"
#include "tc/core/polyhedral/unroll.h"
#include "tc/core/utils/error.h"

#include <algorithm>
#include <numeric>
#include <sstream>

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
      reportError<promotion::PromotionLogicError>(ss.str());
    }

    auto bandNode = node->child({0});
    auto band = bandNode->elemAs<ScheduleTreeElemBand>();
    if (!band) {
      reportError<promotion::PromotionLogicError>("no copy band");
    }

    // Check that we are not mapping to threads below other thread mappings.
    std::unordered_set<mapping::ThreadId, mapping::ThreadId::Hash> usedThreads;
    for (auto n : node->ancestors(root)) {
      if (auto filterNode = n->elemAs<ScheduleTreeElemMappingFilter>()) {
        for (auto id : filterNode->mappingIds) {
          if (id.isThreadId()) {
            reportError<promotion::PromotionBelowThreadsException>(
                "attempted to map memory copies to threads below "
                "another thread mapping");
          }
        }
      }
    }

    // Map band dimensions to threads, in inverse order since the last member
    // iterates over the last subscript and is likely to result in coalescing.
    // Step over band members that iterate over size-1 arrays subscripts as
    // they would have been executed by a single thread.
    // If not all available thread ids are used, fix remaining to 1 thread.
    auto filter = node->elemAs<ScheduleTreeElemFilter>()->filter_;
    auto filterSets = isl::UnionAsVector<isl::union_set>(filter);
    int t = 0;
    for (int i = band->nMember() - 1; i >= 0 && t < mscop.numThreads.size();
         --i) {
      auto skip = std::all_of(
          filterSets.begin(), filterSets.end(), [&mscop, i](isl::set s) {
            auto groupId =
                s.get_space().unwrap().get_tuple_id(isl::dim_type::out);
            if (mscop.scop().promotedDecls().count(groupId) != 1) {
              std::stringstream ss;
              ss << "promoted group " << groupId << " has no declaration";
              reportError<promotion::PromotionLogicError>(ss.str());
            }
            auto decl = mscop.scop().promotedDecls().at(groupId);
            return i >= decl.sizes.size() || decl.sizes[i] == 1;
          });
      if (skip) {
        continue;
      }

      mapToParameterWithExtent(
          root, bandNode, i, mapping::ThreadId::makeId(t), mscop.numThreads[t]);
      ++t;
    }
    mscop.mapRemaining<mapping::ThreadId>(bandNode, t, mscop.numThreads.size());

    // Unroll if requested.
    if (unroll) {
      markUnroll(root, bandNode, mscop.unroll);
    }
  }
}

/*
 * Transform schedule bands into a union_map.
 * Takes all partial schedules at leaves as MUPAs (without accounting for
 * intermediate non-band nodes), transforms them into union maps and intersects
 * their domain with the filters between the root and the
 * current leaves.
 * Mapping filters are ignored.
 */
isl::union_map fullSchedule(const detail::ScheduleTree* root) {
  using namespace tc::polyhedral::detail;

  if (!root->elemAs<ScheduleTreeElemDomain>()) {
    reportError<promotion::PromotionLogicError>(
        "expected root to be a domain node");
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
    auto prefixMupa = prefixScheduleMupa(root, node);
    if (auto band = node->elemAs<ScheduleTreeElemBand>()) {
      prefixMupa = prefixMupa.flat_range_product(band->mupa_);
    }
    auto current = isl::union_map::from(prefixMupa);

    auto pathToRoot = node->ancestors(root);
    pathToRoot.push_back(node);
    for (auto n : pathToRoot) {
      if (auto filterNode = n->elemAs<ScheduleTreeElemFilter>()) {
        current = current.intersect_domain(filterNode->filter_);
      }
    }

    schedule = schedule.unite(current);
    if (!schedule.is_single_valued()) {
      std::stringstream ss;
      ss << "schedules must be single-valued " << schedule << std::endl
         << *root;
      reportError<promotion::PromotionLogicError>(ss.str());
    }
  }
  return schedule;
}

/*
 * Insert map constraints that equate first "nDims" input dimensions to newly
 * introduced parameters.
 */
isl::map fixOuterInputDimsAsParameters(isl::map map, int nDims) {
  if (nDims < 0 || nDims > map.dim(isl::dim_type::in)) {
    std::stringstream ss;
    ss << nDims << "  is out of [0, " << map.dim(isl::dim_type::in)
       << ") range";
    reportError<promotion::OutOfRangeException>(ss.str());
  }

  auto fixedMap = map;
  auto localSpace = isl::local_space(map.get_space().domain());
  auto nParams = map.dim(isl::dim_type::param);
  localSpace = localSpace.add_dims(isl::dim_type::param, nDims);
  for (int i = 0; i < nDims; ++i) {
    localSpace = localSpace.set_dim_name(
        isl::dim_type::param,
        nParams + i,
        "__tcFixerParam" + std::to_string(i));
  }
  for (int i = 0; i < nDims; ++i) {
    auto left = isl::aff(localSpace, isl::dim_type::param, nParams + i);
    auto right = isl::aff(localSpace, isl::dim_type::set, i);
    auto dom = isl::aff_set(left) == right;
    fixedMap = fixedMap.intersect_domain(dom);
  }
  return fixedMap;
}

/*
 * Check if a reference group features reuse at "depth" after applying
 * "schedule". In particular, consider first depth schedule dimensions as fixed
 * by equating them to parameters and check if the resulting relation is not
 * injective.
 */
bool hasReuse(
    const TensorReferenceGroup& group,
    isl::union_map schedule,
    size_t depth) {
  auto scheduledAccessesUMap = group.originalAccesses().apply_domain(schedule);
  auto scheduledAccessMaps =
      isl::UnionAsVector<isl::union_map>(scheduledAccessesUMap);
  return std::any_of(
      scheduledAccessMaps.begin(),
      scheduledAccessMaps.end(),
      [schedule, depth](isl::map access) {
        access = fixOuterInputDimsAsParameters(access, static_cast<int>(depth));
        return !access.is_injective();
      });
}

/*
 * Create a map that increments the "dim"-th dimension and keeps all other
 * dimensions unchanged.
 */
isl::map makeNextElementMap(isl::space setSpace, int dim) {
  if (dim < 0 || dim >= setSpace.dim(isl::dim_type::set)) {
    std::stringstream ss;
    ss << dim << "  is out of [0, " << setSpace.dim(isl::dim_type::set)
       << ") range";
    reportError<promotion::OutOfRangeException>(ss.str());
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
size_t computeThreadIdxxScheduleDepth(
    const ThreadIdxxScheduleDepthState& threadIdxxScheduleDepthState,
    isl::union_set s) {
  std::unordered_set<size_t> depths;
  for (auto p : threadIdxxScheduleDepthState) {
    if (!p.first.intersect(s).is_empty()) {
      depths.insert(p.second);
    }
  }
  if (depths.size() != 1) {
    std::stringstream ss;
    ss << "threadIdx.x depth " << (depths.size() == 0 ? "unknown" : "diverged")
       << " for " << s;
    reportError<promotion::PromotionLogicError>(ss.str());
  }
  return *depths.begin();
}

/*
 * Check if a reference group is accessed in a coalesced way.
 *
 * In particular, check if incrementing the schedule dimension mapped to
 * Thread::x results in the last tensor index being incremented as well.
 * Since accesses in the group may belong to different statements, which are
 * have different loops mapped to Thread::x, perform the check for each basic
 * map in the union of access maps taking into account which dimension is
 * mapped for a particular statement (domain of the basic map).  The group is
 * accessed in a coalesced way if all references in this group are accessed in
 * a coalesced way.
 */
bool isCoalesced(
    const ThreadIdxxScheduleDepthState& threadIdxxScheduleDepthState,
    const TensorReferenceGroup& group,
    isl::union_map schedule) {
  auto originalAccesses = group.originalAccesses();

  for (auto accessMap : isl::UnionAsVector<isl::union_map>(originalAccesses)) {
    for (auto access : accessMap.get_basic_map_list()) {
      auto tensorSpace = access.get_space().range();
      auto elementToNext = makeNextElementMap(
          tensorSpace, tensorSpace.dim(isl::dim_type::set) - 1);
      auto domainUMap = isl::union_set(isl::set(access.domain()));
      int threadIdxxDepth = computeThreadIdxxScheduleDepth(
          threadIdxxScheduleDepthState, domainUMap);
      auto partialScheduleUMap =
          schedule.intersect_domain(domainUMap.universe());
      if (partialScheduleUMap.n_map() != 1) {
        reportError<promotion::PromotionLogicError>(
            "expected single schedule space");
      }
      auto partialSchedule = isl::map::from_union_map(partialScheduleUMap);
      auto scheduleToNextX = makeNextElementMap(
          partialSchedule.get_space().range(), threadIdxxDepth);
      auto scheduledAccess = isl::map(access)
                                 .gist_domain(access.domain())
                                 .apply_domain(partialSchedule);
      auto accessedByAdjacentX = scheduleToNextX.apply_domain(scheduledAccess)
                                     .apply_range(scheduledAccess);

      if (not accessedByAdjacentX.is_subset(elementToNext)) {
        return false;
      }
    }
  }
  return true;
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
    const ThreadIdxxScheduleDepthState& threadIdxxScheduleDepthState,
    const Block& block,
    size_t depth,
    size_t maxMemory) {
  using namespace tc::polyhedral::detail;

  if (depth == 0) {
    reportError<promotion::PromotionNYI>("promotion before any band");
  }

  auto root = scop.scheduleRoot();

  // 1. Starting from the root, find bands where depth is reached.
  // Using DFSPreorder to make sure order is specified and consistent for
  // tests.
  auto bands =
      ScheduleTree::collectDFSPreorder(root, detail::ScheduleTreeType::Band);
  std::function<bool(ScheduleTree * st)> containsDepth = [&](ScheduleTree* st) {
    auto depthBefore = st->scheduleDepth(root);
    auto band = st->elemAs<ScheduleTreeElemBand>();
    auto depthAfter = depthBefore + band->nMember();
    return depthBefore < depth && depthAfter >= depth;
  };
  bands = functional::Filter(containsDepth, bands);

  // 2. Split bands so that the "depth"-th dimension is always the last in some
  // band.  Keep such bands.
  std::function<ScheduleTree*(ScheduleTree*)> splitAtDepth =
      [&](ScheduleTree* st) {
        auto nMember = st->elemAs<ScheduleTreeElemBand>()->nMember();
        auto scheduleDepth = st->scheduleDepth(root);
        auto depthAfter = scheduleDepth + nMember;
        return depthAfter == depth ? st
                                   : bandSplit(root, st, depth - scheduleDepth);
      };
  bands = functional::Map(splitAtDepth, bands);

  // 3. Compute full schedule without mapping filters.  The filters would make
  // it impossible to test for coalescing by incrementing a member of a band as
  // only the values divisible by grid or block size pass through the filter.
  auto fullSched = fullSchedule(root);

  // 4. For each band that ends at "depth", take decisions about promotion
  // immediately below it in the tree.  In particular, promote if the
  // approximated footprint fits into the remaining memory, and the reference
  // group either features reuse or is accessed in a non-coalesced way, or
  // both.
  size_t remainingMemory = maxMemory;
  for (auto bandNode : bands) {
    auto groupMap = TensorReferenceGroup::accessedBySubtree(bandNode, scop);
    auto activeStmts = activeStatements(root, bandNode);
    auto partialSched = partialSchedule(root, bandNode);

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
          reportError<promotion::PromotionLogicError>(
              "cannot promote a scalar");
        }
        if (sizes.back() % 2 == 0) {
          sizes.back() += 1;
        }
        auto nApproximationElements = std::accumulate(
            sizes.begin(), sizes.end(), 1, std::multiplies<size_t>());
        auto memoryRequirement =
            nApproximationElements * scop.findArgument(tensorId).type().bytes();
        if (memoryRequirement > remainingMemory) {
          continue;
        }
        // Do not promote if the group features no reuse and is accessed in a
        // coalesced way.
        if (!hasReuse(*group, fullSched, depth) &&
            isCoalesced(threadIdxxScheduleDepthState, *group, fullSched)) {
          continue;
        }

        scop.promoteGroupToShared(
            tensorId,
            std::move(group),
            bandNode,
            activeStmts,
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
    const ThreadIdxxScheduleDepthState& threadIdxxScheduleDepthState,
    size_t depth,
    size_t sharedMemorySize,
    bool unrollCopies) {
  // 1. Promote using heuristic.
  promoteToSharedGreedy(
      mscop.scop(),
      threadIdxxScheduleDepthState,
      mscop.numThreads,
      depth,
      sharedMemorySize);

  // 2. Map copies to shared, state by copy
  mapCopiesToThreads(mscop, unrollCopies);
}

} // namespace polyhedral
} // namespace tc
