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

#include "tc/external/isl.h"

#include "tc/core/check.h"
#include "tc/core/constants.h"
#include "tc/core/polyhedral/domain_types.h"
#include "tc/core/polyhedral/schedule_tree.h"
#include "tc/core/polyhedral/schedule_tree_elem.h"

#include <functional>
#include <unordered_map>
#include <vector>

namespace tc {
namespace polyhedral {
////////////////////////////////////////////////////////////////////////////////
//                 ScheduleTree utility functions, out-of-class
////////////////////////////////////////////////////////////////////////////////

// Starting from the "start" ScheduleTree, iteratively traverse the subtree
// using the "next" function and collect all nodes along the way.
// Stop when "next" returns nullptr.
// The returned vector begins with "start".
std::vector<detail::ScheduleTree*> collectScheduleTreesPath(
    std::function<detail::ScheduleTree*(detail::ScheduleTree*)> next,
    detail::ScheduleTree* start);
std::vector<const detail::ScheduleTree*> collectScheduleTreesPath(
    std::function<const detail::ScheduleTree*(const detail::ScheduleTree*)>
        next,
    const detail::ScheduleTree* start);

// Given a schedule defined by the ancestors of the given node,
// extend it to a schedule that also covers the node itself.
isl::union_map extendSchedule(
    const detail::ScheduleTree* node,
    isl::union_map schedule);

// Get the partial schedule defined by ancestors of the given node and the node
// itself.
template <typename Schedule>
isl::UnionMap<Statement, Schedule> partialSchedule(
    const detail::ScheduleTree* root,
    const detail::ScheduleTree* node);

// Return the schedule defined by the ancestors of the given node.
template <typename Schedule>
isl::UnionMap<Statement, Schedule> prefixSchedule(
    const detail::ScheduleTree* root,
    const detail::ScheduleTree* node);

// Return the concatenation of all band node partial schedules
// from "relativeRoot" (inclusive) to "tree" (exclusive)
// within a tree rooted at "root".
// If there are no intermediate band nodes, then return a zero-dimensional
// function on the universe domain of the schedule tree.
// Note that this function does not take into account
// any intermediate filter nodes.
isl::multi_union_pw_aff infixScheduleMupa(
    const detail::ScheduleTree* root,
    const detail::ScheduleTree* relativeRoot,
    const detail::ScheduleTree* tree);

// Return the concatenation of all outer band node partial schedules.
// If there are no outer band nodes, then return a zero-dimensional
// function on the universe domain of the schedule tree.
// Note that unlike isl_schedule_node_get_prefix_schedule_multi_union_pw_aff,
// this function does not take into account any intermediate filter nodes.
template <typename Schedule>
isl::MultiUnionPwAff<Statement, Schedule> prefixScheduleMupa(
    const detail::ScheduleTree* root,
    const detail::ScheduleTree* tree);

// Return the concatenation of all outer band node partial schedules,
// including that of the node itself.
// Note that this function does not take into account
// any intermediate filter nodes.
template <typename Schedule>
isl::MultiUnionPwAff<Statement, Schedule> partialScheduleMupa(
    const detail::ScheduleTree* root,
    const detail::ScheduleTree* tree);

// Get the set of domain points active at the given node.  A domain
// point is active if it was not filtered away on the path from the
// root to the node.  The root must be a domain element, otherwise no
// elements would be considered active.
isl::union_set activeDomainPoints(
    const detail::ScheduleTree* root,
    const detail::ScheduleTree* node);

// Get the set of domain points active below the given node.  A domain
// point is active if it was not filtered away on the path from the
// root to the node.  The root must be a domain element, otherwise no
// elements would be considered active.
isl::union_set activeDomainPointsBelow(
    const detail::ScheduleTree* root,
    const detail::ScheduleTree* node);

// Collect the outer block/thread identifier mappings
// into a filter on the active domain elements.
isl::union_set prefixMappingFilter(
    const detail::ScheduleTree* root,
    const detail::ScheduleTree* node);

// Extract a mapping from the domain elements active at "tree" (in a tree
// rooted at "root") to identifiers "ids", where all branches in "tree" are
// assumed to have been mapped to these identifiers.  The result lives in a
// space of the form "tupleId"["ids"...].
template <typename MappingType>
isl::MultiUnionPwAff<Statement, MappingType> extractDomainToIds(
    const detail::ScheduleTree* root,
    const detail::ScheduleTree* tree,
    const std::vector<mapping::MappingId>& ids,
    isl::id tupleId);

// Is "tree" a mapping filter that maps identifiers of the type provided as
// template argument?
template <typename MappingType>
bool isMappingTo(const detail::ScheduleTree* tree) {
  using namespace detail;

  if (auto filterNode = tree->as<ScheduleTreeMapping>()) {
    for (auto& kvp : filterNode->mapping) {
      if (kvp.first.is<MappingType>()) {
        return true;
      }
    }
  }
  return false;
}

} // namespace polyhedral
} // namespace tc

#include "tc/core/polyhedral/schedule_utils-inl.h"
