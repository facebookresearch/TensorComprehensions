/**
 * Copyright (c) 2017-2018, Facebook, Inc.
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

#include <vector>

#include "tc/core/check.h"
#include "tc/core/polyhedral/schedule_tree.h"
#include "tc/core/polyhedral/schedule_tree_elem.h"
#include "tc/external/isl.h"

namespace tc {
namespace polyhedral {

namespace detail {
inline isl::union_map partialScheduleImpl(
    const ScheduleTree* root,
    const ScheduleTree* node,
    bool useNode) {
  auto nodes = node->ancestors(root);
  if (useNode) {
    nodes.push_back(node);
  }
  TC_CHECK_GT(nodes.size(), 0u) << "root node does not have a prefix schedule";
  auto domain = root->as<ScheduleTreeDomain>();
  TC_CHECK(domain);
  auto schedule = isl::union_map::from_domain(domain->domain_);
  for (auto anc : nodes) {
    if (anc->as<ScheduleTreeDomain>()) {
      TC_CHECK(anc == root);
    } else {
      schedule = extendSchedule(anc, schedule);
    }
  }
  return schedule;
}
} // namespace detail

inline isl::union_map prefixSchedule(
    const detail::ScheduleTree* root,
    const detail::ScheduleTree* node) {
  return detail::partialScheduleImpl(root, node, false);
}

inline isl::union_map partialSchedule(
    const detail::ScheduleTree* root,
    const detail::ScheduleTree* node) {
  return detail::partialScheduleImpl(root, node, true);
}

namespace detail {

template <typename T>
inline std::vector<const ScheduleTree*> filterType(
    const std::vector<const ScheduleTree*>& vec) {
  std::vector<const ScheduleTree*> result;
  for (auto e : vec) {
    if (e->as<T>()) {
      result.push_back(e);
    }
  }
  return result;
}

template <typename T, typename Func>
inline T
foldl(const std::vector<const ScheduleTree*> vec, Func op, T init = T()) {
  T value = init;
  for (auto st : vec) {
    value = op(st, value);
  }
  return value;
}

} // namespace detail

inline isl::multi_union_pw_aff infixScheduleMupa(
    const detail::ScheduleTree* root,
    const detail::ScheduleTree* relativeRoot,
    const detail::ScheduleTree* tree) {
  using namespace polyhedral::detail;

  auto domainElem = root->as<ScheduleTreeDomain>();
  TC_CHECK(domainElem);
  auto domain = domainElem->domain_.universe();
  auto zero = isl::multi_val::zero(domain.get_space().set_from_params());
  auto prefix = isl::multi_union_pw_aff(domain, zero);
  prefix = foldl(
      filterType<ScheduleTreeBand>(tree->ancestors(relativeRoot)),
      [](const ScheduleTree* st, isl::multi_union_pw_aff pref) {
        auto mupa = st->as<ScheduleTreeBand>()->mupa_;
        return pref.flat_range_product(mupa);
      },
      prefix);
  return prefix;
}

inline isl::multi_union_pw_aff prefixScheduleMupa(
    const detail::ScheduleTree* root,
    const detail::ScheduleTree* tree) {
  return infixScheduleMupa(root, root, tree);
}

inline isl::multi_union_pw_aff partialScheduleMupa(
    const detail::ScheduleTree* root,
    const detail::ScheduleTree* tree) {
  using namespace polyhedral::detail;

  auto prefix = prefixScheduleMupa(root, tree);
  auto band = tree->as<ScheduleTreeBand>();
  return band ? prefix.flat_range_product(band->mupa_) : prefix;
}

/*
 * Extract a mapping from the domain elements active at "tree"
 * to identifiers "ids", where all branches in "tree"
 * are assumed to have been mapped to these identifiers.
 * The result lives in a space of the form "tupleId"["ids"...].
 */
inline isl::multi_union_pw_aff extractDomainToIds(
    const detail::ScheduleTree* root,
    const detail::ScheduleTree* tree,
    const std::vector<mapping::MappingId>& ids,
    isl::id tupleId) {
  using namespace polyhedral::detail;

  auto space = isl::space(tree->ctx_, 0);
  auto empty = isl::union_set::empty(space);
  space = space.add_named_tuple_id_ui(tupleId, ids.size());
  auto zero = isl::multi_val::zero(space);
  auto domainToIds = isl::multi_union_pw_aff(empty, zero);

  for (auto mapping : tree->collect(tree, ScheduleTreeType::Mapping)) {
    auto mappingNode = mapping->as<ScheduleTreeMapping>();
    auto list = isl::union_pw_aff_list(tree->ctx_, ids.size());
    for (auto id : ids) {
      if (mappingNode->mapping.count(id) == 0) {
        break;
      }
      auto idMap = mappingNode->mapping.at(id);
      list = list.add(idMap);
    }
    // Ignore this node if it does not map to all required ids.
    if (static_cast<size_t>(list.size()) != ids.size()) {
      continue;
    }
    auto nodeToIds = isl::multi_union_pw_aff(space, list);
    auto active = activeDomainPoints(root, mapping);
    TC_CHECK(active.intersect(domainToIds.domain()).is_empty())
        << "conflicting mappings; are the filters in the tree disjoint?";
    nodeToIds = nodeToIds.intersect_domain(active);
    domainToIds = domainToIds.union_add(nodeToIds);
  }

  auto active = activeDomainPoints(root, tree);
  TC_CHECK(active.is_subset(domainToIds.domain()))
      << "not all domain points of\n"
      << active << "\nwere mapped to the required ids";

  return domainToIds;
}

} // namespace polyhedral
} // namespace tc
