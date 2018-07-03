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
#include "tc/core/polyhedral/schedule_utils.h"

#include <iostream>
#include <vector>

namespace tc {
namespace polyhedral {
////////////////////////////////////////////////////////////////////////////////
//                 ScheduleTree utility functions, out-of-class
////////////////////////////////////////////////////////////////////////////////

using namespace detail;
using std::ostream;
using std::vector;

isl::union_map extendSchedule(
    const ScheduleTree* node,
    isl::union_map schedule) {
  if (auto bandElem = node->as<ScheduleTreeElemBand>()) {
    if (bandElem->nMember() > 0) {
      schedule =
          schedule.flat_range_product(isl::union_map::from(bandElem->mupa_));
    }
  } else if (auto filterElem = node->as<ScheduleTreeElemFilter>()) {
    schedule = schedule.intersect_domain(filterElem->filter_);
  } else if (auto extensionElem = node->as<ScheduleTreeElemExtension>()) {
    // FIXME: we may need to restrict the range of reversed extension map to
    // schedule values that correspond to active domain elements at this
    // point.
    schedule = schedule.unite(
        extensionElem->extension_.reverse().intersect_range(schedule.range()));
  }

  return schedule;
}

namespace {
isl::union_map partialScheduleImpl(
    const ScheduleTree* root,
    const ScheduleTree* node,
    bool useNode) {
  auto nodes = node->ancestors(root);
  if (useNode) {
    nodes.push_back(node);
  }
  TC_CHECK_GT(nodes.size(), 0u) << "root node does not have a prefix schedule";
  auto domain = root->as<ScheduleTreeElemDomain>();
  TC_CHECK(domain);
  auto schedule = isl::union_map::from_domain(domain->domain_);
  for (auto anc : nodes) {
    if (anc->as<ScheduleTreeElemDomain>()) {
      TC_CHECK(anc == root);
    } else {
      schedule = extendSchedule(anc, schedule);
    }
  }
  return schedule;
}
} // namespace

isl::union_map prefixSchedule(
    const ScheduleTree* root,
    const ScheduleTree* node) {
  return partialScheduleImpl(root, node, false);
}

isl::union_map partialSchedule(
    const ScheduleTree* root,
    const ScheduleTree* node) {
  return partialScheduleImpl(root, node, true);
}

namespace {
/*
 * If "node" is any filter, then intersect "domain" with that filter.
 */
isl::union_set applyFilter(isl::union_set domain, const ScheduleTree* node) {
  if (auto filterElem = node->as<ScheduleTreeElemFilter>()) {
    return domain.intersect(filterElem->filter_);
  }
  return domain;
}

/*
 * If "node" is a mapping, then intersect "domain" with its filter.
 */
isl::union_set applyMapping(isl::union_set domain, const ScheduleTree* node) {
  if (auto filterElem = node->as<ScheduleTreeElemMapping>()) {
    return domain.intersect(filterElem->filter_);
  }
  return domain;
}

// Get the set of domain elements that are active below
// the given branch of nodes, filtered using "filter".
//
// Domain elements are introduced by the root domain node.  Some nodes
// refine this set of elements based on "filter".  Extension nodes
// are considered to introduce additional domain points.
isl::union_set collectDomain(
    const ScheduleTree* root,
    const vector<const ScheduleTree*>& nodes,
    isl::union_set (*filter)(isl::union_set domain, const ScheduleTree* node)) {
  auto domainElem = root->as<ScheduleTreeElemDomain>();
  TC_CHECK(domainElem) << "root must be a Domain node" << *root;

  auto domain = domainElem->domain_;

  for (auto anc : nodes) {
    domain = filter(domain, anc);
    if (auto extensionElem = anc->as<ScheduleTreeElemExtension>()) {
      auto parentSchedule = prefixSchedule(root, anc);
      auto extension = extensionElem->extension_;
      TC_CHECK(parentSchedule) << "missing root domain node";
      parentSchedule = parentSchedule.intersect_domain(domain);
      domain = domain.unite(parentSchedule.range().apply(extension));
    }
  }
  return domain;
}

// Get the set of domain elements that are active below
// the given branch of nodes.
isl::union_set activeDomainPointsHelper(
    const ScheduleTree* root,
    const vector<const ScheduleTree*>& nodes) {
  return collectDomain(root, nodes, &applyFilter);
}

} // namespace

isl::union_set prefixMappingFilter(
    const ScheduleTree* root,
    const ScheduleTree* node) {
  return collectDomain(root, node->ancestors(root), &applyMapping);
}

isl::union_set activeDomainPoints(
    const ScheduleTree* root,
    const ScheduleTree* node) {
  return activeDomainPointsHelper(root, node->ancestors(root));
}

isl::union_set activeDomainPointsBelow(
    const ScheduleTree* root,
    const ScheduleTree* node) {
  auto ancestors = node->ancestors(root);
  ancestors.emplace_back(node);
  return activeDomainPointsHelper(root, ancestors);
}

vector<ScheduleTree*> collectScheduleTreesPath(
    std::function<ScheduleTree*(ScheduleTree*)> next,
    ScheduleTree* start) {
  vector<ScheduleTree*> res{start};
  auto n = start;
  while ((n = next(n)) != nullptr) {
    res.push_back(n);
  }
  return res;
}

vector<const ScheduleTree*> collectScheduleTreesPath(
    std::function<const ScheduleTree*(const ScheduleTree*)> next,
    const ScheduleTree* start) {
  vector<const ScheduleTree*> res{start};
  auto n = start;
  while ((n = next(n)) != nullptr) {
    res.push_back(n);
  }
  return res;
}

namespace {

template <typename T>
vector<T> reversed(const vector<T>& vec) {
  vector<T> result;
  result.reserve(vec.size());
  result.insert(result.begin(), vec.rbegin(), vec.rend());
  return result;
}

template <typename T>
vector<const ScheduleTree*> filterType(const vector<const ScheduleTree*>& vec) {
  vector<const ScheduleTree*> result;
  for (auto e : vec) {
    if (e->as<T>()) {
      result.push_back(e);
    }
  }
  return result;
}

template <typename T, typename Func>
T foldl(const vector<const ScheduleTree*> vec, Func op, T init = T()) {
  T value = init;
  for (auto st : vec) {
    value = op(st, value);
  }
  return value;
}

template <typename... Args>
ostream& operator<<(ostream& os, const vector<Args...>& v) {
  os << "[";
  bool first = true;
  for (auto const& ve : v) {
    if (!first) {
      os << ", ";
    }
    os << ve;
    first = true;
  }
  os << "]";
  return os;
}
} // namespace

isl::multi_union_pw_aff infixScheduleMupa(
    const ScheduleTree* root,
    const ScheduleTree* relativeRoot,
    const ScheduleTree* tree) {
  auto domainElem = root->as<ScheduleTreeElemDomain>();
  TC_CHECK(domainElem);
  auto domain = domainElem->domain_.universe();
  auto zero = isl::multi_val::zero(domain.get_space().set_from_params());
  auto prefix = isl::multi_union_pw_aff(domain, zero);
  prefix = foldl(
      filterType<ScheduleTreeElemBand>(tree->ancestors(relativeRoot)),
      [](const ScheduleTree* st, isl::multi_union_pw_aff pref) {
        auto mupa = st->as<ScheduleTreeElemBand>()->mupa_;
        return pref.flat_range_product(mupa);
      },
      prefix);
  return prefix;
}

isl::multi_union_pw_aff prefixScheduleMupa(
    const ScheduleTree* root,
    const ScheduleTree* tree) {
  return infixScheduleMupa(root, root, tree);
}

isl::multi_union_pw_aff partialScheduleMupa(
    const detail::ScheduleTree* root,
    const detail::ScheduleTree* tree) {
  auto prefix = prefixScheduleMupa(root, tree);
  auto band = tree->as<ScheduleTreeElemBand>();
  return band ? prefix.flat_range_product(band->mupa_) : prefix;
}

/*
 * Extract a mapping from the domain elements active at "tree"
 * to identifiers "ids", where all branches in "tree"
 * are assumed to have been mapped to these identifiers.
 * The result lives in a space of the form "tupleId"["ids"...].
 */
isl::multi_union_pw_aff extractDomainToIds(
    const detail::ScheduleTree* root,
    const detail::ScheduleTree* tree,
    const std::vector<mapping::MappingId>& ids,
    isl::id tupleId) {
  using namespace polyhedral::detail;

  auto space = isl::space(tree->ctx_, 0);
  auto empty = isl::union_set::empty(space);
  space = space.named_set_from_params_id(tupleId, ids.size());
  auto zero = isl::multi_val::zero(space);
  auto domainToIds = isl::multi_union_pw_aff(empty, zero);

  for (auto mapping : tree->collect(tree, ScheduleTreeType::Mapping)) {
    auto mappingNode = mapping->as<ScheduleTreeElemMapping>();
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
