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
  if (auto bandElem = node->as<ScheduleTreeBand>()) {
    if (bandElem->nMember() > 0) {
      schedule =
          schedule.flat_range_product(isl::union_map::from(bandElem->mupa_));
    }
  } else if (auto filterElem = node->as<ScheduleTreeFilter>()) {
    schedule = schedule.intersect_domain(filterElem->filter_);
  } else if (auto extensionElem = node->as<ScheduleTreeExtension>()) {
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
  if (auto filterElem = node->as<ScheduleTreeFilter>()) {
    return domain.intersect(filterElem->filter_);
  }
  return domain;
}

/*
 * If "node" is a mapping, then intersect "domain" with its filter.
 */
isl::union_set applyMapping(isl::union_set domain, const ScheduleTree* node) {
  if (auto filterElem = node->as<ScheduleTreeMapping>()) {
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
  auto domainElem = root->as<ScheduleTreeDomain>();
  TC_CHECK(domainElem) << "root must be a Domain node" << *root;

  auto domain = domainElem->domain_;

  for (auto anc : nodes) {
    domain = filter(domain, anc);
    if (auto extensionElem = anc->as<ScheduleTreeExtension>()) {
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

} // namespace polyhedral
} // namespace tc
