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

namespace {
/*
 * If "node" is any filter, then intersect "domain" with that filter.
 */
isl::UnionSet<Statement> applyFilter(
    isl::UnionSet<Statement> domain,
    const ScheduleTree* node) {
  if (auto filterElem = node->as<ScheduleTreeFilter>()) {
    return domain.intersect(filterElem->filter_);
  }
  return domain;
}

/*
 * If "node" is a mapping, then intersect "domain" with its filter.
 */
isl::UnionSet<Statement> applyMapping(
    isl::UnionSet<Statement> domain,
    const ScheduleTree* node) {
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
isl::UnionSet<Statement> collectDomain(
    const ScheduleTree* root,
    const vector<const ScheduleTree*>& nodes,
    isl::UnionSet<Statement> (
        *filter)(isl::UnionSet<Statement> domain, const ScheduleTree* node)) {
  auto domainElem = root->as<ScheduleTreeDomain>();
  TC_CHECK(domainElem) << "root must be a Domain node" << *root;

  auto domain = domainElem->domain_;

  for (auto anc : nodes) {
    domain = filter(domain, anc);
    if (auto extensionElem = anc->as<ScheduleTreeExtension>()) {
      auto parentSchedule = prefixSchedule<Prefix>(root, anc);
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

isl::UnionSet<Statement> prefixMappingFilter(
    const ScheduleTree* root,
    const ScheduleTree* node) {
  return collectDomain(root, node->ancestors(root), &applyMapping);
}

isl::UnionSet<Statement> activeDomainPoints(
    const ScheduleTree* root,
    const ScheduleTree* node) {
  return isl::UnionSet<Statement>(
      activeDomainPointsHelper(root, node->ancestors(root)));
}

isl::UnionSet<Statement> activeDomainPointsBelow(
    const ScheduleTree* root,
    const ScheduleTree* node) {
  auto ancestors = node->ancestors(root);
  ancestors.emplace_back(node);
  return isl::UnionSet<Statement>(activeDomainPointsHelper(root, ancestors));
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
