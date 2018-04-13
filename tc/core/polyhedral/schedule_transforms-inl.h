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

namespace tc {
namespace polyhedral {
inline detail::ScheduleTree* insertNodeAbove(
    detail::ScheduleTree* root,
    detail::ScheduleTree* tree,
    ScheduleTreeUPtr&& node) {
  CHECK_EQ(node->numChildren(), 0u);
  auto parent = tree->ancestor(root, 1);
  auto childPos = tree->positionInParent(parent);
  node->appendChild(parent->detachChild(childPos));
  parent->insertChild(childPos, std::move(node));
  return parent->child({childPos});
}

inline detail::ScheduleTree* insertNodeBelow(
    detail::ScheduleTree* tree,
    ScheduleTreeUPtr&& node) {
  CHECK_EQ(node->numChildren(), 0u);
  auto numChildren = tree->numChildren();
  CHECK_LE(numChildren, 1u);
  node->appendChildren(tree->detachChildren());
  tree->appendChild(std::move(node));
  return tree->child({0});
}

template <typename MappingIdType>
inline detail::ScheduleTree* mapToParameterWithExtent(
    detail::ScheduleTree* root,
    detail::ScheduleTree* tree,
    size_t pos,
    MappingIdType id,
    size_t extent) {
  auto band = tree->elemAs<detail::ScheduleTreeElemBand>();
  CHECK(band) << "expected a band, got " << *tree;
  CHECK_GE(pos, 0u) << "dimension underflow";
  CHECK_LT(pos, band->nMember()) << "dimension overflow";
  CHECK_NE(extent, 0u) << "NYI: mapping to 0";

  auto domain = activeDomainPoints(root, tree).universe();

  // Introduce the "mapping" parameter after checking it is not already present
  // in the schedule space.
  CHECK(not band->mupa_.involves_param(id));

  // Create mapping filter by equating the newly introduced
  // parameter "id" to the "pos"-th schedule dimension modulo its extent.
  auto upa =
      band->mupa_.get_union_pw_aff(pos).mod_val(isl::val(tree->ctx_, extent));
  upa = upa.sub(isl::union_pw_aff::param_on_domain(domain, id));
  auto filter = upa.zero_union_set();
  auto mapping =
      detail::ScheduleTree::makeMappingFilter<MappingIdType>(filter, {id});
  return insertNodeAbove(root, tree, std::move(mapping))->child({0});
}
} // namespace polyhedral
} // namespace tc
