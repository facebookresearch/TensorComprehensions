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

#include "tc/core/check.h"

namespace tc {
namespace polyhedral {
inline detail::ScheduleTree* insertNodeAbove(
    detail::ScheduleTree* root,
    detail::ScheduleTree* tree,
    ScheduleTreeUPtr&& node) {
  TC_CHECK_EQ(node->numChildren(), 0u);
  auto parent = tree->ancestor(root, 1);
  auto childPos = tree->positionInParent(parent);
  node->appendChild(parent->detachChild(childPos));
  parent->insertChild(childPos, std::move(node));
  return parent->child({childPos});
}

inline detail::ScheduleTree* insertNodeBelow(
    detail::ScheduleTree* tree,
    ScheduleTreeUPtr&& node) {
  TC_CHECK_EQ(node->numChildren(), 0u);
  auto numChildren = tree->numChildren();
  TC_CHECK_LE(numChildren, 1u);
  node->appendChildren(tree->detachChildren());
  tree->appendChild(std::move(node));
  return tree->child({0});
}
} // namespace polyhedral
} // namespace tc
