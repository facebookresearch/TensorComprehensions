/**
 * Copyright (c) 2017, Facebook, Inc.
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

#include "tc/core/polyhedral/codegen.h"

#include <sstream>

#include "tc/core/polyhedral/schedule_tree.h"
#include "tc/core/polyhedral/schedule_tree_elem.h"

namespace tc {
namespace polyhedral {

isl::id_list Codegen::makeLoopIterators(
    const detail::ScheduleTree* root,
    const std::string& prefix) {
  auto bands =
      detail::ScheduleTree::collect(root, detail::ScheduleTreeType::Band);
  size_t n = 0;
  for (auto const& node : bands) {
    auto bandElem = node->as<detail::ScheduleTreeBand>();
    auto depth = node->scheduleDepth(root) + bandElem->nMember();
    if (depth > n) {
      n = depth;
    }
  }

  auto ctx = root->ctx_;
  isl::id_list loopIterators(ctx, n);
  for (size_t i = 0; i < n; ++i) {
    std::stringstream ss;
    ss << prefix << i;
    loopIterators = loopIterators.add(isl::id(ctx, ss.str()));
  }
  return loopIterators;
}

} // namespace polyhedral
} // namespace tc
