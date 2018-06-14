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
namespace detail {
template <typename MappingIdType>
inline ScheduleTreeUPtr ScheduleTree::makeMapping(
    const std::vector<MappingIdType>& mappedIds,
    isl::union_pw_aff_list mappedAffs,
    std::vector<ScheduleTreeUPtr>&& children) {
  TC_CHECK_EQ(mappedIds.size(), static_cast<size_t>(mappedAffs.n()))
      << "expected as many mapped ids as affs";
  ScheduleTreeElemMapping::Mapping mapping;
  for (size_t i = 0, n = mappedAffs.n(); i < n; ++i) {
    mapping.emplace(mappedIds.at(i), mappedAffs.get(i));
  }
  TC_CHECK_GE(mapping.size(), 1u) << "empty mapping";
  TC_CHECK_EQ(mappedIds.size(), mapping.size())
      << "some id is used more than once in the mapping";
  auto ctx = mappedIds[0].get_ctx();
  ScheduleTreeUPtr res(new ScheduleTree(ctx));
  res->elem_ = std::unique_ptr<ScheduleTreeElemMapping>(
      new ScheduleTreeElemMapping(mapping));
  res->type_ = ScheduleTreeType::Mapping;
  res->appendChildren(std::move(children));
  return res;
}
} // namespace detail
} // namespace polyhedral
} // namespace tc
