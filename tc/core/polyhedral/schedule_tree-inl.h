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
namespace detail {
template <typename MappingIdType>
inline ScheduleTreeUPtr ScheduleTree::makeMappingFilter(
    const std::vector<MappingIdType>& mappedIds,
    isl::union_pw_aff_list mappedAffs,
    std::vector<ScheduleTreeUPtr>&& children) {
  std::vector<mapping::MappingId> ids;
  for (auto id : mappedIds) {
    ids.push_back(id);
  }
  CHECK_GT(ids.size(), 0) << "empty mapping";
  auto ctx = mappedIds[0].get_ctx();
  ScheduleTreeUPtr res(new ScheduleTree(ctx));
  res->elem_ = std::unique_ptr<ScheduleTreeElemMappingFilter>(
      new ScheduleTreeElemMappingFilter(ids, mappedAffs));
  res->type_ = ScheduleTreeType::MappingFilter;
  res->appendChildren(std::move(children));
  return res;
}
} // namespace detail
} // namespace polyhedral
} // namespace tc
