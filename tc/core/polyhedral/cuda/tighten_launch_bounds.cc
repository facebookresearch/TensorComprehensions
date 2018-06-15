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

#include "tc/core/polyhedral/cuda/tighten_launch_bounds.h"

#include "tc/core/check.h"
#include "tc/core/polyhedral/cuda/mapping_types.h"
#include "tc/core/polyhedral/exceptions.h"
#include "tc/core/polyhedral/functional.h"
#include "tc/core/polyhedral/schedule_transforms.h"
#include "tc/core/polyhedral/schedule_tree.h"

namespace tc {
namespace polyhedral {
namespace {
// This returns the (inclusive) range of the mapping parameter "mappingId"
// within the context "mappingContext".
// This range corresponds to the blocks/threads active at the particular
// location in the tree where this mapping is active.
//
// This is used to tighten the kernel to only launch on the necessary amount
// of resources.
//
// When the range is unbounded on the right, we return the maximal positive
// range (0, max_size_t). This needs to be intersected with launch bounds to
// obtain the proper finite range.
// Otherwise, the range is asserted bounded on the left and to lie in the
// positive half of the integer axis.
std::pair<size_t, size_t> rangeOfMappingParameter(
    isl::set mappingContext,
    mapping::MappingId mappingId) {
  if (!mappingContext.involves_param(mappingId)) {
    return std::make_pair(0, std::numeric_limits<size_t>::max());
  }
  auto space = mappingContext.get_space();
  isl::aff a(isl::aff::param_on_domain_space(space, mappingId));
  auto max = mappingContext.max_val(a);
  if (max.is_nan() || max.is_infty()) {
    return std::make_pair(0, std::numeric_limits<size_t>::max());
  }
  TC_CHECK(max.is_int()) << max.to_str();
  TC_CHECK(max.is_nonneg()) << max.to_str();
  auto min = mappingContext.min_val(a);
  TC_CHECK(min.is_int()) << max.to_str();
  TC_CHECK(min.is_nonneg()) << max.to_str();

  return std::make_pair(
      static_cast<size_t>(min.get_num_si()),
      static_cast<size_t>(max.get_num_si()));
}

/*
 * Compute the maximal value attained by the mapping parameter "id".
 */
template <typename MappingIdType>
size_t maxValue(const Scop& scop, const MappingIdType& id) {
  using namespace polyhedral::detail;

  auto root = scop.scheduleRoot();
  auto params = scop.context();
  size_t sizetMax = std::numeric_limits<size_t>::max();
  size_t max = 0;
  size_t min = sizetMax;
  auto filters = root->collect(root, ScheduleTreeType::Mapping);
  filters = functional::Filter(isMappingTo<MappingIdType>, filters);
  for (auto p : filters) {
    auto mappingNode = p->elemAs<ScheduleTreeElemMapping>();
    auto active = activeDomainPoints(root, p).intersect_params(params);
    active = active.intersect(mappingNode->filter_);
    auto range = rangeOfMappingParameter(active.params(), id);
    min = std::min(min, range.first);
    max = std::max(max, range.second);
  }
  // Ignore min for now but there is a future possibility for shifting
  LOG_IF(WARNING, min > 0)
      << "Opportunity for tightening launch bounds with shifting -> min:"
      << min;
  TC_CHECK(max < sizetMax) << "missing mapping to " << id << *root;
  // Inclusive range needs + 1 to translate to sizes
  return max + 1;
}

/*
 * Take grid or block launch bounds "size" and replace them
 * by the tightened, actual, launch bounds used in practice.
 */
template <typename MappingIdType, typename Size>
Size launchBounds(const Scop& scop, Size size) {
  Size tightened;

  for (size_t i = 0; i < size.view.size(); ++i) {
    tightened.view[i] = maxValue(scop, MappingIdType::makeId(i));
  }

  return tightened;
}

} // namespace

// Takes grid/block launch bounds that have been passed to mapping and
// computes the tightened, actual, launch bounds used in practice after
// specialization of the ScheduleTree.
std::pair<tc::Grid, tc::Block> tightenLaunchBounds(
    const Scop& scop,
    const tc::Grid& grid,
    const tc::Block& block) {
  return std::make_pair(
      launchBounds<mapping::BlockId>(scop, grid),
      launchBounds<mapping::ThreadId>(scop, block));
}
} // namespace polyhedral
} // namespace tc
