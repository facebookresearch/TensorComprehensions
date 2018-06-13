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
// This returns the (inclusive) range of the mapping parameter that is active
// at node under root given:
//   1. a context that is the intersection of the specialization context and
//   the mapping context
//   2. a MappingId
// This range corresponds to the blocks/threads active at that particular
// location in the tree.
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
    const detail::ScheduleTree* root,
    const detail::ScheduleTree* node,
    isl::set context,
    mapping::MappingId mappingId) {
  auto active =
      activeDomainPoints(root, node).intersect_params(context).params();
  if (!active.involves_param(mappingId)) {
    return std::make_pair(0, std::numeric_limits<size_t>::max());
  }
  isl::aff a(isl::aff::param_on_domain_space(active.get_space(), mappingId));
  auto max = active.max_val(a);
  if (max.is_nan() || max.is_infty()) {
    return std::make_pair(0, std::numeric_limits<size_t>::max());
  }
  TC_CHECK(max.is_int()) << max.to_str();
  TC_CHECK(max.is_nonneg()) << max.to_str();
  auto min = active.min_val(a);
  TC_CHECK(min.is_int()) << max.to_str();
  TC_CHECK(min.is_nonneg()) << max.to_str();

  return std::make_pair(
      static_cast<size_t>(min.get_num_si()),
      static_cast<size_t>(max.get_num_si()));
}

// Look for nodes with no children.
inline std::vector<const detail::ScheduleTree*> leaves(
    const detail::ScheduleTree* tree) {
  return functional::Filter(
      [](const detail::ScheduleTree* st) { return st->numChildren() == 0; },
      detail::ScheduleTree::collect(tree));
}

/*
 * Compute the maximal value attained by the mapping parameter "id".
 * Return std::numeric_limits<size_t>::max() if this value cannot
 * be determined.
 */
size_t maxValue(const Scop& scop, const mapping::MappingId& id) {
  auto root = scop.scheduleRoot();
  auto params = scop.context();
  size_t sizetMax = std::numeric_limits<size_t>::max();
  size_t max = 0;
  size_t min = sizetMax;
  auto nonSyncLeaves = functional::Filter(
      [root, params](const detail::ScheduleTree* node) {
        auto f = node->elemAsBase<detail::ScheduleTreeElemFilter>();
        if (!f) {
          return true;
        }
        if (f->filter_.n_set() != 1) {
          std::stringstream ss;
          ss << "In tree:\n"
             << *root << "\nnot a single set in filter: " << f->filter_;
          throw tightening::TighteningException(ss.str());
        }
        auto single = isl::set::from_union_set(f->filter_);
        auto single_id = single.get_tuple_id();
        return !Scop::isSyncId(single_id) && !Scop::isWarpSyncId(single_id);
      },
      leaves(root));
  for (auto p : nonSyncLeaves) {
    auto range = rangeOfMappingParameter(root, p, params, id);
    min = std::min(min, range.first);
    max = std::max(max, range.second);
  }
  // Ignore min for now but there is a future possibility for shifting
  LOG_IF(WARNING, min > 0)
      << "Opportunity for tightening launch bounds with shifting -> min:"
      << min;
  // Inclusive range needs + 1 to translate to sizes
  if (max < sizetMax) { // avoid overflow
    return max + 1;
  }
  return sizetMax;
}
} // namespace

// Takes grid/block launch bounds that have been passed to mapping and
// computes the tightened, actual, launch bounds used in practice after
// specialization of the ScheduleTree.
std::pair<tc::Grid, tc::Block> tightenLaunchBounds(
    const Scop& scop,
    const tc::Grid& grid,
    const tc::Block& block) {
  USING_MAPPING_SHORT_NAMES(BX, BY, BZ, TX, TY, TZ);
  // Corner case: take the min with the current size to avoid degenerate
  // range in the unbounded case.
  return std::make_pair(
      tc::Grid({std::min(maxValue(scop, BX), BX.mappingSize(grid)),
                std::min(maxValue(scop, BY), BY.mappingSize(grid)),
                std::min(maxValue(scop, BZ), BZ.mappingSize(grid))}),
      tc::Block({std::min(maxValue(scop, TX), TX.mappingSize(block)),
                 std::min(maxValue(scop, TY), TY.mappingSize(block)),
                 std::min(maxValue(scop, TZ), TZ.mappingSize(block))}));
}
} // namespace polyhedral
} // namespace tc
