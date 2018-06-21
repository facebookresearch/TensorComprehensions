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
#include "tc/core/polyhedral/cuda/mapped_scop.h"
#include "tc/core/polyhedral/cuda/mapping_types.h"
#include "tc/core/polyhedral/exceptions.h"

namespace tc {
namespace polyhedral {
namespace {
/*
 * Return the mapping to MappingTypeId, i.e, either the mapping to blocks or
 * the mapping to threads.
 */
template <typename MappingTypeId>
static isl::multi_union_pw_aff mappingSchedule(const MappedScop& mscop);
template <>
isl::multi_union_pw_aff mappingSchedule<mapping::BlockId>(
    const MappedScop& mscop) {
  return mscop.blockMappingSchedule(mscop.schedule());
}
template <>
isl::multi_union_pw_aff mappingSchedule<mapping::ThreadId>(
    const MappedScop& mscop) {
  return mscop.threadMappingSchedule(mscop.schedule());
}

/*
 * Take grid or block launch bounds "size" and replace them
 * by the tightened, actual, launch bounds used in practice.
 */
template <typename MappingIdType, typename Size>
Size launchBounds(const MappedScop& mscop, Size size) {
  Size tightened;

  auto params = mscop.scop().context();
  auto mapping = mappingSchedule<MappingIdType>(mscop);
  mapping = mapping.intersect_params(params);
  auto max = mapping.max_multi_val();

  for (size_t i = 0; i < size.view.size(); ++i) {
    auto maxVal = max.get_val(i);
    TC_CHECK(maxVal.is_int()) << maxVal.to_str();
    TC_CHECK(maxVal.is_nonneg()) << maxVal.to_str();
    // Inclusive range needs + 1 to translate to sizes
    tightened.view[i] = maxVal.get_num_si() + 1;
  }

  return tightened;
}

} // namespace

// Takes grid/block launch bounds that have been passed to mapping and
// computes the tightened, actual, launch bounds used in practice after
// specialization of the ScheduleTree.
std::pair<tc::Grid, tc::Block> tightenLaunchBounds(
    const MappedScop& mscop,
    const tc::Grid& grid,
    const tc::Block& block) {
  return std::make_pair(
      launchBounds<mapping::BlockId>(mscop, grid),
      launchBounds<mapping::ThreadId>(mscop, block));
}
} // namespace polyhedral
} // namespace tc
