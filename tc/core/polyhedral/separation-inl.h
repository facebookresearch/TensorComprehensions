/**
 * Copyright (c) 2018, Facebook, Inc.
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
#include "tc/core/polyhedral/domain_types.h"

#include "tc/external/isl.h"

namespace tc {
namespace polyhedral {

template <typename Prefix, typename Pretile>
isl::UnionSet<Statement> partialTargetTiles(
    isl::UnionSet<Statement> domain,
    isl::MultiUnionPwAff<Statement, Prefix> prefix,
    isl::MultiUnionPwAff<Statement, Pretile> pretile,
    isl::MultiVal<Pretile> size) {
  auto space = pretile.get_space();
  auto tile = isl::MultiAff<Pretile, Pretile>::identity(space.map_from_set());
  tile = tile.scale_down(size).floor();
  auto tileMap = tile.asMap();
  // Relation between pairs of elements in the same target tile.
  auto sameTile = tileMap.apply_range(tileMap.reverse()).asUnionMap();
  // Mapping between domain elements and pairs of prefix and target values.
  // D -> [P -> T]
  auto schedule = prefix.range_product(pretile);
  auto scheduleMap = schedule.toUnionMap();
  // Mapping between prefix values and target values
  // for some common domain element
  // P -> T
  TC_CHECK(domain.is_subset(scheduleMap.domain()));
  auto target = domain.apply(scheduleMap).template unwrap();
  // Mapping between prefix values and target values
  // for some common domain element, extended to complete target tiles.
  // P -> Tc
  auto extendedTarget = target.apply_range(sameTile);
  // Elements in the complete target tiles
  // that have no matching domain elements for a given value of prefix.
  auto missing = extendedTarget.subtract(target);
  // Elements in those complete target tiles that have at least one tile
  // element not matching any domain element for a given value of prefix,
  // i.e., the partial tiles.
  missing = missing.apply_range(sameTile);
  // The domain elements that map to those partial tiles.
  return missing.wrap().apply(scheduleMap.reverse());
}

} // namespace polyhedral
} // namespace tc
