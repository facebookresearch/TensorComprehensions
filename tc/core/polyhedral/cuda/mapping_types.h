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

#include "tc/external/isl.h"

#include "tc/core/polyhedral/mapping_types.h"

namespace tc {
namespace polyhedral {
namespace mapping {
/*
 * Returns the size of the mapping or Mapping::unmapped if not mapped.
 * This uses traditional internal assumptions that x<=>0, y<=>1, z<=>2.
 */
inline size_t mappingSize(const ThreadId& id, const Block& vals) {
  if (vals.view.size() > id.dim) {
    return vals.view[static_cast<size_t>(id.dim)];
  }
  return MappingId::unmapped;
}

/*
 * Returns the size of the mapping or Mapping::unmapped if not mapped.
 * This uses traditional internal assumptions that x<=>0, y<=>1, z<=>2.
 */
inline size_t mappingSize(const BlockId& id, const Grid& vals) {
  if (vals.view.size() > id.dim) {
    return vals.view[static_cast<size_t>(id.dim)];
  }
  return MappingId::unmapped;
}
} // namespace mapping
} // namespace polyhedral
} // namespace tc
