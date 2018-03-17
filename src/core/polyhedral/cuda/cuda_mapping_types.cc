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
#include "tc/core/polyhedral/cuda/cuda_mapping_types.h"

#include "tc/core/mapping_options.h"

namespace tc {
namespace polyhedral {
namespace mapping {
size_t ThreadId::mappingSize(const Block& vals) const {
  if (vals.size() > dim) {
    return vals[static_cast<size_t>(dim)];
  }
  return MappingId::unmapped;
}

size_t BlockId::mappingSize(const Grid& vals) const {
  if (vals.size() > dim) {
    return vals[static_cast<size_t>(dim)];
  }
  return MappingId::unmapped;
}
} // namespace mapping
} // namespace polyhedral
} // namespace tc
