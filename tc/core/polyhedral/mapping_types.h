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

namespace tc {
namespace polyhedral {
namespace mapping {

struct MappingId : public isl::id {
 protected:
  MappingId(isl::id i, unsigned char l) : isl::id(i), dim(l) {}

 public:
  MappingId(const MappingId& id) : isl::id(id), dim(id.dim) {}

  // For indexing into positional arrays
  // TODO: this should go away but this probably requires tinkering with
  // mapping_options.h::Grid/Block.
  // Also, generally can't have fully static types and dynamic behavior
  // like is used in mapped_scop.cc, so pick your poison:
  //   API bloat/templates or dynamic checks
  const unsigned char dim;

  // Placeholder value to use in absence of mapping size.
  static constexpr size_t unmapped = 1;

  struct Hash {
    size_t operator()(const MappingId& id) const {
      return isl::IslIdIslHash().operator()(id);
    }
  };
};
} // namespace mapping
} // namespace polyhedral
} // namespace tc
