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

#include <iostream>

#include "tc/external/isl.h"

namespace tc {
namespace polyhedral {

// Information about the bodies of the polyhedral statements.
struct Body {
  Body() = default;
  Body(isl::space paramSpace) {
    writes = reads = isl::union_map::empty(paramSpace);
  }

  // Specialize to the given context.
  void specialize(isl::set context) {
    reads = reads.intersect_params(context);
    writes = writes.intersect_params(context);
  }

  // Union maps describing the reads and writes done. Uses the ids in
  // the schedule tree to denote the containing Stmt, and tags each
  // access with a unique reference id of the form __tc_ref_N.
  isl::union_map reads, writes;
};

std::ostream& operator<<(std::ostream& os, const Body& body);

} // namespace polyhedral
} // namespace tc
