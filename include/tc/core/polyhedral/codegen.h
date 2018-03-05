/**
 * Copyright (c) 2017, Facebook, Inc.
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

#include <string>

#include "tc/external/isl.h"

namespace tc {
namespace polyhedral {

class Codegen {
  constexpr static const char* kLoopIteratorDefaultPrefix = "c";

 public:
  // Create a list of isl ids to be used as loop iterators when building the
  // AST.
  //
  // Note that this function can be scrapped as ISL can generate some default
  // iterator names.  However, it may come handy for associating extra info with
  // iterators.
  static isl::list<isl::id> makeLoopIterators(
      isl::ctx ctx,
      int n,
      const std::string& prefix = kLoopIteratorDefaultPrefix);
};

} // namespace polyhedral
} // namespace tc
