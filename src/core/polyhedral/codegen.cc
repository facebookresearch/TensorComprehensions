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

#include "tc/core/polyhedral/codegen.h"
#include <sstream>

namespace tc {
namespace polyhedral {

isl::id_list
Codegen::makeLoopIterators(isl::ctx ctx, int n, const std::string& prefix) {
  isl::id_list loopIterators(ctx, n);
  for (int i = 0; i < n; ++i) {
    std::stringstream ss;
    ss << prefix << i;
    loopIterators = loopIterators.add(isl::id(ctx, ss.str()));
  }
  return loopIterators;
}

} // namespace polyhedral
} // namespace tc
