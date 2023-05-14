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

#include <tuple>
#include <utility>
#include <vector>

#include "tc/core/polyhedral/domain_types.h"
#include "tc/core/polyhedral/schedule_tree.h"
#include "tc/core/polyhedral/scop.h"

namespace tc {
namespace polyhedral {

// Return the union of the reduction update statements
// that appear in "domain".
isl::UnionSet<Statement> reductionUpdates(
    isl::UnionSet<Statement> domain,
    const Scop& scop);

// Does "prefix" partition "domain" into individual reductions?
// In particular, do the elements of "domain" access a single tensor
// element within "prefix"?
template <typename Prefix>
bool isSingleReductionWithin(
    isl::UnionSet<Statement> domain,
    isl::MultiUnionPwAff<Statement, Prefix> prefix,
    const Scop& scop);

} // namespace polyhedral
} // namespace tc

#include "tc/core/polyhedral/schedule_tree_matcher-inl.h"
