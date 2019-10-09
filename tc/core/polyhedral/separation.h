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

#include "tc/core/polyhedral/domain_types.h"

#include "tc/external/isl.h"

namespace tc {
namespace polyhedral {

/*
 * Consider a tiling of size "size" in the target space of "pretile".
 * Return the elements in "domain" that map to partial tiles
 * in this space for fixed values of "prefix".
 */
template <typename Prefix, typename Pretile>
isl::UnionSet<Statement> partialTargetTiles(
    isl::UnionSet<Statement> domain,
    isl::MultiUnionPwAff<Statement, Prefix> prefix,
    isl::MultiUnionPwAff<Statement, Pretile> pretile,
    isl::MultiVal<Pretile> size);

} // namespace polyhedral
} // namespace tc

#include "tc/core/polyhedral/separation-inl.h"
