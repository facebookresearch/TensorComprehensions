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

#include <vector>

#include <isl/cpp.h>

#include "tc/external/isl.h"

namespace isl {

// Via "class-specific" get:
//   pma -> pa
//   umpa -> upa
//
// Via "class-specific" foreach:
//   pa -> foreach_piece (set, aff)
//   pma -> foreach_piece (set, ma)
//   upma -> foreach_pma (pma)
//   upa -> foreach_pa (pa)
//
// Via multi.h' get:
//   multi -> base e.g.:
//     mupa -> upa
//     mpa -> pa
//     ma -> a
//     mv -> v
//
// Via "class-specific" extract(space)
//   mupa -> mpa
//   umpa -> pma
//   upa -> pa
//
// Reverse listing:
//   mv -> v
//
//   ma -> a
//
//   pa -> (set, aff)
//
//   pma -> (set, ma) -> (set, aff)
//
//   upa : extract -> pa -> (set, aff)
//
//   mpa -> pa -> (set, aff)
//
//   upma -> upa -> pa -> (set, aff)
//       : extract -> pma -> (set, ma) -> (set, aff)
//
//   mupa -> upa -> pa -> (set, aff)
//       : extract -> mpa -> pa -> (set, aff)

// You have simple expressions, like isl_val for values and isl_aff for affine
// functions. If you want vectors instead, you do multi expressions, multi_val
// and multi_aff.
// In this particular case vector spaces have named dimensions
// and a tuple name, so there's a set space associated to each multi_* so
// multi_val is a vector in the vector space ```multi_val.get_space()``` and
// as everywhere in isl, spaces can be parametric, so you may have a
// parametric vector in a vector space that has both set and parameter
// dimensions.
//
// union is a different story, union_* are defined over different spaces
// for affine functions, it means they have different domain spaces.
//
// multi_* is a vector, we just need to differentiate vectors in different
// spaces of the same dimensionality. It is a math vector except that we don't
// have non-elementwise operations.
//
// multi_aff is a vector-valued function, but just a plain aff is a
// scalar-valued multivariate function plus parameters everywhere.

template <typename T, isl::dim_type DT>
struct DimIds : public std::vector<isl::id> {
  DimIds(T s) {
    this->reserve(s.dim(DT));
    for (size_t i = 0; i < s.dim(DT); ++i) {
      this->push_back(s.get_dim_id(DT, i));
    }
  }
};

} // namespace isl
