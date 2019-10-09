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

#include "tc/core/polyhedral/domain_types.h"
#include "tc/external/isl.h"

namespace tc {
namespace polyhedral {

// Information about the bodies of the polyhedral statements.
struct Body {
  Body() = default;
  Body(isl::space paramSpace) {
    auto empty = isl::union_map::empty(paramSpace);
    writes = reads = isl::UnionMap<isl::Pair<Statement, Tag>, Tensor>(empty);
    reductions = isl::UnionMap<Statement, Reduction>(empty);
  }

  // Specialize to the given context.
  void specialize(isl::Set<> context) {
    reads = reads.intersect_params(context);
    writes = writes.intersect_params(context);
    reductions = reductions.intersect_params(context);
  }

  // Union maps describing the reads and writes done. Uses the ids in
  // the schedule tree to denote the containing Stmt, and tags each
  // access with a unique reference id of the form __tc_ref_N.
  isl::UnionMap<isl::Pair<Statement, Tag>, Tensor> reads, writes;

  // A function on reduction update statement instances that partitions them
  // into individual reductions, where each reduction consists of
  // associative updates to the same tensor element.
  // Since each reduction involves a single tensor element,
  // the partition of statement instances based
  // on the reductions forms a refinement of the partition based
  // on the element modified by the statement.
  // That is, if W is the write access relation and R is the reduction function,
  // then, in iscc notation, (W.W^-1) >= (R.R^-1).
  // In theory, it is possible for the inclusion to be strict, i.e.,
  // for (W.W^-1) > (R.R^-1) to hold.  For example, for a statement
  //
  //	A += T(i)
  //
  // with T of size 4, the write access relation is
  //
  //	{ S[i] -> A[] : 0 <= i < 4 }
  //
  // and the reduction relation could in theory be something like
  //
  //	{ S[i] -> R[i] : 0 <= i < 4 }
  //
  // or even
  //
  //	{ S[i] -> R1[i] : 0 <= i < 2; S[i] -> R2[i - 2] : 3 <= i < 4 }
  //
  // In practice, the reduction map is usually equal to
  // the write access relation on reduction update statements,
  // with different target spaces.
  // That is, in the example above, it would just be
  //
  //	{ S[i] -> R[] : 0 <= i < 4 }
  isl::UnionMap<Statement, Reduction> reductions;
};

std::ostream& operator<<(std::ostream& os, const Body& body);

} // namespace polyhedral
} // namespace tc
