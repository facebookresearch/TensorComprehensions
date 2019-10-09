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

#include <iostream>
#include <sstream>
#include <string>
#include <unordered_map>

#include <Halide.h>

#include "tc/core/polyhedral/body.h"
#include "tc/core/polyhedral/domain_types.h"
#include "tc/core/polyhedral/schedule_tree.h"
#include "tc/core/tc2halide.h"
#include "tc/external/isl.h"

namespace tc {
namespace halide2isl {

/// \file halide2isl.h
/// Helper functions that participate in translating Halide IR to ISL
///

using ParameterVector = std::vector<Halide::Internal::Parameter>;
/// Find and categorize all variables referenced in a piece of Halide IR
struct SymbolTable {
  std::vector<std::string> reductionVars, idxVars;
  ParameterVector params;
};
SymbolTable makeSymbolTable(const tc2halide::HalideComponents& components);

/// Make the space of all given parameter values
isl::Space<> makeParamSpace(isl::ctx ctx, const ParameterVector& params);

/// Make the parameter set marking all given parameters
/// as non-negative.
isl::Set<> makeParamContext(isl::ctx ctx, const ParameterVector& params);

/// Make a constant-valued affine function over a space.
isl::AffOn<> makeIslAffFromInt(isl::Space<> space, int64_t i);

// Make an affine function over a space from a Halide Expr. Returns a
// null isl::aff if the expression is not affine. Fails if Variable
// does not correspond to a parameter of the space.
// Note that the input space can be either a parameter space or
// a set space, but the expression can only reference
// the parameters in the space.
isl::AffOn<> makeIslAffFromExpr(isl::Space<> space, const Halide::Expr& e);

// Iteration domain information associated to a statement identifier.
struct IterationDomain {
  // All parameters active at the point where the iteration domain
  // was created, including those corresponding to outer loop iterators.
  isl::Space<> paramSpace;
  // The identifier tuple corresponding to the iteration domain.
  // The identifiers in the tuple are the outer loop iterators,
  // from outermost to innermost.
  isl::MultiId<polyhedral::Statement> tuple;
};

typedef std::unordered_map<isl::id, IterationDomain, isl::IslIdIslHash>
    IterationDomainMap;
typedef std::unordered_map<isl::id, Halide::Internal::Stmt, isl::IslIdIslHash>
    StatementMap;
typedef std::unordered_map<const Halide::Internal::IRNode*, isl::id> AccessMap;
struct ScheduleTreeAndAccesses {
  /// The schedule tree. This encodes the loop structure, but not the
  /// leaf statements. Leaf statements are replaced with IDs of the
  /// form S_N. The memory access patterns and the original statement
  /// for each leaf node is captured below.
  tc::polyhedral::ScheduleTreeUPtr tree;

  /// Information extracted from the bodies of the statements.
  tc::polyhedral::Body body;

  /// The correspondence between from Call and Provide nodes and the
  /// reference ids in the reads and writes maps.
  AccessMap accesses;

  /// The correspondence between leaf Stmts and the statement ids
  /// refered to above.
  StatementMap statements;

  /// The correspondence between statement ids and the iteration domain
  /// of the corresponding leaf Stmt.
  IterationDomainMap domains;
};

/// Make a schedule tree from a Halide Stmt, along with auxiliary data
/// structures describing the memory access patterns.
ScheduleTreeAndAccesses makeScheduleTree(
    isl::Space<> paramSpace,
    const Halide::Internal::Stmt& s);

} // namespace halide2isl
} // namespace tc
