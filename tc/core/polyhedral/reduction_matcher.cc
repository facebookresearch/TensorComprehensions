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
#include "tc/core/polyhedral/schedule_tree_matcher.h"

#include <unordered_set>

#include "tc/core/check.h"
#include "tc/core/polyhedral/schedule_tree.h"
#include "tc/core/polyhedral/scop.h"
#include "tc/external/isl.h"

namespace tc {
namespace polyhedral {

using detail::ScheduleTree;
using detail::ScheduleTreeElemBand;
using detail::ScheduleTreeElemFilter;

namespace {

/*
 * Does the given statement perform a supported type of reduction?
 * Only addition is supported for now since it is not clear
 * if other types are supported by the CUB reduction wrapper.
 */
bool isSupportedReduction(Halide::Internal::Stmt stmt) {
  auto provide = stmt.as<Halide::Internal::Provide>();
  auto call = provide->values[0].as<Halide::Internal::Call>();
  if (call && call->args[0].as<Halide::Internal::Add>()) {
    return true;
  }
  return false;
}

// TODO: the function currently available in Scop only works _after_ inserting
// the reduction.  that is a kind of internal state dependence we want to avoid
// If id is the statement identifier of an update statement
// of a supported type of reduction,
// then return the corresponding reduction dimensions in reductionDims.
bool isReductionUpdateId(
    isl::id id,
    const Scop& scop,
    std::vector<size_t>& reductionDims) {
  TC_CHECK_EQ(scop.halide.statements.count(id), 1u)
      << "id is not a statement in scop" << id;
  auto provideNode = scop.halide.statements.at(id);
  if (!isSupportedReduction(provideNode)) {
    return false;
  }
  for (auto const& iup : scop.halide.reductions) {
    if (iup.update.same_as(provideNode)) {
      reductionDims = iup.dims;
      return true;
    }
  }
  return false;
}

/*
 * Does "aff" only involve the specified input dimension (and not
 * any other input dimensions).
 */
bool pwAffInvolvesOnlyInputDim(isl::pw_aff pa, int redDimIdx) {
  auto space = pa.get_space();

  if (!pa.involves_dims(isl::dim_type::in, redDimIdx, 1)) {
    return false;
  }

  if (pa.involves_dims(isl::dim_type::in, 0, redDimIdx) ||
      pa.involves_dims(
          isl::dim_type::in,
          redDimIdx + 1,
          space.dim(isl::dim_type::in) - redDimIdx - 1)) {
    return false;
  }

  return true;
}

// Does pa have the form S(...) -> [(K*r)] where S is either a reduction init
// or update statement and r is a known reduction loop in Scop?
//
// FIXME: now, K can be any value, including nested integer divisions, to
// support detection after tiling; tighten this.
bool isAlmostIdentityReduction(isl::pw_aff pa, const Scop& scop) {
  auto space = pa.get_space();
  if (!space.has_tuple_id(isl::dim_type::in)) {
    return false;
  }
  auto stmtId = space.get_tuple_id(isl::dim_type::in);
  std::vector<size_t> reductionDims;
  if (!isReductionUpdateId(stmtId, scop, reductionDims)) {
    return false;
  }

  for (auto redDimIdx : reductionDims) {
    if (pwAffInvolvesOnlyInputDim(pa, redDimIdx)) {
      return true;
    }
  }
  return false;
}

} // namespace

isl::union_set reductionUpdates(isl::union_set domain, const Scop& scop) {
  auto update = isl::union_set::empty(domain.get_space());
  domain.foreach_set([&update, &scop](isl::set set) {
    auto setId = set.get_tuple_id();
    std::vector<size_t> reductionDims;
    if (isReductionUpdateId(setId, scop, reductionDims)) {
      update = update.unite(set);
    }
  });
  return update;
}

bool isReductionMember(
    isl::union_pw_aff member,
    isl::union_set domain,
    const Scop& scop) {
  return domain.every_set([member, &scop](isl::set set) {
    auto pa = member.extract_on_domain(set.get_space());
    return isAlmostIdentityReduction(pa, scop);
  });
}

} // namespace polyhedral
} // namespace tc
