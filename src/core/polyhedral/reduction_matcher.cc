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

#include "tc/core/polyhedral/cuda/mapped_scop.h"
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
// then return the corresponding init statement in init and
// the corresponding reduction dimensions in reductionDims.
bool isReductionUpdateId(
    isl::id id,
    const Scop& scop,
    Halide::Internal::Stmt& init,
    std::vector<size_t>& reductionDims) {
  CHECK_EQ(scop.halide.statements.count(id), 1)
      << "id is not a statement in scop" << id;
  auto provideNode = scop.halide.statements.at(id);
  if (!isSupportedReduction(provideNode)) {
    return false;
  }
  for (auto const& iup : scop.halide.reductions) {
    if (iup.update.same_as(provideNode)) {
      init = iup.init;
      reductionDims = iup.dims;
      return true;
    }
  }
  return false;
}

bool affInvolvesOnlyDim(isl::aff aff, int redDimIdx) {
  auto space = aff.get_space();

  bool hasPureDim = aff.involves_dims(isl::dim_type::in, redDimIdx, 1);

  bool divsInvolveOtherDims = false;
  bool divsInvolveDim = false;
  for (int i = 0, ei = aff.dim(isl::dim_type::div); i < ei; ++i) {
    // Ignore divs with coefficient 0 that may be referred to by the aff.
    // This is particularly the case when we are already processing the div.
    if (aff.get_coefficient_val(isl::dim_type::div, i).is_zero()) {
      continue;
    }
    bool divR = affInvolvesOnlyDim(aff.get_div(i), redDimIdx);
    divsInvolveDim |=
        divR; // becomes true if at least one involves the given dim
    divsInvolveOtherDims |=
        !divR; // becomes true if at least one involves other dims
  }

  bool involvesOtherDims = aff.involves_dims(isl::dim_type::in, 0, redDimIdx) ||
      aff.involves_dims(
          isl::dim_type::in,
          redDimIdx + 1,
          space.dim(isl::dim_type::in) - redDimIdx - 1) ||
      aff.involves_dims(
          isl::dim_type::param, 0, space.dim(isl::dim_type::param));

  if (involvesOtherDims) {
    return false;
  }
  if (!hasPureDim && !divsInvolveDim) {
    return false;
  }
  if (divsInvolveDim && divsInvolveOtherDims) {
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
  Halide::Internal::Stmt init;
  std::vector<size_t> reductionDims;
  if (!isReductionUpdateId(stmtId, scop, init, reductionDims)) {
    return false;
  }

  auto paWrapper = isl::PA(pa);
  if (paWrapper.size() != 1) {
    return false;
  }

  auto aff = paWrapper[0].second;
  for (auto redDimIdx : reductionDims) {
    if (affInvolvesOnlyDim(aff, redDimIdx)) {
      return true;
    }
  }
  return false;
}

/*
 * Return the identifier that maps to "stmt".
 */
isl::id statementId(const Scop& scop, const Halide::Internal::Stmt& stmt) {
  for (auto kvp : scop.halide.statements) {
    if (kvp.second.same_as(stmt)) {
      return kvp.first;
    }
  }
  CHECK(false) << "no id recorded for statement" << stmt;
  return isl::id();
}

} // namespace

std::pair<isl::union_set, std::vector<isl::id>> reductionInitsUpdates(
    isl::union_set domain,
    const Scop& scop) {
  auto initUnion = isl::union_set::empty(domain.get_space());
  std::vector<isl::id> update;
  std::unordered_set<isl::id, isl::IslIdIslHash> init;
  std::vector<isl::set> nonUpdate;
  // First collect all the update statement identifiers,
  // the corresponding init statement and all non-update statements.
  domain.foreach_set([&init, &update, &nonUpdate, &scop](isl::set set) {
    auto setId = set.get_tuple_id();
    Halide::Internal::Stmt initStmt;
    std::vector<size_t> reductionDims;
    if (isReductionUpdateId(setId, scop, initStmt, reductionDims)) {
      update.emplace_back(setId);
      init.emplace(statementId(scop, initStmt));
    } else {
      nonUpdate.emplace_back(set);
    }
  });
  // Then check if all the non-update statements are init statements
  // that correspond to the update statements found.
  // If not, return an empty list of update statement identifiers.
  for (auto set : nonUpdate) {
    if (init.count(set.get_tuple_id()) != 1) {
      return std::pair<isl::union_set, std::vector<isl::id>>(
          initUnion, std::vector<isl::id>());
    }
    initUnion = initUnion.unite(set);
  }
  return std::pair<isl::union_set, std::vector<isl::id>>(initUnion, update);
}

int findFirstReductionDim(isl::multi_union_pw_aff islMupa, const Scop& scop) {
  auto mupa = isl::MUPA(islMupa);
  int reductionDim = -1;
  int currentDim = 0;
  for (auto const& upa : mupa) {
    for (auto const& pa : upa) {
      if (isAlmostIdentityReduction(pa.pa, scop)) {
        reductionDim = currentDim;
        break;
      }
    }
    if (reductionDim != -1) {
      break;
    }
    ++currentDim;
  }
  return reductionDim;
}

} // namespace polyhedral
} // namespace tc
