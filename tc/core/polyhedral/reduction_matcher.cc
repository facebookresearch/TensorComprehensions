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
#include "tc/core/polyhedral/domain_types.h"
#include "tc/core/polyhedral/schedule_tree.h"
#include "tc/core/polyhedral/scop.h"
#include "tc/external/isl.h"

namespace tc {
namespace polyhedral {

using detail::ScheduleTree;
using detail::ScheduleTreeBand;
using detail::ScheduleTreeFilter;

namespace {

/*
 * Does the given statement perform a supported type of reduction?
 * Only addition is supported for now since it is not clear
 * if other types are supported by the CUB reduction wrapper.
 */
bool isSupportedReduction(Halide::Internal::Stmt stmt) {
  auto provide = stmt.as<Halide::Internal::Provide>();
  auto call = provide->values[0].as<Halide::Internal::Call>();
  return call && call->args.size() > 0 &&
      call->args[0].as<Halide::Internal::Add>();
}

// If id is the statement identifier of an update statement
// of a supported type of reduction, then return true.
bool isSupportedReductionUpdateId(isl::id id, const Scop& scop) {
  TC_CHECK_EQ(scop.halide.statements.count(id), 1u)
      << "id is not a statement in scop" << id;
  auto provideNode = scop.halide.statements.at(id);
  return isSupportedReduction(provideNode);
}

} // namespace

isl::UnionSet<Statement> reductionUpdates(
    isl::UnionSet<Statement> domain,
    const Scop& scop) {
  domain = scop.body.reductions.intersect_domain(domain).domain();
  auto update = isl::UnionSet<Statement>::empty(domain.get_space());
  domain.foreach_set([&update, &scop](isl::Set<Statement> set) {
    auto setId = set.get_tuple_id();
    if (isSupportedReductionUpdateId(setId, scop)) {
      update = update.unite(set);
    }
  });
  return update;
}

} // namespace polyhedral
} // namespace tc
