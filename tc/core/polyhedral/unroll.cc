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

#include "tc/core/polyhedral/unroll.h"

#include "tc/core/polyhedral/schedule_transforms.h"
#include "tc/core/polyhedral/schedule_tree.h"

namespace tc {
namespace polyhedral {

namespace {
/*
 * Return a bound on the range of values attained by "f" for fixed
 * values of "fixed", taking into account basic strides
 * in the range of values attained by "f".
 *
 * First construct a map from values of "fixed" to corresponding
 * values of "f".  If this map is empty, then "f" cannot attain
 * any values and the bound is zero.
 * Otherwise, consider pairs of "f" values for the same value
 * of "fixed" and take their difference over all possible values
 * of the parameters and of the "fixed" values.
 * Take a simple overapproximation as a convex set and
 * determine the stride is the value differences.
 * The possibly quasi-affine set is then overapproximated by an affine set.
 * At this point, the set is a possibly infinite, symmetrical interval.
 * Take the maximal value of the difference divided by the stride plus one as
 * a bound on the number of possible values of "f".
 * That is, take M/s + 1.  Note that 0 is always an element of
 * the difference set, so no offset needs to be taken into account
 * during the stride computation and M is an integer multiple of s.
 */
isl::val relativeRange(isl::union_map fixed, isl::union_pw_aff f) {
  auto ctx = f.get_ctx();
  auto umap = isl::union_map::from(isl::multi_union_pw_aff(f));
  umap = umap.apply_domain(fixed);
  if (umap.is_empty()) {
    return isl::val::zero(ctx);
  }

  umap = umap.range_product(umap);
  umap = umap.range().unwrap();
  umap = umap.project_out_all_params();
  auto delta = isl::map::from_union_map(umap).deltas();
  auto hull = delta.simple_hull();
  auto stride = isl::set(hull).get_stride(0);
  hull = isl::set(hull).polyhedral_hull();
  auto bound = hull.dim_max_val(0);
  bound = bound.div(stride);
  bound = bound.add(isl::val::one(ctx));
  return bound;
}

/*
 * Compute a bound on the number of instances executed by "band" and
 * mark each member that executes at most "unrollFactor" instances
 * for unrolling.
 * "prefix" is the schedule defined by the ancestors.
 * "bound" is a bound on the number of instances executed by
 * the descendants of "band".
 * As soon as the bound exceeds unrollFactor, simply return infinity.
 *
 * The bound for the number of instances executed in the direction
 * of a band member is taken to be the number of values attained by this member
 * for fixed values of the outer bands and members.
 * The total number of instances executed by this band member
 * is taken to be the product of this number with that of
 * inner band members and the bound on the descendants.
 */
isl::val boundInstancesAndMarkUnroll(
    detail::ScheduleTreeElemBand* band,
    isl::union_map prefix,
    isl::val unrollFactor,
    isl::val bound) {
  if (bound.gt(unrollFactor)) {
    return isl::val::infty(bound.get_ctx());
  }

  auto partial = band->mupa_;
  auto n = band->nMember();

  for (int i = n - 1; i >= 0; --i) {
    auto member = partial.get_union_pw_aff(i);
    auto outerMap = prefix;
    if (i > 0) {
      auto outer = partial.drop_dims(isl::dim_type::set, i, n - i);
      outerMap = outerMap.flat_range_product(isl::union_map::from(outer));
    }
    bound = bound.mul(relativeRange(outerMap, member));
    if (bound.gt(unrollFactor)) {
      return isl::val::infty(bound.get_ctx());
    }
    band->unroll_[i] = true;
  }

  return bound;
}

isl::val boundInstancesAndMarkUnroll(
    detail::ScheduleTree* st,
    isl::union_map prefix,
    isl::val unrollFactor);

/*
 * Compute a bound on the number of instances executed by the children of "st"
 * and mark any band member in any descendant band node that executes
 * at most "unrollFactor" instances for unrolling.
 * "prefix" is the schedule defined by the ancestors.
 * If the bound exceeds "unrollFactor", then infinity may be returned.
 *
 * Update the prefix schedule for use in the children.
 *
 * If "st" has no children, then the number of instances is 1.
 * Otherwise, it is the sum of the number of instances executed
 * by the individual children.
 */
isl::val boundChildrenInstancesAndMarkUnroll(
    detail::ScheduleTree* st,
    isl::union_map prefix,
    isl::val unrollFactor) {
  if (st->children().size() == 0) {
    return isl::val::one(unrollFactor.get_ctx());
  }

  prefix = extendSchedule(st, prefix);

  auto bound = isl::val::zero(unrollFactor.get_ctx());
  for (const auto& c : st->children()) {
    bound = bound.add(boundInstancesAndMarkUnroll(c, prefix, unrollFactor));
  }
  return bound;
}

/*
 * Compute a bound on the number of instances executed by "st" and
 * mark any band member in any descendant band node that executes
 * at most "unrollFactor" instances for unrolling.
 * "prefix" is the schedule defined by the ancestors.
 *
 * First obtain a bound on the number of instances executed
 * by the children of "st", marking any descendant band node.
 * If "st" is a band node, then multiply the bound by an estimate
 * of the number of iterations of the loops corresponding to the band node and
 * mark the appropriate members of the node.
 */
isl::val boundInstancesAndMarkUnroll(
    detail::ScheduleTree* st,
    isl::union_map prefix,
    isl::val unrollFactor) {
  auto bound = boundChildrenInstancesAndMarkUnroll(st, prefix, unrollFactor);

  if (auto band = st->elemAs<detail::ScheduleTreeElemBand>()) {
    bound = boundInstancesAndMarkUnroll(band, prefix, unrollFactor, bound);
  }

  return bound;
}
} // namespace

void markUnroll(
    detail::ScheduleTree* root,
    detail::ScheduleTree* st,
    uint64_t unroll) {
  if (unroll <= 1) {
    return;
  }

  auto unrollVal = isl::val(st->ctx_, unroll);
  auto prefix = prefixSchedule(root, st);
  prefix = prefix.intersect_domain(prefixMappingFilter(root, st));
  boundInstancesAndMarkUnroll(st, prefix, unrollVal);
}
} // namespace polyhedral
} // namespace tc
