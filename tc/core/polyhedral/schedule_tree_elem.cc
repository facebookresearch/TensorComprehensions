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
#include "tc/core/polyhedral/schedule_tree_elem.h"

#include <algorithm>
#include <iostream>
#include <memory>
#include <string>
#include <vector>

#include <glog/logging.h>

#include "tc/core/check.h"
#include "tc/core/constants.h"
#include "tc/core/flags.h"
#include "tc/core/polyhedral/schedule_isl_conversion.h"
#include "tc/core/polyhedral/schedule_tree.h"
#include "tc/core/scope_guard.h"
#include "tc/external/isl.h"

using namespace std;

namespace tc {
namespace polyhedral {
namespace detail {

std::unique_ptr<ScheduleTreeContext> ScheduleTreeContext::make(
    isl::set context,
    std::vector<ScheduleTreeUPtr>&& children) {
  auto res =
      std::unique_ptr<ScheduleTreeContext>(new ScheduleTreeContext(context));
  res->appendChildren(std::move(children));
  return res;
}

std::unique_ptr<ScheduleTreeContext> ScheduleTreeContext::make(
    const ScheduleTreeContext* tree,
    std::vector<ScheduleTreeUPtr>&& children) {
  auto res =
      std::unique_ptr<ScheduleTreeContext>(new ScheduleTreeContext(*tree));
  res->appendChildren(std::move(children));
  return res;
}

std::unique_ptr<ScheduleTreeDomain> ScheduleTreeDomain::make(
    isl::union_set domain,
    std::vector<ScheduleTreeUPtr>&& children) {
  auto res =
      std::unique_ptr<ScheduleTreeDomain>(new ScheduleTreeDomain(domain));
  res->appendChildren(std::move(children));
  return res;
}

std::unique_ptr<ScheduleTreeDomain> ScheduleTreeDomain::make(
    const ScheduleTreeDomain* tree,
    std::vector<ScheduleTreeUPtr>&& children) {
  auto res = std::unique_ptr<ScheduleTreeDomain>(new ScheduleTreeDomain(*tree));
  res->appendChildren(std::move(children));
  return res;
}

std::unique_ptr<ScheduleTreeExtension> ScheduleTreeExtension::make(
    isl::union_map extension,
    std::vector<ScheduleTreeUPtr>&& children) {
  auto res = std::unique_ptr<ScheduleTreeExtension>(
      new ScheduleTreeExtension(extension));
  res->appendChildren(std::move(children));
  return res;
}

std::unique_ptr<ScheduleTreeExtension> ScheduleTreeExtension::make(
    const ScheduleTreeExtension* tree,
    std::vector<ScheduleTreeUPtr>&& children) {
  auto res =
      std::unique_ptr<ScheduleTreeExtension>(new ScheduleTreeExtension(*tree));
  res->appendChildren(std::move(children));
  return res;
}

std::unique_ptr<ScheduleTreeFilter> ScheduleTreeFilter::make(
    isl::union_set filter,
    std::vector<ScheduleTreeUPtr>&& children) {
  auto res =
      std::unique_ptr<ScheduleTreeFilter>(new ScheduleTreeFilter(filter));
  res->appendChildren(std::move(children));
  return res;
}

std::unique_ptr<ScheduleTreeFilter> ScheduleTreeFilter::make(
    const ScheduleTreeFilter* tree,
    std::vector<ScheduleTreeUPtr>&& children) {
  auto res = std::unique_ptr<ScheduleTreeFilter>(new ScheduleTreeFilter(*tree));
  res->appendChildren(std::move(children));
  return res;
}

std::unique_ptr<ScheduleTreeMapping> ScheduleTreeMapping::make(
    isl::ctx ctx,
    const ScheduleTreeMapping::Mapping& mapping,
    std::vector<ScheduleTreeUPtr>&& children) {
  auto res = std::unique_ptr<ScheduleTreeMapping>(
      new ScheduleTreeMapping(ctx, mapping));
  res->appendChildren(std::move(children));
  return res;
}

std::unique_ptr<ScheduleTreeMapping> ScheduleTreeMapping::make(
    const ScheduleTreeMapping* tree,
    std::vector<ScheduleTreeUPtr>&& children) {
  auto res =
      std::unique_ptr<ScheduleTreeMapping>(new ScheduleTreeMapping(*tree));
  res->appendChildren(std::move(children));
  return res;
}

ScheduleTreeMapping::ScheduleTreeMapping(
    isl::ctx ctx,
    const ScheduleTreeMapping::Mapping& mapping)
    : ScheduleTree(ctx, {}, NodeType), mapping(mapping) {
  TC_CHECK_GT(mapping.size(), 0u) << "empty mapping filter";

  auto domain = mapping.cbegin()->second.domain();
  for (auto& kvp : mapping) {
    TC_CHECK(domain.is_equal(kvp.second.domain()));
  }
  filter_ = domain.universe();
  for (auto& kvp : mapping) {
    auto upa = kvp.second;
    auto id = kvp.first;
    // Create mapping filter by equating the
    // parameter mappedIds[i] to the "i"-th affine function.
    upa = upa.sub(isl::union_pw_aff::param_on_domain(domain.universe(), id));
    filter_ = filter_.intersect(upa.zero_union_set());
  }
}

std::unique_ptr<ScheduleTreeSequence> ScheduleTreeSequence::make(
    isl::ctx ctx,
    std::vector<ScheduleTreeUPtr>&& children) {
  auto res =
      std::unique_ptr<ScheduleTreeSequence>(new ScheduleTreeSequence(ctx));
  res->appendChildren(std::move(children));
  return res;
}

std::unique_ptr<ScheduleTreeSequence> ScheduleTreeSequence::make(
    const ScheduleTreeSequence* tree,
    std::vector<ScheduleTreeUPtr>&& children) {
  auto res =
      std::unique_ptr<ScheduleTreeSequence>(new ScheduleTreeSequence(*tree));
  res->appendChildren(std::move(children));
  return res;
}

std::unique_ptr<ScheduleTreeSet> ScheduleTreeSet::make(
    isl::ctx ctx,
    std::vector<ScheduleTreeUPtr>&& children) {
  auto res = std::unique_ptr<ScheduleTreeSet>(new ScheduleTreeSet(ctx));
  res->appendChildren(std::move(children));
  return res;
}

std::unique_ptr<ScheduleTreeSet> ScheduleTreeSet::make(
    const ScheduleTreeSet* tree,
    std::vector<ScheduleTreeUPtr>&& children) {
  auto res = std::unique_ptr<ScheduleTreeSet>(new ScheduleTreeSet(*tree));
  res->appendChildren(std::move(children));
  return res;
}

std::unique_ptr<ScheduleTreeBand> ScheduleTreeBand::make(
    isl::multi_union_pw_aff mupa,
    bool permutable,
    std::vector<bool> coincident,
    std::vector<bool> unroll,
    std::vector<ScheduleTreeUPtr>&& children) {
  TC_CHECK_EQ(static_cast<size_t>(mupa.size()), coincident.size());
  TC_CHECK_EQ(static_cast<size_t>(mupa.size()), unroll.size());
  isl::ctx ctx(mupa.get_ctx());
  std::unique_ptr<ScheduleTreeBand> band(new ScheduleTreeBand(ctx));
  band->permutable_ = permutable;
  band->mupa_ = mupa.floor();
  band->coincident_ = coincident;
  band->unroll_ = unroll;
  band->appendChildren(std::move(children));
  return band;
}

std::unique_ptr<ScheduleTreeBand> ScheduleTreeBand::make(
    const ScheduleTreeBand* tree,
    std::vector<ScheduleTreeUPtr>&& children) {
  auto res = std::unique_ptr<ScheduleTreeBand>(new ScheduleTreeBand(*tree));
  res->appendChildren(std::move(children));
  return res;
}

// Return the number of scheduling dimensions in the band
size_t ScheduleTreeBand::nMember() const {
  size_t res = mupa_.size();
  TC_CHECK_EQ(res, coincident_.size());
  TC_CHECK_EQ(res, unroll_.size());
  return res;
}

size_t ScheduleTreeBand::nOuterCoincident() const {
  TC_CHECK_EQ(nMember(), coincident_.size());
  size_t i;
  for (i = 0; i < nMember(); ++i) {
    if (!coincident_[i]) {
      break;
    }
  }
  return i;
}

void ScheduleTreeBand::drop(size_t pos, size_t n) {
  TC_CHECK_LE(0u, n) << "range out of bounds";
  TC_CHECK_LE(0u, pos) << "range  out of bounds";
  TC_CHECK_GE(nMember(), pos + n) << "range out of bounds";
  auto nBegin = nMember();

  auto list = mupa_.get_union_pw_aff_list();
  auto space = mupa_.get_space().params();
  list = list.drop(pos, n);
  space = space.add_unnamed_tuple_ui(list.size());
  mupa_ = isl::multi_union_pw_aff(space, list);

  std::copy(
      coincident_.begin() + pos + n,
      coincident_.end(),
      coincident_.begin() + pos);
  coincident_.resize(nBegin - n);
  std::copy(unroll_.begin() + pos + n, unroll_.end(), unroll_.begin() + pos);
  unroll_.resize(nBegin - n);
  TC_CHECK_EQ(nBegin - n, nMember());
}

isl::multi_union_pw_aff ScheduleTreeBand::memberRange(size_t first, size_t n)
    const {
  auto list = mupa_.get_union_pw_aff_list();
  auto space = mupa_.get_space().params().add_unnamed_tuple_ui(n);
  auto end = first + n;
  TC_CHECK_LE(end, nMember());
  list = list.drop(end, nMember() - end);
  list = list.drop(0, first);
  return isl::multi_union_pw_aff(space, list);
}

std::unique_ptr<ScheduleTreeThreadSpecificMarker>
ScheduleTreeThreadSpecificMarker::make(
    isl::ctx ctx,
    std::vector<ScheduleTreeUPtr>&& children) {
  auto res = std::unique_ptr<ScheduleTreeThreadSpecificMarker>(
      new ScheduleTreeThreadSpecificMarker(ctx));
  res->appendChildren(std::move(children));
  return res;
}

std::unique_ptr<ScheduleTreeThreadSpecificMarker>
ScheduleTreeThreadSpecificMarker::make(
    const ScheduleTreeThreadSpecificMarker* tree,
    std::vector<ScheduleTreeUPtr>&& children) {
  auto res = std::unique_ptr<ScheduleTreeThreadSpecificMarker>(
      new ScheduleTreeThreadSpecificMarker(*tree));
  res->appendChildren(std::move(children));
  return res;
}

bool ScheduleTreeBand::nodeEquals(const ScheduleTreeBand* otherBand) const {
  if (!otherBand) {
    return false;
  }
  if (permutable_ != otherBand->permutable_) {
    return false;
  }
  if (coincident_.size() != otherBand->coincident_.size()) {
    return false;
  }
  if (unroll_.size() != otherBand->unroll_.size()) {
    return false;
  }
  if (!std::equal(
          coincident_.begin(),
          coincident_.end(),
          otherBand->coincident_.begin())) {
    return false;
  }
  if (!std::equal(unroll_.begin(), unroll_.end(), otherBand->unroll_.begin())) {
    return false;
  }

  // Compare partial schedules by converting them to union_maps.  If a partial
  // schedule is zero-dimensional, it is not convertible to a union map for
  // further comparison.  Compare its explicit domains instead.  Note that
  // .domain() returns a zero-dimensional union set (in purely parameter space)
  // if there is no explicit domain.
  bool mupaIs0D = nMember() == 0;
  bool otherMupaIs0D = otherBand->nMember() == 0;
  if (mupaIs0D ^ otherMupaIs0D) {
    return false;
  }
  if (mupaIs0D && otherMupaIs0D) {
    auto d1 = mupa_.domain();
    auto d2 = otherBand->mupa_.domain();
    auto res = d1.is_equal(d2);
    if (!res) {
      LOG_IF(INFO, FLAGS_debug_tc_mapper)
          << "0D band MUPAs have different domains:" << std::endl
          << d1 << std::endl
          << d2 << std::endl;
      return false;
    }
  } else {
    auto m1 = isl::union_map::from(mupa_);
    auto m2 = isl::union_map::from(otherBand->mupa_);
    {
      auto res = m1.is_equal(m2);
      if (!res) {
        LOG_IF(INFO, FLAGS_debug_tc_mapper) << "Band mupa_:\n"
                                            << m1 << "\n\tVS\n"
                                            << m2 << "\n";
        return false;
      }
    }
  }

  return true;
}

bool ScheduleTreeContext::nodeEquals(const ScheduleTreeContext* other) const {
  return other && context_.is_equal(other->context_);
}

bool ScheduleTreeDomain::nodeEquals(const ScheduleTreeDomain* other) const {
  if (!other) {
    return false;
  }
  auto res = domain_.is_equal(other->domain_);
  if (!res) {
    LOG_IF(INFO, FLAGS_debug_tc_mapper)
        << "ScheduleTreeDomain difference: " << domain_ << " VS "
        << other->domain_ << "\n";
  }
  return res;
}

bool ScheduleTreeExtension::nodeEquals(
    const ScheduleTreeExtension* other) const {
  return other && extension_.is_equal(other->extension_);
}

bool ScheduleTreeFilter::nodeEquals(const ScheduleTreeFilter* other) const {
  return other && filter_.is_equal(other->filter_);
}

bool ScheduleTreeMapping::nodeEquals(const ScheduleTreeMapping* other) const {
  if (mapping.size() != other->mapping.size()) {
    return false;
  }
  for (const auto& kvp : mapping) {
    if (other->mapping.count(kvp.first) == 0) {
      return false;
    }
    if (!other->mapping.at(kvp.first).plain_is_equal(kvp.second)) {
      return false;
    }
  }
  return filter_.is_equal(other->filter_);
}

bool ScheduleTreeSequence::nodeEquals(const ScheduleTreeSequence* other) const {
  return true;
}

bool ScheduleTreeSet::nodeEquals(const ScheduleTreeSet* other) const {
  return true;
}

bool ScheduleTreeThreadSpecificMarker::nodeEquals(
    const ScheduleTreeThreadSpecificMarker* other) const {
  return true;
}

} // namespace detail
} // namespace polyhedral
} // namespace tc
