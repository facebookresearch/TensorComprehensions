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
#include "tc/core/polyhedral/domain_types.h"
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
    : ScheduleTree(ctx, {}, NodeType),
      mapping(mapping),
      filter_(isl::union_set()) {
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
    upa = upa.sub(
        isl::UnionPwAffOn<Statement>::param_on_domain(domain.universe(), id));
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
    isl::MultiUnionPwAff<Statement, Band> mupa,
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
  auto spaceBand = space.add_unnamed_tuple_ui<Band>(list.size());
  mupa_ = isl::MultiUnionPwAff<Statement, Band>(spaceBand, list);

  std::copy(
      coincident_.begin() + pos + n,
      coincident_.end(),
      coincident_.begin() + pos);
  coincident_.resize(nBegin - n);
  std::copy(unroll_.begin() + pos + n, unroll_.end(), unroll_.begin() + pos);
  unroll_.resize(nBegin - n);
  TC_CHECK_EQ(nBegin - n, nMember());
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

bool ScheduleTreeBand::operator==(const ScheduleTreeBand& other) const {
  if (permutable_ != other.permutable_) {
    return false;
  }
  if (coincident_.size() != other.coincident_.size()) {
    return false;
  }
  if (unroll_.size() != other.unroll_.size()) {
    return false;
  }
  if (!std::equal(
          coincident_.begin(), coincident_.end(), other.coincident_.begin())) {
    return false;
  }
  if (!std::equal(unroll_.begin(), unroll_.end(), other.unroll_.begin())) {
    return false;
  }

  // Compare partial schedules by converting them to union_maps.  If a partial
  // schedule is zero-dimensional, it is not convertible to a union map for
  // further comparison.  Compare its explicit domains instead.  Note that
  // .domain() returns a zero-dimensional union set (in purely parameter space)
  // if there is no explicit domain.
  bool mupaIs0D = nMember() == 0;
  bool otherMupaIs0D = other.nMember() == 0;
  if (mupaIs0D ^ otherMupaIs0D) {
    return false;
  }
  if (mupaIs0D && otherMupaIs0D) {
    auto d1 = mupa_.domain();
    auto d2 = other.mupa_.domain();
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
    auto m2 = isl::union_map::from(other.mupa_);
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

bool ScheduleTreeContext::operator==(const ScheduleTreeContext& other) const {
  auto res = context_.is_equal(other.context_);
  return res;
}

bool ScheduleTreeDomain::operator==(const ScheduleTreeDomain& other) const {
  auto res = domain_.is_equal(other.domain_);
  if (!res) {
    LOG_IF(INFO, FLAGS_debug_tc_mapper)
        << "ScheduleTreeDomain difference: " << domain_ << " VS "
        << other.domain_ << "\n";
  }
  return res;
}

bool ScheduleTreeExtension::operator==(
    const ScheduleTreeExtension& other) const {
  auto res = extension_.is_equal(other.extension_);
  return res;
}

bool ScheduleTreeFilter::operator==(const ScheduleTreeFilter& other) const {
  auto res = filter_.is_equal(other.filter_);
  return res;
}

bool ScheduleTreeMapping::operator==(const ScheduleTreeMapping& other) const {
  auto res = filter_.is_equal(other.filter_);
  return res;
}

bool ScheduleTreeSequence::operator==(const ScheduleTreeSequence& other) const {
  return true;
}

bool ScheduleTreeSet::operator==(const ScheduleTreeSet& other) const {
  return true;
}

bool elemEquals(
    const ScheduleTree* e1,
    const ScheduleTree* e2,
    detail::ScheduleTreeType type) {
#define ELEM_EQUALS_CASE(CLASS)                                              \
  else if (type == CLASS::NodeType) {                                        \
    return *static_cast<const CLASS*>(e1) == *static_cast<const CLASS*>(e2); \
  }

  if (type == detail::ScheduleTreeType::None) {
    LOG(FATAL) << "Hit Error node!";
  }
  ELEM_EQUALS_CASE(ScheduleTreeBand)
  ELEM_EQUALS_CASE(ScheduleTreeContext)
  ELEM_EQUALS_CASE(ScheduleTreeDomain)
  ELEM_EQUALS_CASE(ScheduleTreeExtension)
  ELEM_EQUALS_CASE(ScheduleTreeFilter)
  ELEM_EQUALS_CASE(ScheduleTreeMapping)
  ELEM_EQUALS_CASE(ScheduleTreeSequence)
  ELEM_EQUALS_CASE(ScheduleTreeSet)
  else {
    LOG(FATAL) << "NYI: ScheduleTree::operator== for type: "
               << static_cast<int>(type);
  }

#undef ELEM_EQUALS_CASE

  return false;
}
} // namespace detail
} // namespace polyhedral
} // namespace tc
