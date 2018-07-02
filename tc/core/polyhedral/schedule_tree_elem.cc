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

std::unique_ptr<ScheduleTreeElemBand> ScheduleTreeElemBand::fromMultiUnionPwAff(
    isl::multi_union_pw_aff mupa) {
  isl::ctx ctx(mupa.get_ctx());
  std::unique_ptr<ScheduleTreeElemBand> band(new ScheduleTreeElemBand(ctx));
  band->mupa_ = mupa.floor();
  size_t n = band->mupa_.size();
  band->coincident_ = vector<bool>(n, false);
  band->unroll_ = vector<bool>(n, false);
  return band;
}

// Return the number of scheduling dimensions in the band
size_t ScheduleTreeElemBand::nMember() const {
  size_t res = mupa_.size();
  TC_CHECK_EQ(res, coincident_.size());
  TC_CHECK_EQ(res, unroll_.size());
  return res;
}

size_t ScheduleTreeElemBand::nOuterCoincident() const {
  TC_CHECK_EQ(nMember(), coincident_.size());
  size_t i;
  for (i = 0; i < nMember(); ++i) {
    if (!coincident_[i]) {
      break;
    }
  }
  return i;
}

void ScheduleTreeElemBand::drop(size_t pos, size_t n) {
  TC_CHECK_LE(0u, n) << "range out of bounds";
  TC_CHECK_LE(0u, pos) << "range  out of bounds";
  TC_CHECK_GE(nMember(), pos + n) << "range out of bounds";
  auto nBegin = nMember();

  auto list = mupa_.get_union_pw_aff_list();
  auto space = mupa_.get_space().domain();
  list = list.drop(pos, n);
  space = addRange(space, list.size());
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

isl::multi_union_pw_aff ScheduleTreeElemBand::memberRange(
    size_t first,
    size_t n) const {
  auto list = mupa_.get_union_pw_aff_list();
  auto space = addRange(mupa_.get_space().domain(), n);
  auto end = first + n;
  TC_CHECK_LE(end, nMember());
  list = list.drop(end, nMember() - end);
  list = list.drop(0, first);
  return isl::multi_union_pw_aff(space, list);
}

bool ScheduleTreeElemBand::operator==(const ScheduleTreeElemBand& other) const {
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

bool ScheduleTreeElemContext::operator==(
    const ScheduleTreeElemContext& other) const {
  auto res = context_.is_equal(other.context_);
  return res;
}

bool ScheduleTreeElemDomain::operator==(
    const ScheduleTreeElemDomain& other) const {
  auto res = domain_.is_equal(other.domain_);
  if (!res) {
    LOG_IF(INFO, FLAGS_debug_tc_mapper)
        << "ScheduleTreeElemDomain difference: " << domain_ << " VS "
        << other.domain_ << "\n";
  }
  return res;
}

bool ScheduleTreeElemExtension::operator==(
    const ScheduleTreeElemExtension& other) const {
  auto res = extension_.is_equal(other.extension_);
  return res;
}

bool ScheduleTreeElemFilter::operator==(
    const ScheduleTreeElemFilter& other) const {
  auto res = filter_.is_equal(other.filter_);
  return res;
}

bool ScheduleTreeElemMapping::operator==(
    const ScheduleTreeElemMapping& other) const {
  auto res = filter_.is_equal(other.filter_);
  return res;
}

bool ScheduleTreeElemSequence::operator==(
    const ScheduleTreeElemSequence& other) const {
  return true;
}

bool ScheduleTreeElemSet::operator==(const ScheduleTreeElemSet& other) const {
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
  ELEM_EQUALS_CASE(ScheduleTreeElemBand)
  ELEM_EQUALS_CASE(ScheduleTreeElemContext)
  ELEM_EQUALS_CASE(ScheduleTreeElemDomain)
  ELEM_EQUALS_CASE(ScheduleTreeElemExtension)
  ELEM_EQUALS_CASE(ScheduleTreeElemFilter)
  ELEM_EQUALS_CASE(ScheduleTreeElemMapping)
  ELEM_EQUALS_CASE(ScheduleTreeElemSequence)
  ELEM_EQUALS_CASE(ScheduleTreeElemSet)
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
