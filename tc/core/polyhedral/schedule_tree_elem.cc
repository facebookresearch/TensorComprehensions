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

namespace {

std::unique_ptr<ScheduleTreeElemBand> fromIslScheduleNodeBand(
    isl::schedule_node_band b) {
  auto res =
      ScheduleTreeElemBand::fromMultiUnionPwAff(b.get_partial_schedule());
  res->permutable_ = b.get_permutable();
  for (size_t i = 0; i < b.n_member(); ++i) {
    res->coincident_[i] = b.member_get_coincident(i);
  }
  return res;
}

} // namespace

std::unique_ptr<ScheduleTreeElemBase> ScheduleTreeElemBase::make(
    isl::schedule_node node) {
  if (auto band = node.as<isl::schedule_node_band>()) {
    return fromIslScheduleNodeBand(band);
  } else if (auto context = node.as<isl::schedule_node_context>()) {
    auto c = context.get_context();
    return std::unique_ptr<ScheduleTreeElemContext>(
        new ScheduleTreeElemContext(c));
  } else if (auto domain = node.as<isl::schedule_node_domain>()) {
    auto c = domain.get_domain();
    return std::unique_ptr<ScheduleTreeElemDomain>(
        new ScheduleTreeElemDomain(c));
  } else if (auto expansion = node.as<isl::schedule_node_expansion>()) {
    LOG(FATAL) << "expansion nodes not supported";
    return nullptr;
  } else if (auto extension = node.as<isl::schedule_node_extension>()) {
    auto e = extension.get_extension();
    return std::unique_ptr<ScheduleTreeElemExtension>(
        new ScheduleTreeElemExtension(e));
  } else if (auto filter = node.as<isl::schedule_node_filter>()) {
    auto f = filter.get_filter();
    return std::unique_ptr<ScheduleTreeElemFilter>(
        new ScheduleTreeElemFilter(f));
  } else if (auto guard = node.as<isl::schedule_node_guard>()) {
    LOG(FATAL) << "guard nodes not supported";
    return nullptr;
  } else if (auto mark = node.as<isl::schedule_node_mark>()) {
    LOG(FATAL) << "mark nodes not supported";
    return nullptr;
  } else if (node.isa<isl::schedule_node_leaf>()) {
    LOG(FATAL) << "ScheduleTreeElemBase::make called on explicit leaf";
    return nullptr;
  } else if (node.isa<isl::schedule_node_sequence>()) {
    return std::unique_ptr<ScheduleTreeElemSequence>(
        new ScheduleTreeElemSequence());
  } else if (node.isa<isl::schedule_node_set>()) {
    return std::unique_ptr<ScheduleTreeElemSet>(new ScheduleTreeElemSet());
  }
  LOG(FATAL) << "NYI: ScheduleTreeElemBase from type: "
             << isl_schedule_node_get_type(node.get());
  return nullptr;
}

std::unique_ptr<ScheduleTreeElemBase> ScheduleTreeElemBase::make(
    const ScheduleTree& st) {
#define ELEM_MAKE_CASE(CLASS)                             \
  else if (st.type_ == CLASS::NodeType) {                 \
    return std::unique_ptr<CLASS>(                        \
        new CLASS(*static_cast<CLASS*>(st.elem_.get()))); \
  }

  if (st.type_ == detail::ScheduleTreeType::None) {
    LOG(FATAL) << "Hit Error node!";
  }
  ELEM_MAKE_CASE(ScheduleTreeElemBand)
  ELEM_MAKE_CASE(ScheduleTreeElemContext)
  ELEM_MAKE_CASE(ScheduleTreeElemDomain)
  ELEM_MAKE_CASE(ScheduleTreeElemExtension)
  ELEM_MAKE_CASE(ScheduleTreeElemFilter)
  ELEM_MAKE_CASE(ScheduleTreeElemMapping)
  ELEM_MAKE_CASE(ScheduleTreeElemSequence)
  ELEM_MAKE_CASE(ScheduleTreeElemSet)
  ELEM_MAKE_CASE(ScheduleTreeElemThreadSpecificMarker)

#undef ELEM_MAKE_CASE

  LOG(FATAL) << "NYI: ScheduleTreeElemBase from type: "
             << static_cast<int>(st.type_);
  return nullptr;
}

std::unique_ptr<ScheduleTreeElemBand> ScheduleTreeElemBand::fromMultiUnionPwAff(
    isl::multi_union_pw_aff mupa) {
  isl::ctx ctx(mupa.get_ctx());
  std::unique_ptr<ScheduleTreeElemBand> band(new ScheduleTreeElemBand);
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

  mupa_ = mupa_.drop_dims(isl::dim_type::set, pos, n);

  std::copy(
      coincident_.begin() + pos + n,
      coincident_.end(),
      coincident_.begin() + pos);
  coincident_.resize(nBegin - n);
  std::copy(unroll_.begin() + pos + n, unroll_.end(), unroll_.begin() + pos);
  unroll_.resize(nBegin - n);
  TC_CHECK_EQ(nBegin - n, nMember());
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
    const ScheduleTreeElemBase* e1,
    const ScheduleTreeElemBase* e2,
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
    LOG(FATAL) << "NYI: ScheduleTreeElemBase::operator== for type: "
               << static_cast<int>(type);
  }

#undef ELEM_EQUALS_CASE

  return false;
}
} // namespace detail
} // namespace polyhedral
} // namespace tc
