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

#include <memory>
#include <sstream>
#include <unordered_map>
#include <vector>

#include "tc/external/isl.h"

#include "tc/core/check.h"
#include "tc/core/polyhedral/mapping_types.h"
#include "tc/core/polyhedral/schedule_tree.h"

namespace tc {
namespace polyhedral {
namespace detail {

struct ScheduleTreeContext : public ScheduleTree {
 public:
  static constexpr detail::ScheduleTreeType NodeType =
      detail::ScheduleTreeType::Context;

 private:
  ScheduleTreeContext() = delete;
  explicit ScheduleTreeContext(isl::set s)
      : ScheduleTree(s.get_ctx(), {}, NodeType), context_(s) {}

 public:
  ScheduleTreeContext(const ScheduleTreeContext& eb)
      : ScheduleTree(eb), context_(eb.context_) {}
  virtual ~ScheduleTreeContext() override {}

  static std::unique_ptr<ScheduleTreeContext> make(
      isl::set context,
      std::vector<ScheduleTreeUPtr>&& children = {});

  bool operator==(const ScheduleTreeContext& other) const;
  bool operator!=(const ScheduleTreeContext& other) const {
    return !(*this == other);
  }

  virtual std::ostream& write(std::ostream& os) const override;

 public:
  isl::set context_;
};

struct ScheduleTreeDomain : public ScheduleTree {
 public:
  static constexpr detail::ScheduleTreeType NodeType =
      detail::ScheduleTreeType::Domain;

 private:
  ScheduleTreeDomain() = delete;
  explicit ScheduleTreeDomain(isl::union_set us)
      : ScheduleTree(us.get_ctx(), {}, NodeType), domain_(us) {}

 public:
  ScheduleTreeDomain(const ScheduleTreeDomain& eb)
      : ScheduleTree(eb), domain_(eb.domain_) {}
  virtual ~ScheduleTreeDomain() override {}

  static std::unique_ptr<ScheduleTreeDomain> make(
      isl::union_set domain,
      std::vector<ScheduleTreeUPtr>&& children = {});

  bool operator==(const ScheduleTreeDomain& other) const;
  bool operator!=(const ScheduleTreeDomain& other) const {
    return !(*this == other);
  }

  virtual std::ostream& write(std::ostream& os) const override;

 public:
  isl::union_set domain_;
};

struct ScheduleTreeExtension : public ScheduleTree {
 public:
  static constexpr detail::ScheduleTreeType NodeType =
      detail::ScheduleTreeType::Extension;

 private:
  ScheduleTreeExtension() = delete;
  explicit ScheduleTreeExtension(isl::union_map m)
      : ScheduleTree(m.get_ctx(), {}, NodeType), extension_(m) {}

 public:
  ScheduleTreeExtension(const ScheduleTreeExtension& eb)
      : ScheduleTree(eb), extension_(eb.extension_) {}
  virtual ~ScheduleTreeExtension() override {}

  static std::unique_ptr<ScheduleTreeExtension> make(
      isl::union_map extension,
      std::vector<ScheduleTreeUPtr>&& children = {});

  bool operator==(const ScheduleTreeExtension& other) const;
  bool operator!=(const ScheduleTreeExtension& other) const {
    return !(*this == other);
  }

  virtual std::ostream& write(std::ostream& os) const override;

 public:
  isl::union_map extension_;
};

struct ScheduleTreeFilter : public ScheduleTree {
 public:
  static constexpr detail::ScheduleTreeType NodeType =
      detail::ScheduleTreeType::Filter;

 private:
  ScheduleTreeFilter() = delete;
  explicit ScheduleTreeFilter(isl::union_set s)
      : ScheduleTree(s.get_ctx(), {}, NodeType), filter_(s) {}

 public:
  ScheduleTreeFilter(const ScheduleTreeFilter& eb)
      : ScheduleTree(eb), filter_(eb.filter_) {}
  virtual ~ScheduleTreeFilter() override {}

  bool operator==(const ScheduleTreeFilter& other) const;
  bool operator!=(const ScheduleTreeFilter& other) const {
    return !(*this == other);
  }

  static std::unique_ptr<ScheduleTreeFilter> make(
      isl::union_set filter,
      std::vector<ScheduleTreeUPtr>&& children = {});

  virtual std::ostream& write(std::ostream& os) const override;

 public:
  isl::union_set filter_;
};

struct ScheduleTreeMapping : public ScheduleTree {
  using Mapping = std::unordered_map<
      mapping::MappingId,
      isl::union_pw_aff,
      typename mapping::MappingId::Hash>;

  static constexpr detail::ScheduleTreeType NodeType =
      detail::ScheduleTreeType::Mapping;

  ScheduleTreeMapping() = delete;
  ScheduleTreeMapping(const ScheduleTreeMapping& eb)
      : ScheduleTree(eb), mapping(eb.mapping), filter_(eb.filter_) {}
  ScheduleTreeMapping(isl::ctx ctx, const Mapping& mapping)
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
      upa = upa.sub(isl::union_pw_aff::param_on_domain(domain.universe(), id));
      filter_ = filter_.intersect(upa.zero_union_set());
    }
  }
  virtual ~ScheduleTreeMapping() override {}

  bool operator==(const ScheduleTreeMapping& other) const;
  bool operator!=(const ScheduleTreeMapping& other) const {
    return !(*this == other);
  }

  virtual std::ostream& write(std::ostream& os) const override;

 public:
  // Mapping from identifiers to affine functions on domain elements.
  const Mapping mapping;
  // Assignment of the affine functions to the identifiers as parameters.
  isl::union_set filter_;
};

struct ScheduleTreeSequence : public ScheduleTree {
  static constexpr detail::ScheduleTreeType NodeType =
      detail::ScheduleTreeType::Sequence;

  explicit ScheduleTreeSequence(isl::ctx ctx)
      : ScheduleTree(ctx, {}, NodeType) {}
  ScheduleTreeSequence(const ScheduleTreeSequence& eb) : ScheduleTree(eb) {}
  virtual ~ScheduleTreeSequence() override {}

  bool operator==(const ScheduleTreeSequence& other) const;
  bool operator!=(const ScheduleTreeSequence& other) const {
    return !(*this == other);
  }

  virtual std::ostream& write(std::ostream& os) const override;
};

struct ScheduleTreeSet : public ScheduleTree {
  static constexpr detail::ScheduleTreeType NodeType =
      detail::ScheduleTreeType::Set;

  explicit ScheduleTreeSet(isl::ctx ctx) : ScheduleTree(ctx, {}, NodeType) {}
  ScheduleTreeSet(const ScheduleTreeSet& eb) : ScheduleTree(eb) {}
  virtual ~ScheduleTreeSet() override {}

  bool operator==(const ScheduleTreeSet& other) const;
  bool operator!=(const ScheduleTreeSet& other) const {
    return !(*this == other);
  }

  virtual std::ostream& write(std::ostream& os) const override;
};

struct ScheduleTreeBand : public ScheduleTree {
 private:
  explicit ScheduleTreeBand(isl::ctx ctx) : ScheduleTree(ctx, {}, NodeType) {}

 public:
  static constexpr detail::ScheduleTreeType NodeType =
      detail::ScheduleTreeType::Band;

  ScheduleTreeBand(const ScheduleTreeBand& eb)
      : ScheduleTree(eb),
        permutable_(eb.permutable_),
        mupa_(eb.mupa_),
        coincident_(eb.coincident_),
        unroll_(eb.unroll_) {}
  virtual ~ScheduleTreeBand() override {}

  bool operator==(const ScheduleTreeBand& other) const;
  bool operator!=(const ScheduleTreeBand& other) const {
    return !(*this == other);
  }

  virtual std::ostream& write(std::ostream& os) const override;

  // First replace "mupa" by its greatest integer part to ensure that the
  // schedule is always integral.
  // The band is not marked permutable, the dimensions are not marked
  // coincident and are not marked for unrolling.
  static std::unique_ptr<ScheduleTreeBand> fromMultiUnionPwAff(
      isl::multi_union_pw_aff mupa);

  // Return the number of scheduling dimensions in the band
  size_t nMember() const;

  // Return the number of outer coincident members in the band.
  size_t nOuterCoincident() const;

  // Drop the "n" dimensions starting at "pos" from "band".
  // We apply the transformation even if "n" is zero to ensure consistent
  // behavior with respect to changes in the schedule space.
  void drop(size_t pos, size_t n);

  // Extract the range of "n" members starting at "first"
  // (in an anonymous space).
  isl::multi_union_pw_aff memberRange(size_t first, size_t n) const;

 public:
  bool permutable_{false};
  isl::multi_union_pw_aff mupa_;

  std::vector<bool> coincident_;
  // For each member, should the corresponding loop in the generated code
  // be (fully) unrolled?
  std::vector<bool> unroll_;
};

/*
 * A node of type ThreadSpecificMarker marks part of a schedule tree
 * that is specific to a thread.  That is, the marker appears right
 * underneath the innermost band member mapped to threads.
 */
struct ScheduleTreeThreadSpecificMarker : public ScheduleTree {
  static constexpr detail::ScheduleTreeType NodeType =
      detail::ScheduleTreeType::ThreadSpecificMarker;

  explicit ScheduleTreeThreadSpecificMarker(isl::ctx ctx)
      : ScheduleTree(ctx, {}, NodeType) {}
  virtual ~ScheduleTreeThreadSpecificMarker() override {}

  bool operator==(const ScheduleTreeThreadSpecificMarker& other) const {
    return true;
  }
  bool operator!=(const ScheduleTreeThreadSpecificMarker& other) const {
    return !(*this == other);
  }

  virtual std::ostream& write(std::ostream& os) const override;
};

bool elemEquals(
    const ScheduleTree* e1,
    const ScheduleTree* e2,
    detail::ScheduleTreeType type);

std::ostream& operator<<(std::ostream& os, detail::ScheduleTreeType nt);
std::ostream& operator<<(
    std::ostream& os,
    const std::vector<std::unique_ptr<ScheduleTree>>& vst);
std::ostream& operator<<(std::ostream& os, const ScheduleTree& eb);

} // namespace detail
} // namespace polyhedral
} // namespace tc
