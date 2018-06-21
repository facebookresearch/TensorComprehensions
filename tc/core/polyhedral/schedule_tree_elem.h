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
#include "tc/core/polyhedral/domain_types.h"
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
  ScheduleTreeContext(const ScheduleTreeContext& eb)
      : ScheduleTree(eb), context_(eb.context_) {}

 public:
  virtual ~ScheduleTreeContext() override {}

  static std::unique_ptr<ScheduleTreeContext> make(
      isl::set context,
      std::vector<ScheduleTreeUPtr>&& children = {});
  static std::unique_ptr<ScheduleTreeContext> make(
      const ScheduleTreeContext* tree,
      std::vector<ScheduleTreeUPtr>&& children = {});

  bool operator==(const ScheduleTreeContext& other) const;
  bool operator!=(const ScheduleTreeContext& other) const {
    return !(*this == other);
  }

  virtual std::ostream& write(std::ostream& os) const override;
  virtual ScheduleTreeUPtr clone() const override {
    return make(this);
  }

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
  ScheduleTreeDomain(const ScheduleTreeDomain& eb)
      : ScheduleTree(eb), domain_(eb.domain_) {}

 public:
  virtual ~ScheduleTreeDomain() override {}

  static std::unique_ptr<ScheduleTreeDomain> make(
      isl::union_set domain,
      std::vector<ScheduleTreeUPtr>&& children = {});
  static std::unique_ptr<ScheduleTreeDomain> make(
      const ScheduleTreeDomain* tree,
      std::vector<ScheduleTreeUPtr>&& children = {});

  bool operator==(const ScheduleTreeDomain& other) const;
  bool operator!=(const ScheduleTreeDomain& other) const {
    return !(*this == other);
  }

  virtual std::ostream& write(std::ostream& os) const override;
  virtual ScheduleTreeUPtr clone() const override {
    return make(this);
  }

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
  ScheduleTreeExtension(const ScheduleTreeExtension& eb)
      : ScheduleTree(eb), extension_(eb.extension_) {}

 public:
  virtual ~ScheduleTreeExtension() override {}

  static std::unique_ptr<ScheduleTreeExtension> make(
      isl::union_map extension,
      std::vector<ScheduleTreeUPtr>&& children = {});
  static std::unique_ptr<ScheduleTreeExtension> make(
      const ScheduleTreeExtension* tree,
      std::vector<ScheduleTreeUPtr>&& children = {});

  bool operator==(const ScheduleTreeExtension& other) const;
  bool operator!=(const ScheduleTreeExtension& other) const {
    return !(*this == other);
  }

  virtual std::ostream& write(std::ostream& os) const override;
  virtual ScheduleTreeUPtr clone() const override {
    return make(this);
  }

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
  ScheduleTreeFilter(const ScheduleTreeFilter& eb)
      : ScheduleTree(eb), filter_(eb.filter_) {}

 public:
  virtual ~ScheduleTreeFilter() override {}

  bool operator==(const ScheduleTreeFilter& other) const;
  bool operator!=(const ScheduleTreeFilter& other) const {
    return !(*this == other);
  }

  static std::unique_ptr<ScheduleTreeFilter> make(
      isl::union_set filter,
      std::vector<ScheduleTreeUPtr>&& children = {});
  static std::unique_ptr<ScheduleTreeFilter> make(
      const ScheduleTreeFilter* tree,
      std::vector<ScheduleTreeUPtr>&& children = {});

  virtual std::ostream& write(std::ostream& os) const override;
  virtual ScheduleTreeUPtr clone() const override {
    return make(this);
  }

 public:
  isl::union_set filter_;
};

struct ScheduleTreeMapping : public ScheduleTree {
 public:
  using Mapping = std::unordered_map<
      mapping::MappingId,
      isl::UnionPwAffOn<Statement>,
      typename mapping::MappingId::Hash>;

  static constexpr detail::ScheduleTreeType NodeType =
      detail::ScheduleTreeType::Mapping;

 private:
  ScheduleTreeMapping() = delete;
  ScheduleTreeMapping(isl::ctx ctx, const Mapping& mapping);
  ScheduleTreeMapping(const ScheduleTreeMapping& eb)
      : ScheduleTree(eb), mapping(eb.mapping), filter_(eb.filter_) {}

 public:
  virtual ~ScheduleTreeMapping() override {}

  bool operator==(const ScheduleTreeMapping& other) const;
  bool operator!=(const ScheduleTreeMapping& other) const {
    return !(*this == other);
  }

  static std::unique_ptr<ScheduleTreeMapping> make(
      isl::ctx ctx,
      const Mapping& mapping,
      std::vector<ScheduleTreeUPtr>&& children = {});
  static std::unique_ptr<ScheduleTreeMapping> make(
      const ScheduleTreeMapping* tree,
      std::vector<ScheduleTreeUPtr>&& children = {});

  virtual std::ostream& write(std::ostream& os) const override;
  virtual ScheduleTreeUPtr clone() const override {
    return make(this);
  }

 public:
  // Mapping from identifiers to affine functions on domain elements.
  const Mapping mapping;
  // Assignment of the affine functions to the identifiers as parameters.
  isl::union_set filter_;
};

struct ScheduleTreeSequence : public ScheduleTree {
 public:
  static constexpr detail::ScheduleTreeType NodeType =
      detail::ScheduleTreeType::Sequence;

 private:
  explicit ScheduleTreeSequence(isl::ctx ctx)
      : ScheduleTree(ctx, {}, NodeType) {}
  ScheduleTreeSequence(const ScheduleTreeSequence& eb) : ScheduleTree(eb) {}

 public:
  virtual ~ScheduleTreeSequence() override {}

  bool operator==(const ScheduleTreeSequence& other) const;
  bool operator!=(const ScheduleTreeSequence& other) const {
    return !(*this == other);
  }

  static std::unique_ptr<ScheduleTreeSequence> make(
      isl::ctx ctx,
      std::vector<ScheduleTreeUPtr>&& children = {});
  static std::unique_ptr<ScheduleTreeSequence> make(
      const ScheduleTreeSequence* tree,
      std::vector<ScheduleTreeUPtr>&& children = {});

  virtual std::ostream& write(std::ostream& os) const override;
  virtual ScheduleTreeUPtr clone() const override {
    return make(this);
  }
};

struct ScheduleTreeSet : public ScheduleTree {
 public:
  static constexpr detail::ScheduleTreeType NodeType =
      detail::ScheduleTreeType::Set;

 private:
  explicit ScheduleTreeSet(isl::ctx ctx) : ScheduleTree(ctx, {}, NodeType) {}
  ScheduleTreeSet(const ScheduleTreeSet& eb) : ScheduleTree(eb) {}

 public:
  virtual ~ScheduleTreeSet() override {}

  bool operator==(const ScheduleTreeSet& other) const;
  bool operator!=(const ScheduleTreeSet& other) const {
    return !(*this == other);
  }

  static std::unique_ptr<ScheduleTreeSet> make(
      isl::ctx ctx,
      std::vector<ScheduleTreeUPtr>&& children = {});
  static std::unique_ptr<ScheduleTreeSet> make(
      const ScheduleTreeSet* tree,
      std::vector<ScheduleTreeUPtr>&& children = {});

  virtual std::ostream& write(std::ostream& os) const override;
  virtual ScheduleTreeUPtr clone() const override {
    return make(this);
  }
};

struct ScheduleTreeBand : public ScheduleTree {
 private:
  explicit ScheduleTreeBand(isl::ctx ctx) : ScheduleTree(ctx, {}, NodeType) {}
  ScheduleTreeBand(const ScheduleTreeBand& eb)
      : ScheduleTree(eb),
        permutable_(eb.permutable_),
        mupa_(eb.mupa_),
        coincident_(eb.coincident_),
        unroll_(eb.unroll_) {}

 public:
  static constexpr detail::ScheduleTreeType NodeType =
      detail::ScheduleTreeType::Band;

  virtual ~ScheduleTreeBand() override {}

  bool operator==(const ScheduleTreeBand& other) const;
  bool operator!=(const ScheduleTreeBand& other) const {
    return !(*this == other);
  }

  virtual std::ostream& write(std::ostream& os) const override;
  virtual ScheduleTreeUPtr clone() const override {
    return make(this);
  }

  // Make a schedule node band from partial schedule.
  // Replace "mupa" by its greatest integer part to ensure that the
  // schedule is always integral.
  static std::unique_ptr<ScheduleTreeBand> make(
      isl::multi_union_pw_aff mupa,
      bool permutable,
      std::vector<bool> coincident,
      std::vector<bool> unroll,
      std::vector<ScheduleTreeUPtr>&& children = {});
  static std::unique_ptr<ScheduleTreeBand> make(
      const ScheduleTreeBand* tree,
      std::vector<ScheduleTreeUPtr>&& children = {});

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
  template <typename Range>
  isl::MultiUnionPwAff<Statement, Range> memberRange(size_t first, size_t n)
      const {
    auto list = mupa_.get_union_pw_aff_list();
    auto space = mupa_.get_space().params().add_unnamed_tuple_ui(n);
    auto end = first + n;
    TC_CHECK_LE(end, nMember());
    list = list.drop(end, nMember() - end);
    list = list.drop(0, first);
    return isl::MultiUnionPwAff<Statement, Range>(
        isl::multi_union_pw_aff(space, list));
  }

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
 public:
  static constexpr detail::ScheduleTreeType NodeType =
      detail::ScheduleTreeType::ThreadSpecificMarker;

 private:
  explicit ScheduleTreeThreadSpecificMarker(isl::ctx ctx)
      : ScheduleTree(ctx, {}, NodeType) {}
  ScheduleTreeThreadSpecificMarker(const ScheduleTreeThreadSpecificMarker& tree)
      : ScheduleTree(tree) {}

 public:
  virtual ~ScheduleTreeThreadSpecificMarker() override {}

  bool operator==(const ScheduleTreeThreadSpecificMarker& other) const {
    return true;
  }
  bool operator!=(const ScheduleTreeThreadSpecificMarker& other) const {
    return !(*this == other);
  }

  static std::unique_ptr<ScheduleTreeThreadSpecificMarker> make(
      isl::ctx ctx,
      std::vector<ScheduleTreeUPtr>&& children = {});
  static std::unique_ptr<ScheduleTreeThreadSpecificMarker> make(
      const ScheduleTreeThreadSpecificMarker* tree,
      std::vector<ScheduleTreeUPtr>&& children = {});

  virtual std::ostream& write(std::ostream& os) const override;
  virtual ScheduleTreeUPtr clone() const override {
    return make(this);
  }
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
