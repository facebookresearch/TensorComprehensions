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
#include <unordered_set>
#include <vector>

#include "tc/external/isl.h"

#include "tc/core/polyhedral/mapping_types.h"

namespace tc {
namespace polyhedral {
namespace detail {

enum class ScheduleTreeType {
  None,
  Band,
  Context,
  Domain,
  Extension,
  Filter,
  Sequence,
  Set,
  MappingFilter,
  ThreadSpecificMarker,
  Any,
};

struct ScheduleTree;

struct ScheduleTreeElemBase {
  static constexpr detail::ScheduleTreeType NodeType =
      detail::ScheduleTreeType::None;
  static std::unique_ptr<ScheduleTreeElemBase> make(isl::schedule_node node);
  static std::unique_ptr<ScheduleTreeElemBase> make(const ScheduleTree& st);
  virtual ~ScheduleTreeElemBase() {}
  virtual std::ostream& write(std::ostream& os) const = 0;
  virtual detail::ScheduleTreeType type() const = 0;
};

struct ScheduleTreeElemContext : public ScheduleTreeElemBase {
  static constexpr std::initializer_list<detail::ScheduleTreeType>
      NodeDerivedTypes{detail::ScheduleTreeType::None};
  static constexpr detail::ScheduleTreeType NodeType =
      detail::ScheduleTreeType::Context;
  isl::set context_;
  ScheduleTreeElemContext() = delete;
  ScheduleTreeElemContext(const ScheduleTreeElemContext& eb)
      : context_(eb.context_) {}
  explicit ScheduleTreeElemContext(isl::set s) : context_(s) {}
  virtual ~ScheduleTreeElemContext() override {}
  bool operator==(const ScheduleTreeElemContext& other) const;
  bool operator!=(const ScheduleTreeElemContext& other) const {
    return !(*this == other);
  }
  virtual std::ostream& write(std::ostream& os) const override;
  virtual detail::ScheduleTreeType type() const override {
    return NodeType;
  }
};

struct ScheduleTreeElemDomain : public ScheduleTreeElemBase {
  static constexpr std::initializer_list<detail::ScheduleTreeType>
      NodeDerivedTypes{detail::ScheduleTreeType::None};
  static constexpr detail::ScheduleTreeType NodeType =
      detail::ScheduleTreeType::Domain;
  isl::union_set domain_;
  ScheduleTreeElemDomain() = delete;
  ScheduleTreeElemDomain(const ScheduleTreeElemDomain& eb)
      : domain_(eb.domain_) {}
  explicit ScheduleTreeElemDomain(isl::union_set us) : domain_(us) {}
  virtual ~ScheduleTreeElemDomain() override {}
  bool operator==(const ScheduleTreeElemDomain& other) const;
  bool operator!=(const ScheduleTreeElemDomain& other) const {
    return !(*this == other);
  }
  virtual std::ostream& write(std::ostream& os) const override;
  virtual detail::ScheduleTreeType type() const override {
    return NodeType;
  }
};

struct ScheduleTreeElemExtension : public ScheduleTreeElemBase {
  static constexpr std::initializer_list<detail::ScheduleTreeType>
      NodeDerivedTypes{detail::ScheduleTreeType::None};
  static constexpr detail::ScheduleTreeType NodeType =
      detail::ScheduleTreeType::Extension;
  isl::union_map extension_;
  ScheduleTreeElemExtension() = delete;
  ScheduleTreeElemExtension(const ScheduleTreeElemExtension& eb)
      : extension_(eb.extension_) {}
  explicit ScheduleTreeElemExtension(isl::union_map m) : extension_(m) {}
  virtual ~ScheduleTreeElemExtension() override {}
  bool operator==(const ScheduleTreeElemExtension& other) const;
  bool operator!=(const ScheduleTreeElemExtension& other) const {
    return !(*this == other);
  }
  virtual std::ostream& write(std::ostream& os) const override;
  virtual detail::ScheduleTreeType type() const override {
    return NodeType;
  }
};

struct ScheduleTreeElemFilter : public ScheduleTreeElemBase {
  static constexpr std::initializer_list<detail::ScheduleTreeType>
      NodeDerivedTypes{detail::ScheduleTreeType::MappingFilter};
  static constexpr detail::ScheduleTreeType NodeType =
      detail::ScheduleTreeType::Filter;
  isl::union_set filter_;
  ScheduleTreeElemFilter() = delete;
  ScheduleTreeElemFilter(const ScheduleTreeElemFilter& eb)
      : filter_(eb.filter_) {}
  explicit ScheduleTreeElemFilter(isl::union_set s) : filter_(s) {}
  virtual ~ScheduleTreeElemFilter() override {}
  bool operator==(const ScheduleTreeElemFilter& other) const;
  bool operator!=(const ScheduleTreeElemFilter& other) const {
    return !(*this == other);
  }
  virtual std::ostream& write(std::ostream& os) const override;
  virtual detail::ScheduleTreeType type() const override {
    return NodeType;
  }
};

struct ScheduleTreeElemMappingFilter : public ScheduleTreeElemFilter {
  static constexpr std::initializer_list<detail::ScheduleTreeType>
      NodeDerivedTypes{detail::ScheduleTreeType::None};
  static constexpr detail::ScheduleTreeType NodeType =
      detail::ScheduleTreeType::MappingFilter;
  ScheduleTreeElemMappingFilter() = delete;
  ScheduleTreeElemMappingFilter(const ScheduleTreeElemMappingFilter& eb)
      : ScheduleTreeElemFilter(eb.filter_), mappingIds(eb.mappingIds) {}
  ScheduleTreeElemMappingFilter(
      isl::union_set us,
      const std::unordered_set<
          mapping::MappingId,
          typename mapping::MappingId::Hash>& ids)
      : ScheduleTreeElemFilter(us), mappingIds(ids) {
    USING_MAPPING_SHORT_NAMES(BX, BY, BZ, TX, TY, TZ);
    for (auto id : std::vector<mapping::MappingId>{BX, BY, BZ, TX, TY, TZ}) {
      if (mappingIds.count(id) > 0) {
        CHECK_EQ(1u, ids.count(id)) << "id: " << id << " mapped >1 times";
        for (auto s : us.get_set_list()) {
          CHECK(s.involves_param(id))
              << "unexpected missing id: " << id << " in filter: " << s;
        }
      } else {
        if (us.involves_param(id)) {
          std::stringstream ss;
          for (auto id : ids) {
            ss << id.to_str() << " ";
          }
          // TODO: will need to relax this if we map the same loop
          // iteratively without stripmining it beforehand
          CHECK(false) << "unexpected involved id: " << id
                       << " in filter: " << us
                       << " but not present in filter id list: " << ss.str();
        }
      }
    }
  }
  virtual ~ScheduleTreeElemMappingFilter() override {}
  bool operator==(const ScheduleTreeElemMappingFilter& other) const;
  bool operator!=(const ScheduleTreeElemMappingFilter& other) const {
    return !(*this == other);
  }
  virtual std::ostream& write(std::ostream& os) const override;
  virtual detail::ScheduleTreeType type() const override {
    return NodeType;
  }

  const std::
      unordered_set<mapping::MappingId, typename mapping::MappingId::Hash>
          mappingIds;
};

struct ScheduleTreeElemSequence : public ScheduleTreeElemBase {
  static constexpr std::initializer_list<detail::ScheduleTreeType>
      NodeDerivedTypes{detail::ScheduleTreeType::None};
  static constexpr detail::ScheduleTreeType NodeType =
      detail::ScheduleTreeType::Sequence;
  explicit ScheduleTreeElemSequence() {}
  ScheduleTreeElemSequence(const ScheduleTreeElemSequence& eb) {}
  virtual ~ScheduleTreeElemSequence() override {}
  bool operator==(const ScheduleTreeElemSequence& other) const;
  bool operator!=(const ScheduleTreeElemSequence& other) const {
    return !(*this == other);
  }
  virtual std::ostream& write(std::ostream& os) const override;
  virtual detail::ScheduleTreeType type() const override {
    return NodeType;
  }
};

struct ScheduleTreeElemSet : public ScheduleTreeElemBase {
  static constexpr std::initializer_list<detail::ScheduleTreeType>
      NodeDerivedTypes{detail::ScheduleTreeType::None};
  static constexpr detail::ScheduleTreeType NodeType =
      detail::ScheduleTreeType::Set;
  explicit ScheduleTreeElemSet() {}
  ScheduleTreeElemSet(const ScheduleTreeElemSet& eb) {}
  virtual ~ScheduleTreeElemSet() override {}
  bool operator==(const ScheduleTreeElemSet& other) const;
  bool operator!=(const ScheduleTreeElemSet& other) const {
    return !(*this == other);
  }
  virtual std::ostream& write(std::ostream& os) const override;
  virtual detail::ScheduleTreeType type() const override {
    return NodeType;
  }
};

struct ScheduleTreeElemBand : public ScheduleTreeElemBase {
 private:
  ScheduleTreeElemBand() = default;

 public:
  static constexpr std::initializer_list<detail::ScheduleTreeType>
      NodeDerivedTypes{detail::ScheduleTreeType::None};
  static constexpr detail::ScheduleTreeType NodeType =
      detail::ScheduleTreeType::Band;

  ScheduleTreeElemBand(const ScheduleTreeElemBand& eb)
      : permutable_(eb.permutable_),
        mupa_(eb.mupa_),
        coincident_(eb.coincident_),
        unroll_(eb.unroll_) {}
  virtual ~ScheduleTreeElemBand() override {}
  bool operator==(const ScheduleTreeElemBand& other) const;
  bool operator!=(const ScheduleTreeElemBand& other) const {
    return !(*this == other);
  }
  virtual std::ostream& write(std::ostream& os) const override;
  virtual detail::ScheduleTreeType type() const override {
    return NodeType;
  }

  // First replace "mupa" by its greatest integer part to ensure that the
  // schedule is always integral.
  // The band is not marked permutable, the dimensions are not marked
  // coincident and are not marked for unrolling.
  static std::unique_ptr<ScheduleTreeElemBand> fromMultiUnionPwAff(
      isl::multi_union_pw_aff mupa);

  // Return the number of scheduling dimensions in the band
  size_t nMember() const;

  // Return the number of outer coincident members in the band.
  size_t nOuterCoincident() const;

  // Drop the "n" dimensions starting at "pos" from "band".
  // We apply the transformation even if "n" is zero to ensure consistent
  // behavior with respect to changes in the schedule space.
  // The caller is responsible for updating the isolate option (Note: why?)
  void drop(size_t pos, size_t n);

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
struct ScheduleTreeElemThreadSpecificMarker : public ScheduleTreeElemBase {
  static constexpr std::initializer_list<detail::ScheduleTreeType>
      NodeDerivedTypes{detail::ScheduleTreeType::None};
  static constexpr detail::ScheduleTreeType NodeType =
      detail::ScheduleTreeType::ThreadSpecificMarker;
  explicit ScheduleTreeElemThreadSpecificMarker() {}
  virtual ~ScheduleTreeElemThreadSpecificMarker() override {}
  bool operator==(const ScheduleTreeElemThreadSpecificMarker& other) const {
    return true;
  }
  bool operator!=(const ScheduleTreeElemThreadSpecificMarker& other) const {
    return !(*this == other);
  }
  virtual std::ostream& write(std::ostream& os) const override;
  virtual detail::ScheduleTreeType type() const override {
    return NodeType;
  }
};

bool elemEquals(
    const ScheduleTreeElemBase* e1,
    const ScheduleTreeElemBase* e2,
    detail::ScheduleTreeType type);

std::ostream& operator<<(std::ostream& os, isl::ast_loop_type lt);
std::ostream& operator<<(std::ostream& os, detail::ScheduleTreeType nt);
std::ostream& operator<<(
    std::ostream& os,
    const std::vector<std::unique_ptr<ScheduleTree>>& vst);
std::ostream& operator<<(std::ostream& os, const ScheduleTreeElemBase& eb);

} // namespace detail
} // namespace polyhedral
} // namespace tc
