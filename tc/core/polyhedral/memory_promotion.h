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

#include <iostream>

#include "tc/core/polyhedral/schedule_tree.h"
#include "tc/core/polyhedral/scop.h"
#include "tc/external/isl.h"

#include <unordered_map>
#include <unordered_set>
#include <vector>

namespace tc {
namespace polyhedral {

enum class AccessType : short { Read, Write };

// Rectangular overapproximation of a tensor elements accessed through a single
// reference.
// Each dimension is overapproximated by a lower bound, an affine function of
// parameters and schedule dimensions visible around the scope, by a
// constant size, and by a pair offset/stride for strided accesses.  If the
// access is not strided, then "offset" is a zero expression and "stride" is 1.
// The lowerBound and the size are computed after removing the potential stride.
// The scope is defined by a specific position in a schedule tree (const
// ScheduleTree*), the user is responsible for maintaining the correspondence
// between schedule tree positions and footprints.
struct ScopedFootprint {
  size_t dim() const {
    return box.get_size().size();
  }
  isl::val size(size_t pos) const {
    return box.get_size().get_val(pos);
  }
  isl::aff lowerBound(size_t pos) const {
    return box.get_offset().get_aff(pos);
  }
  isl::val stride(size_t pos) const {
    return strideValues.get_val(pos);
  }
  isl::aff strideOffset(size_t pos) const {
    return strideOffsets.get_aff(pos);
  }

  isl::fixed_box box;
  isl::multi_val strideValues;
  isl::multi_aff strideOffsets;

  isl::multi_aff lowerBounds() const;
};

// Descriptor of tensor reference in a Scop.
// May be scoped to a specific position in a schedule tree, the user is
// responsible for maintaining the correspondence between schedule tree
// positions and scoped access relations.
class TensorReference {
 public:
  bool isRead() const {
    return type == AccessType::Read;
  }

  bool isWrite() const {
    return type == AccessType::Write;
  }

 public:
  // Original access relation in terms of the Scop domain.
  isl::map originalAccess;

  // Access relation in terms of partial schedule at the point where the
  // reference group is introduced in the tree.
  isl::map scopedAccess;

  // Access direction (read or write).
  AccessType type;

  // Unique identifier of a reference in the Scop.
  isl::id refId;
};

class TensorReferenceGroup;
using TensorGroupsInfo = std::vector<std::unique_ptr<TensorReferenceGroup>>;
typedef std::unordered_map<isl::id, TensorGroupsInfo, isl::IslIdIslHash>
    TensorGroups;

// A group of tensor references that must be handled together during memory
// promotion.  In particular, references that access the same tensor element,
// and at least one of them modifies it, should be placed in the shared/private
// memory together to avoid inconsistent values.
//
// Scoped to a specific position in a schedule tree, the user is responsible
// for maintaining the correspondence between schedule tree positions and scoped
// access relations of each reference as well as scoped footprints.
class TensorReferenceGroup {
 private:
  TensorReferenceGroup() {}

 public:
  static TensorGroups accessedWithin(
      isl::union_map outerSchedule,
      isl::union_map reads,
      isl::union_map writes);

  bool isReadOnly() const;

  // Sets of tensor elements accessed below the scoping point.
  isl::set writeFootprint() const;
  isl::set readFootprint() const;
  isl::set footprint() const {
    return writeFootprint().unite(readFootprint());
  }

  // Access relations in terms of partial schedule of the scoping point.
  isl::map scopedWrites() const;
  isl::map scopedReads() const;
  isl::map scopedAccesses() const {
    return scopedWrites().unite(scopedReads());
  }

  // Access relations in terms of Scop domain elements.
  // The resulting union relations have different domain spaces but identical
  // range spaces.
  isl::union_map originalWrites() const;
  isl::union_map originalReads() const;
  isl::union_map originalAccesses() const {
    return originalWrites().unite(originalReads());
  }

  // Rectangular overapproximation of the set of tensor elements accessed below
  // and relative to the scoping point.
  isl::map approximateFootprint() const;

  isl::multi_aff promotion() const;
  isl::set promotedFootprint() const;

  std::vector<size_t> approximationSizes() const;

  std::unordered_set<isl::id, isl::IslIdIslHash> referenceIds() const;

  static std::unique_ptr<TensorReferenceGroup> makeJoint(
      std::unique_ptr<TensorReferenceGroup>&& g1,
      std::unique_ptr<TensorReferenceGroup>&& g2);
  static std::unique_ptr<TensorReferenceGroup> makeSingleton(
      isl::map originalAccess,
      isl::map scopedAccess,
      AccessType type);

 public:
  std::vector<std::unique_ptr<TensorReference>> references;
  ScopedFootprint approximation;
};

inline std::ostream& operator<<(std::ostream& os, const ScopedFootprint& fp) {
  if (!fp.box) {
    return os;
  }
  os << "{\n";
  os << fp.box.get_offset() << " of size " << fp.box.get_size() << "\n";
  os << "}";
  return os;
}

inline std::ostream& operator<<(std::ostream& os, const TensorReference& tr) {
  os << ((tr.isRead()) ? "rd" : "wr") << " scopedAccess: " << tr.scopedAccess;
  ;
  return os;
}

inline std::ostream& operator<<(
    std::ostream& os,
    const TensorReferenceGroup& tg) {
  os << "Reference with footprint: " << tg.approximation << "\n";
  for (const auto& tr : tg.references) {
    os << *tr << "\n";
  }
  return os;
}

inline std::ostream& operator<<(std::ostream& os, const TensorGroupsInfo& ti) {
  for (const auto& tg : ti) {
    os << *tg << " ";
  }
  return os;
}

inline std::ostream& operator<<(std::ostream& os, const TensorGroups& tg) {
  size_t i = 0;
  for (const auto& kvp : tg) {
    os << "id: " << kvp.first << "; acc: " << kvp.second;
    if (++i < tg.size()) {
      os << std::endl;
    }
  }
  return os;
}

detail::ScheduleTree* insertCopiesUnder(
    Scop& scop,
    detail::ScheduleTree* tree,
    const TensorReferenceGroup& group,
    isl::id tensorId,
    isl::id groupId,
    bool unrollAllCopies);
} // namespace polyhedral
} // namespace tc
