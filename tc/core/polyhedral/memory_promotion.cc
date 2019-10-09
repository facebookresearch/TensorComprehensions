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
#include "tc/core/polyhedral/memory_promotion.h"

#include <memory>
#include <sstream>
#include <string>
#include <unordered_map>

#include "tc/core/check.h"
#include "tc/core/polyhedral/body.h"
#include "tc/core/polyhedral/domain_types.h"
#include "tc/core/polyhedral/exceptions.h"
#include "tc/core/polyhedral/schedule_tree.h"
#include "tc/core/polyhedral/scop.h"
#include "tc/core/polyhedral/utils.h"
#include "tc/external/isl.h"

namespace tc {
namespace polyhedral {

using detail::ScheduleTree;

namespace {
// Remove strides specified by "strides" and "offsets" from the range of
// "relation".  In particular, relation has a shape
//
//   D -> O: o_i = offset_i + stride_i * f(D)
//
// transform it into
//
//   D -> O: o_i = f(D)
//
// by subtracting "offsets" and by dividing the result by "strides".
template <typename Domain, typename Range>
isl::Map<Domain, Range> removeRangeStrides(
    isl::Map<Domain, Range> relation,
    isl::MultiVal<Range> strides,
    isl::MultiAff<Domain, Range> offsets) {
  TC_CHECK_EQ(strides.size(), offsets.size());

  auto space = relation.get_space();
  auto stridesMA =
      isl::MultiAff<Range, Range>::identity(space.range().map_from_set());
  stridesMA = stridesMA / strides;

  return relation.sum(offsets.neg().asMap()).apply_range(stridesMA.asMap());
}

// Compute a box approximation of the range of the given relation,
// including the lower bounds, the box sizes, and the strides.
// If the range has strides, remove them first.
ScopedFootprint outputRanges(isl::Map<Prefix, Tensor> access) {
  ScopedFootprint footprint;
  footprint.strideValues =
      isl::MultiVal<Tensor>::zero(access.get_space().range());
  footprint.strideOffsets =
      isl::MultiAff<Prefix, Tensor>::zero(access.get_space());

  int nSubscripts = footprint.strideValues.size();
  for (int i = 0; i < nSubscripts; ++i) {
    auto si = access.get_range_stride_info(i);
    footprint.strideValues = footprint.strideValues.set_val(i, si.get_stride());
    footprint.strideOffsets =
        footprint.strideOffsets.set_aff(i, si.get_offset());
  }

  auto accessNoStrides = removeRangeStrides(
      access, footprint.strideValues, footprint.strideOffsets);

  footprint.box = accessNoStrides.get_range_simple_fixed_box_hull();
  return footprint;
}
} // namespace

// Access has the shape :: [S -> ref] -> O
// Extract the reference ID, store it separately and simplify the access.
std::unique_ptr<TensorReferenceGroup> TensorReferenceGroup::makeSingleton(
    isl::Map<isl::Pair<Statement, Tag>, Tensor> originalAccess,
    isl::Map<isl::Pair<Prefix, Tag>, Tensor> scopedTaggedAccess,
    AccessType type) {
  auto ref = std::unique_ptr<TensorReference>(new TensorReference);
  auto refId =
      scopedTaggedAccess.get_space().domain().unwrap().get_map_range_tuple_id();
  auto scopedAccess = scopedTaggedAccess.domain_factor_domain();
  ref->originalAccess = originalAccess.domain_factor_domain();
  ref->scopedAccess = scopedAccess;
  ref->type = type;
  ref->refId = refId;
  auto group = std::unique_ptr<TensorReferenceGroup>(new TensorReferenceGroup);
  group->references.push_back(std::move(ref));
  group->approximation = outputRanges(scopedAccess);

  if (!group->approximation.box.is_valid()) {
    std::stringstream ss;
    ss << "could not compute rectangular overapproximation of: "
       << scopedAccess;
    throw promotion::GroupingError(ss.str());
  }

  return group;
}

isl::Map<Prefix, Tensor> TensorReferenceGroup::approximateScopedAccesses()
    const {
  auto scopedDomain = scopedAccesses().domain();
  auto space = approximation.box.get_space();
  auto accessed =
      isl::Map<Prefix, Tensor>::universe(space).intersect_domain(scopedDomain);

  auto identity =
      isl::MultiAff<Tensor, Tensor>::identity(space.range().map_from_set());
  for (size_t i = 0; i < approximation.dim(); ++i) {
    auto offset = approximation.lowerBound(i);
    auto stride = approximation.stride(i);
    auto strideOffset = approximation.strideOffset(i);
    auto size = approximation.size(i);
    auto rhs = identity.get_aff(i);
    auto lowerBound = offset * stride + strideOffset;
    auto upperBound = (offset + size) * stride + strideOffset;
    auto partial = lowerBound.asPwAff().lt_map((rhs + 1).asPwAff()) &
        upperBound.asPwAff().gt_map(rhs.asPwAff());

    accessed = accessed & partial;
  }
  return accessed;
}

isl::MultiAff<Prefix, Tensor> ScopedFootprint::lowerBounds() const {
  if (dim() == 0) {
    throw promotion::PromotionNYI("promotion for scalars");
  }
  return box.get_offset();
}

bool TensorReferenceGroup::isReadOnly() const {
  bool result = true;
  for (auto const& ref : references) {
    result &= !ref->isWrite();
  }
  return result;
}

isl::Set<Tensor> TensorReferenceGroup::promotedFootprint() const {
  auto space = scopedAccesses().get_space().range();
  auto sizes = approximation.box.get_size();
  if (!sizes.get_space().has_equal_tuples(space)) {
    throw promotion::GroupingError("unexpected dimensionality mismatch");
  }

  isl::Set<Tensor> footprint = isl::Set<Tensor>::universe(space);
  auto identity = isl::MultiAff<Tensor, Tensor>::identity(space.map_from_set());
  for (size_t i = 0, e = sizes.size(); i < e; ++i) {
    auto aff = identity.get_aff(i);
    auto size = sizes.get_val(i);
    footprint = footprint & aff.asPwAff().nonneg_set() &
        (size - aff).asPwAff().pos_set();
  }
  return footprint;
}

std::vector<size_t> TensorReferenceGroup::approximationSizes() const {
  std::vector<size_t> result;
  result.reserve(approximation.dim());
  for (const auto& size : approximation.box.get_size().get_val_list()) {
    result.push_back(size.get_num_si());
  }
  return result;
}

namespace {
isl::Map<Prefix, Tensor> referenceScopedAccessesImpl(
    const TensorReferenceGroup& group,
    AccessType type) {
  if (group.references.size() == 0) {
    throw promotion::GroupingError("no references in the group");
  }
  auto accesses = isl::Map<Prefix, Tensor>::empty(
      group.references.front()->scopedAccess.get_space());

  for (const auto& ref : group.references) {
    if (ref->type != type) {
      continue;
    }
    auto current = ref->scopedAccess;
    accesses = accesses.unite(current);
  }
  return accesses;
}
} // namespace

isl::set TensorReferenceGroup::writeFootprint() const {
  return referenceScopedAccessesImpl(*this, AccessType::Write).range();
}

isl::set TensorReferenceGroup::readFootprint() const {
  return referenceScopedAccessesImpl(*this, AccessType::Read).range();
}

isl::Map<Prefix, Tensor> TensorReferenceGroup::scopedWrites() const {
  return referenceScopedAccessesImpl(*this, AccessType::Write);
}

isl::Map<Prefix, Tensor> TensorReferenceGroup::scopedReads() const {
  return referenceScopedAccessesImpl(*this, AccessType::Read);
}

namespace {
isl::UnionMap<Statement, Tensor> referenceOriginalAccessesImpl(
    const TensorReferenceGroup& group,
    AccessType type) {
  if (group.references.size() == 0) {
    throw promotion::GroupingError("no references in the group");
  }
  auto accesses = isl::UnionMap<Statement, Tensor>::empty(
      group.references.front()->originalAccess.get_space().params());

  for (const auto& ref : group.references) {
    if (ref->type != type) {
      continue;
    }
    auto current = ref->originalAccess;
    accesses = accesses.unite(current.asUnionMap());
  }
  return accesses;
}
} // namespace

isl::UnionMap<Statement, Tensor> TensorReferenceGroup::originalWrites() const {
  return referenceOriginalAccessesImpl(*this, AccessType::Write);
}

isl::UnionMap<Statement, Tensor> TensorReferenceGroup::originalReads() const {
  return referenceOriginalAccessesImpl(*this, AccessType::Read);
}

std::unique_ptr<TensorReferenceGroup> TensorReferenceGroup::makeJoint(
    std::unique_ptr<TensorReferenceGroup>&& g1,
    std::unique_ptr<TensorReferenceGroup>&& g2) {
  auto result = std::unique_ptr<TensorReferenceGroup>(new TensorReferenceGroup);
  std::copy(
      std::make_move_iterator(g1->references.begin()),
      std::make_move_iterator(g1->references.end()),
      std::back_inserter(result->references));
  std::copy(
      std::make_move_iterator(g2->references.begin()),
      std::make_move_iterator(g2->references.end()),
      std::back_inserter(result->references));
  result->approximation = outputRanges(result->scopedAccesses());
  return result;
}

namespace {
void joinOverlappingWrites(
    std::vector<std::unique_ptr<TensorReferenceGroup>>& groups) {
  for (size_t i = 0; i < groups.size(); ++i) {
    for (size_t j = i + 1; j < groups.size(); ++j) {
      auto& g1 = groups[i];
      auto& g2 = groups[j];
      if (g1->isReadOnly() && g2->isReadOnly()) {
        continue;
      }
      if (g1->approximateScopedAccesses()
              .intersect(g2->approximateScopedAccesses())
              .is_empty()) {
        continue;
      }
      groups[i] = TensorReferenceGroup::makeJoint(
          std::move(groups[i]), std::move(groups[j]));
      groups.erase(groups.begin() + j);
      --j;
    }
  }
}

void addSingletonReferenceGroup(
    TensorGroups& tensorGroups,
    isl::id targetTensor,
    isl::UnionMap<Statement, Prefix> schedule,
    isl::Map<isl::Pair<Statement, Tag>, Tensor> access,
    AccessType type) {
  auto unionAccess = access.curry().asUnionMap();
  auto scopedUnionAccess = unionAccess.apply_domain(schedule);
  auto scopedAccess = scopedUnionAccess.toMap().uncurry();
  tensorGroups[targetTensor].push_back(
      TensorReferenceGroup::makeSingleton(access, scopedAccess, type));
}

void addSingletonReferenceGroups(
    TensorGroups& tensorGroups,
    isl::UnionMap<isl::Pair<Statement, Tag>, Tensor> accesses,
    isl::UnionSet<Statement> domain,
    isl::UnionMap<Statement, Prefix> schedule,
    AccessType type) {
  // access relations have a shape :: [D -> ref] -> O
  // use currying to isolate the D part before intersecting with the domain
  // Compute initial groups with single reference per group.
  std::unordered_set<isl::id, isl::IslIdIslHash> unapproximatable;
  for (auto a : accesses.get_map_list()) {
    if (a.curry().asUnionMap().intersect_domain(domain).is_empty()) {
      continue;
    }

    auto tensorId = a.get_range_tuple_id();
    if (unapproximatable.count(tensorId) != 0) {
      continue;
    }
    try {
      addSingletonReferenceGroup(tensorGroups, tensorId, schedule, a, type);
    } catch (const promotion::GroupingError& err) {
      unapproximatable.insert(tensorId);
    }
  }
}
} // namespace

// Compute tensor reference groups encapsulating all tensor accesses within
// "outerSchedule".  Only statement instances present in the domain of
// "outerSchedule" are considered.  In particular, if this domain is
// intersected with block and/or thread mapping, the reference groups are
// computed inside one block and/or thread, even if "outerSchedule" does not
// include band members mapped to blocks and/or threads.
//
// Tensor reference descriptors (TensorReference) contain information about
// tensor elements accessed through the given reference within "outerSchedule".
// Several references form a group (TensorReferenceGroup) if the same elements
// may be accessed through these references, and at least one of the accesses
// writes to the element.  A group stores a rectangular overapproximation of
// the set of accessed tensor elements (access footprint).  This
// overappoximation can be used to create copies of the given tensor elements
// in another memory space, i.e., to perform memory promotion.  If the domain
// of "outerSchedule" included thread or block mapping, then the
// overappoximation is computed per-block or per-thread.
//
// Returns a map between tensor ids and vectors of unique pointers to
// TensorReferenceGroup, with each group potentially containing multiple
// references.
TensorGroups TensorReferenceGroup::accessedWithin(
    isl::UnionMap<Statement, Prefix> outerSchedule,
    const Body& body) {
  TensorGroups tensorGroups;
  auto domain = outerSchedule.domain();

  addSingletonReferenceGroups(
      tensorGroups, body.writes, domain, outerSchedule, AccessType::Write);
  addSingletonReferenceGroups(
      tensorGroups, body.reads, domain, outerSchedule, AccessType::Read);

  // For each tensor, join groups whose footprints overlap and at least one
  // access is a write.  Do not join between tensors because no aliasing.
  for (auto& p : tensorGroups) {
    joinOverlappingWrites(p.second);
  }
  return tensorGroups;
}

// Compute the relation between schedule dimensions, original and promoted array
// subscripts, in the space
//   [S -> O] -> O.
// The caller is in charge of updating the tuple of the target space with the
// group identifier.
// The mapping depends on the original schedule dimensions because the same
// elements of the promoted array get assigned different values of the original
// array in different outer loop iterations; it's impossible to project out the
// outer schedule dimensions.
isl::MultiAff<isl::Pair<Prefix, Tensor>, Tensor>
TensorReferenceGroup::promotion() const {
  // access space is S -> O
  auto map = scopedAccesses();
  auto accessSpace = map.get_space();

  // Construct a projection multi-aff in [S -> O] -> S
  // for further precomposition.
  auto originalSpaceInserter =
      isl::MultiAff<isl::Pair<Prefix, Tensor>, Prefix>::domain_map(accessSpace);

  // Lower bounds and offsets space is S -> O; transform into [S -> O] -> O.
  auto lowerBounds =
      approximation.lowerBounds().pullback(originalSpaceInserter);
  auto offsets = approximation.strideOffsets.pullback(originalSpaceInserter);

  // Create promotion starting by identity in [S -> O] -> O.
  auto original =
      isl::MultiAff<isl::Pair<Prefix, Tensor>, Tensor>::range_map(accessSpace);
  auto promotion =
      (original - offsets) / approximation.strideValues - lowerBounds;

  return promotion;
}

std::unordered_set<isl::id, isl::IslIdIslHash>
TensorReferenceGroup::referenceIds() const {
  std::unordered_set<isl::id, isl::IslIdIslHash> ids;
  for (const auto& ref : references) {
    ids.insert(ref->refId);
  }
  return ids;
}

namespace {
// Construct the set containing all tensor elements.
//
// Find the Halide image corresponding to the given tensorId.  Transform its
// min() and extent() into parametric isl affs and construct a set where the
// each dimension of the tensor is contrained by the min_aff on the left and
// by the min_aff + extent_aff on the right.  Intersect this set with the
// context of the scop.
isl::Set<Tensor> tensorElementsSet(const Scop& scop, isl::id tensorId) {
  auto halideParameter = scop.findArgument(tensorId).parameter();
  auto space = scop.domain().get_space();
  auto nDim = halideParameter.dimensions();
  auto tensorTuple = constructTensorTuple(space, tensorId, nDim);
  auto tensorSpace = tensorTuple.get_space();

  auto tensorElements = isl::Set<Tensor>::universe(tensorSpace);
  auto identity =
      isl::MultiAff<Tensor, Tensor>::identity(tensorSpace.map_from_set());
  for (int i = 0; i < nDim; ++i) {
    auto minAff = halide2isl::makeIslAffFromExpr(
        space, halideParameter.min_constraint(i));
    auto extentAff = halide2isl::makeIslAffFromExpr(
        space, halideParameter.extent_constraint(i));
    auto minAff2 = minAff.unbind_params_insert_domain(tensorTuple);
    auto extentAff2 = extentAff.unbind_params_insert_domain(tensorTuple);
    auto aff = identity.get_aff(i);
    tensorElements = tensorElements & (minAff2.le_set(aff)) &
        (aff.lt_set(minAff2 + extentAff2));
  }

  tensorElements = tensorElements.intersect_params(scop.context());
  return tensorElements;
}

/*
 * "schedule" iterates over the elements of the tensor described by "decl".
 * Remove the schedule dimensions that correspond to tensor dimensions
 * of size 1.
 * Note that this function drops the name of the target space of "schedule",
 * but this space is irrelevant for the caller.
 */
template <typename Domain>
isl::MultiAff<Domain, Domain> dropDummyTensorDimensions(
    isl::MultiAff<Domain, Domain> schedule,
    const Scop::PromotedDecl& decl) {
  auto list = schedule.get_aff_list();
  auto domainSpace = schedule.get_space().domain();

  auto n = list.size();
  for (int i = n - 1; i >= 0; --i) {
    if (decl.sizes[i] == 1) {
      list = list.drop(i, 1);
    }
  }

  auto space = domainSpace.template add_unnamed_tuple_ui<Domain>(list.size());
  return isl::MultiAff<Domain, Domain>(space, list);
}

inline void unrollAllMembers(detail::ScheduleTreeBand* band) {
  band->unroll_ = std::vector<bool>(band->nMember(), true);
}

} // namespace

ScheduleTree* insertCopiesUnder(
    Scop& scop,
    ScheduleTree* tree,
    const TensorReferenceGroup& group,
    isl::id tensorId,
    isl::id groupId,
    bool unrollAllCopies) {
  const ScheduleTree* root = scop.scheduleRoot();
  auto ctx = root->ctx_;
  isl::id readId = isl::id(ctx, std::string(kReadIdName));
  isl::id writeId = isl::id(ctx, std::string(kWriteIdName));

  // Take the set of all tensor elements.
  auto tensorElements = tensorElementsSet(scop, tensorId);

  auto promotion =
      group.promotion().asMap().set_range_tuple_id<Promoted>(groupId);
  auto promotionSpace = promotion.get_space();

  auto identityCopySchedule = isl::MultiAff<Promoted, Promoted>::identity(
      promotionSpace.range().map_from_set());
  // Only iterate over significant tensor dimensions.
  auto decl = scop.promotedDecl(groupId);
  identityCopySchedule = dropDummyTensorDimensions(identityCopySchedule, decl);
  auto readSpace = promotionSpace.wrap().set_set_tuple_id<Statement>(readId);
  auto writeSpace = promotionSpace.wrap().set_set_tuple_id<Statement>(writeId);
  auto readSchedule = isl::multi_union_pw_aff(identityCopySchedule.pullback(
      isl::MultiAff<
          isl::NamedPair<Statement, isl::Pair<Prefix, Tensor>, Promoted>,
          Promoted>::wrapped_range_map(readSpace)));
  auto writeSchedule = isl::multi_union_pw_aff(identityCopySchedule.pullback(
      isl::MultiAff<
          isl::NamedPair<Statement, isl::Pair<Prefix, Tensor>, Promoted>,
          Promoted>::wrapped_range_map(writeSpace)));

  auto readBandNode = ScheduleTree::makeBand(
      isl::MultiUnionPwAff<Statement, Band>(readSchedule));
  auto writeBandNode = ScheduleTree::makeBand(
      isl::MultiUnionPwAff<Statement, Band>(writeSchedule));

  if (unrollAllCopies) {
    unrollAllMembers(readBandNode->as<detail::ScheduleTreeBand>());
    unrollAllMembers(writeBandNode->as<detail::ScheduleTreeBand>());
  }

  auto extension =
      promotion.wrap().identity().domain_factor_domain().domain_factor_domain();

  // It's safe to read the overapproximated footprint, and it gives simpler
  // control flow, but we should only write back elements that are actually
  // written to.  In any case, intersect the footprint with the set of existing
  // tensor elements.
  auto promotedFootprint =
      group.promotedFootprint().set_tuple_id<Promoted>(groupId);
  auto scheduleUniverse =
      isl::Set<Prefix>::universe(promotionSpace.domain().unwrap().domain());
  auto arrayId = promotionSpace.domain().unwrap().get_map_range_tuple_id();
  auto approximatedRead =
      group.approximateScopedAccesses().intersect_range(tensorElements).wrap();
  auto product = approximatedRead.product(promotedFootprint);
  auto readExtension =
      extension.intersect_range(product).set_range_tuple_id<Statement>(readId);
  auto writtenElements = group.scopedWrites()
                             .intersect_range(tensorElements)
                             .wrap()
                             .product(promotedFootprint);
  auto writeExtension = extension.intersect_range(writtenElements)
                            .set_range_tuple_id<Statement>(writeId);

  auto readFilterNode = ScheduleTree::makeFilter(
      isl::set::universe(readExtension.get_space().range()),
      std::move(readBandNode));
  auto writeFilterNode = ScheduleTree::makeFilter(
      isl::set::universe(writeExtension.get_space().range()),
      std::move(writeBandNode));

  bool reads = !group.scopedReads().is_empty();
  bool writes = !group.scopedWrites().is_empty();

  if (tree->numChildren() == 0) {
    // The point underneath a leaf node cannot be referenced,
    // so insert a dummy sequence first.  It will be extended
    // with the reads and/or writes.
    insertSequenceBelow(root, tree);
  }

  if (reads) {
    insertExtensionBefore(
        root, tree, tree->child({0}), readExtension, std::move(readFilterNode));
  }
  if (writes) {
    insertExtensionAfter(
        root,
        tree,
        tree->child({0}),
        writeExtension,
        std::move(writeFilterNode));
  }

  return tree;
}
} // namespace polyhedral
} // namespace tc
