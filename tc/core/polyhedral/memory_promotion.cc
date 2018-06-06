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

#include "tc/core/polyhedral/exceptions.h"
#include "tc/core/polyhedral/schedule_tree.h"
#include "tc/core/polyhedral/scop.h"
#include "tc/external/isl.h"

namespace tc {
namespace polyhedral {

using detail::ScheduleTree;

namespace {
std::pair<isl::val, isl::aff> outputRange(
    isl::basic_set wrappedAccess,
    isl::constraint cstr) {
  auto emptyRange =
      std::make_pair(isl::val::nan(wrappedAccess.get_ctx()), isl::aff());
  int pos = cstr.dim(isl::dim_type::set) - 1;
  if (!cstr.is_lower_bound(isl::dim_type::set, pos)) {
    return emptyRange;
  }
  if (cstr.involves_dims(isl::dim_type::div, 0, cstr.dim(isl::dim_type::div))) {
    return emptyRange;
  }

  auto lowerBound = cstr.get_bound(isl::dim_type::set, pos).ceil();
  auto aff = lowerBound.neg().add_coefficient_si(isl::dim_type::in, pos, 1);
  lowerBound = lowerBound.drop_dims(isl::dim_type::in, pos, 1);

  auto range = wrappedAccess.max_val(aff);
  if (range.is_int()) {
    return std::make_pair(range + 1, lowerBound);
  }
  return emptyRange;
}

// Compute the bounds of the single output dimension of the given relation.
// The lower bound is an affine function defined in terms of input dimensions
// and parameters, and the upper bound has the same shape but shifted by a
// constant value.  Look for bounds of the same shape as lower bound
// constraints present in the "access" relation, and pick the one with the
// smallest constant distance between the lower and the upper bound.
//
// TODO: replace this with a call to isl_map_range_box_hull when available.
std::pair<isl::val, isl::aff> outputRangeSingle(isl::map access) {
  CHECK_EQ(access.dim(isl::dim_type::out), 1u)
      << "expected 1-dim output, call outputRanges instead";
  access = access.detect_equalities();
  auto wrappedAccess = access.wrap().flatten().compute_divs().simple_hull();

  isl::val minRange;
  isl::aff lowerBoundWithMinRange;
  for (auto cstr : wrappedAccess.get_constraint_list()) {
    auto range = outputRange(wrappedAccess, cstr);
    if (range.first.is_nan()) {
      continue;
    }
    if (minRange.is_null() || range.first < minRange) {
      minRange = range.first;
      lowerBoundWithMinRange = range.second;
    }
  }
  if (minRange.is_null()) {
    return std::make_pair(
        isl::val::nan(access.get_ctx()), lowerBoundWithMinRange);
  }

  return std::make_pair(minRange, lowerBoundWithMinRange);
}

isl::map removeRangeStrides(
    isl::map relation,
    const std::vector<isl::val>& strides,
    const std::vector<isl::aff>& offsets) {
  CHECK_EQ(strides.size(), offsets.size());

  auto valList = isl::val_list(relation.get_ctx(), strides.size());
  auto affList = isl::aff_list(relation.get_ctx(), 1);
  for (int i = 0; i < strides.size(); ++i) {
    valList = valList.add(strides.at(i));
    affList = affList.add(offsets.at(i).neg());
  }
  auto space = relation.get_space();
  auto stridesMV = isl::multi_val(space.range(), valList);
  auto stridesMA = isl::multi_aff::identity(space.range().map_from_set());
  stridesMA = stridesMA / stridesMV;
  auto negOffsetsMA = isl::multi_aff(space, affList);

  return relation.sum(isl::map(negOffsetsMA)).apply_range(isl::map(stridesMA));
}

// Compute the box approximation of the range of the given relation,
// including the lower bounds, the box sizes, and the strides.
// If the range has strides, remove them first.
// TODO: replace this with a call to isl_map_range_box_hull when available.
ScopedFootprint outputRanges(isl::map access) {
  int nSubscripts = access.dim(isl::dim_type::out);

  if (access.get_space().domain().is_wrapping()) {
    access = access.domain_factor_domain();
  }

  std::vector<isl::val> strides;
  std::vector<isl::aff> offsets;
  for (int i = 0; i < nSubscripts; ++i) {
    auto si = access.get_range_stride_info(i);
    strides.push_back(si.get_stride());
    offsets.push_back(si.get_offset());
  }

  access = removeRangeStrides(access, strides, offsets);

  ScopedFootprint footprint;
  for (int i = 0; i < nSubscripts; ++i) {
    auto singleDim =
        access.project_out(isl::dim_type::out, 0, i)
            .project_out(isl::dim_type::out, 1, nSubscripts - i - 1);
    auto range = outputRangeSingle(singleDim);
    if (range.first.is_nan()) {
      return {};
    }
    footprint.emplace_back(range.second, range.first, offsets[i], strides[i]);
  }
  return footprint;
}

// Given a set space, construct a map space with the input as domain and
// a range of the given size.
isl::space add_range(isl::space space, unsigned dim) {
  auto range = space.params().unnamed_set_from_params(dim);
  return space.map_from_domain_and_range(range);
}

} // namespace

// Access has the shape :: [S -> ref] -> O
// Extract the reference ID, store it separately and simplify the access.
std::unique_ptr<TensorReferenceGroup> TensorReferenceGroup::makeSingleton(
    isl::map originalAccess,
    isl::map scopedAccess,
    AccessType type) {
  auto ref = std::unique_ptr<TensorReference>(new TensorReference);
  auto refId = scopedAccess.get_space().domain().unwrap().get_tuple_id(
      isl::dim_type::out);
  ref->originalAccess = originalAccess.domain_factor_domain();
  ref->scopedAccess = scopedAccess.domain_factor_domain();
  ref->type = type;
  ref->refId = refId;
  auto group = std::unique_ptr<TensorReferenceGroup>(new TensorReferenceGroup);
  group->references.push_back(std::move(ref));
  group->approximation = outputRanges(scopedAccess);

  if (group->approximation.size() != scopedAccess.dim(isl::dim_type::out)) {
    std::stringstream ss;
    ss << "could not compute rectangular overapproximation of: "
       << scopedAccess;
    throw promotion::GroupingError(ss.str());
  }

  return group;
}

isl::set TensorReferenceGroup::approximateFootprint() const {
  auto accessed = scopedAccesses();
  auto lspace = isl::local_space(accessed.get_space().range());

  for (size_t i = 0; i < approximation.size(); ++i) {
    auto dim = approximation.at(i);
    auto rhs = isl::aff(lspace, isl::dim_type::set, i);
    auto lowerBound = dim.lowerBound * dim.stride + dim.offset;
    auto upperBound = (dim.lowerBound + dim.size) * dim.stride + dim.offset;
    isl::map partial =
        (isl::aff_map(lowerBound) <= rhs) & (isl::aff_map(upperBound) > rhs);

    accessed = accessed & partial;
  }
  return accessed.range();
}

namespace {
template <typename Func>
isl::multi_aff createMA(
    const std::vector<ScopedFootprintDim>& dims,
    Func accessor) {
  if (dims.size() == 0) {
    throw promotion::PromotionNYI("promotion for scalars");
  }
  auto space =
      add_range(accessor(dims.at(0)).get_space().domain(), dims.size());
  auto ma = isl::multi_aff::zero(space);

  int i = 0;
  for (const auto& a : dims) {
    ma = ma.set_aff(i++, accessor(a));
  }
  return ma;
}
} // namespace

isl::multi_aff ScopedFootprint::lowerBounds() const {
  return createMA(
      *this, [](const ScopedFootprintDim& dim) { return dim.lowerBound; });
}

isl::multi_aff ScopedFootprint::offsets() const {
  return createMA(
      *this, [](const ScopedFootprintDim& dim) { return dim.offset; });
}

// TODO: this can be abstracted in the template createMA by making the
// return type a template argument, and by exporting set_aff/val as
// __isl_overload for name homogeneity.
isl::multi_val ScopedFootprint::strides() const {
  if (size() == 0) {
    throw promotion::PromotionNYI("promotion for scalars");
  }
  auto space = add_range(at(0).lowerBound.get_space().domain(), size());
  auto mv = isl::multi_val::zero(space);

  int i = 0;
  for (const auto& a : *this) {
    mv = mv.set_val(i++, a.stride);
  }
  return mv;
}

bool TensorReferenceGroup::isReadOnly() const {
  bool result = true;
  for (auto const& ref : references) {
    result &= !ref->isWrite();
  }
  return result;
}

isl::set TensorReferenceGroup::promotedFootprint() const {
  auto space =
      scopedAccesses().get_space().range().reset_tuple_id(isl::dim_type::set);
  auto sizes = approximationSizes();
  if (sizes.size() != space.dim(isl::dim_type::set)) {
    throw promotion::GroupingError("unexpected dimensionality mismatch");
  }

  isl::set footprint = isl::set::universe(space);
  auto lspace = isl::local_space(space);
  for (size_t i = 0, e = sizes.size(); i < e; ++i) {
    auto aff = isl::aff(lspace, isl::dim_type::out, i);
    footprint =
        footprint & (isl::aff_set(aff) >= 0) & (isl::aff_set(aff) < sizes[i]);
  }
  return footprint;
}

std::vector<size_t> TensorReferenceGroup::approximationSizes() const {
  std::vector<size_t> result;
  result.reserve(approximation.size());
  for (const auto& dim : approximation) {
    result.push_back(dim.size.get_num_si());
  }
  return result;
}

namespace {
isl::map referenceScopedAccessesImpl(
    const TensorReferenceGroup& group,
    AccessType type) {
  if (group.references.size() == 0) {
    throw promotion::GroupingError("no references in the group");
  }
  auto accesses =
      isl::map::empty(group.references.front()->scopedAccess.get_space());

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

isl::map TensorReferenceGroup::scopedWrites() const {
  return referenceScopedAccessesImpl(*this, AccessType::Write);
}

isl::map TensorReferenceGroup::scopedReads() const {
  return referenceScopedAccessesImpl(*this, AccessType::Read);
}

namespace {
isl::union_map referenceOriginalAccessesImpl(
    const TensorReferenceGroup& group,
    AccessType type) {
  if (group.references.size() == 0) {
    throw promotion::GroupingError("no references in the group");
  }
  auto accesses = isl::union_map::empty(
      group.references.front()->originalAccess.get_space());

  for (const auto& ref : group.references) {
    if (ref->type != type) {
      continue;
    }
    auto current = ref->originalAccess;
    accesses = accesses.unite(isl::union_map(current));
  }
  return accesses;
}
} // namespace

isl::union_map TensorReferenceGroup::originalWrites() const {
  return referenceOriginalAccessesImpl(*this, AccessType::Write);
}

isl::union_map TensorReferenceGroup::originalReads() const {
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
      if (g1->approximateFootprint()
              .intersect(g2->approximateFootprint())
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
    isl::union_map schedule,
    isl::map access,
    AccessType type) {
  auto scopedUnionAccess = isl::union_map(access.curry());
  scopedUnionAccess = scopedUnionAccess.apply_domain(schedule);
  auto scopedAccess = isl::map::from_union_map(scopedUnionAccess);
  scopedAccess = scopedAccess.uncurry();
  tensorGroups[targetTensor].push_back(
      TensorReferenceGroup::makeSingleton(access, scopedAccess, type));
}

void addSingletonReferenceGroups(
    TensorGroups& tensorGroups,
    isl::union_map accesses,
    isl::union_set domain,
    isl::union_map schedule,
    AccessType type) {
  // access relations have a shape :: [D -> ref] -> O
  // use currying to isolate the D part before intersecting with the domain
  // Compute initial groups with single reference per group.
  std::unordered_set<isl::id, isl::IslIdIslHash> unapproximatable;
  for (auto a : isl::UnionAsVector<isl::union_map>(accesses)) {
    if (isl::union_map(a.curry()).intersect_domain(domain).is_empty()) {
      continue;
    }

    auto tensorId = a.get_tuple_id(isl::dim_type::out);
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
// write to the element.  A group stores a rectangular overapproximation of the
// set of accessed tensor elements (access footprint).  This overappoximation
// can be used to create copies of the given tensor elements in another memory
// space, i.e., to perform memory promotion.  If the domain of "outerSchedule"
// included thread or block mapping, then the overappoximation is computed
// per-block or per-thread.
//
// Returns a map between tensor ids and vectors of unique pointers to
// TensorReferenceGroup, with each group potentially containing multiple
// references.
TensorGroups TensorReferenceGroup::accessedWithin(
    isl::union_map outerSchedule,
    isl::union_map reads,
    isl::union_map writes) {
  TensorGroups tensorGroups;
  auto domain = outerSchedule.domain();

  addSingletonReferenceGroups(
      tensorGroups, writes, domain, outerSchedule, AccessType::Write);
  addSingletonReferenceGroups(
      tensorGroups, reads, domain, outerSchedule, AccessType::Read);

  // For each tensor, join groups whose footprints overlap and at least one
  // access is a write.  Do not join between tensors because no aliasing.
  for (auto& p : tensorGroups) {
    joinOverlappingWrites(p.second);
  }
  return tensorGroups;
}

// Compute the relation between schedule dimensions, orignal and promoted array
// subscripts, in the space
//   [S -> O] -> P
// The mapping depends on the original schedule dimensions because the same
// elements of the promoted array get assigned different values of the original
// array in different outer loop iterations; it's impossible to project out the
// outer schedule dimensions.
isl::multi_aff TensorReferenceGroup::promotion() const {
  // access space is S -> O
  isl::map map = scopedAccesses();
  auto accessSpace = map.get_space();

  // Construct a projection multi-aff in [S -> O] -> S
  // for futher precomposition.
  auto originalSpaceInserter = isl::multi_aff::domain_map(accessSpace);

  // Lower bounds and offsets space is S -> P; transform into [S -> O] -> P.
  auto lowerBounds =
      approximation.lowerBounds().pullback(originalSpaceInserter);
  auto offsets = approximation.offsets().pullback(originalSpaceInserter);

  // Create identity mapping from original to promoted,
  // resetting the tuple name for the promoted tensor.
  auto original =
      isl::multi_aff::range_map(accessSpace).reset_tuple_id(isl::dim_type::out);

  return (original - offsets) / approximation.strides() - lowerBounds;
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
isl::set tensorElementsSet(const Scop& scop, isl::id tensorId) {
  auto halideParameter = scop.findArgument(tensorId).parameter();
  auto space = scop.domain().get_space().params();
  auto nDim = halideParameter.dimensions();
  space = space.named_set_from_params_id(tensorId, nDim);

  auto tensorElements = isl::set::universe(space);
  for (int i = 0; i < nDim; ++i) {
    auto minAff = halide2isl::makeIslAffFromExpr(
        space, halideParameter.min_constraint(i));
    auto extentAff = halide2isl::makeIslAffFromExpr(
        space, halideParameter.extent_constraint(i));
    auto aff = isl::aff(isl::local_space(space), isl::dim_type::set, i);
    tensorElements = tensorElements & (minAff <= isl::aff_set(aff)) &
        (isl::aff_set(aff) < (minAff + extentAff));
  }

  if (scop.globalParameterContext) {
    tensorElements =
        tensorElements.intersect_params(scop.globalParameterContext);
  }
  return tensorElements;
}

/*
 * "schedule" iterates over the elements of the tensor described by "decl".
 * Remove the schedule dimensions that correspond to tensor dimensions
 * of size 1.
 * Note that this function drops the name of the target space of "schedule",
 * but this space is irrelevant for the caller.
 */
isl::multi_aff dropDummyTensorDimensions(
    isl::multi_aff schedule,
    const Scop::PromotedDecl& decl) {
  auto list = schedule.get_aff_list();
  auto space = schedule.get_space().domain();

  auto n = list.n();
  for (int i = n - 1; i >= 0; --i) {
    if (decl.sizes[i] == 1) {
      list = list.drop(i, 1);
    }
  }

  space = add_range(space, list.n());
  return isl::multi_aff(space, list);
}

inline void unrollAllMembers(detail::ScheduleTreeElemBand* band) {
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

  if (groupId.is_null()) {
    throw promotion::GroupingError("expected group id");
  }
  auto promotion =
      isl::map(group.promotion()).set_tuple_id(isl::dim_type::out, groupId);
  auto promotionSpace = promotion.get_space();

  auto identityCopySchedule =
      isl::multi_aff::identity(promotionSpace.range().map_from_set());
  identityCopySchedule =
      identityCopySchedule.pullback(isl::multi_aff::range_map(promotionSpace));
  // Only iterate over significant tensor dimensions.
  auto decl = scop.promotedDecl(groupId);
  identityCopySchedule = dropDummyTensorDimensions(identityCopySchedule, decl);
  auto readSchedule = isl::multi_union_pw_aff(
      identityCopySchedule.set_tuple_id(isl::dim_type::in, readId));
  auto writeSchedule = isl::multi_union_pw_aff(
      identityCopySchedule.set_tuple_id(isl::dim_type::in, writeId));

  auto readBandNode = ScheduleTree::makeBand(readSchedule);
  auto writeBandNode = ScheduleTree::makeBand(writeSchedule);

  if (unrollAllCopies) {
    unrollAllMembers(readBandNode->elemAs<detail::ScheduleTreeElemBand>());
    unrollAllMembers(writeBandNode->elemAs<detail::ScheduleTreeElemBand>());
  }

  auto extension =
      promotion.wrap().identity().domain_factor_domain().domain_factor_domain();

  // It's safe to read the overapproximated footprint, and it gives simpler
  // control flow, but we should only write back elements that are actually
  // written to.  In any case, intersect the footprint with the set of existing
  // tensor elements.
  auto promotedFootprint = group.promotedFootprint().set_tuple_id(groupId);
  auto scheduleUniverse =
      isl::set::universe(promotionSpace.domain().unwrap().domain());
  auto arrayId =
      promotionSpace.domain().unwrap().get_tuple_id(isl::dim_type::out);
  auto approximatedRead = scheduleUniverse.product(
      group.approximateFootprint().set_tuple_id(arrayId).intersect(
          tensorElements));
  approximatedRead = approximatedRead.product(promotedFootprint);
  auto readExtension = extension.intersect_range(approximatedRead)
                           .set_tuple_id(isl::dim_type::out, readId);
  auto writtenElements =
      group.scopedWrites().intersect_range(tensorElements).wrap();
  writtenElements = writtenElements.product(promotedFootprint);
  auto writeExtension = extension.intersect_range(writtenElements)
                            .set_tuple_id(isl::dim_type::out, writeId);

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
