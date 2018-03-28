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

std::pair<isl::val, isl::aff> outputRangeSingle(isl::map access) {
  CHECK_EQ(access.dim(isl::dim_type::out), 1)
      << "expected 1-dim output, call outputRanges instead";
  access = access.detect_equalities();
  auto wrappedAccess = access.wrap().flatten().compute_divs().simple_hull();

  // TODO: also compute strides

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

ScopedFootprint outputRanges(isl::map access) {
  int nSubscripts = access.dim(isl::dim_type::out);
  ScopedFootprint footprint;
  for (int i = 0; i < nSubscripts; ++i) {
    auto singleDim =
        access.project_out(isl::dim_type::out, 0, i)
            .project_out(isl::dim_type::out, 1, nSubscripts - i - 1);
    auto range = outputRangeSingle(singleDim);
    if (range.first.is_nan()) {
      return {};
    }
    footprint.emplace_back(range.second, range.first);
  }
  return footprint;
}
} // namespace

// Access has the shape :: [D -> ref] -> O
// Extract the reference ID, store it separatly and simplify the access.
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

isl::set ScopedFootprint::footprint(isl::set domain) const {
  auto space = domain.get_space().from_domain();
  space = space.add_dims(isl::dim_type::out, size());
  auto accessed = isl::map::universe(space).intersect_domain(domain);
  auto lspace = isl::local_space(accessed.get_space().range());

  for (size_t i = 0; i < size(); ++i) {
    auto dim = at(i);
    auto rhs = isl::aff(lspace, isl::dim_type::set, i);
    isl::map partial = (isl::aff_map(dim.lowerBound) <= rhs) &
        (isl::aff_map(dim.lowerBound + dim.size) > rhs);
    accessed = accessed & partial;
  }
  return accessed.range();
}

isl::multi_aff ScopedFootprint::lowerBounds() const {
  if (size() == 0) {
    throw promotion::PromotionNYI("promotion for scalars");
  }
  auto space = at(0).lowerBound.get_space();
  space = space.add_dims(isl::dim_type::out, size() - 1);
  auto ma = isl::multi_aff::zero(space);

  int i = 0;
  for (const auto& a : *this) {
    ma = ma.set_aff(i++, a.lowerBound);
  }
  return ma;
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

TensorGroups TensorReferenceGroup::accessedBySubtree(
    const ScheduleTree* tree,
    const Scop& scop) {
  TensorGroups tensorGroups;
  auto domain = activeDomainPoints(scop.scheduleRoot(), tree);
  auto schedule = partialSchedule(scop.scheduleRoot(), tree);

  addSingletonReferenceGroups(
      tensorGroups, scop.writes, domain, schedule, AccessType::Write);
  addSingletonReferenceGroups(
      tensorGroups, scop.reads, domain, schedule, AccessType::Read);

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

  // lower bounsd space is S -> O; which we transform into [S -> O] -> P
  auto lowerBounds = approximation.lowerBounds().pullback(
      isl::multi_aff::domain_map(accessSpace));
  auto promotion = isl::multi_aff::range_map(accessSpace)
                       .reset_tuple_id(isl::dim_type::out) -
      lowerBounds;
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
bool hasCopyExtensionSingleChild(const ScheduleTree* tree) {
  if (tree->numChildren() != 1) {
    return false;
  }

  auto extensionNode =
      tree->child({0})->elemAs<detail::ScheduleTreeElemExtension>();
  if (!extensionNode) {
    return false;
  }

  if ((tree->child({0})->numChildren() != 1) &&
      (tree->child({0, 0})->elemAs<detail::ScheduleTreeElemSequence>())) {
    return false;
  }

  for (auto e : isl::UnionAsVector<isl::union_map>(extensionNode->extension_)) {
    if (!e.has_tuple_name(isl::dim_type::out)) {
      return false;
    }
    if (e.get_tuple_name(isl::dim_type::out) != kReadIdName &&
        e.get_tuple_name(isl::dim_type::out) != kWriteIdName) {
      return false;
    }
  }
  return true;
}

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
  space = space.add_dims(isl::dim_type::set, nDim)
              .set_tuple_id(isl::dim_type::set, tensorId);

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
} // namespace

ScheduleTree* insertCopiesUnder_(
    Scop& scop,
    ScheduleTree* tree,
    const TensorReferenceGroup& group,
    isl::map promotion,
    isl::set originalElements,
    isl::set readElements,
    isl::map exactWrites) {
  auto groupId = promotion.get_tuple_id(isl::dim_type::out);
  const ScheduleTree* root = scop.scheduleRoot();
  auto ctx = root->ctx_;
  isl::id readId = isl::id(ctx, std::string(kReadIdName));
  isl::id writeId = isl::id(ctx, std::string(kWriteIdName));

  auto promotionSpace = promotion.get_space();

  auto identityCopySchedule =
      isl::multi_aff::identity(promotionSpace.range().map_from_set());
  identityCopySchedule =
      identityCopySchedule.pullback(isl::multi_aff::range_map(promotionSpace));
  auto readSchedule = isl::multi_union_pw_aff(
      identityCopySchedule.set_tuple_id(isl::dim_type::in, readId));
  auto writeSchedule = isl::multi_union_pw_aff(
      identityCopySchedule.set_tuple_id(isl::dim_type::in, writeId));

  auto readBandNode = ScheduleTree::makeBand(readSchedule);
  auto writeBandNode = ScheduleTree::makeBand(writeSchedule);

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
  auto approximattedRead =
      isl::map(
          scheduleUniverse,
          readElements.set_tuple_id(arrayId).intersect(originalElements))
          .wrap();
  approximattedRead = isl::map(approximattedRead, promotedFootprint).wrap();
  auto readExtension = extension.intersect_range(approximattedRead)
                           .set_tuple_id(isl::dim_type::out, readId);

  auto writtenElements =
      isl::map(
          exactWrites.intersect_range(originalElements).wrap(),
          promotedFootprint)
          .wrap();
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

  if (hasCopyExtensionSingleChild(tree)) {
    auto extensionNode = tree->child({0});
    auto sequenceNode = tree->child({0, 0});

    auto& ext =
        extensionNode->elemAs<detail::ScheduleTreeElemExtension>()->extension_;
    if (reads) {
      ext = ext.unite(isl::union_map(readExtension));
      sequenceNode->insertChild(0, std::move(readFilterNode));
    }
    if (writes) {
      ext = ext.unite(isl::union_map(writeExtension));
      sequenceNode->appendChild(std::move(writeFilterNode));
    }
    return tree;
  }

  auto mainCompFilter = activeDomainPoints(root, tree).universe();
  auto mainCompFilterNode =
      ScheduleTree::makeFilter(mainCompFilter, tree->detachChildren());

  // XXX: I don't really like the syntax-imposed impossibility to create a
  // sequence node with no children.
  auto sequenceNode = ScheduleTree::makeSequence(
      reads ? std::move(readFilterNode) : std::move(mainCompFilterNode));
  if (reads) {
    sequenceNode->appendChild(std::move(mainCompFilterNode));
  }
  if (writes) {
    sequenceNode->appendChild(std::move(writeFilterNode));
  }

  auto extensionUmap = isl::union_map::empty(promotionSpace.params());
  if (reads) {
    extensionUmap = extensionUmap.unite(readExtension);
  }
  if (writes) {
    extensionUmap = extensionUmap.unite(writeExtension);
  }
  auto extensionNode =
      ScheduleTree::makeExtension(extensionUmap, std::move(sequenceNode));
  tree->appendChild(std::move(extensionNode));
  return tree;
}

ScheduleTree* insertIntraCopiesUnder(
    Scop& scop,
    ScheduleTree* tree,
    const TensorReferenceGroup& group,
    const TensorReferenceGroup& outerScopeGroup,
    isl::id tensorId,
    isl::id groupId,
    isl::id outerScopeGroupId) {
  auto innerScopePromotion =
      isl::map(group.promotion()).set_tuple_id(isl::dim_type::out, groupId);
  auto outerScopePromotion =
      isl::map(outerScopeGroup.promotion())
          .set_tuple_id(isl::dim_type::out, outerScopeGroupId);

  auto outerScopeInDims =
      outerScopePromotion.get_space().curry().dim(isl::dim_type::in);
  auto innerScopeInDims =
      innerScopePromotion.get_space().curry().dim(isl::dim_type::in);
  CHECK_GT(innerScopeInDims, outerScopeInDims);
  outerScopePromotion =
      outerScopePromotion.curry()
          .add_dims(isl::dim_type::in, innerScopeInDims - outerScopeInDims)
          .uncurry();
  auto domainAccessToDomainMap = isl::map(isl::multi_aff::domain_map(
      innerScopePromotion.get_space().domain().unwrap()));
  outerScopePromotion =
      domainAccessToDomainMap.range_product(outerScopePromotion);
  innerScopePromotion = innerScopePromotion.apply_domain(outerScopePromotion);

  return insertCopiesUnder_(
      scop,
      tree,
      group,
      innerScopePromotion,
      outerScopeGroup.promotedFootprint().set_tuple_id(outerScopeGroupId),
      outerScopeGroup.promotedFootprint().set_tuple_id(outerScopeGroupId),
      group.scopedWrites().wrap().apply(outerScopePromotion).unwrap());
}

ScheduleTree* insertCopiesUnder(
    Scop& scop,
    ScheduleTree* tree,
    const TensorReferenceGroup& group,
    isl::id tensorId,
    isl::id groupId) {
  // Take the set of all tensor elements.
  auto tensorElements = tensorElementsSet(scop, tensorId);

  if (groupId.is_null()) {
    throw promotion::GroupingError("expected group id");
  }
  auto promotion =
      isl::map(group.promotion()).set_tuple_id(isl::dim_type::out, groupId);

  return insertCopiesUnder_(
      scop,
      tree,
      group,
      promotion,
      tensorElements,
      group.approximateFootprint(),
      group.scopedWrites());
}
} // namespace polyhedral
} // namespace tc
