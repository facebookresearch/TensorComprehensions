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
#include "tc/core/polyhedral/scop.h"

#include <functional>
#include <memory>
#include <numeric>
#include <set>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include "tc/core/halide2isl.h"
#include "tc/core/polyhedral/exceptions.h"
#include "tc/core/polyhedral/functional.h"
#include "tc/core/polyhedral/memory_promotion.h"
#include "tc/core/polyhedral/schedule_isl_conversion.h"
#include "tc/core/polyhedral/schedule_transforms.h"
#include "tc/core/polyhedral/schedule_tree_matcher.h"
#include "tc/core/scope_guard.h"
#include "tc/core/tc2halide.h"

using namespace std;

namespace tc {
namespace polyhedral {

using namespace detail;
using ScopUPtr = std::unique_ptr<Scop>;

ScopUPtr Scop::makeScop(
    isl::ctx ctx,
    const tc2halide::HalideComponents& components) {
  CHECK(components.stmt.defined());

  halide2isl::SymbolTable sym = halide2isl::makeSymbolTable(components);

  isl::space paramSpace = halide2isl::makeParamSpace(ctx, sym);
  auto globalParameterContext = isl::set::universe(paramSpace);
  isl::local_space ls(globalParameterContext.get_space());
  for (int i = 0; i < globalParameterContext.dim(isl::dim_type::param); ++i) {
    globalParameterContext =
        globalParameterContext & (isl::aff(ls, isl::dim_type::param, i) >= 0);
  }

  ScopUPtr scop(new Scop());
  scop->globalParameterContext = globalParameterContext;
  scop->halide.params = sym.params;
  scop->halide.idx = sym.idxVars;
  scop->halide.reductionIdx = sym.reductionVars;
  scop->halide.inputs = components.inputs;
  scop->halide.outputs = components.outputs;

  auto tree = halide2isl::makeScheduleTree(paramSpace, components.stmt);
  scop->scheduleTreeUPtr = std::move(tree.tree);
  scop->reads = tree.reads;
  scop->writes = tree.writes;
  scop->halide.statements = std::move(tree.statements);
  scop->halide.accesses = std::move(tree.accesses);
  scop->halide.reductions = halide2isl::findReductions(components.stmt);

  // Set partial schedule tuples for proper comparison with ISL
  // schedules (needs DFSPreorder numbering). Just for testing.
  auto bands = ScheduleTree::collectDFSPreorder(
      scop->scheduleRoot(), detail::ScheduleTreeType::Band);
  for (size_t i = 0; i < bands.size(); ++i) {
    auto b = bands[i]->elemAs<ScheduleTreeElemBand>();
    CHECK(b);
    b->mupa_ = b->mupa_.set_tuple_name(
        isl::dim_type::set, kPartialScheduleLabel + std::to_string(i));
  }

  return scop;
}

ScopUPtr Scop::makeScop(isl::ctx ctx, const string& tc) {
  return makeScop(ctx, tc2halide::translate(ctx, tc));
}

ScopUPtr Scop::makeScop(isl::ctx ctx, const lang::TreeRef& treeRef) {
  return makeScop(ctx, tc2halide::translate(ctx, treeRef));
}

isl::union_set& Scop::domain() {
  auto dom = scheduleRoot()->elemAs<ScheduleTreeElemDomain>();
  CHECK(dom) << "root is not a domain in: " << *scheduleRoot();
  // TODO: activate this when the invariant has a chance of working (i.e. we
  // don't use a Context node for specifying parameter values that iterate in
  // spacetime).
  // TODO: find a proper place for the invariant.
  // auto noCont =
  //   scheduleRoot()->child({0})->elemAs<ScheduleTreeElemContext>();
  // CHECK(!noCont) << "root is not a domain in: " << *scheduleRoot();
  return dom->domain_;
}

const isl::union_set Scop::domain() const {
  return const_cast<Scop*>(this)->domain();
}

std::ostream& operator<<(std::ostream& os, const Scop& s) {
  os << "domain: " << s.domain() << "\n";
  os << "reads: " << s.reads << "\n";
  os << "writes: " << s.writes << "\n";
  os << "schedule: " << *s.scheduleRoot() << "\n";
  os << "idx: { ";
  for (auto i : s.halide.idx) {
    os << i << ", ";
  }
  os << "reductionIdx: { ";
  for (auto i : s.halide.reductionIdx) {
    os << i << ", ";
  }
  os << "params: {";
  for (auto p : s.halide.params) {
    os << p.type() << " " << p.name() << ", ";
  }
  os << "}";
  return os;
}

isl::id Scop::insertReductionSync1D(
    detail::ScheduleTree* redPoint,
    isl::id updateId) {
  isl::id treeSyncId, reductionInitId;
  std::tie(treeSyncId, reductionInitId) = makeReductionSpecialIds(updateId);

  // Append a filter for the sync after the two initial filters of the
  // redPoint.
  insertExtensionLabelAfter(scheduleRoot(), redPoint, treeSyncId);

  // Prepend a filter for the reduction register initialization before the two
  // initial filters of the redPoint.
  insertExtensionLabelBefore(scheduleRoot(), redPoint, reductionInitId);

  return treeSyncId;
}

void Scop::insertSync(detail::ScheduleTree* seqNode, size_t i) {
  insertExtensionLabelAt(scheduleRoot(), seqNode, i, makeSyncId());
}

namespace {

void checkFiltersDisjointStatements(const ScheduleTree* root) {
  for (auto node :
       ScheduleTree::collect(root, detail::ScheduleTreeType::Sequence)) {
    isl::union_set alreadyVisitedStmts;
    for (auto child : node->children()) {
      auto filterNode = child->elemAsBase<ScheduleTreeElemFilter>();
      CHECK(filterNode) << "expected children of seqence to be filters";
      auto filter = filterNode->filter_.universe();
      if (!alreadyVisitedStmts.get()) {
        alreadyVisitedStmts = filter;
      } else {
        // This may break if we implement recomputation or index-set splitting.
        // In these cases, promotion no longer applies to the entire statement,
        // but only to a part of it.  Possible solution -- introduce "scope"
        // mark nodes into the schedule tree that will contain information
        // about the promotion and process these marks when generating the AST.
        CHECK(alreadyVisitedStmts.intersect(filter).is_empty())
            << "filters are expected to be disjoint as stmt level";
        alreadyVisitedStmts = alreadyVisitedStmts.unite(filter);
      }
    }
  }
}
} // namespace

std::vector<size_t> Scop::activePromotionsIndexes(
    isl::union_set activePoints,
    isl::id tensorId) const {
  std::vector<size_t> result;

  for (size_t i = 0, e = activePromotions_.size(); i < e; ++i) {
    const auto& kvp = activePromotions_[i];
    if (kvp.first.intersect(activePoints).is_empty()) {
      continue;
    }

    auto groupId = kvp.second.groupId;
    if (promotedDecls_.count(groupId) != 0 &&
        promotedDecls_.at(groupId).tensorId == tensorId) {
      result.push_back(i);
    }
  }

  return result;
}

std::vector<std::pair<isl::union_set, Scop::PromotionInfo>>
Scop::promotionsAtIndexes(const std::vector<size_t>& indexes) const {
  std::vector<std::pair<isl::union_set, Scop::PromotionInfo>> result;

  for (auto idx : indexes) {
    result.emplace_back(activePromotions_[idx]);
  }

  return result;
}

namespace {
template <typename T>
T projectOutNamedParam(T t, isl::id paramId) {
  auto space = t.get_space();
  int pos = space.find_dim_by_id(isl::dim_type::param, paramId);
  return (pos == -1) ? t : t.project_out(isl::dim_type::param, pos, 1);
}
} // namespace

void Scop::promoteWithCopyFromGlobal(
    isl::union_set activePoints,
    PromotedDecl::Kind kind,
    isl::id tensorId,
    std::unique_ptr<TensorReferenceGroup>&& gr,
    ScheduleTree* tree,
    isl::union_map schedule,
    bool forceLastExtentOdd) {
  auto groupId = nextGroupIdForTensor(tensorId);
  insertCopiesUnder(*this, tree, *gr, kind == PromotedDecl::Kind::Register,
      tensorId, groupId);
  auto sizes = gr->approximationSizes();
  if (sizes.size() > 0 && forceLastExtentOdd && (sizes.back() % 2) == 0) {
    sizes.back() += 1;
  }
  promotedDecls_[groupId] = PromotedDecl{tensorId, sizes, kind};

  // FIXME: we can now store a unique pointer...
  auto group = std::shared_ptr<TensorReferenceGroup>(std::move(gr));
  activePromotions_.emplace_back(
      std::make_pair(activePoints, PromotionInfo{group, schedule, groupId}));
}

void Scop::promoteGroup(
    PromotedDecl::Kind kind,
    isl::id tensorId,
    std::unique_ptr<TensorReferenceGroup>&& gr,
    ScheduleTree* tree,
    isl::union_map schedule,
    bool forceLastExtentOdd) {
  auto activePoints = activeDomainPoints(scheduleRoot(), tree);
  // Allow promoting the second group the same tensor if:
  // - footprints don't overlap => copy from global
  // - footprints do overlap but
  //   - the footprint of the new group is a subset some existing group and the
  //     new promotion is deeper
  //     => copy from existing
  //   - all groups are read-only and [the footprint of the new group is not a
  //     subset of any other group OR the new promotion is not deeper]
  //     => copy from global

  auto activePromIndexes = activePromotionsIndexes(activePoints, tensorId);
  auto activeProms = promotionsAtIndexes(activePromIndexes);

  auto footprints = isl::set::empty(gr->approximateFootprint().get_space());
  auto allReadOnly = gr->isReadOnly();
  for (const auto& prom : activeProms) {
    footprints = footprints.unite(prom.second.group->approximateFootprint());
    allReadOnly = allReadOnly && prom.second.group->isReadOnly();
  }
  auto footprintsOverlap =
      !footprints.intersect(gr->approximateFootprint()).is_empty();

  if (!footprintsOverlap) {
    promoteWithCopyFromGlobal(
        activePoints,
        kind,
        tensorId,
        std::move(gr),
        tree,
        schedule,
        forceLastExtentOdd);
  } else {
    std::vector<size_t> possibleParents;
    // If the new promotion is a subset of some old promotion, and the new has
    // writes, then the old one also must have writes and must have been
    // grouped with other references reading from the same value.  If the new
    // one is read-only, and is a subset of some old promotion that has a
    // write, all other read-only promotions at the previous level must have
    // been grouped with it.  If everything is read-only, we just have multiple
    // cached copies.  Therefore, we can find the first old promotion that is a
    // superset of the new one, and copy to/from that.
    for (auto i : activePromIndexes) {
      if (gr->approximateFootprint().is_subset(
              activePromotions_[i].second.group->approximateFootprint())) {
        possibleParents.emplace_back(i);
      } else if (gr->approximateFootprint().intersect(
                     activePromotions_[i]
                         .second.group->approximateFootprint())) {
        // If the new promotion is not a subset of some other promotion, but
        // overlaps with it, can only promote if all accesses are reads (no
        // consistency problem).  Warn and return otherwise.
        if (allReadOnly) {
          // TODO: This would break the codegen invariant that only one
          // promotion is active in a statement instance for a tensor.
          // We need to "prioritize" promotions and select "faster" ones
          // in case when multiple read-only promotions are present.
#if 0
          promoteWithCopyFromGlobal(
              activePoints,
              kind,
              tensorId,
              std::move(gr),
              tree,
              schedule,
              forceLastExtentOdd);
#endif
          return;
        }
        LOG(WARNING)
            << "not performing nested promotion because the inner footprint\n"
            << gr->approximateFootprint() << "\n"
            << "overlaps with one of the outer footprints\n"
            << activePromotions_[i].second.group->approximateFootprint() << "\n"
            << "without being its subset";
        return;
      }
    }
    // This should not happen: if the footprint of the current group is not a
    // subset of some other group but overlaps with some (top-level branch
    // condition), it must have been picked up in the loop above and caused
    // early return.
    if (possibleParents.size() == 0) {
      throw promotion::PromotionLogicError(
          "group overlaps with existing groups and can't be read from global");
    }
    auto parentPromIdx = possibleParents.front();

    auto groupId = nextGroupIdForTensor(tensorId);
    insertIntraCopiesUnder(
        *this,
        tree,
        *gr,
        *activePromotions_[parentPromIdx].second.group,
        kind == PromotedDecl::Kind::SharedMem,
        tensorId,
        groupId,
        activePromotions_[parentPromIdx].second.groupId);
    promotedDecls_[groupId] =
        PromotedDecl{tensorId, gr->approximationSizes(), kind};

    for (auto i : possibleParents) {
      auto pts = projectOutNamedParam(activePoints, mapping::ThreadId::makeId(0));
      pts = projectOutNamedParam(pts, mapping::ThreadId::makeId(1));
      pts = projectOutNamedParam(pts, mapping::ThreadId::makeId(2));
      activePromotions_[i].first = activePromotions_[i].first.subtract(pts);
    }

    auto group = std::shared_ptr<TensorReferenceGroup>(std::move(gr));
    activePromotions_.emplace_back(
        std::make_pair(activePoints, PromotionInfo{group, schedule, groupId}));
  }
}

namespace {
inline bool rangeOfUMapContainsTupleId(isl::union_map umap, isl::id id) {
  for (auto s : isl::UnionAsVector<isl::union_set>(umap.range())) {
    if (s.get_tuple_id() == id) {
      return true;
    }
  }
  return false;
}

inline isl::union_map dropMapsWithRangeTupleId(
    isl::union_map umap,
    isl::id id) {
  isl::union_map result = isl::union_map::empty(umap.get_space());
  for (auto m : isl::UnionAsVector<isl::union_map>(umap)) {
    if (!m.can_uncurry()) {
      result = result.add_map(m);
      continue;
    }
    if (m.uncurry().get_tuple_id(isl::dim_type::out) != id) {
      result = result.add_map(m);
    }
  }
  return result;
}
} // namespace

void Scop::demoteGroup(isl::id groupId) {
  using namespace polyhedral::detail;

  auto extensions = match(
      extension(
          [groupId](isl::union_map m) {
            return rangeOfUMapContainsTupleId(m.range().unwrap(), groupId);
          },
          sequence(any())),
      scheduleRoot());

  CHECK_EQ(extensions.size(), 1)
      << "group " << groupId << " is not present as schedule extension.";

  auto extensionTree = const_cast<ScheduleTree*>(extensions[0]);

  auto sequenceTree = extensionTree->child({0});
  for (size_t i = sequenceTree->numChildren(); i > 0; --i) {
    auto filterElem =
        sequenceTree->child({i - 1})->elemAs<ScheduleTreeElemFilter>();
    CHECK(filterElem) << "expected children of a sequence node to be filters "
                      << "got\n"
                      << *sequenceTree;
    if (!rangeOfUMapContainsTupleId(filterElem->filter_.unwrap(), groupId)) {
      continue;
    }
    CHECK_EQ(filterElem->filter_.n_set(), 1)
        << "filter for copy code contains more than one statement";
    sequenceTree->detachChild({i - 1});
  }

  auto extensionElem = extensionTree->elemAs<ScheduleTreeElemExtension>();
  extensionElem->extension_ =
      dropMapsWithRangeTupleId(extensionElem->extension_, groupId);

  if (extensionElem->extension_.is_empty()) {
    auto parent = extensionTree->ancestor(scheduleRoot(), 1);
    auto pos = extensionTree->positionInParent(parent);
    if (sequenceTree->numChildren() > 1) {
      auto ownedSequenceTree = extensionTree->detachChildren();
      parent->detachChild(pos);
      parent->insertChildren(pos, std::move(ownedSequenceTree));
    } else {
      auto ownedChildren = sequenceTree->detachChildren();
      parent->detachChild(pos);
      parent->insertChildren(pos, std::move(ownedChildren));
    }
  }

  for (size_t i = activePromotions_.size(); i > 0; --i) {
    if (activePromotions_[i - 1].second.groupId == groupId) {
      activePromotions_.erase(activePromotions_.begin() + (i - 1));
    }
  }
  promotedDecls_.erase(groupId);
}

void Scop::insertSyncsAroundCopies(ScheduleTree* tree) {
  // Return immediately if nothing was inserted
  auto extensionNode =
      tree->child({0})->elemAs<detail::ScheduleTreeElemExtension>();
  if (!extensionNode) {
    return;
  }

  // Insert syncs before and after copies (FIXME: this is excessive)
  auto seqNode = tree->child({0, 0});
  CHECK(seqNode->elemAs<detail::ScheduleTreeElemSequence>())
      << "unexpected tree structure";

  int foundMainComputations = 0;
  for (size_t i = 0; i < seqNode->numChildren(); ++i) {
    auto filterNode =
        seqNode->child({i})->elemAs<detail::ScheduleTreeElemFilter>();
    CHECK(filterNode) << "expected filters below sequence";
    auto filters = isl::UnionAsVector<isl::union_set>(filterNode->filter_);
    bool isCopyFilter = filters.size() == 1 && filters[0].has_tuple_name() &&
        (filters[0].get_tuple_name() == kReadIdName ||
         filters[0].get_tuple_name() == kWriteIdName);
    if ((foundMainComputations != 0) ^ isCopyFilter) {
      continue;
    }
    if (!isCopyFilter) {
      ++foundMainComputations;
    }
    CHECK_LT(foundMainComputations, 2)
        << "copies are interleaved with computation" << *seqNode;
    insertSync(seqNode, i);
    ++i;
  }
  insertSync(seqNode, 0);
  insertSync(seqNode, seqNode->numChildren());
}

void Scop::promoteEverythingAt(std::vector<size_t> pos) {
  auto root = scheduleRoot();
  auto tree = scheduleRoot()->child(pos);

  checkFiltersDisjointStatements(scheduleRoot());
  auto schedule = partialSchedule(root, tree);

  auto groupMap = TensorReferenceGroup::accessedBySubtree(tree, *this);
  for (auto& p : groupMap) {
    for (auto& gr : p.second) {
      promoteGroup(
          PromotedDecl::Kind::SharedMem,
          p.first,
          std::move(gr),
          tree,
          schedule);
    }
  }
  insertSyncsAroundCopies(tree);
}

namespace {
typedef std::unordered_map<isl::id, long, isl::IslIdIslHash> IslParamValueMap;

// Extract the fixed values of the parameters from the given (context) set.
IslParamValueMap extractParamValueMap(isl::set set) {
  CHECK(set.is_singleton()) << "set must be singleton to extract fixed values";

  auto ctx = set.get_ctx();
  auto longMax = isl::val(ctx, std::numeric_limits<long>::max());
  auto p = set.sample_point();
  auto space = p.get_space();

  IslParamValueMap paramValueMap;
  int i = 0;
  for (auto id : isl::DimIds<isl::space, isl::dim_type::param>(space)) {
    auto val = p.get_coordinate_val(isl::dim_type::param, i);
    CHECK_EQ(val.get_den_si(), 1) << "fractional parameters unsupported";
    CHECK(val.le(longMax)) << "parameter value overflows long";
    paramValueMap[id] = val.get_num_si();
    ++i;
  }

  return paramValueMap;
}
} // namespace

// Compute the values of parameters based on the effective sizes of the
// tensors provided as arguments and their parametric expressions stored in
// Halide InputImage.  We only know input sizes, output sizes are inferred.
// Result is an isl set directly usable as context.
//
// TODO(ntv)
isl::set Scop::makeContextFromInputs(
    const std::vector<const DLTensor*>& inputs) const {
  CHECK_EQ(halide.inputs.size(), inputs.size());

  auto paramSpace = domain().get_space().params();
  auto paramSet = isl::set::universe(paramSpace);
  for (size_t i = 0, ei = inputs.size(); i < ei; ++i) {
    CHECK_EQ(halide.inputs[i].dimensions(), inputs[i]->ndim);
    for (size_t j = 0, ej = halide.inputs[i].dimensions(); j < ej; ++j) {
      auto parametricAff = halide2isl::makeIslAffFromExpr(
          paramSpace, halide.inputs[i].parameter().extent_constraint(j));
      paramSet =
          paramSet & (isl::aff_set(parametricAff) == inputs[i]->shape[j]);
    }
  }
  CHECK(paramSet.is_singleton()) << "could not infer the values of parameters";
  return paramSet;
}

std::vector<long> Scop::getParameterValues(isl::set context) const {
  IslParamValueMap pvm = extractParamValueMap(context);

  // Scop holds a vector of Variables, which also appear as user pointers
  // of the ids.  Iterate over parameters in order, checking if the
  // ParamValueMap contains an id whose user pointer corresponds to a
  // Variable and push respective parameter values.
  std::vector<long> paramValues;
  for (auto const& param : halide.params) {
    size_t previousSize = paramValues.size();
    for (auto p : pvm) {
      isl::id id = p.first;
      if (id.get_name() == param.name()) {
        paramValues.push_back(p.second);
      }
    }
    CHECK_EQ(previousSize + 1, paramValues.size())
        << "parameter " << param.name() << " is not present in the context "
        << context << "; mind identical names in Halide.";
  }
  return paramValues;
}

namespace {

using namespace tc::polyhedral;

isl::union_map computeDependences(
    isl::union_map sources,
    isl::union_map sinks,
    isl::schedule schedule) {
  auto uai = isl::union_access_info(sinks);
  uai = uai.set_may_source(sources);
  uai = uai.set_schedule(schedule);
  auto flow = uai.compute_flow();
  return flow.get_may_dependence();
}

// Do the simplest possible dependence analysis.
// Live-range reordering needs tagged access relations to be available.
// The domain of the constraints is intersected with "restrictDomain" if it is
// provided.
isl::schedule_constraints makeScheduleConstraints(
    const Scop& scop,
    const SchedulerOptionsView& schedulerOptions,
    isl::union_set restrictDomain = isl::union_set()) {
  auto schedule = toIslSchedule(scop.scheduleRoot());
  auto firstChildNode = scop.scheduleRoot()->child({0});
  auto reads = scop.reads.domain_factor_domain();
  auto writes = scop.writes.domain_factor_domain();

  // RAW
  auto flowDeps = computeDependences(writes, reads, schedule);
  // WAR and WAW
  auto falseDeps = computeDependences(writes.unite(reads), writes, schedule);

  auto allDeps = flowDeps.unite(falseDeps).coalesce();

  auto constraints = isl::schedule_constraints::on_domain(scop.domain())
                         .set_validity(allDeps)
                         .set_proximity(allDeps)
                         .set_coincidence(allDeps);
  if (restrictDomain) {
    constraints = constraints.intersect_domain(restrictDomain);
  }
  if (auto contextNode =
          firstChildNode->elemAs<detail::ScheduleTreeElemContext>()) {
    constraints = constraints.set_context(contextNode->context_);
  }

  // Set up "add_schedule_constraints" and "merge_callback"
  // depending on the scheduler options.
  isl_schedule_constraints* sc = constraints.release();
  {
    if (schedulerOptions.proto.positive_orthant()) {
      sc = isl_schedule_constraints_set_custom_constraint_callback(
          sc, callbacks::AddPositiveCoefficientConstraints, nullptr);
    }
  }
  {
    auto fusionStrategy = schedulerOptions.proto.fusion_strategy();
    if (fusionStrategy == FusionStrategy::Max) {
      sc = isl_schedule_constraints_set_merge_callback(
          sc, callbacks::FuseAll, nullptr);
    } else if (fusionStrategy == FusionStrategy::Preserve3Coincident) {
      sc = isl_schedule_constraints_set_merge_callback(
          sc, callbacks::FuseAllPreserve3Coincident, nullptr);
    } else if (fusionStrategy == FusionStrategy::Min) {
      sc = isl_schedule_constraints_set_merge_callback(
          sc, callbacks::FuseNone, nullptr);
    } else {
      throw std::runtime_error{"NYI: unknown fusion strategy requested"};
    }
  }
  constraints = isl::manage(sc);

  return constraints;
}
} // namespace

std::unique_ptr<detail::ScheduleTree> Scop::computeSchedule(
    isl::schedule_constraints constraints,
    const SchedulerOptionsView& schedulerOptions) {
  auto ctx = constraints.get_ctx();
  auto usedWholeComponent = isl_options_get_schedule_whole_component(ctx.get());
  auto wasSerializingSccs = isl_options_get_schedule_serialize_sccs(ctx.get());
  auto wasUnit =
      isl_options_get_schedule_unit_max_var_coefficient_sum(ctx.get());
  isl_options_set_schedule_whole_component(ctx.get(), 0);
  if (schedulerOptions.proto.fusion_strategy() == FusionStrategy::Min) {
    isl_options_set_schedule_serialize_sccs(ctx.get(), 1);
  }
  if (!schedulerOptions.proto.allow_skewing()) {
    isl_options_set_schedule_unit_max_var_coefficient_sum(ctx.get(), 1);
  }
  tc::ScopeGuard islOptionsResetter([&]() {
    isl_options_set_schedule_whole_component(ctx.get(), usedWholeComponent);
    isl_options_set_schedule_serialize_sccs(ctx.get(), wasSerializingSccs);
    isl_options_set_schedule_unit_max_var_coefficient_sum(ctx.get(), wasUnit);
  });

  return detail::fromIslSchedule(constraints.compute_schedule());
}

std::unique_ptr<Scop> Scop::makeScheduled(
    const Scop& scop,
    const SchedulerOptionsView& schedulerOptions) {
  auto s = makeScop(scop);
  auto constraints = makeScheduleConstraints(*s, schedulerOptions);
  s->scheduleTreeUPtr = computeSchedule(constraints, schedulerOptions);
  LOG_IF(INFO, FLAGS_debug_tc_mapper) << "After scheduling:" << std::endl
                                      << *s->scheduleTreeUPtr;
  return s;
}

namespace {

/*
 * Mark the band node at "tree" permutable.
 */
detail::ScheduleTree* setPermutable(detail::ScheduleTree* tree) {
  auto band = tree->elemAs<detail::ScheduleTreeElemBand>();
  CHECK(band);
  band->permutable_ = true;
  return tree;
}

/*
 * Return the outermost band in the schedule tree with the given root.
 * If there is no single outermost band, then insert a (permutable)
 * zero-dimensional band and return that.
 * In particular, if the leaf of the tree has been reached, then
 * insert the band in the leaf.  If branching is encountered, then
 * insert the band above the branching.
 */
detail::ScheduleTree* obtainOuterBand(detail::ScheduleTree* root) {
  auto tree = root;
  while (!tree->elemAs<ScheduleTreeElemBand>()) {
    auto n = tree->numChildren();
    if (n == 1) {
      tree = tree->child({0});
      continue;
    }

    auto domain = root->elemAs<ScheduleTreeElemDomain>();
    CHECK(domain);
    auto space = domain->domain_.get_space().set_from_params();
    auto zero = isl::multi_union_pw_aff::zero(space);
    if (n == 0) {
      return setPermutable(insertBandBelow(tree, zero));
    } else {
      return setPermutable(insertBandAbove(root, tree, zero));
    }
  }
  return tree;
}
} // namespace

detail::ScheduleTree* Scop::tileOuterBand(const TilingView& tileSizes) {
  using namespace tc::polyhedral::detail;
  auto band = obtainOuterBand(scheduleRoot());
  auto bandNode = band->elemAs<ScheduleTreeElemBand>();
  std::vector<size_t> sizes = tileSizes.extractVector();
  if (bandNode->nMember() < sizes.size()) {
    sizes.resize(bandNode->nMember());
  }
  auto res = bandTile(band, sizes, TileOptions::ShiftPointLoops);
  LOG_IF(INFO, FLAGS_debug_tc_mapper) << "After tiling outer:" << std::endl
                                      << *scheduleTreeUPtr;
  return res;
}

void Scop::reschedule(
    ScheduleTree* tree,
    const SchedulerOptionsView& schedulerOptions) {
  auto root = scheduleTreeUPtr.get();
  auto parentTree = tree->ancestor(root, 1);
  auto treePos = tree->positionInParent(parentTree);
  auto domain = activeDomainPoints(root, tree);
  auto prefix = prefixScheduleMupa(root, tree);

  // Restrict the constraints to domain points reachable from point loops
  // and update the current prefix.
  auto constraints = makeScheduleConstraints(*this, schedulerOptions, domain)
                         .set_prefix(prefix);
  auto newTree = computeSchedule(constraints, schedulerOptions);
  parentTree->detachChild(treePos);
  parentTree->insertChildren(treePos, newTree->detachChildren());
  LOG_IF(INFO, FLAGS_debug_tc_mapper) << "After rescheduling:" << std::endl
                                      << *scheduleTreeUPtr;
}

const Halide::OutputImageParam& Scop::findArgument(isl::id id) const {
  std::string name = id.get_name();

  for (const auto& i : halide.inputs) {
    if (i.name() == name) {
      return i;
    }
  }
  for (const auto& i : halide.outputs) {
    if (i.name() == name) {
      return i;
    }
  }

  CHECK(false) << "name \"" << name << "\" not found";
  return *halide.inputs.begin();
}

} // namespace polyhedral
} // namespace tc
