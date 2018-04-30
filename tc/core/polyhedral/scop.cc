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

  auto globalParameterContext = halide2isl::makeParamContext(ctx, sym);
  isl::space paramSpace = globalParameterContext.get_space();

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
  scop->halide.iterators = std::move(tree.iterators);

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

void Scop::promoteGroup(
    PromotedDecl::Kind kind,
    isl::id tensorId,
    std::unique_ptr<TensorReferenceGroup>&& gr,
    ScheduleTree* tree,
    isl::union_map schedule,
    bool forceLastExtentOdd) {
  auto activePoints = activeDomainPoints(scheduleRoot(), tree);

  for (const auto& kvp : activePromotions_) {
    if (kvp.first.intersect(activePoints).is_empty()) {
      continue;
    }

    auto groupId = kvp.second.groupId;
    if (promotedDecls_.count(groupId) != 0 &&
        promotedDecls_[groupId].tensorId == tensorId) {
      // FIXME: allow double promotion if copies are inserted properly,
      // in particular if the new promotion is strictly smaller in scope
      // and size than the existing ones (otherwise we would need to find
      // the all the existing ones and change their copy relations).
      return;
    }
  }

  auto groupId = nextGroupIdForTensor(tensorId);
  auto sizes = gr->approximationSizes();
  if (sizes.size() > 0 && forceLastExtentOdd && (sizes.back() % 2) == 0) {
    sizes.back() += 1;
  }
  promotedDecls_[groupId] = PromotedDecl{tensorId, sizes, kind};
  insertCopiesUnder(*this, tree, *gr, tensorId, groupId);

  // FIXME: we can now store a unique pointer...
  auto group = std::shared_ptr<TensorReferenceGroup>(std::move(gr));
  activePromotions_.emplace_back(
      std::make_pair(activePoints, PromotionInfo{group, schedule, groupId}));
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

// Compute the values of parameters based on the effective sizes of the
// tensors provided as arguments and their parametric expressions stored in
// Halide InputImage.  We only know input sizes, output sizes are inferred.
// Result is an isl set directly usable as context.
//
// TODO(ntv)
isl::set Scop::makeContextFromInputs(
    const std::vector<const DLConstTensor*>& inputs) const {
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
  CHECK(paramSet.is_equal(paramSet.sample()))
      << "could not infer the values of parameters";
  return paramSet;
}

std::vector<long> Scop::getParameterValues(isl::set context) const {
  auto ctx = context.get_ctx();
  auto longMax = isl::val(ctx, std::numeric_limits<long>::max());
  auto space = context.get_space();
  auto p = context.sample_point();
  CHECK(context.is_equal(p));

  // Scop holds a vector of Variables.
  // Iterate over parameters in order, checking if the
  // context contains a parameter whose name corresponds to that
  // Variable and push respective parameter values.
  std::vector<long> paramValues;
  for (auto const& param : halide.params) {
    isl::id id(ctx, param.name());
    CHECK(context.involves_param(id));
    auto val = isl::aff::param_on_domain_space(space, id).eval(p);
    CHECK(val.is_int()) << "fractional parameters unsupported";
    CHECK(val.le(longMax)) << "parameter value overflows long";
    paramValues.push_back(val.get_num_si());
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

    auto band = ScheduleTree::makeEmptyBand(root);
    if (n == 0) {
      return setPermutable(insertNodeBelow(tree, std::move(band)));
    } else {
      return setPermutable(insertNodeAbove(root, tree, std::move(band)));
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

isl::aff Scop::makeIslAffFromStmtExpr(
    isl::id stmtId,
    isl::space paramSpace,
    const Halide::Expr& e) const {
  auto ctx = stmtId.get_ctx();
  auto iterators = halide.iterators.at(stmtId);
  auto space = paramSpace.set_from_params();
  space = space.add_dims(isl::dim_type::set, iterators.size());
  // Set the names of the set dimensions of "space" for use
  // by halide2isl::makeIslAffFromExpr.
  for (size_t i = 0; i < iterators.size(); ++i) {
    isl::id id(ctx, iterators[i]);
    space = space.set_dim_id(isl::dim_type::set, i, id);
  }
  space = space.set_tuple_id(isl::dim_type::set, stmtId);
  return halide2isl::makeIslAffFromExpr(space, e);
}

} // namespace polyhedral
} // namespace tc
