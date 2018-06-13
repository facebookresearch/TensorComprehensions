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

#include "tc/core/check.h"
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
  TC_CHECK(components.stmt.defined());

  halide2isl::SymbolTable sym = halide2isl::makeSymbolTable(components);

  isl::space paramSpace = halide2isl::makeParamSpace(ctx, sym.params);

  ScopUPtr scop(new Scop());
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

  return scop;
}

ScopUPtr Scop::makeScop(isl::ctx ctx, const string& tc) {
  return makeScop(ctx, tc2halide::translate(ctx, tc));
}

ScopUPtr Scop::makeScop(isl::ctx ctx, const lang::TreeRef& treeRef) {
  return makeScop(ctx, tc2halide::translate(ctx, treeRef));
}

isl::union_set& Scop::domainRef() {
  auto dom = scheduleRoot()->elemAs<ScheduleTreeElemDomain>();
  TC_CHECK(dom) << "root is not a domain in: " << *scheduleRoot();
  // TODO: activate this when the invariant has a chance of working (i.e. we
  // don't use a Context node for specifying parameter values that iterate in
  // spacetime).
  // TODO: find a proper place for the invariant.
  // auto noCont =
  //   scheduleRoot()->child({0})->elemAs<ScheduleTreeElemContext>();
  // TC_CHECK(!noCont) << "root is not a domain in: " << *scheduleRoot();
  return dom->domain_;
}

const isl::union_set Scop::domain() const {
  return const_cast<Scop*>(this)->domainRef();
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
  os << "}, ";
  os << "reductionIdx: { ";
  for (auto i : s.halide.reductionIdx) {
    os << i << ", ";
  }
  os << "}, ";
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

namespace {

void checkFiltersDisjointStatements(const ScheduleTree* root) {
  for (auto node :
       ScheduleTree::collect(root, detail::ScheduleTreeType::Sequence)) {
    isl::union_set alreadyVisitedStmts;
    for (auto child : node->children()) {
      auto filterNode = child->elemAs<ScheduleTreeElemFilter>();
      TC_CHECK(filterNode) << "expected children of sequence to be filters";
      auto filter = filterNode->filter_.universe();
      if (!alreadyVisitedStmts.get()) {
        alreadyVisitedStmts = filter;
      } else {
        // This may break if we implement recomputation or index-set splitting.
        // In these cases, promotion no longer applies to the entire statement,
        // but only to a part of it.  Possible solution -- introduce "scope"
        // mark nodes into the schedule tree that will contain information
        // about the promotion and process these marks when generating the AST.
        TC_CHECK(alreadyVisitedStmts.intersect(filter).is_empty())
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
  insertCopiesUnder(
      *this,
      tree,
      *gr,
      tensorId,
      groupId,
      kind == PromotedDecl::Kind::Register);

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
  TC_CHECK(seqNode->elemAs<detail::ScheduleTreeElemSequence>())
      << "unexpected tree structure";

  int foundMainComputations = 0;
  std::string lastTupleName = "";
  for (size_t i = 0; i < seqNode->numChildren(); ++i) {
    auto filterNode =
        seqNode->child({i})->elemAs<detail::ScheduleTreeElemFilter>();
    TC_CHECK(filterNode) << "expected filters below sequence";
    auto filters = isl::UnionAsVector<isl::union_set>(filterNode->filter_);
    bool isCopyFilter = filters.size() == 1 && filters[0].has_tuple_name() &&
        (filters[0].get_tuple_name() == kReadIdName ||
         filters[0].get_tuple_name() == kWriteIdName);
    if ((foundMainComputations != 0) ^ isCopyFilter) {
      lastTupleName = "";
      continue;
    }
    if (!isCopyFilter) {
      ++foundMainComputations;
    }
    TC_CHECK_LT(foundMainComputations, 2)
        << "copies are interleaved with computation" << *seqNode;
    if (filters[0].get_tuple_name() != lastTupleName) {
      lastTupleName = filters[0].get_tuple_name();
      insertSync(seqNode, i);
      ++i;
    }
  }
  insertSync(seqNode, 0);
  insertSync(seqNode, seqNode->numChildren());
}

void Scop::promoteEverythingAt(std::vector<size_t> pos) {
  auto root = scheduleRoot();
  auto tree = scheduleRoot()->child(pos);

  checkFiltersDisjointStatements(scheduleRoot());
  auto schedule = partialSchedule(root, tree);

  auto groupMap = TensorReferenceGroup::accessedWithin(schedule, reads, writes);
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

std::vector<long> Scop::getParameterValues() const {
  // Scop holds a vector of Variables.
  // Iterate over parameters in order, checking if the
  // map of known parameter values contains a parameter
  // whose name corresponds to that
  // Variable and push respective parameter values.
  std::vector<long> paramValues;
  for (auto const& param : halide.params) {
    auto name = param.name();
    TC_CHECK(parameterValues.count(name) == 1);
    paramValues.push_back(parameterValues.at(name));
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

// The domain of the constraints is intersected with "restrictDomain" if it is
// provided.
isl::schedule_constraints makeScheduleConstraints(
    const Scop& scop,
    const SchedulerOptionsView& schedulerOptions,
    isl::union_set restrictDomain = isl::union_set()) {
  auto constraints = isl::schedule_constraints::on_domain(scop.domain())
                         .set_validity(scop.dependences)
                         .set_proximity(scop.dependences)
                         .set_coincidence(scop.dependences);
  if (restrictDomain) {
    constraints = constraints.intersect_domain(restrictDomain);
  }
  auto root = scop.scheduleRoot();
  if (root->numChildren() > 0) {
    if (auto contextNode =
            root->child({0})->elemAs<detail::ScheduleTreeElemContext>()) {
      constraints = constraints.set_context(contextNode->context_);
    }
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

void Scop::computeAllDependences() {
  auto schedule = toIslSchedule(scheduleRoot());
  auto allReads = reads.domain_factor_domain();
  auto allWrites = writes.domain_factor_domain();
  // RAW
  auto flowDeps = computeDependences(allWrites, allReads, schedule);
  // WAR and WAW
  auto falseDeps =
      computeDependences(allWrites.unite(allReads), allWrites, schedule);

  dependences = flowDeps.unite(falseDeps).coalesce();
}

isl::union_map Scop::activeDependences(detail::ScheduleTree* tree) {
  auto prefix = prefixScheduleMupa(scheduleRoot(), tree);
  auto domain = activeDomainPoints(scheduleRoot(), tree);
  auto active = dependences;
  active = active.intersect_domain(domain);
  active = active.intersect_range(domain);
  active = active.eq_at(prefix);
  return active;
}

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
  if (not s->dependences) {
    s->computeAllDependences();
  }
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
  TC_CHECK(band);
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

  TC_CHECK(false) << "name \"" << name << "\" not found";
  return *halide.inputs.begin();
}

isl::aff Scop::makeIslAffFromStmtExpr(
    isl::id stmtId,
    isl::space paramSpace,
    const Halide::Expr& e) const {
  auto ctx = stmtId.get_ctx();
  auto iterators = halide.iterators.at(stmtId);
  auto space = paramSpace.named_set_from_params_id(stmtId, iterators.size());
  // Set the names of the set dimensions of "space" for use
  // by halide2isl::makeIslAffFromExpr.
  for (size_t i = 0; i < iterators.size(); ++i) {
    isl::id id(ctx, iterators[i]);
    space = space.set_dim_id(isl::dim_type::set, i, id);
  }
  return halide2isl::makeIslAffFromExpr(space, e);
}

} // namespace polyhedral
} // namespace tc
