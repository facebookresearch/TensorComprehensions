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
#include "tc/core/halide2isl.h"

#include <algorithm>
#include <unordered_set>

#include "tc/core/check.h"
#include "tc/core/constants.h"
#include "tc/core/polyhedral/body.h"
#include "tc/core/polyhedral/domain_types.h"
#include "tc/core/polyhedral/schedule_isl_conversion.h"
#include "tc/core/polyhedral/schedule_transforms.h"
#include "tc/core/polyhedral/schedule_tree.h"
#include "tc/core/polyhedral/utils.h"
#include "tc/core/tc2halide.h"

namespace tc {
namespace halide2isl {

using namespace Halide;
using namespace Halide::Internal;
using namespace tc::polyhedral;
using namespace tc::polyhedral::detail;

SymbolTable makeSymbolTable(const tc2halide::HalideComponents& components) {
  // const Stmt& s) {
  // Collect and categorize all the Halide Variable symbols as reduction
  // or index variables
  class BuildSymbolTable : public IRVisitor {
    using IRVisitor::visit;
    std::set<std::string> included;
    void visit(const Variable* op) {
      if (!included.count(op->name)) {
        if (op->param.defined()) {
          // Param may exist only in the function definition and not in a
          // Halide::Stmt. Skip it here and just get parameters directly from
          // components.
        } else if (op->reduction_domain.defined()) {
          table.reductionVars.push_back(op->name);
        } else {
          table.idxVars.push_back(op->name);
        }
        included.insert(op->name);
      }
    }

   public:
    halide2isl::SymbolTable table;
  } builder;

  components.stmt.accept(&builder);
  // Get params from components.params which contain everything declared in
  // TC Def. However, the 0-D tensors are registered as both params and inputs,
  // filter those out.
  for (auto kvp : components.params) {
    bool skip = false;
    for (auto& o : components.inputs) {
      if (o.name() == kvp.second.name()) {
        skip = true;
        break;
      }
    }
    if (!skip) {
      builder.table.params.push_back(kvp.second);
    }
  }
  return builder.table;
}

isl::AffOn<> makeIslAffFromInt(isl::Space<> space, int64_t val) {
  isl::val v = isl::val(space.get_ctx(), val);
  return isl::AffOn<>(isl::aff(isl::local_space(space), v));
}

std::vector<isl::AffOn<>> makeIslAffBoundsFromExpr(
    isl::Space<> space,
    const Expr& e,
    bool allowMin,
    bool allowMax);

namespace {
/*
 * Convert Halide binary expression "op" into a list of isl affine functions by
 * converting its LHS and RHS into lists of affs and concatenating those lists.
 * This is intended to be used with Min/Max operations in upper/lower bound
 * computations, respectively.  Essentially, this allows for replacements
 *   x < min(a,min(b,c)) <=> x < a AND x < b AND x < c
 *   x > max(a,max(b,c)) <=> x > a AND x > b AND x > c
 */
template <typename T>
inline std::vector<isl::AffOn<>>
concatAffs(isl::Space<> space, T op, bool allowMin, bool allowMax) {
  std::vector<isl::AffOn<>> result;

  for (const auto& aff :
       makeIslAffBoundsFromExpr(space, op->a, allowMin, allowMax)) {
    result.push_back(aff);
  }
  for (const auto& aff :
       makeIslAffBoundsFromExpr(space, op->b, allowMin, allowMax)) {
    result.push_back(aff);
  }

  return result;
}

/*
 * Convert Halide binary expression "op" into an isl affine function by
 * converting its LHS and RHS into affs and combining them with "combine"
 * into a single expression.  LHS and RHS are expected to only produce at most
 * one expression.  If either of them produces zero expressions, meaning the
 * bound is not affine, return an empty vector.  Otherwise return a vector with
 * a single expression that is the result of applying LHS.combine(RHS).
 * This is intended for use with operations other than Min/Max that do not
 * commute nicely in bounds, for example
 *   x < a + max(b,c)  NOT <=>  x < a + b AND x < a + c for negative values.
 */
template <typename T>
inline std::vector<isl::AffOn<>> combineSingleAffs(
    isl::Space<> space,
    T op,
    isl::AffOn<> (isl::AffOn<>::*combine)(const isl::AffOn<>&) const) {
  auto left = makeIslAffBoundsFromExpr(space, op->a, false, false);
  auto right = makeIslAffBoundsFromExpr(space, op->b, false, false);
  TC_CHECK_LE(left.size(), 1u);
  TC_CHECK_LE(right.size(), 1u);

  if (left.size() == 0 || right.size() == 0) {
    return {};
  }

  return {(left[0].*combine)(right[0])};
}

} // end namespace

/*
 * Convert Halide expression into list of isl affine expressions usable for
 * defining constraints.  In particular, an expression starting with (nested)
 * Max operations can be used for lower bounds
 *   x > max(a,b) <=> x > a AND x > b
 * while an expression starting with (nested) Min operations can be used for
 * upper bounds
 *   x < min(a,b) <=> x < a AND x < b.
 * Arguments "allowMin" and "allowMax" control whether Min and Max operations,
 * respectively, are allowed to be present in the expression. Note that they
 * can only appear before any other operation and cannot appear together in an
 * expression.
 * If a Halide expression cannot be converted into a list of affine expressions,
 * return an empty list.
 */
std::vector<isl::AffOn<>> makeIslAffBoundsFromExpr(
    isl::Space<> space,
    const Expr& e,
    bool allowMin,
    bool allowMax) {
  TC_CHECK(!(allowMin && allowMax));

  using Halide::Internal::Max;
  using Halide::Internal::Min;

  const Min* minOp = e.as<Min>();
  const Max* maxOp = e.as<Max>();

  if (const Variable* op = e.as<Variable>()) {
    isl::id id(space.get_ctx(), op->name);
    if (space.has_param(id)) {
      return {isl::AffOn<>::param_on_domain_space(space, id)};
    }
    LOG(FATAL) << "Variable not found in isl::space: " << space << ": " << op
               << ": " << op->name << '\n';
    return {};
  } else if (minOp != nullptr && allowMin) {
    return concatAffs(space, minOp, allowMin, allowMax);
  } else if (maxOp != nullptr && allowMax) {
    return concatAffs(space, maxOp, allowMin, allowMax);
  } else if (const Add* op = e.as<Add>()) {
    return combineSingleAffs(space, op, &isl::AffOn<>::add);
  } else if (const Sub* op = e.as<Sub>()) {
    return combineSingleAffs(space, op, &isl::AffOn<>::sub);
  } else if (const Mul* op = e.as<Mul>()) {
    return combineSingleAffs(space, op, &isl::AffOn<>::mul);
  } else if (const Div* op = e.as<Div>()) {
    return combineSingleAffs(space, op, &isl::AffOn<>::div);
  } else if (const Mod* op = e.as<Mod>()) {
    std::vector<isl::aff> result;
    // We cannot span multiple constraints if a modulo operation is involved.
    // x > max(a,b) % C is not equivalent to (x > a % C && x > b % C).
    auto lhs = makeIslAffBoundsFromExpr(space, op->a, false, false);
    TC_CHECK_EQ(lhs.size(), 1u);
    if (const int64_t* b = as_const_int(op->b)) {
      return {lhs[0].mod(isl::val(space.get_ctx(), *b))};
    }
  } else if (const int64_t* i = as_const_int(e)) {
    return {makeIslAffFromInt(space, *i)};
  }

  return {};
}

isl::AffOn<> makeIslAffFromExpr(isl::Space<> space, const Expr& e) {
  auto list = makeIslAffBoundsFromExpr(space, e, false, false);
  TC_CHECK_LE(list.size(), 1u)
      << "Halide expr " << e << " unrolled into more than 1 isl aff"
      << " but min/max operations were disabled";

  // Non-affine
  if (list.size() == 0) {
    return isl::AffOn<>();
  }
  return list[0];
}

isl::Space<> makeParamSpace(isl::ctx ctx, const ParameterVector& params) {
  auto space = isl::Space<>(ctx, 0);
  // set parameter names
  for (auto p : params) {
    space = space.add_param(isl::id(ctx, p.name()));
  }
  return space;
}

isl::Set<> makeParamContext(isl::ctx ctx, const ParameterVector& params) {
  auto space = makeParamSpace(ctx, params);
  auto context = isl::Set<>::universe(space);
  for (auto p : params) {
    auto a(isl::AffOn<>::param_on_domain_space(space, isl::id(ctx, p.name())));
    context = context & a.asPwAff().nonneg_set();
  }
  return context;
}

namespace {

/*
 * Call the domain_map factory method of the isl::MultiAff
 * with appropriate template arguments.
 */
template <typename Domain, typename Range>
static isl::MultiAff<isl::Pair<Domain, Range>, Domain> domainMap(
    isl::Space<Domain, Range> space) {
  return isl::MultiAff<isl::Pair<Domain, Range>, Domain>::domain_map(space);
}

isl::Map<isl::Pair<Statement, Tag>, Tensor> extractAccess(
    const IterationDomain& domain,
    const IRNode* op,
    const std::string& tensor,
    const std::vector<Expr>& args,
    AccessMap* accesses) {
  // Make an isl::map representing this access. It maps from the iteration space
  // to the tensor's storage space, using the coordinates accessed.
  // First construct a set describing the accessed element
  // in terms of the parameters (including those corresponding
  // to the outer loop iterators) and then convert this set
  // into a map in terms of the iteration domain.

  auto paramSpace = domain.paramSpace;
  isl::id tensorID(paramSpace.get_ctx(), tensor);
  auto tensorTuple = constructTensorTuple(paramSpace, tensorID, args.size());
  auto tensorSpace = tensorTuple.get_space();

  // Start with a totally unconstrained set - every point in
  // the allocation could be accessed.
  auto access = isl::Set<Tensor>::universe(tensorSpace);

  auto identity =
      isl::MultiAff<Tensor, Tensor>::identity(tensorSpace.map_from_set());
  for (size_t i = 0; i < args.size(); i++) {
    // Then add one equality constraint per dimension to encode the
    // point in the allocation actually read/written for each point in
    // the iteration space. In the case of gathers or scatters, we may
    // have to leave some things unconstrained.

    // The coordinate written to in the range ...
    auto rangePoint = identity.get_aff(i);
    // ... equals the coordinate accessed as a function of the parameters.
    auto paramPoint = halide2isl::makeIslAffFromExpr(paramSpace, args[i]);
    if (!paramPoint.is_null()) {
      auto domainPoint = paramPoint.unbind_params_insert_domain(tensorTuple);
      access = access.intersect(domainPoint.eq_set(rangePoint));
    }
  }

  // Now convert the set into a relation with respect to the iteration domain.
  auto map = access.unbind_params_insert_domain(domain.tuple);

  // Add a tag to the domain space so that we can maintain a mapping
  // between each access in the IR and the reads/writes maps.
  std::string tag = "__tc_ref_" + std::to_string(accesses->size());
  isl::id tagID(domain.paramSpace.get_ctx(), tag);
  accesses->emplace(op, tagID);
  auto domainSpace = map.get_space().domain();
  auto tagSpace = domainSpace.params().add_named_tuple_id_ui<Tag>(tagID, 0);
  auto taggedSpace = domainSpace.product(tagSpace).unwrap();
  return map.preimage_domain(domainMap(taggedSpace));
}

std::pair<
    isl::UnionMap<isl::Pair<Statement, Tag>, Tensor>,
    isl::UnionMap<isl::Pair<Statement, Tag>, Tensor>>
extractAccesses(
    const IterationDomain& domain,
    const Stmt& s,
    AccessMap* accesses) {
  class FindAccesses : public IRGraphVisitor {
    using IRGraphVisitor::visit;

    void visit(const Call* op) override {
      IRGraphVisitor::visit(op);
      if (op->call_type == Call::Halide || op->call_type == Call::Image) {
        reads = reads.unite(
            extractAccess(domain, op, op->name, op->args, accesses));
      }
    }

    void visit(const Provide* op) override {
      IRGraphVisitor::visit(op);
      writes =
          writes.unite(extractAccess(domain, op, op->name, op->args, accesses));
    }

    const IterationDomain& domain;
    AccessMap* accesses;

   public:
    isl::UnionMap<isl::Pair<Statement, Tag>, Tensor> reads, writes;

    FindAccesses(const IterationDomain& domain, AccessMap* accesses)
        : domain(domain),
          accesses(accesses),
          reads(isl::union_map::empty(domain.tuple.get_space())),
          writes(isl::union_map::empty(domain.tuple.get_space())) {}
  } finder(domain, accesses);
  s.accept(&finder);
  return {finder.reads, finder.writes};
}

bool isReductionUpdate(const Provide* op) {
  if (const Call* call = op->values[0].as<Call>()) {
    return call->is_intrinsic(tc2halide::kReductionUpdate);
  } else {
    return false;
  }
}

/* Construct a multi-dimensional affine function mapping
 * the given iteration domain
 * to the outer loop iterators that do not appear in "skip".
 * "id" is used as the identifier of the target space.
 * For each of these outer loop iterators, an affine function
 * is first constructed in terms of the parameter space
 * active at the point where the iteration domain was created and
 * then converted into an expression on that iteration domain
 * by reinterpreting the parameters as input dimensions.
 */
template <typename Other>
static isl::MultiAff<Statement, Other> mapToOther(
    const IterationDomain& iterationDomain,
    std::unordered_set<std::string> skip,
    isl::id id) {
  auto ctx = iterationDomain.tuple.get_ctx();
  auto list = isl::AffListOn<Statement>(ctx, 0);
  for (auto id : iterationDomain.tuple.get_id_list()) {
    if (skip.count(id.get_name()) == 1) {
      continue;
    }
    auto aff =
        isl::AffOn<>::param_on_domain_space(iterationDomain.paramSpace, id);
    list = list.add(aff.unbind_params_insert_domain(iterationDomain.tuple));
  }
  auto domainSpace = iterationDomain.tuple.get_space();
  auto space =
      domainSpace.params().add_named_tuple_id_ui<Other>(id, list.size());
  auto productSpace = domainSpace.product(space).unwrap();
  return isl::MultiAff<Statement, Other>(productSpace, list);
}

/*
 * If "op" performs a reduction, then return a mapping from
 * the statement instances to the individual reductions.
 * Otherwise, return an empty isl::union_map.
 *
 * "op" is considered to be a reduction if it has been marked
 * as performing a reduction and if more than one statement instance
 * is involved in the individual reductions.
 *
 * The space of the reduction has a name of the form R_<op->name>_<index>.
 * Each reduction is indexed by the outer loop variables
 * that are not marked as reduction variables.
 * Since the loop variables that iterate over output tensor elements
 * are never marked as reduction variables, this means in particular
 * that all statement instances that belong to the same reduction
 * write to the same tensor element.
 */
isl::UnionMap<Statement, Reduction> extractReduction(
    const IterationDomain& iterationDomain,
    const Provide* op,
    size_t index) {
  class FindReductionVars : public IRVisitor {
    void visit(const Variable* op) {
      if (op->reduction_domain.defined()) {
        reductionVars.insert(op->name);
      }
    }

   public:
    // The variables that are known to be reduction variables.
    std::unordered_set<std::string> reductionVars;
  } finder;

  if (!isReductionUpdate(op)) {
    auto space = iterationDomain.tuple.get_space().params();
    return isl::UnionMap<Statement, Reduction>::empty(space);
  }
  op->accept(&finder);
  if (finder.reductionVars.size() == 0) {
    auto space = iterationDomain.tuple.get_space().params();
    return isl::UnionMap<Statement, Reduction>(isl::union_map::empty(space));
  }
  auto ctx = iterationDomain.tuple.get_ctx();
  isl::id id(ctx, kReductionLabel + op->name + "_" + std::to_string(index));
  auto reduction =
      mapToOther<Reduction>(iterationDomain, finder.reductionVars, id);
  return reduction.asMap().asUnionMap();
}

/*
 * Take a parametric expression "f" and convert it into an expression
 * on the iteration domains in "domain" by reinterpreting the parameters
 * as set dimensions according to the corresponding tuples in "map".
 */
isl::union_pw_aff
onDomains(isl::aff f, isl::union_set domain, const IterationDomainMap& map) {
  auto upa = isl::union_pw_aff::empty(domain.get_space());
  for (auto set : domain.get_set_list()) {
    auto tuple = map.at(set.get_tuple_id()).tuple;
    auto onSet = isl::union_pw_aff(f.unbind_params_insert_domain(tuple));
    upa = upa.union_add(onSet);
  }
  return upa;
}

} // namespace

/*
 * Helper function for extracting a schedule from a Halide Stmt,
 * recursively descending over the Stmt.
 * "s" is the current position in the recursive descent.
 * "set" describes the bounds on the outer loop iterators.
 * "outer" contains the identifiers of the outer loop iterators
 * from outermost to innermost.
 * Return the schedule corresponding to the subtree at "s".
 *
 * "body" collects the accesses and reductions found along the way.
 * "accesses" collects the mapping from Call (for the reads) and Provide nodes
 * (for the writes) to the corresponding tag in the access relations.
 * "statements" collects the mapping from instance set tuple identifiers
 * to the corresponding Provide node.
 * "domains" collects the mapping from instance set tuple identifiers
 * to the corresponding iteration domain information.
 */
isl::schedule makeScheduleTreeHelper(
    const Stmt& s,
    isl::Set<> set,
    isl::id_list outer,
    Body* body,
    AccessMap* accesses,
    StatementMap* statements,
    IterationDomainMap* domains) {
  isl::schedule schedule;
  if (auto op = s.as<For>()) {
    // Make an id for this loop var.  It starts out as a parameter.
    isl::id id(set.get_ctx(), op->name);
    auto space = set.get_space().add_param(id);

    // Construct a variable (affine function) that references
    // the new parameter.
    auto loopVar = isl::AffOn<>::param_on_domain_space(space, id);

    // Then we add our new loop bound constraints.
    auto lbs =
        halide2isl::makeIslAffBoundsFromExpr(space, op->min, false, true);
    TC_CHECK_GT(lbs.size(), 0u)
        << "could not obtain polyhedral lower bounds from " << op->min;
    for (auto lb : lbs) {
      set = set.intersect(loopVar.ge_set(lb));
    }

    Expr max = simplify(op->min + op->extent - 1);
    auto ubs = halide2isl::makeIslAffBoundsFromExpr(space, max, true, false);
    TC_CHECK_GT(ubs.size(), 0u)
        << "could not obtain polyhedral upper bounds from " << max;
    for (auto ub : ubs) {
      set = set.intersect(ub.ge_set(loopVar));
    }

    // Recursively descend.
    auto outerNext = outer.add(isl::id(set.get_ctx(), op->name));
    auto bodySchedule = makeScheduleTreeHelper(
        op->body, set, outerNext, body, accesses, statements, domains);

    // Create an affine function that defines an ordering for all
    // the statements in the body of this loop over the values of
    // this loop.  Start from a parametric expression equal
    // to the current loop iterator and then convert it to
    // a function on the statements in the domain of the body schedule.
    auto aff = isl::aff::param_on_domain_space(space, id);
    auto domain = bodySchedule.get_domain();
    auto mupa = isl::multi_union_pw_aff(onDomains(aff, domain, *domains));

    schedule = bodySchedule.insert_partial_schedule(mupa);
  } else if (auto op = s.as<Halide::Internal::Block>()) {
    std::vector<Stmt> stmts;
    stmts.push_back(op->first);
    stmts.push_back(op->rest);

    // Build a schedule tree for both members of the block and
    // combine them in a sequence.
    std::vector<isl::schedule> schedules;
    for (Stmt stmt : stmts) {
      schedules.push_back(makeScheduleTreeHelper(
          stmt, set, outer, body, accesses, statements, domains));
    }
    schedule = schedules[0].sequence(schedules[1]);

  } else if (auto op = s.as<Provide>()) {
    // Make an ID for this leaf statement. This *is* semantically
    // meaningful - it is used as a key to identify the provide
    // node.
    size_t stmtIndex = statements->size();
    isl::id id(set.get_ctx(), kStatementLabel + std::to_string(stmtIndex));
    statements->emplace(id, op);
    auto space = isl::Space<>(set.get_ctx(), 0);
    auto tupleSpace = space.add_named_tuple_id_ui<Statement>(id, outer.size());
    IterationDomain iterationDomain;
    iterationDomain.paramSpace = set.get_space();
    iterationDomain.tuple = isl::MultiId<Statement>(tupleSpace, outer);
    domains->emplace(id, iterationDomain);
    auto domain = set.unbind_params(iterationDomain.tuple);
    schedule = isl::schedule::from_domain(domain);

    isl::UnionMap<isl::Pair<Statement, Tag>, Tensor> newReads, newWrites;
    std::tie(newReads, newWrites) =
        extractAccesses(iterationDomain, op, accesses);
    // A tensor may be involved in multiple reductions.
    // Use the statement index to differentiate between them.
    auto newReduction = extractReduction(iterationDomain, op, stmtIndex);

    body->reads = body->reads.unite(newReads);
    body->writes = body->writes.unite(newWrites);
    body->reductions = body->reductions.unite(newReduction);

  } else {
    LOG(FATAL) << "Unhandled Halide stmt: " << s;
  }
  return schedule;
};

ScheduleTreeAndAccesses makeScheduleTree(
    isl::Space<> paramSpace,
    const Stmt& s) {
  ScheduleTreeAndAccesses result;

  Body body(paramSpace);

  // Walk the IR building a schedule tree
  isl::id_list outer(paramSpace.get_ctx(), 0);
  auto schedule = makeScheduleTreeHelper(
      s,
      isl::Set<>::universe(paramSpace),
      outer,
      &body,
      &result.accesses,
      &result.statements,
      &result.domains);

  result.body = body;
  result.tree = fromIslSchedule(schedule);

  return result;
}

} // namespace halide2isl
} // namespace tc
