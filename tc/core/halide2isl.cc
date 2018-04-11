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
#include "tc/core/polyhedral/schedule_isl_conversion.h"
#include "tc/core/polyhedral/schedule_transforms.h"
#include "tc/core/polyhedral/schedule_tree.h"
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

isl::aff makeIslAffFromInt(isl::space space, int64_t val) {
  isl::val v = isl::val(space.get_ctx(), val);
  return isl::aff(isl::local_space(space), v);
}

std::vector<isl::aff> makeIslAffBoundsFromExpr(
    isl::space space,
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
inline std::vector<isl::aff>
concatAffs(isl::space space, T op, bool allowMin, bool allowMax) {
  std::vector<isl::aff> result;

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
inline std::vector<isl::aff> combineSingleAffs(
    isl::space space,
    T op,
    isl::aff (isl::aff::*combine)(isl::aff) const) {
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
std::vector<isl::aff> makeIslAffBoundsFromExpr(
    isl::space space,
    const Expr& e,
    bool allowMin,
    bool allowMax) {
  TC_CHECK(!(allowMin && allowMax));

  using Halide::Internal::Max;
  using Halide::Internal::Min;

  const Min* minOp = e.as<Min>();
  const Max* maxOp = e.as<Max>();

  if (const Variable* op = e.as<Variable>()) {
    isl::local_space ls = isl::local_space(space);
    int pos = space.find_dim_by_name(isl::dim_type::param, op->name);
    if (pos >= 0) {
      return {isl::aff(ls, isl::dim_type::param, pos)};
    } else {
      // FIXME: thou shalt not rely upon set dimension names
      pos = space.find_dim_by_name(isl::dim_type::set, op->name);
      if (pos >= 0) {
        return {isl::aff(ls, isl::dim_type::set, pos)};
      }
    }
    LOG(FATAL) << "Variable not found in isl::space: " << space << ": " << op
               << ": " << op->name << '\n';
    return {};
  } else if (minOp != nullptr && allowMin) {
    return concatAffs(space, minOp, allowMin, allowMax);
  } else if (maxOp != nullptr && allowMax) {
    return concatAffs(space, maxOp, allowMin, allowMax);
  } else if (const Add* op = e.as<Add>()) {
    return combineSingleAffs(space, op, &isl::aff::add);
  } else if (const Sub* op = e.as<Sub>()) {
    return combineSingleAffs(space, op, &isl::aff::sub);
  } else if (const Mul* op = e.as<Mul>()) {
    return combineSingleAffs(space, op, &isl::aff::mul);
  } else if (const Div* op = e.as<Div>()) {
    return combineSingleAffs(space, op, &isl::aff::div);
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

isl::aff makeIslAffFromExpr(isl::space space, const Expr& e) {
  auto list = makeIslAffBoundsFromExpr(space, e, false, false);
  TC_CHECK_LE(list.size(), 1u)
      << "Halide expr " << e << " unrolled into more than 1 isl aff"
      << " but min/max operations were disabled";

  // Non-affine
  if (list.size() == 0) {
    return isl::aff();
  }
  return list[0];
}

isl::space makeParamSpace(isl::ctx ctx, const ParameterVector& params) {
  auto space = isl::space(ctx, 0);
  // set parameter names
  for (auto p : params) {
    space = space.add_param(isl::id(ctx, p.name()));
  }
  return space;
}

isl::set makeParamContext(isl::ctx ctx, const ParameterVector& params) {
  auto space = makeParamSpace(ctx, params);
  auto context = isl::set::universe(space);
  for (auto p : params) {
    isl::aff a(isl::aff::param_on_domain_space(space, isl::id(ctx, p.name())));
    context = context & (a >= 0);
  }
  return context;
}

namespace {

isl::map extractAccess(
    isl::set domain,
    const IRNode* op,
    const std::string& tensor,
    const std::vector<Expr>& args,
    AccessMap* accesses) {
  // Make an isl::map representing this access. It maps from the iteration space
  // to the tensor's storage space, using the coordinates accessed.

  isl::space domainSpace = domain.get_space();
  isl::space paramSpace = domainSpace.params();
  isl::id tensorID(paramSpace.get_ctx(), tensor);
  auto rangeSpace = paramSpace.named_set_from_params_id(tensorID, args.size());

  // Add a tag to the domain space so that we can maintain a mapping
  // between each access in the IR and the reads/writes maps.
  std::string tag = "__tc_ref_" + std::to_string(accesses->size());
  isl::id tagID(domain.get_ctx(), tag);
  accesses->emplace(op, tagID);
  isl::space tagSpace = paramSpace.named_set_from_params_id(tagID, 0);
  domainSpace = domainSpace.product(tagSpace);

  // Start with a totally unconstrained relation - every point in
  // the iteration domain could write to every point in the allocation.
  isl::map map =
      isl::map::universe(domainSpace.map_from_domain_and_range(rangeSpace));

  for (size_t i = 0; i < args.size(); i++) {
    // Then add one equality constraint per dimension to encode the
    // point in the allocation actually read/written for each point in
    // the iteration space. In the case of gathers or scatters, we may
    // have to leave some things unconstrained.

    // The coordinate written to in the range ...
    auto rangePoint =
        isl::pw_aff(isl::local_space(rangeSpace), isl::dim_type::set, i);
    // ... equals the coordinate accessed as a function of the domain.
    auto domainPoint = halide2isl::makeIslAffFromExpr(domainSpace, args[i]);
    if (!domainPoint.is_null()) {
      map = map.intersect(isl::pw_aff(domainPoint).eq_map(rangePoint));
    }
  }

  return map;
}

std::pair<isl::union_map, isl::union_map>
extractAccesses(isl::set domain, const Stmt& s, AccessMap* accesses) {
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

    const isl::set& domain;
    AccessMap* accesses;

   public:
    isl::union_map reads, writes;

    FindAccesses(const isl::set& domain, AccessMap* accesses)
        : domain(domain),
          accesses(accesses),
          reads(isl::union_map::empty(domain.get_space())),
          writes(isl::union_map::empty(domain.get_space())) {}
  } finder(domain, accesses);
  s.accept(&finder);
  return {finder.reads, finder.writes};
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
 * "reads" and "writes" collect the accesses found along the way.
 * "accesses" collects the mapping from Call (for the reads) and Provide nodes
 * (for the writes) to the corresponding tag in the access relations.
 * "statements" collects the mapping from instance set tuple identifiers
 * to the corresponding Provide node.
 * "domains" collects the mapping from instance set tuple identifiers
 * to the corresponding iteration domain information.
 */
isl::schedule makeScheduleTreeHelper(
    const Stmt& s,
    isl::set set,
    isl::id_list outer,
    isl::union_map* reads,
    isl::union_map* writes,
    AccessMap* accesses,
    StatementMap* statements,
    IterationDomainMap* domains) {
  isl::schedule schedule;
  if (auto op = s.as<For>()) {
    // Add one additional dimension to our set of loop variables
    int thisLoopIdx = set.dim(isl::dim_type::set);
    set = set.add_dims(isl::dim_type::set, 1);

    // Make an id for this loop var. For set dimensions this is
    // really just for pretty-printing.
    isl::id id(set.get_ctx(), op->name);
    set = set.set_dim_id(isl::dim_type::set, thisLoopIdx, id);

    // Construct a variable (affine function) that indexes the new dimension of
    // this space.
    isl::aff loopVar(
        isl::local_space(set.get_space()), isl::dim_type::set, thisLoopIdx);

    // Then we add our new loop bound constraints.
    auto lbs = halide2isl::makeIslAffBoundsFromExpr(
        set.get_space(), op->min, false, true);
    TC_CHECK_GT(lbs.size(), 0u)
        << "could not obtain polyhedral lower bounds from " << op->min;
    for (auto lb : lbs) {
      set = set.intersect(loopVar.ge_set(lb));
    }

    Expr max = simplify(op->min + op->extent - 1);
    auto ubs =
        halide2isl::makeIslAffBoundsFromExpr(set.get_space(), max, true, false);
    TC_CHECK_GT(ubs.size(), 0u)
        << "could not obtain polyhedral upper bounds from " << max;
    for (auto ub : ubs) {
      set = set.intersect(ub.ge_set(loopVar));
    }

    // Recursively descend.
    auto outerNext = outer.add(isl::id(set.get_ctx(), op->name));
    auto body = makeScheduleTreeHelper(
        op->body, set, outerNext, reads, writes, accesses, statements, domains);

    // Create an affine function that defines an ordering for all
    // the statements in the body of this loop over the values of
    // this loop. For each statement in the children we want the
    // function that maps everything in its space to this
    // dimension. The spaces may be different, but they'll all have
    // this loop var at the same index.
    isl::multi_union_pw_aff mupa;
    body.get_domain().foreach_set([&](isl::set s) {
      isl::aff newLoopVar(
          isl::local_space(s.get_space()), isl::dim_type::set, thisLoopIdx);
      if (mupa) {
        mupa = mupa.union_add(isl::union_pw_aff(isl::pw_aff(newLoopVar)));
      } else {
        mupa = isl::union_pw_aff(isl::pw_aff(newLoopVar));
      }
    });

    schedule = body.insert_partial_schedule(mupa);
  } else if (auto op = s.as<Halide::Internal::Block>()) {
    std::vector<Stmt> stmts;
    stmts.push_back(op->first);
    stmts.push_back(op->rest);

    // Build a schedule tree for both members of the block and
    // combine them in a sequence.
    std::vector<isl::schedule> schedules;
    for (Stmt stmt : stmts) {
      schedules.push_back(makeScheduleTreeHelper(
          stmt, set, outer, reads, writes, accesses, statements, domains));
    }
    schedule = schedules[0].sequence(schedules[1]);

  } else if (auto op = s.as<Provide>()) {
    // Make an ID for this leaf statement. This *is* semantically
    // meaningful - it is used as a key to identify the provide
    // node.
    size_t stmtIndex = statements->size();
    isl::id id(set.get_ctx(), kStatementLabel + std::to_string(stmtIndex));
    statements->emplace(id, op);
    auto tupleSpace = isl::space(set.get_ctx(), 0);
    tupleSpace = tupleSpace.named_set_from_params_id(id, outer.n());
    IterationDomain iterationDomain;
    iterationDomain.tuple = isl::multi_id(tupleSpace, outer);
    domains->emplace(id, iterationDomain);
    isl::set domain = set.set_tuple_id(id);
    schedule = isl::schedule::from_domain(domain);

    isl::union_map newReads, newWrites;
    std::tie(newReads, newWrites) = extractAccesses(domain, op, accesses);

    *reads = reads->unite(newReads);
    *writes = writes->unite(newWrites);

  } else {
    LOG(FATAL) << "Unhandled Halide stmt: " << s;
  }
  return schedule;
};

ScheduleTreeAndAccesses makeScheduleTree(isl::space paramSpace, const Stmt& s) {
  ScheduleTreeAndAccesses result;

  result.writes = result.reads = isl::union_map::empty(paramSpace);

  // Walk the IR building a schedule tree
  isl::id_list outer(paramSpace.get_ctx(), 0);
  auto schedule = makeScheduleTreeHelper(
      s,
      isl::set::universe(paramSpace),
      outer,
      &result.reads,
      &result.writes,
      &result.accesses,
      &result.statements,
      &result.domains);

  result.tree = fromIslSchedule(schedule);

  return result;
}

std::vector<Reduction> findReductions(const Stmt& s) {
  class FindReductions : public IRVisitor {
    using IRVisitor::visit;

    bool isReductionUpdate(const Provide* op) {
      if (const Call* call = op->values[0].as<Call>()) {
        return call->is_intrinsic(tc2halide::kReductionUpdate);
      } else {
        return false;
      }
    }

    // Keep track of any reduction variable name for use in visit(Provide*)
    void visit(const Variable* op) {
      if (op->reduction_domain.defined()) {
        reductionVars.insert(op->name);
      }
    }

    // Keep track of the names of the outer For nodes.
    void visit(const For* op) {
      vars.push_back(op->name);
      IRVisitor::visit(op);
      vars.pop_back();
    }

    // Check if the node is an update node with at least one reduction
    // dimension, keeping track of the information about the reduction.
    // In particular, collect the positions of the reduction
    // dimensions in the update statement domain.
    // Visit the children first to ensure that all relevant
    // reduction variables have been found first.
    void visit(const Provide* op) {
      IRVisitor::visit(op);
      if (isReductionUpdate(op)) {
        std::vector<size_t> dims;
        auto n = vars.size();
        for (size_t i = 0; i < n; ++i) {
          if (reductionVars.count(vars[i]) != 0) {
            dims.emplace_back(i);
          }
        }
        if (dims.size() > 0) {
          Reduction p;
          p.update = op;
          p.dims = dims;
          reductions.emplace_back(p);
        }
      }
    }

   public:
    // The variables that are known to be reduction variables.
    std::unordered_set<std::string> reductionVars;
    // The names of the outer For nodes, outermost to innermost.
    std::vector<std::string> vars;
    std::vector<Reduction> reductions;
  } finder;
  s.accept(&finder);

  return finder.reductions;
}

} // namespace halide2isl
} // namespace tc
