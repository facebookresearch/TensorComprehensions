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
#include <numeric>
#include <unordered_set>

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
  // Collect and categorize all the Variable symbols
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
  // tcdef. However, the 0-D tensors are registered as both params and inputs,
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

isl::aff makeIslAffFromExpr(isl::space space, const Expr& e) {
  if (const Variable* op = e.as<Variable>()) {
    isl::local_space ls = isl::local_space(space);
    int pos = space.find_dim_by_name(isl::dim_type::param, op->name);
    if (pos >= 0) {
      return isl::aff(ls, isl::dim_type::param, pos);
    } else {
      pos = space.find_dim_by_name(isl::dim_type::set, op->name);
      if (pos >= 0) {
        return isl::aff(ls, isl::dim_type::set, pos);
      }
    }
    LOG(FATAL) << "Variable not found in isl::space: " << space << ": " << op
               << ": " << op->name << '\n';
    return isl::aff();
  } else if (const Add* op = e.as<Add>()) {
    return makeIslAffFromExpr(space, op->a)
        .add(makeIslAffFromExpr(space, op->b));
  } else if (const Sub* op = e.as<Sub>()) {
    return makeIslAffFromExpr(space, op->a)
        .sub(makeIslAffFromExpr(space, op->b));
  } else if (const Mul* op = e.as<Mul>()) {
    return makeIslAffFromExpr(space, op->a)
        .mul(makeIslAffFromExpr(space, op->b));
  } else if (const Div* op = e.as<Div>()) {
    return makeIslAffFromExpr(space, op->a)
        .div(makeIslAffFromExpr(space, op->b));
  } else if (const Mod* op = e.as<Mod>()) {
    if (const int64_t* b = as_const_int(op->b)) {
      return makeIslAffFromExpr(space, op->a)
          .mod(isl::val(space.get_ctx(), *b));
    }
  } else if (const int64_t* i = as_const_int(e)) {
    return makeIslAffFromInt(space, *i);
  }

  // Non-affine
  return isl::aff();
}

isl::space makeParamSpace(isl::ctx ctx, const SymbolTable& symbolTable) {
  auto space = isl::space(ctx, 0);
  // set parameter names
  for (auto p : symbolTable.params) {
    space = space.add_param(isl::id(ctx, p.name()));
  }
  return space;
}

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

  isl::space rangeSpace = paramSpace.add_dims(isl::dim_type::set, args.size());

  rangeSpace = rangeSpace.set_tuple_name(isl::dim_type::set, tensor);

  // Add a tag to the domain space so that we can maintain a mapping
  // between each access in the IR and the reads/writes maps.
  std::string tag = "__tc_ref_" + std::to_string(accesses->size());
  isl::id tagID(domain.get_ctx(), tag);
  accesses->emplace(op, tagID);
  isl::space tagSpace = paramSpace.set_tuple_name(isl::dim_type::set, tag);
  domainSpace = domainSpace.product(tagSpace);

  // The tensor's storage space is really a set, rather than a
  // space, but for now we'll ignore the fact that it is bounded.
  isl::set range = isl::set::universe(rangeSpace);

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

struct ScheduleTreeAndDomain {
  ScheduleTreeUPtr tree;
  isl::union_set domain;
};

ScheduleTreeAndDomain makeScheduleTreeHelper(
    const Stmt& s,
    isl::set set,
    isl::union_map* reads,
    isl::union_map* writes,
    AccessMap* accesses,
    StatementMap* statements) {
  ScheduleTreeAndDomain result;
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
    isl::aff lb = halide2isl::makeIslAffFromExpr(set.get_space(), op->min);
    Expr max = simplify(op->min + op->extent - 1);
    isl::aff ub = halide2isl::makeIslAffFromExpr(set.get_space(), max);
    set = set.intersect(loopVar.ge_set(lb).intersect(ub.ge_set(loopVar)));

    // Recursively descend.
    auto body = makeScheduleTreeHelper(
        op->body, set, reads, writes, accesses, statements);

    // Create an affine function that defines an ordering for all
    // the statements in the body of this loop over the values of
    // this loop. For each statement in the children we want the
    // function that maps everything in its space to this
    // dimension. The spaces may be different, but they'll all have
    // this loop var at the same index.
    isl::multi_union_pw_aff mupa;
    body.domain.foreach_set([&](isl::set s) {
      isl::aff loopVar(
          isl::local_space(s.get_space()), isl::dim_type::set, thisLoopIdx);
      if (mupa) {
        mupa = mupa.union_add(isl::union_pw_aff(isl::pw_aff(loopVar)));
      } else {
        mupa = isl::union_pw_aff(isl::pw_aff(loopVar));
      }
    });

    if (body.tree) {
      result.tree = ScheduleTree::makeBand(mupa, std::move(body.tree));
    } else {
      result.tree = ScheduleTree::makeBand(mupa);
    }
    result.domain = body.domain;
  } else if (auto op = s.as<Halide::Internal::Block>()) {
    // Flatten a nested block. Halide Block statements always nest
    // rightwards. Flattening it is not strictly necessary, but it
    // keeps things uniform with the PET lowering path.
    std::vector<Stmt> stmts;
    stmts.push_back(op->first);
    stmts.push_back(op->rest);
    while (const Halide::Internal::Block* b =
               stmts.back().as<Halide::Internal::Block>()) {
      Stmt f = b->first;
      Stmt r = b->rest;
      stmts.pop_back();
      stmts.push_back(f);
      stmts.push_back(r);
    }

    // Build a schedule tree for each member of the block, then set up
    // appropriate filters that state which statements lie in which
    // children.
    std::vector<ScheduleTreeUPtr> trees;
    for (Stmt s : stmts) {
      auto mem =
          makeScheduleTreeHelper(s, set, reads, writes, accesses, statements);
      ScheduleTreeUPtr filter;
      if (mem.tree) {
        // No statement instances are shared between the blocks, so we
        // can drop the constraints on the spaces. This makes the
        // schedule tree slightly simpler.
        filter = ScheduleTree::makeFilter(
            mem.domain.universe(), std::move(mem.tree));
      } else {
        filter = ScheduleTree::makeFilter(mem.domain.universe());
      }
      if (result.domain) {
        result.domain = result.domain.unite(mem.domain);
      } else {
        result.domain = mem.domain;
      }
      trees.push_back(std::move(filter));
    }
    CHECK_GE(trees.size(), 1);

    result.tree = ScheduleTree::makeSequence(std::move(trees[0]));
    for (size_t i = 1; i < trees.size(); i++) {
      result.tree->appendChild(std::move(trees[i]));
    }

  } else if (auto op = s.as<Provide>()) {
    // Make an ID for this leaf statement. This *is* semantically
    // meaningful - it is used as a key to identify the provide
    // node.
    size_t stmtIndex = statements->size();
    isl::id id(set.get_ctx(), kStatementLabel + std::to_string(stmtIndex));
    statements->emplace(id, op);
    isl::set domain = set.set_tuple_id(id);
    result.domain = domain;

    isl::union_map newReads, newWrites;
    std::tie(newReads, newWrites) =
        halide2isl::extractAccesses(domain, op, accesses);

    *reads = reads->unite(newReads);
    *writes = writes->unite(newWrites);

  } else {
    LOG(FATAL) << "Unhandled Halide stmt: " << s;
  }
  return result;
};

ScheduleTreeAndAccesses makeScheduleTree(isl::space paramSpace, const Stmt& s) {
  ScheduleTreeAndAccesses result;

  result.writes = result.reads = isl::union_map::empty(paramSpace);

  // Walk the IR building a schedule tree
  auto treeAndDomain = makeScheduleTreeHelper(
      s,
      isl::set::universe(paramSpace),
      &result.reads,
      &result.writes,
      &result.accesses,
      &result.statements);

  // TODO: This fails if the stmt is just a Provide node, I'm not sure
  // what the schedule tree should look like in that case.
  CHECK(treeAndDomain.tree);

  // Add the outermost domain node
  result.tree = ScheduleTree::makeDomain(
      treeAndDomain.domain, std::move(treeAndDomain.tree));

  // Check we have obeyed the ISL invariants
  checkValidIslSchedule(result.tree.get());

  return result;
}

std::vector<Reduction> findReductions(const Stmt& s) {
  class FindReductions : public IRVisitor {
    using IRVisitor::visit;

    bool isReductionInit(const Provide* op) {
      if (const Call* call = op->values[0].as<Call>()) {
        return call->is_intrinsic(tc2halide::kReductionInit);
      } else {
        return false;
      }
    }

    bool isReductionUpdate(const Provide* op) {
      if (const Call* call = op->values[0].as<Call>()) {
        return call->is_intrinsic(tc2halide::kReductionUpdate);
      } else {
        return false;
      }
    }

    // Keep track of any reduction variable name for use in isValidReduction
    void visit(const Variable* op) {
      if (op->reduction_domain.defined()) {
        reductionVars.insert(op->name);
      }
    }

    // Check that the given update node, together with the corresponding
    // init node form a proper reduction pair.
    // In particular, check that they share some outer For nodes and
    // that the variables of the additional For nodes surrounding
    // the update node are all reduction variables.
    bool isValidReductionUpdate(const Provide* op) {
      const auto& opInitVars = initVars[op->name];
      auto n = opInitVars.size();
      if (vars.size() <= n) {
        return false;
      }
      if (!std::equal(opInitVars.begin(), opInitVars.end(), vars.begin())) {
        return false;
      }
      for (auto i = vars.begin() + n; i != vars.end(); ++i) {
        if (reductionVars.count(*i) == 0) {
          return false;
        }
      }
      return true;
    }

    // Keep track of the names of the outer For nodes.
    void visit(const For* op) {
      vars.push_back(op->name);
      IRVisitor::visit(op);
      vars.pop_back();
    }

    // Check if the node is an init node, keeping track of it,
    // or an update node corresponding to init node that was found before,
    // updating the information about the reduction.
    // In particular, double-check that the pair are in the right
    // relative positions and collect the positions of the reduction
    // dimensions in the update statement domain.
    // Visit the children first to ensure that all relevant
    // reduction variables have been found first.
    void visit(const Provide* op) {
      IRVisitor::visit(op);
      if (isReductionInit(op)) {
        reductions[op->name].init = op;
        initVars[op->name] = vars;
      } else if (isReductionUpdate(op)) {
        if (isValidReductionUpdate(op)) {
          auto& p = reductions[op->name];
          CHECK(p.init.defined())
              << "Missing reduction init or (unsupported) multiple updates";
          CHECK(!p.update.defined())
              << "Multiple reduction updates not yet implemented";
          p.update = op;
          auto n = initVars[op->name].size();
          p.dims.resize(vars.size() - n);
          std::iota(p.dims.begin(), p.dims.end(), n);
        } else {
          reductions.erase(op->name);
        }
      }
    }

   public:
    // The variables that are known to be reduction variables.
    std::unordered_set<std::string> reductionVars;
    // The names of the outer For nodes, outermost to innermost.
    std::vector<std::string> vars;
    // For each init node, the names of its outer For nodes.
    std::map<std::string, std::vector<std::string>> initVars;
    std::map<std::string, Reduction> reductions;
  } finder;
  s.accept(&finder);

  std::vector<Reduction> result;
  for (auto p : finder.reductions) {
    result.push_back(p.second);
  }
  return result;
}

} // namespace halide2isl
} // namespace tc
