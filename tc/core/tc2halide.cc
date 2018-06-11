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
#include <glog/logging.h>

#include "tc/core/check.h"
#include "tc/core/flags.h"
#include "tc/core/tc2halide.h"
#include "tc/lang/parser.h"
#include "tc/lang/sema.h"

namespace tc2halide {

using namespace Halide;
using namespace Halide::Internal;

using std::map;
using std::set;
using std::string;
using std::vector;

namespace {

Type translateScalarType(int tcType) {
  switch (tcType) {
    case lang::TK_BOOL:
      return UInt(1);
    case lang::TK_UINT8:
      return UInt(8);
    case lang::TK_UINT16:
      return UInt(16);
    case lang::TK_UINT32:
      return UInt(32);
    case lang::TK_UINT64:
      return UInt(64);
    case lang::TK_INT8:
      return Int(8);
    case lang::TK_INT16:
      return Int(16);
    case lang::TK_INT32:
      return Int(32);
    case lang::TK_INT64:
      return Int(64);
    case lang::TK_FLOAT:
      return Float(32);
    case lang::TK_DOUBLE:
      return Float(64);
    default:
      LOG(FATAL) << "Unhandled TC scalar type: " << tcType << '\n';
      return Type();
  }
}

// Translate the TC def input params to corresponding Halide components.
// params, inputs will be populated here.
void translateParam(
    const lang::Param& p,
    map<string, Parameter>* params,
    vector<ImageParam>* inputs) {
  // Check if the param has already been converted to halide components.
  if (params->find(p.ident().name()) != params->end()) {
    return;
  }
  lang::TensorType type = p.tensorType();
  int dimensions = (int)type.dims().size();
  ImageParam imageParam(
      translateScalarType(type.scalarType()), dimensions, p.ident().name());
  inputs->push_back(imageParam);
  vector<Expr> dims;
  for (auto d_ : type.dims()) {
    if (d_->kind() == lang::TK_IDENT) {
      auto d = lang::Ident(d_);
      auto it = params->find(d.name());
      Parameter p;
      if (it != params->end()) {
        p = it->second;
      } else {
        p = Parameter(Int(32), false, 0, d.name(), true);
        (*params)[d.name()] = p;
      }
      dims.push_back(Variable::make(Int(32), p.name(), p));
    } else {
      TC_CHECK(d_->kind() == lang::TK_CONST);
      int32_t value = lang::Const(d_).value();
      dims.push_back(Expr(value));
    }
  }

  for (int i = 0; i < imageParam.dimensions(); i++) {
    imageParam.dim(i).set_bounds(0, dims[i]);
  }
  (*params)[imageParam.name()] = imageParam.parameter();
}

void translateOutput(
    const lang::Param& p,
    const map<string, Function>& funcs,
    vector<Function>* outputs) {
  outputs->push_back(funcs.at(p.ident().name()));
}

Expr translateExpr(
    const lang::TreeRef& expr,
    const map<string, Parameter>& params,
    const map<string, Function>& funcs,
    const map<string, Expr>& lets) {
  auto t = [&](int idx) {
    return translateExpr(expr->tree(idx), params, funcs, lets);
  };
  switch (expr->kind()) {
    case lang::TK_IDENT: {
      const auto& name = lang::Ident(expr).name();
      auto it = lets.find(name);
      if (it != lets.end())
        return it->second;
      return Var(name);
    }
    case lang::TK_ACCESS: {
      auto a = lang::Access(expr);
      string tensorName = a.name().name();
      auto paramIt = params.find(tensorName);
      auto funcIt = funcs.find(tensorName);
      vector<Expr> args;
      for (auto e : a.arguments()) {
        args.push_back(translateExpr(e, params, funcs, lets));
      }
      if (paramIt != params.end()) {
        // Accessing an input tensor
        return Call::make(paramIt->second, args);
      } else if (funcIt != funcs.end()) {
        // Call to a Func
        return Call::make(funcIt->second, args);
      } else {
        LOG(FATAL) << "Access to unknown symbol: " << a << '\n';
        return Expr();
      }
    }
    case '+':
      return t(0) + t(1);
    case '-':
      if (expr->trees().size() == 1) {
        return 0 - t(0);
      } else {
        return t(0) - t(1);
      }
    case '*':
      return t(0) * t(1);
    case '/':
      return t(0) / t(1);
    case '%':
      return t(0) % t(1);
    case lang::TK_MIN:
      return min(t(0), t(1));
    case lang::TK_MAX:
      return max(t(0), t(1));
    case '?': {
      Expr cond = t(0), true_val = t(1), false_val = t(2);
      return Call::make(
          true_val.type(),
          Call::if_then_else,
          {cond, true_val, false_val},
          Call::Intrinsic);
    }
    case lang::TK_EQ:
      return t(0) == t(1);
    case lang::TK_NE:
      return t(0) != t(1);
    case lang::TK_LE:
      return t(0) <= t(1);
    case lang::TK_GE:
      return t(0) >= t(1);
    case '<':
      return t(0) < t(1);
    case '>':
      return t(0) > t(1);
    case '!':
      return !t(0);
    case lang::TK_AND:
      return t(0) && t(1);
    case lang::TK_OR:
      return t(0) || t(1);
    case lang::TK_BUILT_IN: {
      auto b = lang::BuiltIn(expr);
      vector<Expr> exprs;
      for (auto a : b.arguments()) {
        exprs.push_back(translateExpr(a, params, funcs, lets));
      }
      auto output_type = translateScalarType(b.type()->kind());
      return Call::make(output_type, b.name(), exprs, Call::PureExtern);
    }
    case lang::TK_CONST: {
      auto c = lang::Const(expr);
      // current Const can only be a a TK_FLOAT or a TK_INT32
      if (c.type()->kind() == lang::TK_FLOAT) {
        return static_cast<float>(c.value());
      } else {
        TC_ASSERT(c, c.type()->kind() == lang::TK_INT32);
        return static_cast<int>(c.value());
      }
    }
    case lang::TK_CAST: {
      auto c = lang::Cast(expr);
      auto v = translateExpr(c.value(), params, funcs, lets);
      return cast(translateScalarType(c.type()->kind()), v);
    }
    default:
      LOG(FATAL) << "Unhandled TC expr: " << expr << '\n';
      return Expr();
  }
}

vector<const Variable*> unboundVariables(const vector<Var>& lhs, Expr rhs) {
  class FindUnboundVariables : public IRVisitor {
    using IRVisitor::visit;

    void visit(const Variable* op) {
      if (!op->param.defined() && !op->reduction_domain.defined() &&
          !op->image.defined() && !bound.contains(op->name) &&
          !visited.count(op->name)) {
        result.push_back(op);
        visited.insert(op->name);
      }
    }

    void visit(const Let* op) {
      op->value.accept(this);
      {
        ScopedBinding<> bind(bound, op->name);
        op->body.accept(this);
      }
    }

    Scope<> bound;
    set<string> visited;

   public:
    FindUnboundVariables(const vector<Var>& lhs) {
      for (auto v : lhs) {
        bound.push(v.name());
      }
    }
    vector<const Variable*> result;
  } finder(lhs);
  rhs.accept(&finder);
  return finder.result;
}

typedef map<Function, map<string, Interval>, Function::Compare> FunctionBounds;

void forwardBoundsInference(
    const std::vector<Expr>& exprs,
    const FunctionBounds& bounds,
    const lang::TreeRef& comprehension,
    bool throwWarnings,
    Scope<Interval>* solution) {
  class CreateConstraints : public IRVisitor {
    using IRVisitor::visit;

    void visit(const Variable* op) {
      if (!op->param.defined() && !op->reduction_domain.defined() &&
          !op->image.defined()) {
        freeVars.insert(op->name);
      }
    }

    void visit(const Call* op) {
      IRVisitor::visit(op);

      // Create inequalities that assert this is not an out-of-bounds access.
      if (op->call_type == Call::Halide) {
        TC_CHECK(op->func.defined())
            << "Expected a Call of type Halide to have an associated Function\n";
        const auto& it = bounds.find(Function(op->func));
        if (it != bounds.end()) {
          const map<string, Interval>& b = it->second;
          for (size_t i = 0; i < op->args.size(); i++) {
            const string& dim = Function(op->func).args()[i];
            const auto& it = b.find(dim);
            if (it != b.end()) {
              Interval interval = it->second;
              TC_CHECK(interval.is_bounded())
                  << "Expected explicit constraints on every dimension of every Func\n";
              result.push_back(op->args[i] >= interval.min);
              result.push_back(op->args[i] <= interval.max);
            }
          }
        }
      } else if (op->call_type == Call::Image) {
        TC_CHECK(op->param.defined())
            << "Expected a Call of type Image to have an associated Parameter\n";
        for (size_t i = 0; i < op->args.size(); i++) {
          TC_CHECK(
              op->param.min_constraint(i).defined() &&
              op->param.extent_constraint(i).defined())
              << "Expected explicit constraints on every dimension of every input\n";
          Expr lb = op->param.min_constraint(i);
          Expr ub = lb + op->param.extent_constraint(i);
          result.push_back(op->args[i] >= lb);
          result.push_back(op->args[i] < ub);
        }
      }
    }

    void visit(const Let* op) {
      // To handle lets correctly we'll need to see which constraints
      // found in the body depend on the let variable and wrap a let
      // expression around each such constraint.
      LOG(FATAL) << "Lets not yet handled by this visitor: " << Expr(op)
                 << '\n';
    };

   public:
    vector<Expr> result;
    set<string> freeVars;
    const FunctionBounds& bounds;
    CreateConstraints(const FunctionBounds& b) : bounds(b) {}
  } constraints(bounds);
  for (auto& expr : exprs) {
    expr.accept(&constraints);
  }
  // Use Halide tools for solving the system. If this falls down in
  // some way we should swap in ISL, but if it doesn't, it's
  // convenient to stay in Halide IR and not convert Exprs back and
  // forth. My (Andrew's) other motivation for writing it this way is
  // to be sure I understand the range inference semantics in TC.

  // The precise model for bounds inference is still under
  // discussion. I'll proceed with inequalities for now, because it
  // makes the tests pass, and it's the first thing I happened to
  // write. It's way too lenient though - in the face of ambiguities
  // it just picks something semi-reasonable instead of throwing an
  // error.

  // A visitor to detect if an expression depends on a single free var
  // only.
  class FreeVarCounter : public IRVisitor {
    using IRVisitor::visit;
    void visit(const Variable* op) {
      if (op->param.defined()) {
        return;
      }
      if (op->name == var) {
        this_var_count++;
      }
      free_vars_count++;
    }
    string var;
    int this_var_count;
    int free_vars_count;

   public:
    bool depends_on_single_free_var(Expr e, string v) {
      this_var_count = 0;
      free_vars_count = 0;
      var = v;
      e.accept(this);
      return this_var_count && (this_var_count == free_vars_count);
    }

    bool free_vars(Expr e) {
      this_var_count = 0;
      free_vars_count = 0;
      var.clear();
      e.accept(this);
      return free_vars_count;
    }

  } counter;

  // Iteratively solve for the free vars. Requires that the system be
  // upper triangular (i.e. we find a variable with no interactions
  // with other variables, then back substitute).
  set<string> unsolved = constraints.freeVars;
  vector<Expr> inequalities = constraints.result;

  Expr unchecked_preconditions = const_true();

  // First fix all the things explicitly specified with 'where' clauses
  for (auto it = solution->begin(); it != solution->end(); ++it) {
    unsolved.erase(it.name());
  }
  for (Expr& e : inequalities) {
    Expr solved = and_condition_over_domain(e, *solution);
    if (is_zero(solved)) {
      unchecked_preconditions = unchecked_preconditions && e;
      e = const_true();
    } else {
      e = solved;
    }
  }

  while (!unsolved.empty()) {
    set<string> still_unsolved;
    for (string v : unsolved) {
      // Collect the terms that depend on this var only
      Expr c = const_true();
      vector<Expr> remaining_inequalities;
      for (Expr e : inequalities) {
        if (counter.depends_on_single_free_var(e, v)) {
          c = c && e;
        } else {
          remaining_inequalities.push_back(e);
        }
      }

      // Find the largest possible range for the free variable that
      // satisfies these inequalities.
      Interval i = solve_for_inner_interval(c, v);

      // TODO: Also check outer interval and assert equal to enforce no slack,
      // no disjoint subsets, etc.

      if (i.is_empty()) {
        throw lang::ErrorReport(comprehension)
            << "Inferred an empty range for variable " << v
            << ". Derived condition: " << c;
      }

      if (!i.is_bounded()) {
        // This var is underconstrained without solving for more vars first.
        still_unsolved.insert(v);
      } else {
        // At this point we could eliminate this var from the system
        // before proceeding to the next var, but we don't, in order
        // to keep things invariant to the order in which we iterate
        // over the set of unsolved vars.
        solution->push(v, i);
        inequalities.swap(remaining_inequalities);
      }
    }

    // At the end of each round, eliminate the variables solved from
    // the system.
    for (Expr& e : inequalities) {
      int before = counter.free_vars(e);
      Expr orig = e;
      e = and_condition_over_domain(e, *solution);
      int after = counter.free_vars(e);
      // Check that we didn't eliminate all vars from a single
      // expression. We would have to check the resulting condition at
      // runtime, and we consider this too confusing for the user - we
      // only do runtime-checked bounds constraints on access
      // dimensions that do not have any free vars in them
      // (e.g. constants).
      if (before && !after && !can_prove(e)) {
        unchecked_preconditions = unchecked_preconditions && orig;
        e = const_true();
      }
    }

    if (still_unsolved.size() == unsolved.size()) {
      Expr combined_expr = fold_left(inequalities, And::make);
      lang::ErrorReport err(comprehension);
      for (string v : unsolved) {
        err << "Unsolved variable: " << v << '\n';
      }
      err << "Inferred constraint: " << combined_expr << '\n'
          << "Could not infer ranges for free variables. Use a where clause";
      throw err;
    }
    unsolved.swap(still_unsolved);
  }

  // Let's take a look at the remaining inequalities. We'll have to
  // assert these at runtime, but that isn't implemented yet.
  inequalities.push_back(unchecked_preconditions);
  Expr remaining = simplify(fold_left(inequalities, And::make));
  if (!is_one(remaining)) {
    lang::ErrorReport err(comprehension);
    err << "Required precondition will not be checked at runtime: "
        << remaining;
    if (throwWarnings) {
      throw err;
    } else {
      warn(err);
    }
  }
}

Expr reductionUpdate(Expr e) {
  return Call::make(e.type(), kReductionUpdate, {e}, Call::Intrinsic);
}

// Translate a single TC comprehension/statement to Halide components: funcs,
// bounds, reductions.
//
// Note that the function definitions created by translateComprehension may
// contain kReductionUpdate intrinsics.  These may have to be removed
// in order to be able to apply internal Halide analysis passes on them.
void translateComprehension(
    const lang::Comprehension& comprehension,
    const map<string, Parameter>& params,
    bool throwWarnings,
    map<string, Function>* funcs,
    FunctionBounds* bounds) {
  Function f;
  auto it = funcs->find(comprehension.ident().name());
  if (it != funcs->end()) {
    f = it->second;
  } else {
    f = Function(comprehension.ident().name());
    (*funcs)[comprehension.ident().name()] = f;
  }
  // Function is the internal Halide IR type for a pipeline
  // stage. Func is the front-end class that wraps it. Here it's
  // convenient to use both.
  Func func(f);

  vector<Var> lhs;
  vector<Expr> lhs_as_exprs;
  for (lang::Ident id : comprehension.indices()) {
    lhs.push_back(Var(id.name()));
    lhs_as_exprs.push_back(lhs.back());
  }

  // we currently inline all of the let bindings generated in where clauses
  // in the future we may consider using Halide Let bindings when they
  // are supported later
  map<string, Expr> lets;
  for (auto wc : comprehension.whereClauses()) {
    if (wc->kind() == lang::TK_LET) {
      auto let = lang::Let(wc);
      lets[let.name().name()] = translateExpr(let.rhs(), params, *funcs, lets);
    }
  }

  Expr rhs = translateExpr(comprehension.rhs(), params, *funcs, lets);

  std::vector<Expr> all_exprs;
  for (auto wc : comprehension.whereClauses()) {
    if (wc->kind() == lang::TK_EXISTS) {
      all_exprs.push_back(
          translateExpr(lang::Exists(wc).exp(), params, *funcs, lets));
    }
  }

  // Halide doesn't have first-class reductions. We map reductions to recursion.
  bool added_implicit_initialization = false;

  auto setupIdentity = [&](const Expr& identity, bool zero) {
    if (!f.has_pure_definition()) {
      added_implicit_initialization = true;
      func(lhs) = (zero) ? identity
                         : undef(rhs.type()); // undef causes the original value
                                              // to remain in input arrays
    }
  };

  // Each reduction operator has two variants
  // (1) +=, TK_PLUS_EQ which updates the tensor inplace using its existing
  // values (2) +=!, TK_PLUS_EQ_B which first sets the tensor to the identity
  // for the reduction and then applies the reduction.
  bool should_zero = false;
  switch (comprehension.assignment()->kind()) {
    case lang::TK_PLUS_EQ_B:
      should_zero = true; // fallthrough
    case lang::TK_PLUS_EQ:
      setupIdentity(make_zero(rhs.type()), should_zero);
      rhs = func(lhs) + rhs;
      break;

    case lang::TK_TIMES_EQ_B:
      should_zero = true; // fallthrough
    case lang::TK_TIMES_EQ:
      setupIdentity(make_one(rhs.type()), should_zero);
      rhs = func(lhs) * rhs;
      break;

    case lang::TK_MIN_EQ_B:
      should_zero = true; // fallthrough
    case lang::TK_MIN_EQ:
      setupIdentity(rhs.type().max(), should_zero);
      rhs = min(func(lhs), rhs);
      break;

    case lang::TK_MAX_EQ_B:
      should_zero = true; // fallthrough
    case lang::TK_MAX_EQ:
      setupIdentity(rhs.type().min(), should_zero);
      rhs = max(func(lhs), rhs);
      break;

    case '=':
      break;
    default:
      throw lang::ErrorReport(comprehension)
          << "Unimplemented reduction "
          << comprehension.assignment()->range().text() << "\n";
  }

  // Tag reductions as such
  if (comprehension.assignment()->kind() != '=') {
    rhs = reductionUpdate(rhs);
  }

  // Bind any scalar params on the rhs to their parameter objects.
  class BindParams : public IRMutator2 {
    using IRMutator2::visit;
    Expr visit(const Variable* op) {
      auto it = params.find(op->name);
      if (it != params.end()) {
        return Variable::make(op->type, op->name, it->second);
      } else {
        return op;
      }
    }
    const map<string, Parameter>& params;

   public:
    BindParams(const map<string, Parameter>& params) : params(params) {}
  } bindParams(params);

  rhs = bindParams.mutate(rhs);
  for (auto& exp : all_exprs) {
    exp = bindParams.mutate(exp);
  }

  // TODO: When the LHS incorporates general expressions we'll need to
  // bind params there too.

  // Do forward bounds inference -- construct an expression that says
  // this expression never reads out of bounds on its inputs, and
  // solve it for the largest possible extent in the free variables to
  // get a bound for each. Note that doing it here eagerly,
  // comprehension-by-comprehension, means we can't mix forwards and
  // backwards bounds inference yet (e.g. resolve bounds of earlier
  // underconstrained comprehensions using later ones based on
  // demand).
  Scope<Interval> solution;

  // Put anything explicitly specified with a 'where' class in the solution
  for (auto constraint_ : comprehension.whereClauses()) {
    if (constraint_->kind() != lang::TK_RANGE_CONSTRAINT)
      continue;
    auto constraint = lang::RangeConstraint(constraint_);
    Interval i;
    i.min = translateExpr(constraint.start(), params, *funcs, lets);
    i.max = translateExpr(constraint.end(), params, *funcs, lets) - 1;

    // TODO: In the future we'll want to make any non-trivial bounds
    // into hidden scalar parameters, and just pass variables to the
    // polyhedral layer instead of potentially complex
    // expressions. This will not be simple. These expressions can
    // potentially affect the bounds of the output tensors, so they
    // need to be efficiently computable outside the emitted kernel by
    // the execution engine.

    solution.push(constraint.ident().name(), i);
  }

  // Infer the rest
  all_exprs.push_back(rhs);
  forwardBoundsInference(
      all_exprs, *bounds, comprehension, throwWarnings, &solution);

  // TODO: What if subsequent updates have incompatible bounds
  // (e.g. an in-place stencil)?. The .bound directive will use the
  // bounds of the last stage for all stages.

  // Does a tensor have a single bound, or can its bounds shrink over
  // time? Solve for a single bound for now.

  for (Var v : lhs) {
    if (!solution.contains(v.name())) {
      throw lang::ErrorReport(comprehension)
          << "Free variable " << v
          << " was not solved in range inference. May not be used right-hand side";
    }
    // TODO: We're enforcing a single bound across all comprehensions
    // for now. We should really check later ones are equal to earlier
    // ones instead of just clobbering.
    (*bounds)[f][v.name()] = solution.get(v.name());
  }

  // Free variables that appear on the rhs but not the lhs are
  // reduction variables. Make a reduction domain for them.

  // TODO: the nesting order of the variables in the reduction domain
  // is currently in the order found, from left to right. This means
  // reordering the expression can change the result for
  // non-commutative reductions.
  vector<const Variable*> unbound = unboundVariables(lhs, rhs);
  RDom rdom;
  if (!unbound.empty()) {
    vector<ReductionVariable> rVars;
    for (size_t i = 0; i < unbound.size(); i++) {
      auto v = unbound[unbound.size() - 1 - i];
      if (!solution.contains(v->name)) {
        throw lang::ErrorReport(comprehension)
            << "Free variable " << v << " is unconstrained. "
            << "Use a 'where' clause to set its range.";
      }
      Interval bound = solution.get(v->name);
      Expr v_min = bound.min;
      Expr v_extent = simplify(bound.max - bound.min + 1);
      rVars.push_back({v->name, v_min, v_extent});
      (*bounds)[f][v->name] = bound;
    }
    ReductionDomain domain(rVars);
    for (auto v : unbound) {
      Expr rv = Variable::make(Int(32), v->name, domain);
      rhs = substitute(v->name, rv, rhs);
    }
    rdom = RDom(domain);
  }

  Stage stage{func(lhs) = rhs};

  // Use the simplest possible Halide schedule, but reorder the loop
  // indices to match TC convention.
  vector<VarOrRVar> loop_nest;
  if (rdom.defined()) {
    for (int i = 0; i < rdom.dimensions(); i++) {
      loop_nest.push_back(rdom[i]);
    }
  }
  while (!lhs.empty()) {
    loop_nest.push_back(lhs.back());
    lhs.pop_back();
  }

  if (added_implicit_initialization) {
    // Also reorder reduction initializations to the TC convention
    vector<Var> funcArgs = func.args();
    loop_nest.clear();
    while (!funcArgs.empty()) {
      loop_nest.push_back(funcArgs.back());
      funcArgs.pop_back();
    }
    func.reorder(loop_nest);
  }

  func.compute_root();
  stage.reorder(loop_nest);
}

// Translate a semantically checked TC def to HalideComponents struct.
HalideComponents translateDef(const lang::Def& def, bool throwWarnings) {
  map<string, Function> funcs;
  HalideComponents components;
  components.def = def;
  FunctionBounds bounds;

  for (auto p : def.params()) {
    translateParam(p, &components.params, &components.inputs);
  }
  for (auto c : def.statements()) {
    translateComprehension(
        c, components.params, throwWarnings, &funcs, &bounds);
  }
  vector<Function> outputs;
  for (auto p : def.returns()) {
    translateOutput(p, funcs, &outputs);
  }

  // Now apply an extremely simplified version of Halide lowering

  // Compute an environment
  map<string, Function> env;
  for (auto f : outputs) {
    populate_environment(f, env);
  }

  // Finalize all the LoopLevels
  for (auto& iter : env) {
    iter.second.lock_loop_levels();
  }

  // Compute a realization order. This is a topological order on the
  // pipeline of groups of Funcs. For our purposes, each group has a
  // single Func in it. The Halide scheduling directive compute_with,
  // (which does general loop fusion) can create groups with multiple
  // Funcs in it, but we don't use it here.
  vector<string> order;
  vector<vector<string>> fused_groups;
  std::tie(order, fused_groups) = realization_order(outputs, env);

  // Create loop nests
  bool any_memoized = false;
  // This part of lowering requires a target, but it will never be
  // used in the pipelines we construct here, so just make a host target.
  Target target("host");
  Stmt s = schedule_functions(outputs, fused_groups, env, target, any_memoized);
  // we insert these to allow for inplace mutation of in/out tensors
  s = remove_undef(s);
  // Apply forward bounds inference results. This replaces the usual Halide
  // bounds inference.
  for (auto p : bounds) {
    const Function& f = p.first;
    for (auto b : p.second) {
      const string& var = b.first;
      const Interval& bound = b.second;
      for (size_t i = 0; i <= f.updates().size(); i++) {
        // Halide lowers function loop bounds as follows:
        string qualified_var_name =
            f.name() + ".s" + std::to_string(i) + "." + var;
        s = LetStmt::make(qualified_var_name + ".min", bound.min, s);
        s = LetStmt::make(qualified_var_name + ".max", bound.max, s);
      }
    }
  }

  // Collect the arguments (inputs and outputs)
  s = uniquify_variable_names(s);
  s = simplify(s);

  // Trim ProducerConsumer annotations. TC doesn't use them.
  class RemoveProducerConsumer : public IRMutator2 {
    using IRMutator2::visit;
    Stmt visit(const ProducerConsumer* op) {
      return mutate(op->body);
    }
  } removeProducerConsumer;

  s = removeProducerConsumer.mutate(s);

  // Rename all loop variables to be valid C identifiers, to ease
  // conversion to isl.
  class RenameVariables : public IRMutator2 {
    using IRMutator2::visit;

    map<string, string> new_names;

    Expr visit(const Variable* op) override {
      auto it = new_names.find(op->name);
      if (it != new_names.end()) {
        return Variable::make(
            op->type, it->second, op->image, op->param, op->reduction_domain);
      } else {
        return op;
      }
    }

    Stmt visit(const For* op) override {
      string sanitized = replace_all(op->name, ".", "_");
      Expr min = mutate(op->min);
      Expr extent = mutate(op->extent);
      new_names[op->name] = sanitized;
      Stmt body = mutate(op->body);
      return For::make(
          sanitized,
          std::move(min),
          std::move(extent),
          op->for_type,
          op->device_api,
          std::move(body));
    }
  } renameVariables;

  s = renameVariables.mutate(s);

  // We don't handle Let nodes after this point
  class SubstituteAllLets : public IRMutator2 {
    Scope<Expr> scope;
    Stmt visit(const LetStmt* op) override {
      ScopedBinding<Expr> bind(scope, op->name, mutate(op->value));
      return mutate(op->body);
    }
    Expr visit(const Let* op) override {
      ScopedBinding<Expr> bind(scope, op->name, mutate(op->value));
      return mutate(op->body);
    }
    Expr visit(const Variable* op) override {
      if (scope.contains(op->name)) {
        return scope.get(op->name);
      } else {
        return op;
      }
    }
  };
  s = SubstituteAllLets().mutate(s);

  components.stmt = s;

  for (Function f : outputs) {
    OutputImageParam o = Func(f).output_buffers()[0];
    // Apply forward bounds inference results to the output buffers.
    const auto& b = bounds[f];
    for (int i = 0; i < o.dimensions(); i++) {
      const Interval& bound = b.at(f.args()[i]);
      o.dim(i).set_bounds(bound.min, simplify(bound.max - bound.min + 1));
    }
    components.outputs.push_back(o);
  }

  return components;
}
} // namespace

HalideComponents
translate(isl::ctx ctx, const lang::TreeRef& treeRef, bool throwWarnings) {
  LOG_IF(INFO, tc::FLAGS_debug_halide) << treeRef;
  return translateDef(
      lang::Def(lang::Sema().checkFunction(treeRef)), throwWarnings);
}

// NOTE: there is no guarantee here that the tc string has only one def. It
// could have many defs. Only first def will be converted in that case.
HalideComponents
translate(isl::ctx ctx, const std::string& tc, bool throwWarnings) {
  LOG_IF(INFO, tc::FLAGS_debug_halide) << tc;
  return translate(ctx, lang::Parser(tc).parseFunction(), throwWarnings);
}

} // namespace tc2halide
