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
#include "tc/utils/compiler_options.h"

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
    case lang::TK_FLOAT16:
      return Float(16);
    case lang::TK_FLOAT32:
      return Float(32);
    case lang::TK_FLOAT64:
      return Float(64);
    case lang::TK_FLOAT:
      return Float(32);
    case lang::TK_DOUBLE:
      return Float(64);

    default:
      LOG(FATAL) << "Unhandled TC scalar type: " << tcType << '\n';
      return Type();
  }
}

struct TensorInfo {
  Type type;
  vector<string> args;
  map<string, Interval> bounds;
};

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

Expr translateExpr(
    const lang::TreeRef& expr,
    const map<string, Parameter>& params,
    const map<string, TensorInfo>& tensors,
    const map<string, Expr>& lets) {
  auto t = [&](int idx) {
    return translateExpr(expr->tree(idx), params, tensors, lets);
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
      auto tensorIt = tensors.find(tensorName);
      vector<Expr> args;
      for (auto e : a.arguments()) {
        args.push_back(translateExpr(e, params, tensors, lets));
      }
      if (paramIt != params.end()) {
        // Accessing an input tensor
        return Call::make(paramIt->second, args);
      } else if (tensorIt != tensors.end()) {
        // Call to a Func
        return Call::make(
            tensorIt->second.type, tensorName, args, Call::Halide);
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
        exprs.push_back(translateExpr(a, params, tensors, lets));
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
      auto v = translateExpr(c.value(), params, tensors, lets);
      return cast(translateScalarType(c.type()->kind()), v);
    }
    default:
      LOG(FATAL) << "Unhandled TC expr: " << expr << '\n';
      return Expr();
  }
}

vector<const Variable*> unboundVariables(const vector<Expr>& lhs, Expr rhs) {
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
    FindUnboundVariables(const vector<Expr>& lhs) {
      for (auto v : lhs) {
        bound.push(v.as<Variable>()->name);
      }
    }
    vector<const Variable*> result;
  } finder(lhs);
  rhs.accept(&finder);
  return finder.result;
}

void forwardBoundsInference(
    const std::vector<Expr>& exprs,
    const map<string, TensorInfo>& tensors,
    const lang::TreeRef& comprehension,
    const tc::CompilerOptions& compilerOptions,
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
        const auto& tensorInfo = tensors.find(op->name);
        if (tensorInfo != tensors.end()) {
          const map<string, Interval>& b = tensorInfo->second.bounds;
          for (size_t i = 0; i < op->args.size(); i++) {
            const string& dim = tensorInfo->second.args[i];
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
    const map<string, TensorInfo>& tensors;
    CreateConstraints(const map<string, TensorInfo>& t) : tensors(t) {}
  } constraints(tensors);
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
    if (compilerOptions.throwWarnings) {
      throw err;
    } else {
      warn(err, compilerOptions);
    }
  }
}

Expr reductionUpdate(Expr e) {
  return Call::make(e.type(), kReductionUpdate, {e}, Call::Intrinsic);
}

// Translate a single TC comprehension/statement to a Halide Stmt
//
// Note that the function definitions created by translateComprehension may
// contain kReductionUpdate intrinsics.  These may have to be removed
// in order to be able to apply internal Halide analysis passes on them.
Stmt translateComprehension(
    const lang::Comprehension& comprehension,
    const map<string, Parameter>& params,
    const tc::CompilerOptions& compilerOptions,
    map<string, TensorInfo>* tensors) {
  TensorInfo info;

  auto tensorName = comprehension.ident().name();

  vector<Expr> lhs;
  for (lang::Ident id : comprehension.indices()) {
    lhs.push_back(Var(id.name()));
    info.args.push_back(id.name());
  }

  // we currently inline all of the let bindings generated in where clauses
  // in the future we may consider using Halide Let bindings when they
  // are supported later
  map<string, Expr> lets;
  for (auto wc : comprehension.whereClauses()) {
    if (wc->kind() == lang::TK_LET) {
      auto let = lang::Let(wc);
      lets[let.name().name()] =
          translateExpr(let.rhs(), params, *tensors, lets);
    }
  }

  Expr rhs = translateExpr(comprehension.rhs(), params, *tensors, lets);

  info.type = rhs.type();

  std::vector<Expr> all_exprs;
  for (auto wc : comprehension.whereClauses()) {
    if (wc->kind() == lang::TK_EXISTS) {
      all_exprs.push_back(
          translateExpr(lang::Exists(wc).exp(), params, *tensors, lets));
    }
  }

  // Each reduction operator has two variants
  // (1) +=, TK_PLUS_EQ which updates the tensor inplace using its existing
  // values (2) +=!, TK_PLUS_EQ_B which first sets the tensor to the identity
  // for the reduction and then applies the reduction.
  Expr currentVal = Call::make(rhs.type(), tensorName, lhs, Call::Halide);
  Expr identity;
  switch (comprehension.assignment()->kind()) {
    case lang::TK_PLUS_EQ_B:
      identity = make_zero(rhs.type());
    case lang::TK_PLUS_EQ: // fallthrough
      rhs = currentVal + rhs;
      break;
    case lang::TK_TIMES_EQ_B:
      identity = make_one(rhs.type());
    case lang::TK_TIMES_EQ: // fallthrough
      rhs = currentVal * rhs;
      break;
    case lang::TK_MIN_EQ_B:
      identity = rhs.type().max();
    case lang::TK_MIN_EQ: // fallthrough
      rhs = min(currentVal, rhs);
      break;
    case lang::TK_MAX_EQ_B:
      identity = rhs.type().min();
    case lang::TK_MAX_EQ: // fallthrough
      rhs = max(currentVal, rhs);
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
    i.min = translateExpr(constraint.start(), params, *tensors, lets);
    i.max = translateExpr(constraint.end(), params, *tensors, lets) - 1;

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
      all_exprs, *tensors, comprehension, compilerOptions, &solution);

  // TODO: What if subsequent updates have incompatible bounds
  // (e.g. an in-place stencil)?. The .bound directive will use the
  // bounds of the last stage for all stages.

  // Does a tensor have a single bound, or can its bounds shrink over
  // time? Solve for a single bound for now.

  for (lang::Ident id : comprehension.indices()) {
    if (!solution.contains(id.name())) {
      throw lang::ErrorReport(comprehension)
          << "Free variable " << id.name()
          << " was not solved in range inference. May not be used right-hand side";
    }
    // TODO: We're enforcing a single bound across all comprehensions
    // for now. We should really check later ones are equal to earlier
    // ones instead of just clobbering.
    info.bounds[id.name()] = solution.get(id.name());
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
      info.bounds[v->name] = bound;
    }
    ReductionDomain domain(rVars);
    for (auto v : unbound) {
      Expr rv = Variable::make(Int(32), v->name, domain);
      rhs = substitute(v->name, rv, rhs);
    }
    rdom = RDom(domain);
  }

  // Now construct the Stmt
  Stmt stmt = Provide::make(tensorName, {rhs}, lhs);

  // Wrap the reduction loops
  if (rdom.defined()) {
    for (int i = 0; i < rdom.dimensions(); i++) {
      stmt = For::make(
          rdom[i].name(),
          rdom[i].min(),
          rdom[i].extent(),
          ForType::Serial,
          DeviceAPI::None,
          stmt);
    }
  }

  // Add an initialization if needed
  Stmt init;
  if (identity.defined()) {
    init = Provide::make(tensorName, {identity}, lhs);
  }

  // Wrap the rest of the loops
  for (auto id = info.args.rbegin(); id != info.args.rend(); id++) {
    Interval in = info.bounds[*id];
    Expr extent = simplify(in.max - in.min + 1);
    stmt =
        For::make(*id, in.min, extent, ForType::Serial, DeviceAPI::None, stmt);
    if (init.defined()) {
      init = For::make(
          *id, in.min, extent, ForType::Serial, DeviceAPI::None, init);
    }
  }

  if (init.defined()) {
    stmt = Block::make(init, stmt);
  }

  auto existingInfo = tensors->find(tensorName);

  // Record information about this tensor for later stages of
  // translation to refer to.
  if (existingInfo == tensors->end()) {
    tensors->emplace(tensorName, std::move(info));
  } else {
    // Clobber the bounds information with the possibly-updated
    // constraints.
    existingInfo->second.bounds = info.bounds;
  }

  return stmt;
}

// Translate a semantically checked TC def to HalideComponents struct.
HalideComponents translateDef(
    const lang::Def& def,
    const tc::CompilerOptions& compilerOptions) {
  HalideComponents components;
  components.def = def;

  map<string, TensorInfo> tensors;

  for (auto p : def.params()) {
    translateParam(p, &components.params, &components.inputs);
  }

  for (auto c : def.statements()) {
    Stmt next =
        translateComprehension(c, components.params, compilerOptions, &tensors);
    if (!components.stmt.defined()) {
      components.stmt = next;
    } else {
      components.stmt = Block::make(components.stmt, next);
    }
  }

  // Populate the output bounds
  for (auto p : def.returns()) {
    // TODO: unify bounds and tensors map?
    const auto& t = tensors[p.ident().name()];
    ImageParam o(t.type, t.args.size(), p.ident().name());
    for (int i = 0; i < o.dimensions(); i++) {
      string arg = t.args[i];
      const Interval& bound = t.bounds.at(arg);
      o.dim(i).set_bounds(bound.min, simplify(bound.max - bound.min + 1));
    }
    components.outputs.push_back(o);
  }

  return components;
}
} // namespace

HalideComponents translate(
    isl::ctx ctx,
    const lang::TreeRef& treeRef,
    const tc::CompilerOptions& compilerOptions = tc::CompilerOptions()) {
  LOG_IF(INFO, tc::FLAGS_debug_halide) << treeRef;
  return translateDef(
      lang::Def(lang::Sema(compilerOptions).checkFunction(treeRef)),
      compilerOptions);
}

// NOTE: there is no guarantee here that the tc string has only one def. It
// could have many defs. Only first def will be converted in that case.
HalideComponents translate(
    isl::ctx ctx,
    const std::string& tc,
    const tc::CompilerOptions& compilerOptions = tc::CompilerOptions()) {
  LOG_IF(INFO, tc::FLAGS_debug_halide) << tc;
  return translate(ctx, lang::Parser(tc).parseFunction(), compilerOptions);
}

} // namespace tc2halide
