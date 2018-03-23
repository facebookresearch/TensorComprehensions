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
#include "tc/core/autodiff.h"
#include "tc/core/tc2halide.h"
#include "tc/lang/parser.h"
#include "tc/lang/sema.h"
#include "tc/lang/tc_format.h"
#include "tc/lang/tree_views.h"

#include <unordered_set>
#include <vector>

namespace tc {

using namespace lang;

static const lang::SourceRange dummyRange{std::make_shared<std::string>(""),
                                          0,
                                          0};

int32_t getTcType(Halide::Type t) {
  if (t.is_int()) {
    switch (t.bits()) {
      case 64:
        return TK_INT64;
      case 32:
        return TK_INT32;
      case 16:
        return TK_INT16;
      case 8:
        return TK_INT8;
    }
  } else if (t.is_uint()) {
    switch (t.bits()) {
      case 64:
        return TK_UINT64;
      case 32:
        return TK_UINT32;
      case 16:
        return TK_UINT16;
      case 8:
        return TK_UINT8;
    }
  } else if (t.is_float()) {
    switch (t.bits()) {
      case 64:
        return TK_DOUBLE;
      case 32:
        return TK_FLOAT;
    }
  }
  throw std::runtime_error("Unknown Halide type");
}

void findAccessedTensors(
    std::unordered_set<std::string>& read_only,
    const TreeRef& tree) {
  if (tree->kind() == TK_ACCESS) {
    read_only.insert(Access(tree).name().name());
  } else {
    for (const TreeRef& subtree : tree->trees()) {
      findAccessedTensors(read_only, subtree);
    }
  }
}

void assertNoWriteAfterRead(Def def) {
  std::unordered_set<std::string> read_only;
  // Inputs are always read-only
  for (Param input : def.params())
    read_only.insert(input.ident().name());
  for (Comprehension comp : def.statements()) {
    findAccessedTensors(read_only, comp.rhs());
    auto lhs_name = comp.ident().name();
    if (read_only.count(lhs_name) > 0)
      throw std::runtime_error(
          "AD not supported in TCs that write to a value after reading it");
  }
}

void findIndexVars(
    std::unordered_set<std::string>& index_vars,
    const TreeRef& tree,
    bool gather_idents) {
  if (tree->kind() == TK_IDENT && gather_idents) {
    index_vars.insert(Ident(tree).name());
  } else if (tree->kind() == TK_ACCESS) {
    for (const TreeRef& idx : Access(tree).arguments()) {
      findIndexVars(index_vars, idx, true);
    }
  } else if (tree->kind() == TK_BUILT_IN) {
    // BuiltIn holds the name of a function as an ident, so we have to skip it
    for (const TreeRef& subtree : BuiltIn(tree).arguments()) {
      findIndexVars(index_vars, subtree, gather_idents);
    }
  } else {
    for (const TreeRef& subtree : tree->trees()) {
      findIndexVars(index_vars, subtree, gather_idents);
    }
  }
}

// XXX: this is a bit of a fragile hack, and can easily break when the AST will
// get more idents in different nodes, but it's quite simple and the worst thing
// that can happen is that we will be too conservative and throw, so it's ok.
std::unordered_set<std::string> usedIndexVars(Comprehension comp) {
  std::unordered_set<std::string> index_vars;
  for (Ident idx : comp.indices())
    index_vars.insert(idx.name());
  findIndexVars(index_vars, comp.rhs(), false);
  return index_vars;
}

// This struct holds a lot of the information required to perform bookkeeping
// of gradient values. For example:
// - do we already have a writeable gradient for a value, or only a seed
// - should we use the seed value, or the writeable gradient in an expression
// - is the gradient implicitly zero now (because the value was overwritten)
struct GradInfo {
  GradInfo(const ListView<Param>& primal_outputs) {
    for (const Param& output : primal_outputs) {
      primal_outputs_.insert(output.ident().name());
    }
  }

  void addGradComprehension(
      Ident primal_lhs_name,
      ListView<lang::TreeRef> lhs_indices,
      TreeRef rhs_expr) {
    auto lhs_name = makeGradName(primal_lhs_name);
    if (has_writeable_grad_.count(primal_lhs_name.name()) == 0) {
      auto rhs_expr = primal_outputs_.count(primal_lhs_name.name()) > 0
          ? Access::create(dummyRange, seedNameOf(lhs_name), lhs_indices)
          : Const::create(
                dummyRange,
                Number::create(0),
                Compound::create(TK_FLOAT, dummyRange, {}));

      grad_comps_.push_back(Comprehension::create(
          dummyRange,
          lhs_name,
          lhs_indices,
          Compound::create('=', dummyRange, {}),
          rhs_expr,
          ListView<TreeRef>::create(dummyRange, TreeList{}),
          Compound::create(TK_OPTION, dummyRange, {}),
          ListView<TreeRef>::create(dummyRange, TreeList{})));
      has_writeable_grad_.insert(primal_lhs_name.name());
    }
    grad_comps_.push_back(Comprehension::create(
        dummyRange,
        lhs_name,
        lhs_indices,
        Compound::create(TK_PLUS_EQ, dummyRange, {}),
        rhs_expr,
        ListView<TreeRef>::create(dummyRange, TreeList{}),
        Compound::create(TK_OPTION, dummyRange, {}),
        ListView<TreeRef>::create(dummyRange, TreeList{})));
    if (usedIndexVars(Comprehension(grad_comps_.back())) !=
        required_index_vars_)
      throw std::runtime_error(
          "Not all index variables are used in gradient comprehension. "
          "AD will require range inference to support this case.");
  }

  bool hasZeroGrad(const std::string& name) {
    return has_zero_grad_.count(name) > 0;
  }
  void markZeroGrad(const std::string& name) {
    has_zero_grad_.count(name);
  }

  std::vector<lang::TreeRef>&& getGradComps() {
    return std::move(grad_comps_);
  }

  void requireAllIndexVarsOf(const Comprehension& comp) {
    required_index_vars_ = usedIndexVars(comp);
  }

  Ident gradNameOf(const Ident& primal_name) {
    if (has_writeable_grad_.count(primal_name.name()) > 0) {
      return makeGradName(primal_name);
    }
    return seedNameOf(primal_name);
  }
  Ident seedNameOf(const Ident& primal_name) {
    return makeSeedName(makeGradName(primal_name));
  }

 private:
  Ident makeGradName(const Ident& name) {
    return Ident(Ident::create(dummyRange, std::string("d_") + name.name()));
  }

  Ident makeSeedName(const Ident& name) {
    return Ident(Ident::create(dummyRange, std::string("seed_") + name.name()));
  }

  std::unordered_set<std::string> required_index_vars_;
  std::vector<lang::TreeRef> grad_comps_;
  // Keys in these sets are always names of primal variables.
  std::unordered_set<std::string> primal_outputs_;
  std::unordered_set<std::string> has_writeable_grad_;
  std::unordered_set<std::string> has_zero_grad_;
};

void differentiateExpr(
    GradInfo& grad_info,
    lang::TreeRef expr,
    lang::TreeRef grad_output_expr) {
  using namespace lang;
  switch (expr->kind()) {
    case TK_ACCESS: {
      Access acc{expr};
      grad_info.addGradComprehension(
          acc.name(), acc.arguments(), grad_output_expr);
      break;
    }
    case '+': {
      differentiateExpr(grad_info, expr->tree(0), grad_output_expr);
      differentiateExpr(grad_info, expr->tree(1), grad_output_expr);
      break;
    }
    case '*': {
      differentiateExpr(
          grad_info,
          expr->tree(0),
          Compound::create(
              '*', expr->range(), {grad_output_expr, expr->tree(1)}));
      differentiateExpr(
          grad_info,
          expr->tree(1),
          Compound::create(
              '*', expr->range(), {grad_output_expr, expr->tree(0)}));
      break;
    }
    case TK_CONST: {
      // There's nothing we have to do to handle constants, because we don't
      // differentiate w.r.t. them.
      break;
    }
    default:
      throw ErrorReport(expr) << "Unsupported expression kind in AD: "
                              << kindToString(expr->kind());
  }
}

// XXX: Sema isn't nilpotent, so we have to reparse the source
std::vector<TreeRef> inferOutputTypes(const std::string& source) {
  auto halide_def =
      tc2halide::translate(isl::with_exceptions::globalIslCtx(), source, true);
  std::vector<TreeRef> output_types;
  for (const auto& halide_output : halide_def.outputs) {
    std::vector<TreeRef> dim_exprs;
    for (int d = 0; d < halide_output.dimensions(); ++d) {
      auto halide_constr = halide_output.parameter().extent_constraint(d);
      if (auto* param = halide_constr.as<Halide::Internal::Variable>()) {
        dim_exprs.push_back(Ident::create(dummyRange, param->name));
      } else if (auto* num = halide_constr.as<Halide::Internal::IntImm>()) {
        dim_exprs.push_back(Const::create(
            dummyRange,
            Number::create(num->value),
            Compound::create(TK_INT32, dummyRange, {})));
      } else {
        std::stringstream s;
        s << "AD only supports TCs in which sizes of outputs can be expressed as "
             "size parameters or constants. This is not the case for "
          << halide_output.name() << " which has an inferred size of (";
        for (int d = 0; d < halide_output.dimensions(); ++d) {
          s << halide_output.parameter().extent_constraint(d);
          if (d != halide_output.dimensions() - 1)
            s << ", ";
        }
        s << ")";
        throw std::runtime_error(s.str());
      }
    }

    auto dim_sizes =
        ListView<TreeRef>::create(dummyRange, std::move(dim_exprs));
    auto scalar_type =
        Compound::create(getTcType(halide_output.type()), dummyRange, {});
    output_types.push_back(
        TensorType::create(dummyRange, scalar_type, dim_sizes));
  }

  return output_types;
}

std::string differentiate(const std::string& source) {
  // Parse and check the source
  auto def = Def(Sema().checkFunction(Parser(source).parseFunction()));
  assertNoWriteAfterRead(def);

  GradInfo grad_info{def.returns()};

  // --------------------------------------------------------------------------
  // Prepare inputs of the gradient Def.
  std::vector<TreeRef> reverse_inputs;
  auto output_types = inferOutputTypes(source);
  auto returns = def.returns();
  for (Param input : def.params()) {
    reverse_inputs.push_back(input);
  }
  for (size_t i = 0, num_returns = returns.size(); i < num_returns; ++i) {
    reverse_inputs.push_back(
        Param::create(dummyRange, returns[i].ident(), output_types.at(i)));
  }
  for (size_t i = 0, num_returns = returns.size(); i < num_returns; ++i) {
    reverse_inputs.push_back(Param::create(
        dummyRange,
        grad_info.seedNameOf(returns[i].ident()),
        output_types.at(i)));
  }

  // --------------------------------------------------------------------------
  // Differentiate the body
  auto body = def.statements();
  auto it = body.end();
  if (it == body.begin())
    throw std::runtime_error("empty body");
  do {
    Comprehension comp = *(--it);

    int assign_kind = comp.assignment()->kind();
    if (assign_kind != '=' && assign_kind != TK_PLUS_EQ_B &&
        assign_kind != TK_PLUS_EQ)
      throw ErrorReport(comp)
          << "Only =, += and +=! assignments are supported in AD";
    if (comp.whereClauses().size() > 0 || comp.equivalent().present())
      throw ErrorReport(comp)
          << "Comprehensions with range constraints or equivalent are not supported in AD";

    // See note [Implicit zero gradients] below.
    auto primal_output = comp.ident();
    if (grad_info.hasZeroGrad(primal_output.name()))
      continue;

    grad_info.requireAllIndexVarsOf(comp);
    auto grad_output_expr = Access::create(
        dummyRange, grad_info.gradNameOf(primal_output), comp.indices());
    differentiateExpr(grad_info, comp.rhs(), grad_output_expr);

    // Note [Implicit zero gradients]
    // If we see one of the overwriting assignments, then we know that all
    // previous values of primal output didn't have any effect on TC outputs
    // and so their gradients are implicitly zero.
    if (assign_kind == '=' || assign_kind == TK_PLUS_EQ_B) {
      grad_info.markZeroGrad(primal_output.name());
    }
  } while (it != body.begin());

  // --------------------------------------------------------------------------
  // Prepare outputs, create the gradient Def, and print it
  auto inferred_type = Compound::create(TK_INFERRED, dummyRange, {});
  std::vector<TreeRef> reverse_outputs;
  for (Param input : def.params())
    reverse_outputs.push_back(Param::create(
        dummyRange, grad_info.gradNameOf(input.ident()), inferred_type));

  auto reverseDef = Def::create(
      dummyRange,
      Ident::create(dummyRange, "grad_" + def.name().name()),
      ListView<Param>::create(dummyRange, std::move(reverse_inputs)),
      ListView<Param>::create(dummyRange, std::move(reverse_outputs)),
      ListView<Comprehension>::create(dummyRange, grad_info.getGradComps()));

  std::ostringstream s;
  tcFormat(s, reverseDef);
  return s.str();
}

} // namespace tc
