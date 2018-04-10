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
#pragma once

#include <sstream>
#include <string>
#include <unordered_map>

#include "tc/core/halide2isl.h"
#include "tc/core/polyhedral/cuda/mapped_scop.h"
#include "tc/core/polyhedral/scop.h"
#include "tc/external/isl.h"

namespace tc {
namespace polyhedral {

struct CodegenContext;
struct CodegenStatementContext;

namespace detail {

isl::pw_aff makeAffFromMappedExpr(
    const Halide::Expr& expr,
    const CodegenStatementContext& context);

void emitHalideExpr(
    const Halide::Expr& e,
    const CodegenStatementContext& context);

void emitHalideExpr(
    const Halide::Expr& e,
    const CodegenStatementContext& context,
    const std::map<std::string, std::string>& substitutions);

void emitMappedSubscripts(
    const std::vector<Halide::Expr>& exprs,
    const CodegenStatementContext& context);

void emitMappedArguments(
    const std::vector<Halide::Expr>& exprs,
    const CodegenStatementContext& context);

void emitMappedTensorAccess(
    std::string name,
    const Halide::Internal::IRNode* node,
    const std::vector<Halide::Expr>& subscripts,
    const CodegenStatementContext& context);

} // namespace detail

/*
 * Information attached to an AST node during printing of the AST.
 * iteratorMap is the inverse schedule, mapping schedule dimensions
 * to the indices of the statement corresponding to the AST node.
 * build is the AST build at the point where the AST node is generated.
 * It is used to generate AST expressions in that context.
 */
struct NodeInfo {
  isl::pw_multi_aff iteratorMap;
  isl::ast_build build;
};
/*
 * Type used for mapping AST node identifier to the corresponding
 * AST node information.
 */
using NodeInfoMapType =
    std::unordered_map<isl::id, NodeInfo, isl::IslIdIslHash>;

struct CodegenContext {
  CodegenContext(
      std::stringstream& ss_,
      const MappedScop& s,
      const NodeInfoMapType& i)
      : ss(ss_), mappedScop(s), nodeInfoMap(i) {}
  CodegenContext(const CodegenContext& c)
      : ss(c.ss), mappedScop(c.mappedScop), nodeInfoMap(c.nodeInfoMap) {}

  const Scop& scop() const {
    return mappedScop.scop();
  }

  std::stringstream& ss;
  const MappedScop& mappedScop;
  const NodeInfoMapType& nodeInfoMap;
};

struct CodegenStatementContext : CodegenContext {
  CodegenStatementContext(const CodegenContext& c, isl::id astId)
      : CodegenContext(c), astNodeId(astId) {}
  isl::pw_multi_aff iteratorMap() const {
    return this->nodeInfoMap.at(astNodeId).iteratorMap;
  }
  // Return the build where the AST node of this CodegenStatementContext
  // was constructed.
  isl::ast_build build() const {
    return this->nodeInfoMap.at(astNodeId).build;
  }
  isl::id statementId() const {
    return this->iteratorMap().get_tuple_id(isl::dim_type::out);
  }
  isl::set domain() const {
    return isl::map::from(this->iteratorMap()).range();
  }
  std::vector<Scop::PromotionInfo> activePromotions() const {
    std::vector<Scop::PromotionInfo> result;
    auto dom = isl::union_set(this->domain());
    for (const auto& kvp : this->scop().activePromotions()) {
      if (!kvp.first.intersect(dom).is_empty()) {
        result.emplace_back(kvp.second);
      }
    }
    return result;
  }
  // Make an affine function from a Halide Expr that is defined
  // over the instance set of the statement corresponding to
  // the AST node of this CodegenStatementContext.  Return a
  // null isl::aff if the expression is not affine.  Fail if any
  // of the variables does not correspond to a parameter or
  // an instance identifier of the statement.
  isl::aff makeIslAffFromExpr(const Halide::Expr& e) const {
    auto space = iteratorMap().get_space().params();
    return scop().makeIslAffFromStmtExpr(statementId(), space, e);
  }

  isl::id astNodeId;
};

std::string emitCudaKernel(
    const std::string& specializedName,
    const MappedScop& scop);

} // namespace polyhedral
} // namespace tc
