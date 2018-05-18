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
#include <algorithm>
#include <set>
#include <sstream>
#include <string>
#include <unordered_map>
#include <utility>

#include "tc/core/flags.h"
#include "tc/core/islpp_wrap.h"
#include "tc/core/libraries.h"
#include "tc/core/polyhedral/codegen.h"
#include "tc/core/polyhedral/cuda/codegen.h"
#include "tc/core/polyhedral/cuda/mapping_types.h"
#include "tc/core/polyhedral/memory_promotion.h"
#include "tc/core/polyhedral/schedule_isl_conversion.h"
#include "tc/core/polyhedral/schedule_transforms.h"

using namespace std;

namespace tc {
namespace polyhedral {

namespace {

struct WS {
  static thread_local int n;
  WS() {
    n += 2;
  }
  ~WS() {
    n -= 2;
  }
  string tab() {
    stringstream ss;
    for (int i = 0; i < n; ++i) {
      ss << " ";
    }
    return ss.str();
  }
};
thread_local int WS::n = 0;

std::string makePointerName(std::string n) {
  return string("p") + n;
}

std::string makeReductionTmpName(isl::id updateId, const Scop& scop) {
  int pos = scop.reductionUpdatePos(updateId);
  return "acc_" + std::to_string(pos);
}

template <typename T>
inline vector<T> operator+(vector<T> a, const vector<T>& b) {
  vector<T> res{a};
  res.insert(res.begin() + res.size(), b.begin(), b.end());
  return res;
}

struct AstPrinter {
 public:
  AstPrinter(const CodegenContext& context) : context_(context) {}
  void emit(isl::ast_node node) {
    emitAst(node);
  }

 private:
  void emitFor(isl::ast_node_for node);
  void emitIf(isl::ast_node_if node);
  void emitStmt(isl::ast_node_user node);
  void emitAst(isl::ast_node node);

 private:
  const CodegenContext& context_;
  // Identifier of reduction update node processed by emitStmt for use
  // in a tree synchronization in a subsequent call to emitStmt.
  isl::id reductionUpdateNodeId_;
  // Has a reduction init statement been encountered in a previous
  // call to emitStmt without a subsequent tree synchronization?
  bool inReduction_ = false;
};

vector<string> emitParams(const Scop& scop) {
  vector<string> res;
  res.reserve(scop.halide.params.size());
  // Halide params. One of these two vectors will be empty.
  for (auto p : scop.halide.params) {
    stringstream ss;
    ss << p.type() << " " << p.name();
    res.push_back(ss.str());
  }
  return res;
}

// Returns number of names printed, i.e. tensors.size().
string emitTypedTensorName(
    Halide::OutputImageParam t,
    bool constInput = false) {
  stringstream ss;
  ss << (constInput ? "const " : "") << t.type() << "* "
     << makePointerName(t.name());
  return ss.str();
}

vector<string> emitTypedTensorNames(
    const vector<Halide::OutputImageParam>& tensors) {
  vector<string> res;
  res.reserve(tensors.size());
  for (auto t : tensors) {
    res.push_back(emitTypedTensorName(t));
  }
  return res;
}

vector<string> emitTypedTensorNames(const vector<Halide::ImageParam>& tensors) {
  vector<string> res;
  res.reserve(tensors.size());
  for (auto t : tensors) {
    res.push_back(emitTypedTensorName(t, true));
  }
  return res;
}

void emitArgs(stringstream& ss, const Scop& scop) {
  // Order is: params, outs, ins
  auto sigVec = emitParams(scop);
  sigVec = sigVec + emitTypedTensorNames(scop.halide.outputs);
  sigVec = sigVec + emitTypedTensorNames(scop.halide.inputs);
  for (auto& s : sigVec) {
    ss << s;
    if (s != sigVec.back()) {
      ss << ", ";
    }
  }
}

void emitKernelSignature(
    stringstream& ss,
    const std::string& specializedName,
    const Scop& scop) {
  CHECK_NE(specializedName, "") << "name not provided";
  ss << "__global__ void " << specializedName << "(";
  emitArgs(ss, scop);
  ss << ") {" << endl;
}

// This is similar to the pass unpack_buffers in
// Halide, which unpacks strides, grabs alignment constraints,
// etc.
// TODO: this is still incorrect because at this point we only use the
// DLTensor shape (i.e. sizes) of the computations.
// To be correct we need the strides.
// Unfortunately, strides are related to memory allocation and are ML
// framework specific.
// Halide has its own facilities to allocate memory and handles concrete
// allocated memory at the (linearized) Buffer level.
// We don't want that, and we are even at a higher level of IR where Buffer to
// not exist.
// So we must pass an additional structure to save strides that we collect at
// runtime from the actual tensors that are passed to TcOp.
// We could go parametric but then we need to pass all the strides as
// parameters to the kernel call. This is doable, we've been doing it since
// day 1 with fbcuda's DeviceTensor but it loses runtime alignment information
// (or we need to jump through hoops to make proper use of it).
// So the better path here is probably to JIT everything, except people want
// as parametric code as possible, **sigh**.
void emitTensorView(
    stringstream& ss,
    Halide::OutputImageParam p,
    const map<string, Halide::Expr>& paramValues,
    bool constInput = false) {
  WS ws;
  stringstream ssViewType;
  for (int i = 1; i < p.dimensions(); ++i) { // Skip the outermost dimension
    Halide::Expr extent = p.parameter().extent_constraint(i);
    extent = Halide::Internal::substitute(paramValues, extent);
    CHECK(extent.defined())
        << "Undefined extent on input/output tensor. Forward bounds inference should have set these\n";
    ssViewType << "[" << extent << "]";
  }
  ss << ws.tab();
  ss << (constInput ? "const " : "") << p.type() << " (*" << p.name() << ")"
     << ssViewType.str();
  ss << " = ";
  ss << "reinterpret_cast<" << (constInput ? "const " : "") << p.type()
     << " (*)" << ssViewType.str() << ">";
  ss << "(" << makePointerName(p.name()) << ")";
  ss << ";";
  ss << endl;
}

void emitTensorViews(
    stringstream& ss,
    const vector<Halide::OutputImageParam>& params,
    const map<string, Halide::Expr>& paramValues) {
  for (auto p : params) {
    emitTensorView(ss, p, paramValues);
  }
}

void emitTensorViews(
    stringstream& ss,
    const vector<Halide::ImageParam>& params,
    const map<string, Halide::Expr>& paramValues) {
  for (auto p : params) {
    emitTensorView(ss, p, paramValues, true);
  }
}

void AstPrinter::emitFor(isl::ast_node_for node) {
  WS ws;
  context_.ss << ws.tab();
  string iter = node.get_iterator().to_C_str();
  context_.ss << "for (int " << iter << " = " << node.get_init().to_C_str()
              << "; " << node.get_cond().to_C_str() << "; " << iter
              << " += " << node.get_inc().to_C_str() << ") {" << endl;
  emitAst(node.get_body());
  context_.ss << ws.tab() << "}" << endl;
}

void AstPrinter::emitIf(isl::ast_node_if node) {
  WS ws;
  context_.ss << ws.tab();
  context_.ss << "if (" << node.get_cond().to_C_str() << ") {" << endl;
  emitAst(node.get_then());
  context_.ss << ws.tab() << "}";
  if (node.has_else()) {
    context_.ss << " else {" << endl;
    emitAst(node.get_else());
    context_.ss << ws.tab() << "}";
  }
  context_.ss << endl;
}

void emitReductionOpName(const Halide::Expr& e, const CodegenContext& context) {
  auto call = e.as<Halide::Internal::Call>();
  CHECK(call);
  CHECK(call->is_intrinsic(tc2halide::kReductionUpdate));
  context.ss << "__tc::ReductionOp::";
  if (call->args[0].as<Halide::Internal::Add>()) {
    context.ss << "Sum";
  } else if (call->args[0].as<Halide::Internal::Mul>()) {
    context.ss << "Prod";
  } else if (call->args[0].as<Halide::Internal::Min>()) {
    context.ss << "Min";
  } else if (call->args[0].as<Halide::Internal::Max>()) {
    context.ss << "Max";
  } else {
    CHECK(false) << "unsupported reduction type: " << e << "\n";
  }
}

// Emit a cross-thread tree reduce.
// For now this is only expected to work with threadIdx.x.
void emitTreeSyncCall(
    isl::id id,
    isl::id reductionUpdateNodeId,
    const CodegenStatementContext& context) {
  CHECK_EQ(1u, context.scop().treeSyncUpdateMap.count(id));
  isl::id updateId = context.scop().treeSyncUpdateMap.at(id);

  // Halide reduction.
  auto provide = context.scop()
                     .halide.statements.at(updateId)
                     .as<Halide::Internal::Provide>();

  USING_MAPPING_SHORT_NAMES(BX, BY, BZ, TX, TY, TZ);
  std::array<size_t, 3> dims = {TX.mappingSize(context.mappedScop.numThreads),
                                TY.mappingSize(context.mappedScop.numThreads),
                                TZ.mappingSize(context.mappedScop.numThreads)};

  context.ss << tc::code::cuda::kCUBReductionName;

  // Template mapping dimension
  context.ss << "<";
  context.ss << dims[0];
  context.ss << ",";
  context.ss << dims[1];
  context.ss << ",";
  context.ss << dims[2];
  context.ss << ",";
  emitReductionOpName(provide->values[0], context);
  context.ss << ">(";
  // Reference to final target element
  auto arrayName = provide->name;
  // Pass T* (i.e. address) for template type deduction
  context.ss << "&";
  detail::emitMappedTensorAccess(
      provide->name,
      provide,
      provide->args,
      CodegenStatementContext(context, reductionUpdateNodeId));
  context.ss << ", ";
  // Reduction temporary
  context.ss << makeReductionTmpName(updateId, context.scop());
  context.ss << ");" << endl;
}

void emitUserStmt(isl::id stmtId, const CodegenStatementContext& context) {
  CHECK(context.scop().halide.statements.count(stmtId))
      << "No stmt with id " << stmtId << "\n";
  auto provide = context.scop().halide.statements.at(stmtId);
  auto op = provide.as<Halide::Internal::Provide>();
  CHECK(op) << "Expected a Provide node: " << provide << '\n';
  detail::emitMappedTensorAccess(op->name, op, op->args, context);
  context.ss << " = ";
  CHECK(op->values.size() == 1)
      << "Multi-valued Provide: " << Halide::Internal::Stmt(provide) << "\n";
  detail::emitHalideExpr(op->values[0], context);
  context.ss << ";" << endl;
}

void emitReductionUpdate(
    isl::id stmtId,
    const CodegenStatementContext& context) {
  // This is a Halide reduction. The reduction update is stored as a
  // recursive expression (e.g. f(x, y) = f(x, y) + foo). Replace
  // the recursive call with a variable representing the temporary
  // accumulator. It's probably at the root of the expression tree,
  // but it's easy enough to be generic here to accommodate more
  // complex reductions in the future.
  string tmp = makeReductionTmpName(stmtId, context.scop());
  context.ss << tmp << " = ";
  auto provide = context.scop()
                     .halide.statements.at(stmtId)
                     .as<Halide::Internal::Provide>();
  Halide::Expr rhs = provide->values[0];
  map<string, string> substitutions;
  substitutions[provide->name] = tmp;
  detail::emitHalideExpr(rhs, context, substitutions);
  context.ss << ";" << endl;
}

void emitReductionInit(
    isl::id stmtId,
    isl::id updateId,
    const CodegenContext& context) {
  // Emit the identity of a reduction, to initialize a local accumulator.
  auto provide = context.scop()
                     .halide.statements.at(updateId)
                     .as<Halide::Internal::Provide>();
  context.ss << makeReductionTmpName(updateId, context.scop()) << " = ";
  auto call = provide->values[0].as<Halide::Internal::Call>();
  CHECK(call && call->is_intrinsic(tc2halide::kReductionUpdate));
  auto assoc = prove_associativity(provide->name, provide->args, call->args);
  CHECK(assoc.associative());
  auto statementContext = CodegenStatementContext(context, stmtId);
  CHECK_EQ(assoc.pattern.identities.size(), 1u);
  detail::emitHalideExpr(assoc.pattern.identities[0], statementContext);
  context.ss << ";" << endl;
}

namespace {
template <typename AFF>
void emitAccess(AFF access, const CodegenStatementContext& context) {
  bool readOnly =
      context.readOnlySet.count(access.get_tuple_id(isl::dim_type::out)) > 0;
  if (readOnly) {
    context.ss << "__ldg(&";
  }
  context.ss << context.build().access_from(access).to_C_str();
  if (readOnly) {
    context.ss << ")";
  }
}
} // namespace

void emitCopyStmt(const CodegenStatementContext& context) {
  auto stmtId = context.statementId();

  auto iteratorMap = context.iteratorMap();
  auto promoted = iteratorMap.range_factor_range();
  auto original = iteratorMap.range_factor_domain().range_factor_range();
  auto isRead = stmtId.get_name() == kReadIdName;

  if (isRead) {
    emitAccess(isl::multi_pw_aff(promoted), context);
    context.ss << " = ";
    emitAccess(isl::multi_pw_aff(original), context);
  } else {
    emitAccess(isl::multi_pw_aff(original), context);
    context.ss << " = ";
    emitAccess(isl::multi_pw_aff(promoted), context);
  }
  context.ss << ";" << std::endl;
}

void AstPrinter::emitStmt(isl::ast_node_user node) {
  isl::ast_expr_op usrExp = node.get_expr().as<isl::ast_expr_op>();
  auto stmtId = usrExp.get_arg(0).as<isl::ast_expr_id>().get_id();
  auto nodeId = node.get_annotation();
  auto statementContext = CodegenStatementContext(context_, nodeId);
  CHECK_EQ(context_.nodeInfoMap.count(nodeId), 1u)
      << "no info for node " << nodeId;

  WS ws;
  context_.ss << ws.tab();

  if (context_.scop().isTreeSyncId(stmtId)) {
    emitTreeSyncCall(stmtId, reductionUpdateNodeId_, statementContext);
    reductionUpdateNodeId_ = isl::id();
    inReduction_ = false;
  } else if (context_.scop().isDefaultReductionInitId(stmtId)) {
    auto updateId = context_.scop().getReductionUpdateForDefaultInit(stmtId);
    emitReductionInit(stmtId, updateId, context_);
    inReduction_ = true;
  } else if (inReduction_ && context_.scop().isReductionUpdate(stmtId)) {
    emitReductionUpdate(stmtId, statementContext);
    reductionUpdateNodeId_ = nodeId;
  } else if (context_.scop().isSyncId(stmtId)) {
    context_.ss << "__syncthreads();" << std::endl;
  } else if (context_.scop().isWarpSyncId(stmtId)) {
    context_.ss << "__syncwarp();" << std::endl;
  } else if (
      stmtId.get_name() == kReadIdName || stmtId.get_name() == kWriteIdName) {
    emitCopyStmt(statementContext);
  } else { // regular statement
    auto mappedStmtId = statementContext.statementId();
    CHECK_EQ(stmtId, mappedStmtId)
        << "statement ids in expr (" << stmtId << ") and in iteratorMaps ("
        << mappedStmtId << ") do not match";
    emitUserStmt(stmtId, statementContext);
  }
}

void AstPrinter::emitAst(isl::ast_node node) {
  if (auto forNode = node.as<isl::ast_node_for>()) {
    emitFor(forNode);
  } else if (auto ifNode = node.as<isl::ast_node_if>()) {
    emitIf(ifNode);
  } else if (auto blockNode = node.as<isl::ast_node_block>()) {
    for (auto child : blockNode.get_children()) {
      emitAst(child);
    }
  } else if (node.as<isl::ast_node_mark>()) {
    CHECK(false) << "mark";
    // emitAst(node.mark_get_node());
  } else if (auto userNode = node.as<isl::ast_node_user>()) {
    emitStmt(userNode);
  } else {
    LOG(FATAL) << "NYI " << node << endl;
  }
}

} // namespace

namespace detail {

isl::pw_aff makeAffFromMappedExpr(
    const Halide::Expr& expr,
    const CodegenStatementContext& context) {
  // We only expect this to be called on encountering a free
  // variable. Compound expressions should be emitted as Halide.
  CHECK(expr.as<Halide::Internal::Variable>());
  auto aff = context.makeIslAffFromExpr(expr);
  auto pwaff = isl::pw_aff(aff).pullback(context.iteratorMap());
  return pwaff;
}

isl::space findDomainSpaceById(const CodegenStatementContext& context) {
  for (auto d : isl::UnionAsVector<isl::union_set>(context.scop().domain())) {
    if (d.get_tuple_id() == context.statementId()) {
      return d.get_space();
    }
  }
  CHECK(false) << "could not find domain for " << context.statementId()
               << " in " << context.scop().domain();
  return isl::space();
}

isl::multi_aff makeMultiAffAccess(
    isl::id tensorId,
    const std::vector<Halide::Expr>& subscripts,
    const CodegenStatementContext& context) {
  CHECK_NE(subscripts.size(), 0u) << "cannot build subscript aff for a scalar";

  auto domainSpace = findDomainSpaceById(context);
  auto tensorSpace = domainSpace.params().named_set_from_params_id(
      tensorId, subscripts.size());
  auto space = domainSpace.map_from_domain_and_range(tensorSpace);

  auto ma = isl::multi_aff::zero(space);
  for (size_t i = 0; i < subscripts.size(); ++i) {
    ma = ma.set_aff(i, context.makeIslAffFromExpr(subscripts[i]));
  }
  return ma;
}

namespace {
bool is_identifier_or_nonnegative_integer(isl::ast_expr expr) {
  if (isl_ast_expr_get_type(expr.get()) == isl_ast_expr_id)
    return true;
  if (isl_ast_expr_get_type(expr.get()) != isl_ast_expr_int)
    return false;
  return isl::manage(isl_ast_expr_get_val(expr.get())).is_nonneg();
}
} // namespace

void emitHalideExpr(
    const Halide::Expr& e,
    const CodegenStatementContext& context,
    const map<string, string>& substitutions) {
  class EmitHalide : public Halide::Internal::IRPrinter {
    using Halide::Internal::IRPrinter::visit;
    void visit(const Halide::Internal::Variable* op) {
      auto pwAff = tc::polyhedral::detail::makeAffFromMappedExpr(
          Halide::Expr(op), context);
      auto expr = context.build().expr_from(pwAff);
      auto s = expr.to_C_str();
      if (!is_identifier_or_nonnegative_integer(expr)) {
        s = "(" + s + ")";
      }
      context.ss << s;
    }
    void visit(const Halide::Internal::Call* op) {
      if (substitutions.count(op->name)) {
        context.ss << substitutions.at(op->name);
      } else if (
          op->call_type == Halide::Internal::Call::CallType::Halide ||
          op->call_type == Halide::Internal::Call::CallType::Image) {
        tc::polyhedral::detail::emitMappedTensorAccess(
            op->name, op, op->args, context);
      } else if (
          op->is_intrinsic(tc2halide::kReductionInit) ||
          op->is_intrinsic(tc2halide::kReductionUpdate)) {
        op->args[0].accept(this);
      } else {
        IRPrinter::visit(op);
      }
    }
    // TODO: handle casts
    const CodegenStatementContext& context;
    const map<string, string>& substitutions;

   public:
    EmitHalide(
        const CodegenStatementContext& ctx,
        const map<string, string>& substitutions)
        : IRPrinter(ctx.ss), context(ctx), substitutions(substitutions) {}
  } printer(context, substitutions);

  e.accept(&printer);
}

void emitHalideExpr(
    const Halide::Expr& e,
    const CodegenStatementContext& context) {
  map<string, string> substitutions;
  emitHalideExpr(e, context, substitutions);
}

void emitMappedTensorAccess(
    std::string name,
    const Halide::Internal::IRNode* node,
    const vector<Halide::Expr>& subscripts,
    const CodegenStatementContext& context) {
  // Scalars are not promoted or remapped.
  if (subscripts.empty()) {
    context.ss << name << "[0]";
    return;
  }

  CHECK_EQ(context.scop().halide.accesses.count(node), 1u)
      << "attempting to generate code for tensor " << name
      << " reference not present in Scop" << node;
  auto refId = context.scop().halide.accesses.at(node);

  Scop::PromotionInfo promotionInfo;
  for (auto pi : context.activePromotions()) {
    if (pi.group->referenceIds().count(refId)) {
      CHECK(!promotionInfo.groupId)
          << "reference " << refId
          << " belongs to two groups: " << promotionInfo.groupId << " and "
          << pi.groupId;
      promotionInfo = pi;
    }
  }

  // Not promoted, emitting just the mapped subscript.
  if (!promotionInfo.groupId) {
    // Almost certainly not the proper way but how do we get to a proper
    // isl::id here?
    bool readOnly = false;
    for (auto id : context.readOnlySet) {
      if (id.to_str() == name) {
        readOnly = true;
        break;
      }
    }
    if (readOnly) {
      context.ss << "__ldg(&";
    }
    context.ss << name;
    for (auto e : subscripts) {
      context.ss << "[";
      emitHalideExpr(e, context);
      context.ss << "]";
    }
    if (readOnly) {
      context.ss << ")";
    }
    return;
  }

  auto tensorId = context.scop().promotedDecl(promotionInfo.groupId).tensorId;

  // Here and below in comments: D = domain, O = original tensor, P = promoted
  // tensor, S = partial schedule, A = AST loops;
  // MA = multi_aff, PMA = pw_multi_aff
  auto access =
      makeMultiAffAccess(tensorId, subscripts, context); // MA :: D -> O
  auto promotion = promotionInfo.group->promotion(); // MA :: [S -> O] -> P
  promotion = promotion.set_tuple_id(isl::dim_type::out, promotionInfo.groupId);
  auto iteratorMap = context.iteratorMap(); // PMA :: A -> D
  auto schedule =
      isl::map::from_union_map(promotionInfo.outerSchedule.intersect_domain(
          context.domain())); // map :: D -> S

  CHECK(schedule.is_single_valued())
      << "expected single-valued schedule, got " << schedule;
  // PMA :: A -> S
  auto astToSchedule = isl::pw_multi_aff(schedule).pullback(iteratorMap);
  // PMA :: A -> O
  auto astToOriginal = isl::pw_multi_aff(access).pullback(iteratorMap);
  // PMA :: A -> [S -> O]
  auto astToScheduledOriginal = astToSchedule.range_product(astToOriginal);
  // PMA :: A -> P
  auto astToPromoted =
      isl::pw_multi_aff(promotion).pullback(astToScheduledOriginal);

  emitAccess(astToPromoted, context);
}

} // namespace detail

// TODO: b0,b1,b2 and t0,t1,t2 are actually hardcoded in codegen_cuda
//       bx,by,bz and tx,ty,tz do not work and this is actually scary!!
// TODO: This is terrible and needs to be changed. Funny enough it is already
//       strictly better than the previous implementation...
void emitThreadIdInit(stringstream& ss, const MappedScop& scop) {
  WS ws;
  ss << ws.tab();
  ss << "int b0 = blockIdx.x; int b1 = blockIdx.y; int b2 = blockIdx.z;\n";
  ss << ws.tab();
  ss << "int t0 = threadIdx.x; int t1 = threadIdx.y; int t2 = threadIdx.z;\n";
}

void emitTmpDecl(stringstream& ss, const Scop& scop) {
  for (const auto& kvp : scop.treeSyncUpdateMap) {
    WS ws;
    ss << ws.tab();
    auto updateId = kvp.second;
    auto provide =
        scop.halide.statements.at(updateId).as<Halide::Internal::Provide>();
    ss << provide->values[0].type() << " "
       << makeReductionTmpName(updateId, scop) << ";" << endl;
  }
}

void emitPromotedArrayViewsHalide(stringstream& ss, const Scop& scop) {
  for (const auto& p : scop.promotedDecls()) {
    WS ws;
    ss << ws.tab();
    auto viewName = p.first.get_name();
    auto tensorName = p.second.tensorId.get_name();
    Halide::Type t;
    for (auto o : scop.halide.outputs) {
      if (o.name() == tensorName) {
        t = o.type();
      }
    }
    for (auto i : scop.halide.inputs) {
      if (i.name() == tensorName) {
        t = i.type();
      }
    }
    if (p.second.kind == Scop::PromotedDecl::Kind::SharedMem) {
      ss << "__shared__ ";
    }
    ss << t << " " << viewName;
    for (auto s : p.second.sizes) {
      ss << "[" << s << "]";
    }
    ss << ";" << endl;
  }
}

size_t& nAstNodes() {
  static thread_local size_t n = 0;
  return n;
}

std::unordered_set<isl::id, isl::IslIdIslHash> gatherReadOnlySet(
    const MappedScop& mscop) {
  auto root = mscop.schedule();
  const auto& scop = mscop.scop();
  std::unordered_set<isl::id, isl::IslIdIslHash> readOnlySet;
  if (mscop.useReadOnlyCache) {
    // TensorReferenceGroup::accessedBySubtree seems to require a
    // MappingFilter. It does not gather references on the root only, so we
    // iterate on all MappingFilter but ideally scubtrees == root
    auto subtrees = detail::ScheduleTree::collect(
        root, detail::ScheduleTreeType::MappingFilter);
    std::unordered_map<isl::id, bool, isl::IslIdIslHash> isReadOnly;
    for (auto t : subtrees) {
      auto groupMap = TensorReferenceGroup::accessedBySubtree(t, scop);
      for (const auto& kvp : groupMap) {
        if (isReadOnly.count(kvp.first) == 0) {
          isReadOnly[kvp.first] = true;
        }
        for (const auto& group : kvp.second) {
          isReadOnly[kvp.first] &= group->isReadOnly();
        }
      }
    }
    for (auto kvp : isReadOnly) {
      if (kvp.second) {
        readOnlySet.emplace(kvp.first);
      }
    }
  }
  return readOnlySet;
}

string emitCudaKernel(
    const std::string& specializedName,
    const MappedScop& mscop) {
  // Expecting a schedule with domain root and context first child.
  CHECK(mscop.schedule()->elemAs<detail::ScheduleTreeElemDomain>());
  CHECK(
      mscop.schedule()->child({0})->elemAs<detail::ScheduleTreeElemContext>());
  const auto& scop = mscop.scop();

  // Make a map of the specialized scalar parameter values
  map<string, Halide::Expr> paramValues;
  {
    auto set = scop.globalParameterContext;
    for (unsigned i = 0; i < set.n_param(); i++) {
      auto val = set.plain_get_val_if_fixed(isl::dim_type::param, i);
      auto name = set.get_space().get_dim_name(isl::dim_type::param, i);
      if (!val.is_nan()) {
        paramValues[name] = static_cast<int>(val.get_num_si());
      }
    }
  }

  stringstream ss;
  emitKernelSignature(ss, specializedName, scop);
  emitThreadIdInit(ss, mscop);
  emitTensorViews(ss, scop.halide.outputs, paramValues);
  emitTensorViews(ss, scop.halide.inputs, paramValues);
  emitTmpDecl(ss, scop);
  emitPromotedArrayViewsHalide(ss, scop);
  NodeInfoMapType nodeInfoMap;
  auto collect = [&nodeInfoMap](
                     isl::ast_node n, isl::ast_build b) -> isl::ast_node {
    auto collectIteratorMaps =
        [](isl::ast_node node,
           isl::ast_build build,
           NodeInfoMapType* nodeInfoMap) -> isl::ast_node {
      auto user = node.as<isl::ast_node_user>();
      CHECK(user);
      auto expr = user.get_expr().as<isl::ast_expr_op>();
      auto stmtId = expr.get_arg(0).as<isl::ast_expr_id>().get_id();
      auto schedule = build.get_schedule();
      auto scheduleMap = isl::map::from_union_map(schedule);

      auto nodeId = isl::id(
          node.get_ctx(),
          std::string(kAstNodeIdPrefix) + std::to_string(nAstNodes()++));
      CHECK_EQ(0u, nodeInfoMap->count(nodeId)) << "entry exists: " << nodeId;

      auto& nodeInfo = (*nodeInfoMap)[nodeId];
      nodeInfo.iteratorMap = isl::pw_multi_aff(scheduleMap.reverse());
      nodeInfo.build = build;
      return node.set_annotation(nodeId);
    };

    return collectIteratorMaps(n, b, &nodeInfoMap);
  };

  checkValidIslSchedule(mscop.schedule());
  auto schedule = detail::toIslSchedule(mscop.schedule());
  auto astBuild = isl::ast_build(schedule.get_ctx());
  astBuild = astBuild.set_at_each_domain(collect);
  auto root = mscop.schedule();
  astBuild = astBuild.set_iterators(Codegen::makeLoopIterators(root));
  auto astNode = astBuild.node_from(schedule);

  AstPrinter(CodegenContext(ss, mscop, nodeInfoMap, gatherReadOnlySet(mscop)))
      .emit(astNode);
  ss << "}" << endl;

  return ss.str();
}

} // namespace polyhedral
} // namespace tc
