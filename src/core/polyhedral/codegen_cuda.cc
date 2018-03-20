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
#include "tc/core/halide2isl.h"
#include "tc/core/islpp_wrap.h"
#include "tc/core/libraries.h"
#include "tc/core/polyhedral/codegen.h"
#include "tc/core/polyhedral/codegen_cuda.h"
#include "tc/core/polyhedral/cuda/cuda_mapping_types.h"
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
  void emitFor(isl::ast_node node);
  void emitIf(isl::ast_node node);
  void emitStmt(isl::ast_node node);
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
string emitTypedTensorName(Halide::OutputImageParam t) {
  stringstream ss;
  ss << t.type() << "* " << makePointerName(t.name());
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
    res.push_back(emitTypedTensorName(t));
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
    const map<string, Halide::Expr>& paramValues) {
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
  ss << p.type() << " (*" << p.name() << ")" << ssViewType.str();
  ss << " = ";
  ss << "reinterpret_cast<" << p.type() << " (*)" << ssViewType.str() << ">";
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
    emitTensorView(ss, p, paramValues);
  }
}

void AstPrinter::emitFor(isl::ast_node node) {
  WS ws;
  context_.ss << ws.tab();
  string iter = node.for_get_iterator().to_C_str();
  context_.ss << "for (int " << iter << " = " << node.for_get_init().to_C_str()
              << "; " << node.for_get_cond().to_C_str() << "; " << iter
              << " += " << node.for_get_inc().to_C_str() << ") {" << endl;
  emitAst(node.for_get_body());
  context_.ss << ws.tab() << "}" << endl;
}

void AstPrinter::emitIf(isl::ast_node node) {
  WS ws;
  context_.ss << ws.tab();
  context_.ss << "if (" << node.if_get_cond().to_C_str() << ") {" << endl;
  emitAst(node.if_get_then());
  context_.ss << ws.tab() << "}";
  if (node.if_has_else()) {
    context_.ss << " else {" << endl;
    emitAst(node.if_get_else());
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

namespace {
// Compute the range of parameter values in a given set.  Both sides of the
// range are inclusive.
std::pair<isl::val, isl::val> computeParamRange(isl::set domain, int pos) {
  // Coerce the set to the shape [N] -> {[i]: only N here }
  domain = domain.params().from_params();
  domain = domain.project_out(isl::dim_type::param, 0, pos);
  domain = domain.project_out(
      isl::dim_type::param, 1, domain.dim(isl::dim_type::param) - 1);
  domain = domain.insert_dims(isl::dim_type::set, 0, 1);

  // Connect parameter to a set dimension [N] -> {[i]: i = N and ...}
  auto lspace = isl::local_space(domain.get_space());
  auto paramAff = isl::aff(lspace, isl::dim_type::param, 0);
  auto varAff = isl::aff(lspace, isl::dim_type::set, 0);
  domain = domain & (isl::aff_set(paramAff) == varAff);

  // Remove the remaining parameter to move its constraints to the set dimension
  domain = domain.project_out(isl::dim_type::param, 0, 1);

  // Get min and max.
  auto lower = domain.dim_min(0);
  auto upper = domain.dim_max(0);

  // Compute the range
  CHECK(lower.is_cst() && upper.is_cst())
      << "expected constant lower and upper bounds";

  // Without parameters at all, we must have a single piece in the bound PA.
  auto lowerPA = isl::PA(lower);
  auto upperPA = isl::PA(upper);
  CHECK(lowerPA.size() == 1 && upperPA.size() == 1);

  return std::make_pair(
      lowerPA[0].second.get_constant_val(),
      upperPA[0].second.get_constant_val());
}

// Given the iteratorMaps, whose domain was affected by the mapping filters, in
// the provided context, compute the range of thread mapping parameters.  If
// the statement is not mapped to some threads, they will not appear in the
// result.
std::unordered_map<isl::id, long, isl::IslIdIslHash> activeThreadsInBlock(
    const CodegenStatementContext& context) {
  auto iterMap = context.iteratorMap();
  auto dom =
      iterMap.domain()
          .intersect_params(context.mappedScop.scop().globalParameterContext)
          .params()
          .from_params();

  USING_MAPPING_SHORT_NAMES(BX, BY, BZ, TX, TY, TZ);
  std::vector<isl::id> threadIds{TX, TY, TZ};
  std::unordered_map<isl::id, long, isl::IslIdIslHash> activeThreads;

  for (auto id : threadIds) {
    int pos = dom.find_dim_by_id(isl::dim_type::param, id);
    if (pos < 0) {
      continue;
    }
    auto range = computeParamRange(dom, pos);
    CHECK_EQ(range.first.get_den_si(), 1) << "fractional parameters?";
    CHECK_EQ(range.second.get_den_si(), 1) << "fractional parameters?";
    CHECK_EQ(range.first.get_num_si(), 0)
        << "NYI: active threads starting not from 0";

    activeThreads[id] =
        range.second.get_num_si() - range.first.get_num_si() + 1;
  }
  return activeThreads;
}

// Given the iteratorMaps, whose domain was affected by the mapping filters, in
// the provided context, compute the range of thread mapping parameters.  If
// the statement is not mapped to some threads, they will _still appear_ in the
// result with the range 1.
std::array<long, 3> activeThreadsInBlockWithDefaults(
    const CodegenStatementContext& context) {
  auto active = activeThreadsInBlock(context);
  std::array<long, 3> result;

  USING_MAPPING_SHORT_NAMES(BX, BY, BZ, TX, TY, TZ);
  std::vector<isl::id> threadIds{TX, TY, TZ};
  int i = 0;
  for (auto id : threadIds) {
    if (active.count(id) != 1) {
      result[i] = MappingId::unmapped;
    } else {
      result[i] = active[id];
    }
    ++i;
  }
  return result;
}
} // namespace

// Emit a cross-thread tree reduce.
// For now this is only expected to work with threadIdx.x.
void emitTreeSyncCall(
    isl::id id,
    isl::id reductionUpdateNodeId,
    const CodegenStatementContext& context) {
  CHECK_EQ(1, context.scop().treeSyncUpdateMap.count(id));
  isl::id updateId = context.scop().treeSyncUpdateMap.at(id);

  // Halide reduction.
  auto provide = context.scop()
                     .halide.statements.at(updateId)
                     .as<Halide::Internal::Provide>();

  USING_MAPPING_SHORT_NAMES(BX, BY, BZ, TX, TY, TZ);
  std::array<size_t, 3> dims = {TX.mappingSize(context.mappedScop.numThreads),
                                TY.mappingSize(context.mappedScop.numThreads),
                                TZ.mappingSize(context.mappedScop.numThreads)};
  std::array<long, 3> active = activeThreadsInBlockWithDefaults(context);

  for (int i = 0; i < 3; ++i) {
    if (active[i] < dims[i]) {
      LOG(INFO) << "Reduction statement " << updateId << " mapped to "
                << dims[i] << "threads along dim: " << i << "but only "
                << active[i] << " are non-empty";
    }
  }

  context.ss << tc::code::cuda::kCUBReductionName;

  // Template mapping dimension
  context.ss << "<";
  context.ss << active[0];
  context.ss << ",";
  context.ss << active[1];
  context.ss << ",";
  context.ss << active[2];
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
  CHECK_EQ(assoc.pattern.identities.size(), 1);
  detail::emitHalideExpr(assoc.pattern.identities[0], statementContext);
  context.ss << ";" << endl;
}

void emitCopyStmt(const CodegenStatementContext& context) {
  using detail::emitDirectSubscripts;

  auto stmtId = context.statementId();

  // Casting to map for more advanced projection functionality.  No information
  // loss expected.
  auto map = isl::map::from(context.iteratorMap());
  auto promoted = isl::pw_multi_aff(map.range_factor_range());
  auto original =
      isl::pw_multi_aff(map.range_factor_domain().range_factor_range());
  auto isRead = stmtId.get_name() == kReadIdName;
  auto originalName = original.get_tuple_id(isl::dim_type::out).get_name();
  auto promotedName = promoted.get_tuple_id(isl::dim_type::out).get_name();

  if (isRead) {
    context.ss << promotedName;
    emitDirectSubscripts(promoted, context);
    context.ss << " = " << originalName;
    emitDirectSubscripts(original, context);
  } else {
    context.ss << originalName;
    emitDirectSubscripts(original, context);
    context.ss << " = " << promotedName;
    emitDirectSubscripts(promoted, context);
  }
  context.ss << ";" << std::endl;
}

void AstPrinter::emitStmt(isl::ast_node node) {
  isl::ast_expr usrExp = node.user_get_expr();
  auto stmtId = usrExp.get_op_arg(0).get_id();
  auto nodeId = node.get_annotation();
  auto statementContext = CodegenStatementContext(context_, nodeId);

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
    CHECK_EQ(context_.iteratorMaps.count(nodeId), 1)
        << "no iterator remapping for op " << nodeId;
    emitReductionUpdate(stmtId, statementContext);
    reductionUpdateNodeId_ = nodeId;
  } else if (context_.scop().isSyncId(stmtId)) {
    context_.ss << "__syncthreads();" << std::endl;
  } else if (
      stmtId.get_name() == kReadIdName || stmtId.get_name() == kWriteIdName) {
    emitCopyStmt(statementContext);
  } else { // regular statement
    CHECK_EQ(context_.iteratorMaps.count(nodeId), 1)
        << "no iterator remapping for op " << nodeId;
    auto mappedStmtId =
        context_.iteratorMaps.at(nodeId).get_tuple_id(isl::dim_type::out);
    CHECK_EQ(stmtId, mappedStmtId)
        << "statement ids in expr (" << stmtId << ") and in iteratorMaps ("
        << mappedStmtId << ") do not match";
    emitUserStmt(stmtId, CodegenStatementContext(context_, nodeId));
  }
}

void AstPrinter::emitAst(isl::ast_node node) {
  switch (node.get_type()) {
    case isl::ast_node_type::_for:
      emitFor(node);
      break;
    case isl::ast_node_type::_if:
      emitIf(node);
      break;
    case isl::ast_node_type::block:
      for (auto child : node.block_get_children()) {
        emitAst(child);
      }
      break;
    case isl::ast_node_type::mark:
      CHECK(false) << "mark";
      // emitAst(node.mark_get_node());
      break;
    case isl::ast_node_type::user:
      emitStmt(node);
      break;
    default:
      LOG(FATAL) << "NYI " << node << endl;
      return;
  }
}

} // namespace

namespace detail {

std::string toString(isl::aff subscript) {
  stringstream ss;
  // TODO: isl printer is not exported
  isl_printer* prn = isl_printer_to_str(subscript.get_ctx().get());
  prn = isl_printer_set_output_format(prn, ISL_FORMAT_C);
  prn = isl_printer_print_aff(prn, subscript.get());
  char* str = isl_printer_get_str(prn);
  ss << str;
  free(str);
  isl_printer_free(prn);
  return ss.str();
}

std::string toString(isl::pw_aff subscript) {
  isl::aff subscriptAff = isl::null<isl::aff>();
  subscript.foreach_piece([&](isl::set domain, isl::aff aff) {
    CHECK(!subscriptAff.get()) << "expected one piece";
    subscriptAff = aff;
  });

  return toString(subscriptAff);
}

isl::pw_aff makeAffFromMappedExpr(
    const Halide::Expr& expr,
    const CodegenStatementContext& context) {
  auto space = context.iteratorMap().get_space().range();
  // We only expect this to be called on encountering a free
  // variable. Compound expressions should be emitted as Halide.
  CHECK(expr.as<Halide::Internal::Variable>());
  auto aff = halide2isl::makeIslAffFromExpr(space, expr);
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

isl::map findScheduleByStmtId(isl::union_map schedule, isl::id stmtId) {
  for (auto s : isl::UnionAsVector<isl::union_map>(schedule)) {
    if (s.get_tuple_id(isl::dim_type::in) == stmtId) {
      return s;
    }
  }
  CHECK(false) << "could not find schedule for " << stmtId << " in "
               << schedule;
  return isl::map();
}

isl::multi_aff makeMultiAffAccess(
    isl::id tensorId,
    const std::vector<Halide::Expr>& subscripts,
    const CodegenStatementContext& context) {
  CHECK_NE(subscripts.size(), 0) << "cannot build subscript aff for a scalar";

  auto domainSpace = findDomainSpaceById(context);
  auto tensorSpace = domainSpace.params().set_from_params().add_dims(
      isl::dim_type::set, subscripts.size());
  tensorSpace = tensorSpace.set_tuple_id(isl::dim_type::set, tensorId);
  auto space = domainSpace.map_from_domain_and_range(tensorSpace);

  auto ma = isl::multi_aff::zero(space);
  for (size_t i = 0; i < subscripts.size(); ++i) {
    ma = ma.set_aff(
        i, halide2isl::makeIslAffFromExpr(domainSpace, subscripts[i]));
  }
  return ma;
}

void emitHalideExpr(
    const Halide::Expr& e,
    const CodegenStatementContext& context,
    const map<string, string>& substitutions) {
  class EmitHalide : public Halide::Internal::IRPrinter {
    using Halide::Internal::IRPrinter::visit;
    void visit(const Halide::Internal::Variable* op) {
      // This is probably needlessly indirect, given that we just have
      // a name to look up somewhere.
      auto pwAff = tc::polyhedral::detail::makeAffFromMappedExpr(
          Halide::Expr(op), context);
      context.ss << tc::polyhedral::detail::toString(pwAff);
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

  CHECK_EQ(context.scop().halide.accesses.count(node), 1)
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
    context.ss << name;
    for (auto e : subscripts) {
      context.ss << "[";
      emitHalideExpr(e, context);
      context.ss << "]";
    }
    return;
  }

  auto tensorId =
      context.scop().promotedDecls().at(promotionInfo.groupId).tensorId;

  // Here and below in comments: D = domain, O = original tensor, P = promoted
  // tensor, S = partial schedule, A = AST loops;
  // MA = multi_aff, PMA = pw_multi_aff
  auto access =
      makeMultiAffAccess(tensorId, subscripts, context); // MA :: D -> O
  auto promotion = promotionInfo.group->promotion(); // MA :: [S -> O] -> P
  promotion = promotion.set_tuple_id(isl::dim_type::out, promotionInfo.groupId);
  auto iteratorMap = context.iteratorMap(); // PMA :: A -> D
  auto schedule = findScheduleByStmtId(
      promotionInfo.outerSchedule,
      context.statementId()); // map :: D -> S

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

  auto pma = isl::PMA(astToPromoted);
  CHECK_EQ(pma.size(), 1) << "expected one piece, got " << astToPromoted;
  auto ma = isl::MA(pma[0].second);
  context.ss << promotionInfo.groupId.get_name();
  for (int i = 0; i < ma.size(); ++i) {
    context.ss << "[" << toString(ma[i]) << "]";
  }
}

void emitDirectSubscripts(
    isl::pw_multi_aff subscripts,
    const CodegenStatementContext& context) {
  auto mpa = isl::multi_pw_aff(subscripts); // this conversion is safe
  for (auto pa : isl::MPA(mpa)) {
    context.ss << "[";
    context.ss << toString(pa.pa);
    context.ss << "]";
  }
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
    ss << "__shared__ " << t << " " << viewName;
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
    for (int i = 0; i < set.n_param(); i++) {
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
  // TODO: improve support for C++ callbacks in isl bindings generator
  // see https://github.com/PollyLabs/isl/issues/24
  // This cannot be done via islpp_wrap because the callback is stored for
  // later use while islpp_wrap passes a pointer to a stack-allocated
  // object to the call as a means to support capturing lambdas.
  auto collect =
      [](isl_ast_node* n, isl_ast_build* b, void* u) -> isl_ast_node* {
    auto collectIteratorMaps =
        [](isl::ast_node node,
           isl::ast_build build,
           IteratorMapsType* iteratorMaps) -> isl::ast_node {
      auto expr = node.user_get_expr();
      auto stmtId = expr.get_op_arg(0).get_id();
      // We rename loop-related dimensions manually.
      auto schedule = build.get_schedule();
      auto scheduleSpace = build.get_schedule_space();
      auto scheduleMap = isl::map::from_union_map(schedule);

      auto nodeId = isl::id(
          node.get_ctx(),
          std::string(kAstNodeIdPrefix) + std::to_string(nAstNodes()++));
      CHECK_EQ(0, iteratorMaps->count(nodeId)) << "entry exists: " << nodeId;
      CHECK_EQ(
          scheduleMap.dim(isl::dim_type::out),
          scheduleSpace.dim(isl::dim_type::set));
      for (int i = 0; i < scheduleSpace.dim(isl::dim_type::set); ++i) {
        scheduleMap = scheduleMap.set_dim_id(
            isl::dim_type::out,
            i,
            scheduleSpace.get_dim_id(isl::dim_type::set, i));
      }

      auto iteratorMap = isl::pw_multi_aff(scheduleMap.reverse());
      iteratorMaps->emplace(nodeId, iteratorMap);
      return node.set_annotation(nodeId);
    };

    auto uv = static_cast<IteratorMapsType*>(u);
    return collectIteratorMaps(isl::manage(n), isl::manage_copy(b), uv)
        .release();
  };

  auto bands = detail::ScheduleTree::collect(
      mscop.schedule(), detail::ScheduleTreeType::Band);
  int maxDepth = 0;
  for (auto const& node : bands) {
    auto bandElem = node->elemAs<detail::ScheduleTreeElemBand>();
    auto depth = node->scheduleDepth(mscop.schedule()) +
        bandElem->mupa_.dim(isl::dim_type::set);
    if (depth > maxDepth) {
      maxDepth = depth;
    }
  }

  checkValidIslSchedule(mscop.schedule());
  auto schedule = detail::toIslSchedule(mscop.schedule());
  auto ctx = schedule.get_ctx();
  IteratorMapsType iteratorMaps;
  auto astBuild = isl::ast_build(schedule.get_ctx());
  astBuild = isl::manage(isl_ast_build_set_at_each_domain(
      astBuild.release(), collect, &iteratorMaps));
  astBuild = astBuild.set_iterators(Codegen::makeLoopIterators(ctx, maxDepth));
  auto astNode = astBuild.node_from(schedule);
  AstPrinter(CodegenContext(ss, mscop, iteratorMaps)).emit(astNode);
  ss << "}" << endl;

  return ss.str();
}

} // namespace polyhedral
} // namespace tc
