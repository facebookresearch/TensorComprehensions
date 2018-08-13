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
#include "tc/core/polyhedral/codegen_llvm.h"

#include <sstream>
#include <vector>

#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Analysis/TargetTransformInfo.h"
#include "llvm/Config/llvm-config.h"
#include "llvm/ExecutionEngine/ExecutionEngine.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/LegacyPassManager.h"
#include "llvm/IR/Verifier.h"
#include "llvm/Support/TargetSelect.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/Transforms/IPO.h"
#include "llvm/Transforms/IPO/PassManagerBuilder.h"

#include "Halide.h"

#include "tc/core/check.h"
#include "tc/core/constants.h"
#include "tc/core/flags.h"
#include "tc/core/halide2isl.h"
#include "tc/core/polyhedral/codegen.h"
#include "tc/core/polyhedral/schedule_isl_conversion.h"
#include "tc/core/polyhedral/scop.h"
#include "tc/core/scope_guard.h"
#include "tc/external/isl.h"

#ifndef LLVM_VERSION_MAJOR
#error LLVM_VERSION_MAJOR not set
#endif

using namespace Halide;

namespace tc {

namespace polyhedral {

using IteratorMapType = std::unordered_map<std::string, isl::ast_expr>;
using IteratorMapsType =
    std::unordered_map<isl::id, IteratorMapType, isl::IslIdIslHash>;

using StmtSubscriptExprMapType =
    std::unordered_map<isl::id, std::vector<isl::ast_expr>, isl::IslIdIslHash>;

namespace {

thread_local llvm::LLVMContext llvmCtx;

int64_t toSInt(isl::val v) {
  TC_CHECK(v.is_int());
  static_assert(sizeof(long) <= 8, "long is assumed to fit into 64bits");
  return v.get_num_si();
}

int64_t getTensorSize(isl::Set<> context, const Halide::Expr& e) {
  // isl will take care of substituting parameter values if they are known and
  // simplifying the expression.
  auto aff = halide2isl::makeIslAffFromExpr(context.get_space(), e);
  auto p = context.sample_point();
  TC_CHECK(context.is_equal(p));
  return toSInt(aff.eval(p));
}

std::vector<int64_t> getTensorSizesWithoutLeadingDim(
    const Halide::OutputImageParam& t,
    isl::Set<> context) {
  auto dims = t.dimensions();
  std::vector<int64_t> sizes;
  sizes.reserve(dims);
  for (int d = 1; d < dims; ++d) {
    Halide::Expr extent = t.parameter().extent_constraint(d);
    TC_CHECK(extent.defined())
        << "Undefined extent on input/output tensor. "
        << "Forward bounds inference should have set these\n";
    sizes.push_back(getTensorSize(context, extent));
  }
  return sizes;
}

static constexpr int kOptLevel = 3;

class CodeGen_TC : public Halide::Internal::CodeGen_X86 {
 public:
  const IteratorMapType* iteratorMap_;
  CodeGen_TC(Target t) : CodeGen_X86(t), iteratorMap_(nullptr) {}

  using CodeGen_X86::codegen;
  using CodeGen_X86::llvm_type_of;
  using CodeGen_X86::sym_get;
  using CodeGen_X86::sym_pop;
  using CodeGen_X86::sym_push;

  void init_module() override {
    const char* llvm_args[] = {"tc (LLVM argument parsing)", nullptr};
    llvm::cl::ParseCommandLineOptions(
        sizeof(llvm_args) / sizeof(*llvm_args) - 1, llvm_args);
    init_context();
    module =
        llvm::make_unique<llvm::Module>("TensorComprehensionsModule", *context);
  }

  llvm::IRBuilder<llvm::ConstantFolder, llvm::IRBuilderDefaultInserter>&
  get_builder() {
    return *builder;
  }
  void set_function(llvm::Function* function_) {
    function = function_;
  }

  llvm::Module* get_module() {
    return module.get();
  }

  llvm::Module* get_module() const {
    return module.get();
  }

  std::unique_ptr<llvm::Module> move_module() {
    return std::move(module);
  }

  Halide::Expr makeHalideExpr(isl::ast_expr expr);
  llvm::Value* codegen(isl::ast_expr expr) {
    return codegen(makeHalideExpr(expr));
  }

 protected:
  using CodeGen_X86::visit;
  void visit(const Halide::Internal::Call* call) override {
    if (call->call_type == Halide::Internal::Call::CallType::Image ||
        call->call_type == Halide::Internal::Call::CallType::Halide) {
      auto baseAddr = sym_get(call->name);
      std::vector<llvm::Value*> args(call->args.size());
      for (size_t i = 0; i < call->args.size(); i++) {
        args[i] = codegen(call->args[i]);
      }
      auto addr = builder->CreateInBoundsGEP(baseAddr, args);
      value = builder->CreateLoad(addr);
      return;
    } else if (call->is_intrinsic(tc2halide::kReductionUpdate)) {
      call->args[0].accept(this);
      return;
    } else {
      CodeGen_X86::visit(call);
    }
  }

  // The type of Variables in TC come in 2 flavors:
  // 1. the ones for which we explicitly registered LLVMIR with sym_push
  //    (i.e. the Halide way).
  // 2. the ones coming from an isl::ast_build which need to be translated
  //    through the iteratorMap_,
  void visit(const Halide::Internal::Variable* op) override {
    if ((value = sym_get(op->name, false))) {
      return;
    }
    TC_CHECK(iteratorMap_) << "IteratorMap must be set";
    value = codegen(iteratorMap_->at(op->name));

    // Generate code for type casting if necessary.
    llvm::Type* ty = llvm_type_of(op->type);
    if (value->getType() != ty) {
      if (op->type.is_int()) {
        value = builder->CreateIntCast(value, ty, true);
      } else if (op->type.is_uint()) {
        value = builder->CreateIntCast(value, ty, false);
      } else if (op->type.is_float()) {
        value = builder->CreateFPCast(value, ty);
      } else {
        TC_CHECK(false) << "Type inconsistency not handled. "
                        << "Variable " << op->name << " is " << op->type
                        << ", but its corresponding llvm::Value is "
                        << toString(value->getType()) << ".";
      }
    }
  }

 public:
  void optimize_module() {
    LOG_IF(INFO, FLAGS_llvm_dump_before_opt)
        << "[LLVM-IR] Before optimization:\n"
        << toString(module.get());

    llvm::legacy::FunctionPassManager functionPassManager(module.get());
    llvm::legacy::PassManager modulePassManager;

    std::unique_ptr<llvm::TargetMachine> targetMachine =
        Halide::Internal::make_target_machine(*module);
    modulePassManager.add(llvm::createTargetTransformInfoWrapperPass(
        targetMachine ? targetMachine->getTargetIRAnalysis()
                      : llvm::TargetIRAnalysis()));
    functionPassManager.add(llvm::createTargetTransformInfoWrapperPass(
        targetMachine ? targetMachine->getTargetIRAnalysis()
                      : llvm::TargetIRAnalysis()));

    llvm::PassManagerBuilder b;
    b.OptLevel = kOptLevel;
    b.Inliner = llvm::createFunctionInliningPass(b.OptLevel, 0, false);
    b.LoopVectorize = true;
    b.SLPVectorize = true;

    if (targetMachine) {
      targetMachine->adjustPassManager(b);
    }

    b.populateFunctionPassManager(functionPassManager);
    b.populateModulePassManager(modulePassManager);

    // Run optimization passes
    functionPassManager.doInitialization();
    for (llvm::Module::iterator i = module->begin(); i != module->end(); i++) {
      functionPassManager.run(*i);
    }

    functionPassManager.doFinalization();
    modulePassManager.run(*module);

    LOG_IF(INFO, FLAGS_llvm_dump_after_opt) << "[LLVM-IR] After optimization:\n"
                                            << toString(module.get());
  }
};

// For now we always generate int32 for induction variables and integer
// constants. Note that we may soon need to also support int64.
Halide::Expr CodeGen_TC::makeHalideExpr(isl::ast_expr expr) {
  if (auto idExpr = expr.as<isl::ast_expr_id>()) {
    return Halide::Internal::Variable::make(
        Halide::Int(32), idExpr.get_id().get_name());
  } else if (auto intExpr = expr.as<isl::ast_expr_int>()) {
    return Halide::Internal::IntImm::make(
        Halide::Int(32), toSInt(intExpr.get_val()));
  } else if (auto op = expr.as<isl::ast_expr_op>()) {
#define MAKE_UN_OP(ISL_TYPE, HALIDE_TYPE)                    \
  if (auto ty = op.as<ISL_TYPE>()) {                         \
    return HALIDE_TYPE::make(makeHalideExpr(op.get_arg(0))); \
  }

#define MAKE_BIN_OP(ISL_TYPE, HALIDE_TYPE)                             \
  if (auto ty = op.as<ISL_TYPE>()) {                                   \
    return HALIDE_TYPE::make(                                          \
        makeHalideExpr(op.get_arg(0)), makeHalideExpr(op.get_arg(1))); \
  }

    // Minus in Halide is done with a binary operator, this is a special case
    // for us.
    if (auto ty = op.as<isl::ast_op_minus>()) {
      auto a = makeHalideExpr(op.get_arg(0));
      auto zero = Halide::Internal::make_zero(a.type());
      return Halide::Internal::Sub::make(zero, a);
    }

    // clang-format off
    MAKE_BIN_OP(isl::ast_op_eq, Halide::Internal::EQ);
    MAKE_BIN_OP(isl::ast_op_le, Halide::Internal::LE);
    MAKE_BIN_OP(isl::ast_op_lt, Halide::Internal::LT);
    MAKE_BIN_OP(isl::ast_op_ge, Halide::Internal::GE);
    MAKE_BIN_OP(isl::ast_op_gt, Halide::Internal::GT);
    MAKE_BIN_OP(isl::ast_op_and, Halide::Internal::And);
    MAKE_BIN_OP(isl::ast_op_or, Halide::Internal::Or);
    MAKE_BIN_OP(isl::ast_op_min, Halide::Internal::Min);
    MAKE_BIN_OP(isl::ast_op_max, Halide::Internal::Max);
    MAKE_BIN_OP(isl::ast_op_add, Halide::Internal::Add);
    MAKE_BIN_OP(isl::ast_op_sub, Halide::Internal::Sub);
    MAKE_BIN_OP(isl::ast_op_mul, Halide::Internal::Mul);
    // clang-format on

#undef MAKE_UN_OP
#undef MAKE_BIN_OP
  }

  LOG(FATAL) << "NYI: " << expr;
  return Halide::Internal::IntImm::make(Halide::Int(32), 0);
}

class LLVMCodegen {
  void collectTensor(const Halide::OutputImageParam& t) {
    auto sizes = getTensorSizesWithoutLeadingDim(t, scop_.context());
    if (not sizes.empty()) {
      args_.emplace_back(
          makePtrToArrayType(halide_cg.llvm_type_of(t.type()), sizes));
    } else {
      args_.emplace_back(halide_cg.llvm_type_of(t.type())->getPointerTo());
    }
    argNames_.emplace_back(t.name());
  }

  void collectInputs(const std::vector<Halide::ImageParam>& inputs) {
    for (const auto& t : inputs) {
      collectTensor(t);
    }
  }

  void collectOutputs(const std::vector<Halide::OutputImageParam>& outputs) {
    for (const auto& t : outputs) {
      collectTensor(t);
    }
  }

  void collectParams(const std::vector<Halide::Internal::Parameter>& params) {
    for (const auto& p : params) {
      args_.emplace_back(halide_cg.llvm_type_of(p.type()));
      argNames_.emplace_back(p.name());
    }
  }

 public:
  LLVMCodegen(
      const Scop& scop,
      const IteratorMapsType& iteratorMaps,
      const StmtSubscriptExprMapType& stmtSubscripts)
      : scop_(scop),
        iteratorMaps_(iteratorMaps),
        stmtSubscripts_(stmtSubscripts),
        halide_cg(Halide::Target(
            Halide::Target::OSUnknown,
            Halide::Target::X86,
            64)) {
    halide_cg.set_context(llvmCtx);

    halide_cg.init_module();
  }

  // This creates a signature of the form:
  //    input_data_types, output_data_types, parameters
  llvm::BasicBlock* createSignature(
      const std::vector<Halide::ImageParam>& inputs,
      const std::vector<Halide::OutputImageParam>& outputs,
      const std::vector<Halide::Internal::Parameter>& params,
      const std::string& fname) {
    auto size = inputs.size() + outputs.size();
    args_.reserve(size);
    argNames_.reserve(size);

    collectInputs(inputs);
    collectOutputs(outputs);
    collectParams(params);

    auto* functionType =
        llvm::FunctionType::get(llvm::Type::getVoidTy(llvmCtx), args_, false);

    auto* function = llvm::Function::Create(
        functionType,
        llvm::Function::ExternalLinkage,
        fname,
        halide_cg.get_module());
    halide_cg.set_function(function);

    size_t idx = 0;
    for (auto& arg : function->args()) {
      halide_cg.sym_push(argNames_.at(idx), &arg);
      arg.setName(argNames_.at(idx++));
    }

    // Only pointer arguments can be NoAlias and NonNull
    for (auto it = function->arg_begin(),
              end = function->arg_begin() + inputs.size() + outputs.size();
         it != end;
         ++it) {
      it->addAttr(llvm::Attribute::NoAlias);
      it->addAttr(llvm::Attribute::NonNull);
    }
    // Only input arguments are ReadOnly
    for (auto it = function->arg_begin(), end = it + inputs.size(); it != end;
         ++it) {
      it->addAttr(llvm::Attribute::ReadOnly);
    }

    return llvm::BasicBlock::Create(llvmCtx, "entry", function);
  }

  // This is the main entry point to emit pieces of LLVM IR
  // LLVM IR insertion is stateful, configured by SetInsertPoint
  // We make this an explicit parameter to avoid implicit conventions
  // All TC IR builder methods take an explicit insertionPoint.
  // The invariant in all emit* (except for emitAst) is that:
  //    TC_CHECK_EQ(halide_cg.get_builder().GetInsertBlock(), insertionPoint);
  llvm::BasicBlock* emitAst(
      isl::ast_node node,
      llvm::BasicBlock* insertionPoint) {
    halide_cg.get_builder().SetInsertPoint(insertionPoint);
    if (auto forNode = node.as<isl::ast_node_for>()) {
      return emitFor(forNode, insertionPoint);
    } else if (auto userNode = node.as<isl::ast_node_user>()) {
      return emitStmt(userNode, insertionPoint);
    } else if (auto blockNode = node.as<isl::ast_node_block>()) {
      llvm::BasicBlock* curBB = insertionPoint;
      for (auto child : blockNode.get_children()) {
        curBB = emitAst(child, curBB);
      }
      return curBB;
    } else {
      if (auto cond = node.as<isl::ast_node_if>()) {
        return emitIf(cond, insertionPoint);
      } else {
        LOG(FATAL) << "NYI " << node << std::endl;
      }
    }
    return nullptr;
  }

 private:
  llvm::Type* makePtrToArrayType(
      llvm::Type* baseTy,
      const std::vector<int64_t>& sizes) {
    TC_CHECK_GE(sizes.size(), 1u);
    TC_CHECK(baseTy);
    llvm::Type* arrTy = llvm::ArrayType::get(baseTy, sizes.back());
    for (auto s = sizes.rbegin() + 1; s != sizes.rend(); ++s) {
      arrTy = llvm::ArrayType::get(arrTy, *s);
    }
    return arrTy->getPointerTo();
  }

  llvm::BasicBlock* emitIf(
      isl::ast_node_if node,
      llvm::BasicBlock* insertionPoint) {
    TC_CHECK_EQ(halide_cg.get_builder().GetInsertBlock(), insertionPoint);
    auto* function = insertionPoint->getParent();

    llvm::Value* condVal = halide_cg.codegen(node.get_cond());
    auto* thenBB = llvm::BasicBlock::Create(llvmCtx, "then", function);
    // Recursively emit "then" in a new thenBB
    auto innerBB = emitAst(node.get_then(), thenBB);

    // outer -> thenBB
    halide_cg.get_builder().SetInsertPoint(insertionPoint);
    // outer ---------> if_exit
    // TODO: When we support "else", go to elseBB instead of exit
    auto* exit = llvm::BasicBlock::Create(llvmCtx, "if_exit", function);
    halide_cg.get_builder().CreateCondBr(condVal, thenBB, exit);

    //          then -> if_exit
    halide_cg.get_builder().SetInsertPoint(innerBB);
    halide_cg.get_builder().CreateBr(exit);

    // Else is often empty in the absence of full tile extraction
    if (node.has_else()) {
      LOG(FATAL) << "NYI: else conditional branch";
      return exit;
    }

    return exit;
  }

  llvm::BasicBlock* emitFor(
      isl::ast_node_for node,
      llvm::BasicBlock* insertionPoint) {
    TC_CHECK_EQ(halide_cg.get_builder().GetInsertBlock(), insertionPoint);
    auto* function = insertionPoint->getParent();
    auto* headerBB = llvm::BasicBlock::Create(llvmCtx, "loop_header", function);
    auto* loopBodyBB = llvm::BasicBlock::Create(llvmCtx, "loop_body", function);
    auto* loopLatchBB =
        llvm::BasicBlock::Create(llvmCtx, "loop_latch", function);
    auto* loopExitBB = llvm::BasicBlock::Create(llvmCtx, "loop_exit", function);

    halide_cg.get_builder().CreateBr(headerBB);

    llvm::PHINode* phi = nullptr;
    auto iterator = node.get_iterator().as<isl::ast_expr_id>().get_id();

    // Loop Header
    {
      auto initVal = halide_cg.codegen(node.get_init());
      halide_cg.get_builder().SetInsertPoint(headerBB);
      phi = halide_cg.get_builder().CreatePHI(
          initVal->getType(), 2, iterator.get_name());
      halide_cg.sym_push(iterator.get_name(), phi);
      phi->addIncoming(initVal, insertionPoint);

      auto cond = halide_cg.codegen(node.get_cond());
      halide_cg.get_builder().CreateCondBr(cond, loopBodyBB, loopExitBB);
    }

    // Create Body
    {
      auto* currentBB = emitAst(node.get_body(), loopBodyBB);
      halide_cg.get_builder().SetInsertPoint(currentBB);
      halide_cg.get_builder().CreateBr(loopLatchBB);
    }

    // Create Latch
    {
      halide_cg.get_builder().SetInsertPoint(loopLatchBB);
      auto incVal = halide_cg.codegen(node.get_inc());
      phi->addIncoming(
          halide_cg.get_builder().CreateAdd(phi, incVal), loopLatchBB);
      halide_cg.get_builder().CreateBr(headerBB);
    }

    halide_cg.sym_pop(iterator.get_name());
    return loopExitBB;
  }

  llvm::BasicBlock* emitStmt(
      isl::ast_node_user node,
      llvm::BasicBlock* insertionPoint) {
    TC_CHECK_EQ(halide_cg.get_builder().GetInsertBlock(), insertionPoint);
    isl::ast_expr_op usrExp = node.get_expr().as<isl::ast_expr_op>();
    auto id = usrExp.get_arg(0).as<isl::ast_expr_id>().get_id();
    auto provide = scop_.halide.statements.at(id);
    auto op = provide.as<Halide::Internal::Provide>();
    TC_CHECK(op) << "Expected a Provide node: " << provide << '\n';
    TC_CHECK(op->values.size() == 1)
        << "Multi-valued Provide: " << Halide::Internal::Stmt(provide) << "\n";
    auto arrayName = op->name;
    const auto& subscripts = stmtSubscripts_.at(id);
    llvm::SmallVector<llvm::Value*, 5> subscriptValues;

    halide_cg.iteratorMap_ = &iteratorMaps_.at(id);
    for (const auto& subscript : subscripts) {
      subscriptValues.push_back(halide_cg.codegen(subscript));
    }

    auto destAddr = halide_cg.get_builder().CreateInBoundsGEP(
        halide_cg.sym_get(arrayName), subscriptValues);

    llvm::Value* rhs = halide_cg.codegen(op->values[0]);
    halide_cg.get_builder().CreateStore(rhs, destAddr);
    // We must return halide_cg.get_builder().GetInsertBlock() because
    // Halide does not adhere to our conventions and when it emits multiple
    // blocks things may go haywire.
    return halide_cg.get_builder().GetInsertBlock();
  }

 public:
  std::string str() const {
    return toString(halide_cg.get_module());
  }

 private:
  const Scop& scop_;
  const IteratorMapsType& iteratorMaps_;
  const StmtSubscriptExprMapType& stmtSubscripts_;

  std::vector<llvm::Type*> args_;
  std::vector<std::string> argNames_;

 public:
  CodeGen_TC halide_cg;
};

struct IslCodegenRes {
  IteratorMapsType iteratorMaps;
  StmtSubscriptExprMapType stmtSubscripts;
  isl::ast_node astNode;
};

isl::ast_node collectIteratorMaps(
    isl::ast_node node,
    isl::ast_build build,
    IteratorMapsType& iteratorMaps,
    const Scop& scop,
    StmtSubscriptExprMapType& stmtSubscripts) {
  auto user = node.as<isl::ast_node_user>();
  TC_CHECK(user);
  auto expr = user.get_expr().as<isl::ast_expr_op>();
  auto schedule = build.get_schedule();
  auto scheduleMap = isl::map::from(schedule);

  auto stmtId = expr.get_arg(0).as<isl::ast_expr_id>().get_id();
  TC_CHECK_EQ(0u, iteratorMaps.count(stmtId)) << "entry exists: " << stmtId;
  auto iteratorMap = isl::pw_multi_aff(scheduleMap.reverse());
  auto tuple = scop.halide.domains.at(stmtId).tuple;
  auto& stmtIteratorMap = iteratorMaps[stmtId];
  for (int i = 0; i < tuple.size(); ++i) {
    auto expr = build.expr_from(iteratorMap.get_pw_aff(i));
    stmtIteratorMap.emplace(tuple.get_id(i).get_name(), expr);
  }
  auto& subscripts = stmtSubscripts[stmtId];
  auto provide =
      scop.halide.statements.at(stmtId).as<Halide::Internal::Provide>();
  for (auto e : provide->args) {
    const auto& map = iteratorMap;
    auto aff = scop.makeIslAffFromStmtExpr(stmtId, e);
    auto pulled = isl::pw_aff(aff).pullback(map);
    TC_CHECK_EQ(pulled.n_piece(), 1);
    subscripts.push_back(build.expr_from(pulled));
  }
  return node.set_annotation(stmtId);
}

IslCodegenRes codegenISL(const Scop& scop) {
  IteratorMapsType iteratorMaps;
  StmtSubscriptExprMapType stmtSubscripts;
  auto collect = [&iteratorMaps, &scop, &stmtSubscripts](
                     isl::ast_node n, isl::ast_build b) -> isl::ast_node {
    auto& uv = iteratorMaps;
    return collectIteratorMaps(n, b, uv, scop, stmtSubscripts);
  };

  auto schedule = detail::toIslSchedule(scop.scheduleRoot());
  auto astBuild = isl::ast_build(schedule.get_ctx());
  astBuild = astBuild.set_at_each_domain(collect);
  auto root = scop.scheduleRoot();
  astBuild = astBuild.set_iterators(Codegen::makeLoopIterators(root));
  auto astNode = astBuild.node_from(schedule);
  return {
      std::move(iteratorMaps), std::move(stmtSubscripts), std::move(astNode)};
}

} // namespace

std::unique_ptr<llvm::Module> emitLLVMKernel(
    const std::string& specializedName,
    const Scop& scop,
    const llvm::DataLayout& dataLayout) {
  auto islCg = codegenISL(scop);
  LLVMCodegen cg(scop, islCg.iteratorMaps, islCg.stmtSubscripts);
  cg.halide_cg.get_module()->setDataLayout(dataLayout);
  cg.halide_cg.get_module()->setTargetTriple(
      llvm::EngineBuilder().selectTarget()->getTargetTriple().str());
  auto entry = cg.createSignature(
      scop.halide.inputs,
      scop.halide.outputs,
      scop.halide.params,
      specializedName);
  auto exit = cg.emitAst(islCg.astNode, entry);
  cg.halide_cg.get_builder().SetInsertPoint(exit);
  cg.halide_cg.get_builder().CreateRetVoid();

  TC_CHECK(!llvm::verifyModule(*cg.halide_cg.get_module()))
      << "LLVM generated module is invalid." << cg.str().c_str();

  cg.halide_cg.optimize_module();
  return cg.halide_cg.move_module();
}

} // namespace polyhedral
} // namespace tc
