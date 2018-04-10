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

#include "isl/ast.h"

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

#ifdef TAPIR_VERSION_MAJOR
#include "llvm/Transforms/Tapir/CilkABI.h"
#endif

using namespace Halide;

namespace tc {

namespace polyhedral {

using IteratorMapType = std::unordered_map<std::string, isl::ast_expr>;
using IteratorMapsType =
    std::unordered_map<isl::id, IteratorMapType, isl::IslIdIslHash>;

using IteratorLLVMValueMapType =
    std::unordered_map<isl::id, llvm::Value*, isl::IslIdIslHash>;

using StmtSubscriptExprMapType =
    std::unordered_map<isl::id, std::vector<isl::ast_expr>, isl::IslIdIslHash>;

namespace {

thread_local llvm::LLVMContext llvmCtx;

int64_t toSInt(isl::val v) {
  CHECK(v.is_int());
  static_assert(sizeof(long) <= 8, "long is assumed to fit into 64bits");
  return v.get_num_si();
}

llvm::Value* getLLVMConstantSignedInt64(int64_t v) {
  return llvm::ConstantInt::get(llvm::Type::getInt64Ty(llvmCtx), v, true);
}

int64_t IslExprToSInt(isl::ast_expr e) {
  CHECK(isl_ast_expr_get_type(e.get()) == isl_ast_expr_type::isl_ast_expr_int);
  return toSInt(isl::manage(isl_ast_expr_get_val(e.get())));
}

int64_t islIdToInt(isl::ast_expr e, isl::set context) {
  CHECK(isl_ast_expr_get_type(e.get()) == isl_ast_expr_type::isl_ast_expr_id);
  auto space = context.get_space();
  isl::aff param(isl::aff::param_on_domain_space(space, e.get_id()));
  auto p = context.sample_point();
  CHECK(context.is_equal(p));
  return toSInt(param.eval(p));
}

int64_t getTensorSize(isl::set context, const Halide::Expr& e) {
  // isl will take care of substituting parameter values if they are known and
  // simplifying the expression.
  auto aff = halide2isl::makeIslAffFromExpr(context.get_space(), e);
  auto p = context.sample_point();
  CHECK(context.is_equal(p));
  return toSInt(aff.eval(p));
}

std::vector<int64_t> getTensorSizesWithoutLeadingDim(
    const Halide::OutputImageParam& t,
    isl::set context) {
  auto dims = t.dimensions();
  std::vector<int64_t> sizes;
  sizes.reserve(dims);
  for (size_t d = 1; d < dims; ++d) {
    Halide::Expr extent = t.parameter().extent_constraint(d);
    CHECK(extent.defined())
        << "Undefined extent on input/output tensor. Forward bounds inference should have set these\n";
    sizes.push_back(getTensorSize(context, extent));
  }
  return sizes;
}

class IslAstExprInterpeter {
  isl::set context_;

 public:
  IslAstExprInterpeter(isl::set context) : context_(context){};

  int64_t interpret(isl::ast_expr e) {
    switch (isl_ast_expr_get_type(e.get())) {
      case isl_ast_expr_type::isl_ast_expr_int:
        return IslExprToSInt(e);
      case isl_ast_expr_type::isl_ast_expr_id:
        return islIdToInt(e, context_);
      case isl_ast_expr_type::isl_ast_expr_op:
        return interpretOp(e);
      default:
        CHECK(false) << "NYI";
        return 0; // avoid warning
    }
  };

 private:
  int64_t interpretOp(isl::ast_expr e) {
    switch (e.get_op_n_arg()) {
      case 1:
        return interpretUnaryOp(e);
      case 2:
        return interpretBinaryOp(e);
      default:
        CHECK(false) << "NYI: " << e;
        return 0; // avoid warning
    }
  }

  int64_t interpretBinaryOp(isl::ast_expr e) {
    auto left = interpret(e.get_op_arg(0));
    auto right = interpret(e.get_op_arg(1));
    switch (e.get_op_type()) {
      case isl::ast_op_type::add:
        return left + right;
      case isl::ast_op_type::sub:
        return left - right;
      default:
        CHECK(false) << "NYI: " << e;
        return 0; // avoid warning
    }
  }

  int64_t interpretUnaryOp(isl::ast_expr e) {
    auto val = interpret(e.get_op_arg(0));
    switch (e.get_op_type()) {
      case isl::ast_op_type::minus:
        return -val;
      default:
        CHECK(false) << "NYI";
        return 0; // avoid warning
    }
  }
};

static constexpr int kOptLevel = 3;

class CodeGen_TC : public Halide::Internal::CodeGen_X86 {
 public:
  const IteratorMapType* iteratorMap_;
  CodeGen_TC(Target t) : CodeGen_X86(t) {}

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

  // Convert an isl AST expression into an llvm::Value.
  // Only expressions that consist of a pure identifier or
  // a pure integer constant are currently supported.
  llvm::Value* getValue(isl::ast_expr expr);

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
    } else if (
        call->is_intrinsic(tc2halide::kReductionInit) ||
        call->is_intrinsic(tc2halide::kReductionUpdate)) {
      call->args[0].accept(this);
      return;
    } else {
      CodeGen_X86::visit(call);
    }
  }
  void visit(const Halide::Internal::Variable* op) override {
    value = getValue(iteratorMap_->at(op->name));
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
#ifdef TAPIR_VERSION_MAJOR
    b.tapirTarget = new llvm::CilkABI();
#endif
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

llvm::Value* CodeGen_TC::getValue(isl::ast_expr expr) {
  switch (isl_ast_expr_get_type(expr.get())) {
    case isl_ast_expr_type::isl_ast_expr_id:
      return sym_get(expr.get_id().get_name());
    case isl_ast_expr_type::isl_ast_expr_int: {
      auto val = isl::manage(isl_ast_expr_get_val(expr.get()));
      return getLLVMConstantSignedInt64(toSInt(val));
    }
    default:
      LOG(FATAL) << "NYI";
      return nullptr;
  }
}

class LLVMCodegen {
  llvm::Type* convertTensorToType(const Halide::OutputImageParam& t) {
    auto sizes =
        getTensorSizesWithoutLeadingDim(t, scop_.globalParameterContext);
    if (not sizes.empty()) {
      return makePtrToArrayType(halide_cg.llvm_type_of(t.type()), sizes);
    } else {
      return halide_cg.llvm_type_of(t.type())->getPointerTo();
    }
  }

  void collectTensor(const Halide::OutputImageParam& t) {
    args_.emplace_back(convertTensorToType(t));
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

  void createSignature(
      const std::vector<Halide::ImageParam>& inputs,
      const std::vector<Halide::OutputImageParam>& outputs,
      const std::string& fname) {
    auto size = inputs.size() + outputs.size();
    args_.reserve(size);
    argNames_.reserve(size);

    collectInputs(inputs);
    collectOutputs(outputs);

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

    for (auto it = function->arg_begin(), end = function->arg_end(); it != end;
         ++it) {
      it->addAttr(llvm::Attribute::NoAlias);
      it->addAttr(llvm::Attribute::NonNull);
    }
    for (auto it = function->arg_begin(), end = it + inputs.size(); it != end;
         ++it) {
      it->addAttr(llvm::Attribute::ReadOnly);
    }

    auto entryBB_ = llvm::BasicBlock::Create(llvmCtx, "entry", function);
    halide_cg.get_builder().SetInsertPoint(entryBB_);
  }

  void CodeGen(isl::ast_node node) {
    emitAst(node);
    halide_cg.get_builder().CreateRetVoid();

    if (llvm::verifyModule(*halide_cg.get_module())) {
      LOG(ERROR) << str();
      llvm::verifyModule(*halide_cg.get_module(), &llvm::outs());
      throw std::runtime_error("LLVM generated module is invalid.");
    }
  }

  void createWrapper(
      const std::vector<Halide::ImageParam>& inputs,
      const std::vector<Halide::OutputImageParam>& outputs,
      const std::string& fname) {
    CHECK(not inputs.empty());
    CHECK(not outputs.empty());

    auto pred = [&inputs](const Halide::OutputImageParam& p) {
      return p.type() == inputs.front().type();
    };

    if (not std::all_of(inputs.begin(), inputs.end(), pred) or
        not std::all_of(outputs.begin(), outputs.end(), pred)) {
      throw std::invalid_argument("All arguments must be of the same type.");
    }

    llvm::Type* wrapperArg = halide_cg.llvm_type_of(inputs.at(0).type())
                                 ->getPointerTo()
                                 ->getPointerTo();

    auto* wrapperFType = llvm::FunctionType::get(
        llvm::Type::getVoidTy(llvmCtx), {wrapperArg}, false);

    auto* wrapper = llvm::Function::Create(
        wrapperFType,
        llvm::Function::ExternalLinkage,
        fname,
        halide_cg.get_module());

    auto builder = halide_cg.get_builder();
    auto* kernel = builder.GetInsertBlock()->getParent();
    CHECK_EQ(
        inputs.size() + outputs.size(),
        kernel->getFunctionType()->getNumParams());

    wrapper->arg_begin()->setName("args");

    std::vector<llvm::Value*> implArgs;
    implArgs.reserve(kernel->getFunctionType()->getNumParams());

    builder.SetInsertPoint(llvm::BasicBlock::Create(llvmCtx, "entry", wrapper));

    size_t idx = 0;
    for (auto* implParam : kernel->getFunctionType()->params()) {
      auto argAddr = builder.CreateInBoundsGEP(
          &*wrapper->arg_begin(), {getLLVMConstantSignedInt64(idx++)});
      auto arg = builder.CreateLoad(argAddr);
      auto castedToSpecialized = builder.CreateBitCast(arg, &*implParam);
      implArgs.push_back(castedToSpecialized);
    }

    builder.CreateCall(kernel, implArgs);
    builder.CreateRetVoid();
  }

  llvm::BasicBlock* emitAst(isl::ast_node node) {
    if (auto forNode = node.as<isl::ast_node_for>()) {
      return emitFor(forNode);
    } else if (auto userNode = node.as<isl::ast_node_user>()) {
      return emitStmt(userNode);
    } else if (auto blockNode = node.as<isl::ast_node_block>()) {
      return emitBlock(blockNode);
    } else {
      if (node.as<isl::ast_node_if>()) {
        LOG(FATAL) << "NYI if node: " << node << std::endl;
      } else {
        LOG(FATAL) << "NYI " << node << std::endl;
      }
      return static_cast<llvm::BasicBlock*>(nullptr); // avoid warning
    }
  }

 private:
  llvm::BasicBlock* emitBlock(isl::ast_node_block node) {
    auto* function = halide_cg.get_builder().GetInsertBlock()->getParent();
    auto* currBB = llvm::BasicBlock::Create(llvmCtx, "block_exit", function);
    halide_cg.get_builder().CreateBr(currBB);
    halide_cg.get_builder().SetInsertPoint(currBB);

    for (auto child : node.get_children()) {
      currBB = emitAst(child);
      halide_cg.get_builder().SetInsertPoint(currBB);
    }

    auto* exit = llvm::BasicBlock::Create(llvmCtx, "block_exit", function);
    halide_cg.get_builder().SetInsertPoint(currBB);
    halide_cg.get_builder().CreateBr(exit);
    halide_cg.get_builder().SetInsertPoint(exit);
    return exit;
  }

  llvm::Type* makePtrToArrayType(
      llvm::Type* baseTy,
      const std::vector<int64_t>& sizes) {
    CHECK_GE(sizes.size(), 1);
    CHECK(baseTy);
    llvm::Type* arrTy = llvm::ArrayType::get(baseTy, sizes.back());
    for (auto s = sizes.rbegin() + 1; s != sizes.rend(); ++s) {
      arrTy = llvm::ArrayType::get(arrTy, *s);
    }
    return arrTy->getPointerTo();
  }

  llvm::BasicBlock* emitFor(isl::ast_node_for node) {
    IteratorLLVMValueMapType iterPHIs;

    auto* incoming = halide_cg.get_builder().GetInsertBlock();
    auto* function = incoming->getParent();
    auto* headerBB = llvm::BasicBlock::Create(llvmCtx, "loop_header", function);
    auto* loopBodyBB = llvm::BasicBlock::Create(llvmCtx, "loop_body", function);
    auto* loopLatchBB =
        llvm::BasicBlock::Create(llvmCtx, "loop_latch", function);
    auto* loopExitBB = llvm::BasicBlock::Create(llvmCtx, "loop_exit", function);

    bool parallel = node.is_coincident();
    llvm::Value* SyncRegion = nullptr;

#ifdef TAPIR_VERSION_MAJOR
    if (parallel) {
      SyncRegion = halide_cg.get_builder().CreateCall(
          llvm::Intrinsic::getDeclaration(
              function->getParent(), llvm::Intrinsic::syncregion_start),
          {},
          "syncreg");
    }
#endif

    halide_cg.get_builder().CreateBr(headerBB);

    llvm::PHINode* phi = nullptr;
    auto iterator = node.get_iterator().get_id();

    // Loop Header
    {
      auto initVal = IslExprToSInt(node.get_init());
      halide_cg.get_builder().SetInsertPoint(headerBB);
      phi = halide_cg.get_builder().CreatePHI(
          llvm::Type::getInt64Ty(llvmCtx), 2, iterator.get_name());
      halide_cg.sym_push(iterator.get_name(), phi);
      phi->addIncoming(getLLVMConstantSignedInt64(initVal), incoming);

      auto cond_expr = node.get_cond();
      CHECK(
          cond_expr.get_op_type() == isl::ast_op_type::lt or
          cond_expr.get_op_type() == isl::ast_op_type::le)
          << "I only know how to codegen lt and le";
      auto condLHS = cond_expr.get_op_arg(0);
      CHECK(
          isl_ast_expr_get_type(condLHS.get()) ==
          isl_ast_expr_type::isl_ast_expr_id);
      CHECK_EQ(condLHS.get_id(), iterator);

      IslAstExprInterpeter i(scop_.globalParameterContext);
      auto condRHSVal = i.interpret(cond_expr.get_op_arg(1));

      auto cond = [&]() {
        auto constant = getLLVMConstantSignedInt64(condRHSVal);
        switch (cond_expr.get_op_type()) {
          case isl::ast_op_type::lt:
            return halide_cg.get_builder().CreateICmpSLT(phi, constant);
          case isl::ast_op_type::le:
            return halide_cg.get_builder().CreateICmpSLE(phi, constant);
          default:
            CHECK(false) << "NYI";
            return static_cast<llvm::Value*>(nullptr); // avoid warning
        }
      }();
      halide_cg.get_builder().CreateCondBr(cond, loopBodyBB, loopExitBB);
    }

    // Create Body
    {
      halide_cg.get_builder().SetInsertPoint(loopBodyBB);

#ifdef TAPIR_VERSION_MAJOR
      if (parallel) {
        auto* detachedBB =
            llvm::BasicBlock::Create(llvmCtx, "det.achd", function);
        halide_cg.get_builder().CreateDetach(
            detachedBB, loopLatchBB, SyncRegion);
        halide_cg.get_builder().SetInsertPoint(detachedBB);
      }
#endif
      auto* currentBB = emitAst(node.get_body());
      halide_cg.get_builder().SetInsertPoint(currentBB);

      if (parallel) {
#ifdef TAPIR_VERSION_MAJOR
        halide_cg.get_builder().CreateReattach(loopLatchBB, SyncRegion);
#endif
      } else {
        halide_cg.get_builder().CreateBr(loopLatchBB);
      }
    }

    // Create Latch
    {
      halide_cg.get_builder().SetInsertPoint(loopLatchBB);
      auto incVal = IslExprToSInt(node.get_inc());
      phi->addIncoming(
          halide_cg.get_builder().CreateAdd(
              phi, getLLVMConstantSignedInt64(incVal)),
          loopLatchBB);
      halide_cg.get_builder().CreateBr(headerBB);
    }

    halide_cg.get_builder().SetInsertPoint(loopExitBB);
    halide_cg.sym_pop(iterator.get_name());
#ifdef TAPIR_VERSION_MAJOR
    if (parallel) {
      auto* syncBB = llvm::BasicBlock::Create(llvmCtx, "synced", function);
      halide_cg.get_builder().CreateSync(syncBB, SyncRegion);
      halide_cg.get_builder().SetInsertPoint(syncBB);
    }
#endif
    return halide_cg.get_builder().GetInsertBlock();
  }

  llvm::BasicBlock* emitStmt(isl::ast_node_user node) {
    isl::ast_expr usrExp = node.get_expr();
    auto id = usrExp.get_op_arg(0).get_id();
    auto provide = scop_.halide.statements.at(id);
    auto op = provide.as<Halide::Internal::Provide>();
    CHECK(op) << "Expected a Provide node: " << provide << '\n';
    CHECK(op->values.size() == 1)
        << "Multi-valued Provide: " << Halide::Internal::Stmt(provide) << "\n";
    auto arrayName = op->name;
    const auto& subscripts = stmtSubscripts_.at(id);
    llvm::SmallVector<llvm::Value*, 5> subscriptValues;

    for (const auto& subscript : subscripts) {
      subscriptValues.push_back(halide_cg.getValue(subscript));
    }

    auto destAddr = halide_cg.get_builder().CreateInBoundsGEP(
        halide_cg.sym_get(arrayName), subscriptValues);

    halide_cg.iteratorMap_ = &iteratorMaps_.at(id);
    llvm::Value* rhs = halide_cg.codegen(op->values[0]);
    halide_cg.get_builder().CreateStore(rhs, destAddr);
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

IslCodegenRes codegenISL(const Scop& scop) {
  IteratorMapsType iteratorMaps;
  StmtSubscriptExprMapType stmtSubscripts;
  auto collect = [&iteratorMaps, &scop, &stmtSubscripts](
                     isl::ast_node n, isl::ast_build b) -> isl::ast_node {
    auto collectIteratorMaps =
        [](isl::ast_node node,
           isl::ast_build build,
           IteratorMapsType& iteratorMaps,
           const Scop& scop,
           StmtSubscriptExprMapType& stmtSubscripts) -> isl::ast_node {
      auto user = node.as<isl::ast_node_user>();
      CHECK(user);
      auto expr = user.get_expr();
      auto schedule = build.get_schedule();
      auto scheduleMap = isl::map::from_union_map(schedule);

      auto stmtId = expr.get_op_arg(0).get_id();
      CHECK_EQ(0, iteratorMaps.count(stmtId)) << "entry exists: " << stmtId;
      auto iteratorMap = isl::pw_multi_aff(scheduleMap.reverse());
      auto iterators = scop.halide.iterators.at(stmtId);
      auto& stmtIteratorMap = iteratorMaps[stmtId];
      for (int i = 0; i < iterators.size(); ++i) {
        auto expr = build.expr_from(iteratorMap.get_pw_aff(i));
        stmtIteratorMap.emplace(iterators[i], expr);
      }
      auto& subscripts = stmtSubscripts[stmtId];
      auto provide =
          scop.halide.statements.at(stmtId).as<Halide::Internal::Provide>();
      for (auto e : provide->args) {
        const auto& map = iteratorMap;
        auto space = map.get_space().params();
        auto aff = scop.makeIslAffFromStmtExpr(stmtId, space, e);
        auto pulled = isl::pw_aff(aff).pullback(map);
        CHECK_EQ(pulled.n_piece(), 1);
        subscripts.push_back(build.expr_from(pulled));
      }
      return node.set_annotation(stmtId);
    };

    auto& uv = iteratorMaps;
    return collectIteratorMaps(n, b, uv, scop, stmtSubscripts);
  };

  auto bands = detail::ScheduleTree::collect(
      scop.scheduleRoot(), detail::ScheduleTreeType::Band);
  int maxDepth = 0;
  for (auto const& node : bands) {
    auto bandElem = node->elemAs<detail::ScheduleTreeElemBand>();
    auto depth = node->scheduleDepth(scop.scheduleRoot()) +
        bandElem->mupa_.dim(isl::dim_type::set);
    if (depth > maxDepth) {
      maxDepth = depth;
    }
  }

  checkValidIslSchedule(scop.scheduleRoot());
  auto schedule = detail::toIslSchedule(scop.scheduleRoot());
  auto ctx = schedule.get_ctx();
  auto astBuild = isl::ast_build(schedule.get_ctx());
  astBuild = astBuild.set_at_each_domain(collect);
  astBuild = astBuild.set_iterators(Codegen::makeLoopIterators(ctx, maxDepth));
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
  cg.createSignature(
      scop.halide.inputs, scop.halide.outputs, specializedName + "_imlp");
  cg.CodeGen(islCg.astNode);
  cg.createWrapper(scop.halide.inputs, scop.halide.outputs, specializedName);
  cg.halide_cg.optimize_module();
  return cg.halide_cg.move_module();
}

} // namespace polyhedral
} // namespace tc
