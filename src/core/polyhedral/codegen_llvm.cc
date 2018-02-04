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
#include "llvm/ExecutionEngine/ExecutionEngine.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/LegacyPassManager.h"
#include "llvm/IR/Verifier.h"
#include "llvm/Support/TargetSelect.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/Transforms/IPO/PassManagerBuilder.h"
#include "llvm/Transforms/IPO.h"
#include "llvm/Transforms/Tapir/CilkABI.h"

#include "Halide/Halide.h"

#include "isl/ast.h"

#include "tc/core/constants.h"
//#include "tc/core/polyhedral/isl_mu_wrappers.h"
#include "tc/core/flags.h"
#include "tc/core/polyhedral/schedule_isl_conversion.h"
#include "tc/core/polyhedral/scop.h"
#include "tc/core/scope_guard.h"

using namespace Halide;

namespace tc {

namespace halide2isl {
isl::aff makeIslAffFromExpr(isl::space space, const Halide::Expr& e);
}

namespace polyhedral {

using IteratorMapsType =
    std::unordered_map<isl::id, isl::pw_multi_aff, isl::IslIdIslHash>;

using IteratorLLVMValueMapType =
    std::unordered_map<isl::id, llvm::Value*, isl::IslIdIslHash>;

using StmtSubscriptExprMapType =
    std::unordered_map<isl::id, std::vector<isl::ast_expr>, isl::IslIdIslHash>;

namespace {

thread_local llvm::LLVMContext llvmCtx;

int64_t toSInt(isl::val v) {
  auto n = v.get_num_si();
  auto d = v.get_den_si();
  CHECK_EQ(n % d, 0);
  return n / d;
}

llvm::Value* getLLVMConstantSignedInt64(int64_t v) {
  return llvm::ConstantInt::get(llvm::Type::getInt64Ty(llvmCtx), v, true);
}

isl::aff extractAff(isl::pw_multi_aff pma) {
  isl::PMA pma_(pma);
  CHECK_EQ(pma_.size(), 1);
  isl::MA ma(pma_[0].second);
  CHECK_EQ(ma.size(), 1);
  return ma[0];
}

int64_t IslExprToSInt(isl::ast_expr e) {
  CHECK(isl_ast_expr_get_type(e.get()) == isl_ast_expr_type::isl_ast_expr_int);
  assert(sizeof(long) <= 8); // long is assumed to fit to 64bits
  return toSInt(isl::manage(isl_ast_expr_get_val(e.get())));
}

int64_t islIdToInt(isl::ast_expr e, isl::set context) {
  CHECK(isl_ast_expr_get_type(e.get()) == isl_ast_expr_type::isl_ast_expr_id);
  CHECK_NE(-1, context.find_dim_by_id(isl::dim_type::param, e.get_id()));
  while (context.dim(isl::dim_type::param) > 1) {
    for (unsigned int d = 0; d < context.dim(isl::dim_type::param); ++d) {
      if (d == context.find_dim_by_id(isl::dim_type::param, e.get_id())) {
        continue;
      }
      context = context.remove_dims(isl::dim_type::param, d, 1);
    }
  }
  auto p = context.sample_point();

  auto val = toSInt(p.get_coordinate_val(isl::dim_type::param, 0));
  return val;
}

int64_t getTensorSize(isl::set context, const Halide::Expr& e) {
  if (context.get_space().is_params()) {
    context = context.from_params();
  }
  // isl will take care of substituting parameter values if they are known and
  // simplifying the expression.
  auto pwAff =
      isl::pw_aff(halide2isl::makeIslAffFromExpr(context.get_space(), e));
  pwAff = pwAff.intersect_params(context);
  isl::PA pwAffs(pwAff);
  CHECK_EQ(pwAffs.size(), 1);
  isl::map m(pwAffs[0].second);
  auto r = m.range();
  r = r.project_out(isl::dim_type::param, 0, r.n_dim());
  CHECK(r.is_singleton());
  auto p = r.sample_point();
  return toSInt(p.get_coordinate_val(isl::dim_type::out, 0));
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

class CodeGen_TC : public Halide::Internal::CodeGen_X86 {
 public:
  const isl::pw_multi_aff* iteratorMap_;
  CodeGen_TC(Target t) : CodeGen_X86(t) {}

  using CodeGen_X86::codegen;
  using CodeGen_X86::llvm_type_of;
  using CodeGen_X86::sym_get;
  using CodeGen_X86::sym_pop;
  using CodeGen_X86::sym_push;

  void init_module() override {
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

 protected:
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
    auto aff = halide2isl::makeIslAffFromExpr(
        iteratorMap_->get_space().range(), Halide::Expr(op));

    auto subscriptPma = isl::pw_aff(aff).pullback(*iteratorMap_);
    auto subscriptAff = extractAff(subscriptPma);

    // sanity checks
    CHECK_EQ(subscriptAff.dim(isl::dim_type::div), 0);
    CHECK_EQ(subscriptAff.dim(isl::dim_type::out), 1);
    for (int d = 0; d < subscriptAff.dim(isl::dim_type::param); ++d) {
      auto v = subscriptAff.get_coefficient_val(isl::dim_type::param, d);
      CHECK(v.is_zero());
    }

    llvm::Optional<int> posOne;
    int sum = 0;
    for (int d = 0; d < subscriptAff.dim(isl::dim_type::in); ++d) {
      auto v = subscriptAff.get_coefficient_val(isl::dim_type::in, d);
      CHECK(v.is_zero() or v.is_one());
      if (v.is_zero()) {
        continue;
      }
      ++sum;
      posOne = d;
    }
    CHECK_LE(sum, 1);

    if (sum == 0) {
      value =
          getLLVMConstantSignedInt64(toSInt(subscriptAff.get_constant_val()));
      return;
    }
    CHECK(posOne);

    std::string name(
        isl_aff_get_dim_name(subscriptAff.get(), isl_dim_in, *posOne));

    value = sym_get(name);
  }
public:
  void optimize_module() {
    Halide::Internal::debug(3) << "Optimizing module\n";

    if (Halide::Internal::debug::debug_level() >= 3) {
        #if LLVM_VERSION >= 50
        module->print(dbgs(), nullptr, false, true);
        #else
        module->dump();
        #endif
    }

    // We override PassManager::add so that we have an opportunity to
    // blacklist problematic LLVM passes.
    class MyFunctionPassManager : public llvm::legacy::FunctionPassManager {
    public:
        MyFunctionPassManager(llvm::Module *m) : llvm::legacy::FunctionPassManager(m) {}
        virtual void add(llvm::Pass *p) override {
            Halide::Internal::debug(2) << "Adding function pass: " << p->getPassName().str() << "\n";
            llvm::legacy::FunctionPassManager::add(p);
        }
    };

    class MyModulePassManager : public llvm::legacy::PassManager {
    public:
        virtual void add(llvm::Pass *p) override {
            Halide::Internal::debug(2) << "Adding module pass: " << p->getPassName().str() << "\n";
            llvm::legacy::PassManager::add(p);
        }
    };

    MyFunctionPassManager function_pass_manager(module.get());
    MyModulePassManager module_pass_manager;

    std::unique_ptr<llvm::TargetMachine> TM = Halide::Internal::make_target_machine(*module);
    module_pass_manager.add(llvm::createTargetTransformInfoWrapperPass(TM ? TM->getTargetIRAnalysis() : llvm::TargetIRAnalysis()));
    function_pass_manager.add(llvm::createTargetTransformInfoWrapperPass(TM ? TM->getTargetIRAnalysis() : llvm::TargetIRAnalysis()));

    llvm::PassManagerBuilder b;
    b.OptLevel = 3;
    b.tapirTarget = new llvm::tapir::CilkABI();
#if LLVM_VERSION >= 50
    b.Inliner = llvm::createFunctionInliningPass(b.OptLevel, 0, false);
#else
    b.Inliner = llvm::createFunctionInliningPass(b.OptLevel, 0);
#endif
    b.LoopVectorize = true;
    b.SLPVectorize = true;

#if LLVM_VERSION >= 50
    if (TM) {
        TM->adjustPassManager(b);
    }
#endif

    b.populateFunctionPassManager(function_pass_manager);
    b.populateModulePassManager(module_pass_manager);

    // Run optimization passes
    function_pass_manager.doInitialization();
    for (llvm::Module::iterator i = module->begin(); i != module->end(); i++) {
        function_pass_manager.run(*i);
    }
    function_pass_manager.doFinalization();
    module_pass_manager.run(*module);

    Halide::Internal::debug(3) << "After LLVM optimizations:\n";
    if (Halide::Internal::debug::debug_level() >= 2) {
        #if LLVM_VERSION >= 50
        module->print(dbgs(), nullptr, false, true);
        #else
        module->dump();
        #endif
    }
}
};

class LLVMCodegen {
  void collectTensor(const Halide::OutputImageParam& t) {
    auto sizes =
        getTensorSizesWithoutLeadingDim(t, scop_.globalParameterContext);
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
      std::cout << str() << std::endl;
      llvm::verifyModule(*halide_cg.get_module(), &llvm::outs());
      throw std::runtime_error("LLVM generated module is invalid.");
    }
  }

  llvm::BasicBlock* emitAst(isl::ast_node node) {
    switch (node.get_type()) {
      case isl::ast_node_type::_for:
        return emitFor(node);
      case isl::ast_node_type::user:
        return emitStmt(node);
      case isl::ast_node_type::block:
        return emitBlock(node);
      case isl::ast_node_type::_if:
        LOG(FATAL) << "NYI if node: " << node << std::endl;
      default:
        LOG(FATAL) << "NYI " << node << std::endl;
        return static_cast<llvm::BasicBlock*>(nullptr); // avoid warning
    }
  }

 private:
  llvm::BasicBlock* emitBlock(isl::ast_node node) {
    auto* function = halide_cg.get_builder().GetInsertBlock()->getParent();
    auto* currBB = llvm::BasicBlock::Create(llvmCtx, "block_exit", function);
    halide_cg.get_builder().CreateBr(currBB);
    halide_cg.get_builder().SetInsertPoint(currBB);

    CHECK(node.get_type() == isl::ast_node_type::block);
    for (auto child : node.block_get_children()) {
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

  llvm::BasicBlock* emitFor(isl::ast_node node) {
    CHECK(node.get_type() == isl::ast_node_type::_for);

    IteratorLLVMValueMapType iterPHIs;

    auto* incoming = halide_cg.get_builder().GetInsertBlock();
    auto* function = incoming->getParent();
    auto* headerBB = llvm::BasicBlock::Create(llvmCtx, "loop_header", function);
    auto* loopBodyBB = llvm::BasicBlock::Create(llvmCtx, "loop_body", function);
    auto* loopLatchBB =
        llvm::BasicBlock::Create(llvmCtx, "loop_latch", function);
    auto* loopExitBB = llvm::BasicBlock::Create(llvmCtx, "loop_exit", function);

    bool parallel = true;

    llvm::Value* SyncRegion = nullptr;
    if (parallel) {
      SyncRegion = halide_cg.get_builder().CreateCall(
        llvm::Intrinsic::getDeclaration(function->getParent(), llvm::Intrinsic::syncregion_start),
        {},
        "syncreg"
      );
    }

    halide_cg.get_builder().CreateBr(headerBB);

    llvm::PHINode* phi = nullptr;

    // Loop Header
    {
      auto initVal = IslExprToSInt(node.for_get_init());
      halide_cg.get_builder().SetInsertPoint(headerBB);
      phi = halide_cg.get_builder().CreatePHI(
          llvm::Type::getInt64Ty(llvmCtx),
          2,
          node.for_get_iterator().get_id().get_name());
      halide_cg.sym_push(node.for_get_iterator().get_id().get_name(), phi);
      phi->addIncoming(getLLVMConstantSignedInt64(initVal), incoming);

      auto cond_expr = node.for_get_cond();
      CHECK(
          cond_expr.get_op_type() == isl::ast_op_type::lt or
          cond_expr.get_op_type() == isl::ast_op_type::le)
          << "I only know how to codegen lt and le";
      auto condLHS = cond_expr.get_op_arg(0);
      CHECK(
          isl_ast_expr_get_type(condLHS.get()) ==
          isl_ast_expr_type::isl_ast_expr_id);
      CHECK_EQ(condLHS.get_id(), node.for_get_iterator().get_id());

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

      if (parallel) {
        auto* detachedBB = llvm::BasicBlock::Create(llvmCtx, "det.achd", function);
        halide_cg.get_builder().CreateDetach(detachedBB, loopLatchBB, SyncRegion);
        halide_cg.get_builder().SetInsertPoint(detachedBB);
      }
      auto* currentBB = emitAst(node.for_get_body());
      halide_cg.get_builder().SetInsertPoint(currentBB);

      if (parallel) {
        halide_cg.get_builder().CreateReattach(loopLatchBB, SyncRegion);
      } else {
        halide_cg.get_builder().CreateBr(loopLatchBB);
      }
    }

    // Create Latch
    {
      halide_cg.get_builder().SetInsertPoint(loopLatchBB);
      auto incVal = IslExprToSInt(node.for_get_inc());
      phi->addIncoming(
          halide_cg.get_builder().CreateAdd(
              phi, getLLVMConstantSignedInt64(incVal)),
          loopLatchBB);
      halide_cg.get_builder().CreateBr(headerBB);
    }

    halide_cg.get_builder().SetInsertPoint(loopExitBB);
    halide_cg.sym_pop(node.for_get_iterator().get_id().get_name());
    if (parallel) {
        auto* syncBB = llvm::BasicBlock::Create(llvmCtx, "synced", function);
        halide_cg.get_builder().CreateSync(syncBB, SyncRegion);
        halide_cg.get_builder().SetInsertPoint(syncBB);
    }
    return halide_cg.get_builder().GetInsertBlock();
  }

  llvm::BasicBlock* emitStmt(isl::ast_node node) {
    CHECK(node.get_type() == isl::ast_node_type::user);
    isl::ast_expr usrExp = node.user_get_expr();
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
      switch (isl_ast_expr_get_type(subscript.get())) {
        case isl_ast_expr_type::isl_ast_expr_id: {
          subscriptValues.push_back(
              halide_cg.sym_get(subscript.get_id().get_name()));
          break;
        }
        case isl_ast_expr_type::isl_ast_expr_int: {
          auto val = isl::manage(isl_ast_expr_get_val(subscript.get()));
          CHECK_EQ(val.get_den_si(), 1);
          subscriptValues.push_back(
              getLLVMConstantSignedInt64(val.get_num_si()));
          break;
        }
        default:
          LOG(FATAL) << "NYI";
      }
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
    std::string output;
    {
      llvm::raw_string_ostream rso(output);
      halide_cg.get_module()->print(rso, nullptr);
      rso.str();
    }
    return output;
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

// Create a list of isl ids to be used as loop iterators when building the AST.
//
// Note that this function can be scrapped as ISL can generate some default
// iterator names.  However, it may come handy for associating extra info with
// iterators.
isl::list<isl::id>
makeLoopIterators(isl::ctx ctx, int n, const std::string& prefix = "c") {
  std::vector<isl::id> loopIterators;
  for (int i = 0; i < n; ++i) {
    std::stringstream ss;
    ss << prefix << i;
    loopIterators.emplace_back(ctx, ss.str());
  }
  return isl::list<isl::id>(ctx, loopIterators.begin(), loopIterators.end());
}

struct IslCodegenRes {
  IteratorMapsType iteratorMaps;
  StmtSubscriptExprMapType stmtSubscripts;
  isl::ast_node astNode;
};

IslCodegenRes codegenISL(const Scop& scop) {
  // TODO: improve support for C++ callbacks in isl bindings generator
  // see https://github.com/PollyLabs/isl/issues/24
  // This cannot be done via islpp_wrap because the callback is stored for
  // later use while islpp_wrap passes a pointer to a stack-allocated
  // object to the call as a means to support capturing lambdas.
  auto collect =
      [](isl_ast_node* n, isl_ast_build* b, void* uTuple) -> isl_ast_node* {
    auto collectIteratorMaps =
        [](isl::ast_node node,
           isl::ast_build build,
           IteratorMapsType& iteratorMaps,
           const Scop& scop,
           StmtSubscriptExprMapType& stmtSubscripts) -> isl::ast_node {
      auto expr = node.user_get_expr();
      // Note that the schedule obtained from build does NOT live in the
      // schedule space obtained from build, despite the naming.
      // We rename loop-related dimensions manually.
      auto schedule = build.get_schedule();
      auto scheduleSpace = build.get_schedule_space();
      auto scheduleMap = isl::map::from_union_map(schedule);

      auto stmtId = expr.get_op_arg(0).get_id();
      // auto nodeId = isl::id(
      // node.get_ctx(),
      // std::string(kAstNodeIdPrefix) + std::to_string(nAstNodes()++));
      CHECK_EQ(0, iteratorMaps.count(stmtId)) << "entry exists: " << stmtId;
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
      iteratorMaps.emplace(stmtId, iteratorMap);
      auto& subscripts = stmtSubscripts[stmtId];
      auto provide =
          scop.halide.statements.at(stmtId).as<Halide::Internal::Provide>();
      for (auto e : provide->args) {
        const auto& map = iteratorMap;
        auto space = map.get_space().range();
        auto aff = halide2isl::makeIslAffFromExpr(space, e);
        auto pulled = isl::pw_aff(aff).pullback(map);
        CHECK_EQ(pulled.n_piece(), 1);
        subscripts.push_back(build.expr_from(pulled));
      }
      return node.set_annotation(stmtId);
    };

    auto& t = *static_cast<
        std::tuple<IteratorMapsType&, Scop&, StmtSubscriptExprMapType&>*>(
        uTuple);

    auto& uv = std::get<0>(t);
    auto& scop = std::get<1>(t);
    auto& stmtSubscripts = std::get<2>(t);
    return collectIteratorMaps(
               isl::manage(n), isl::manage_copy(b), uv, scop, stmtSubscripts)
        .release();
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
  IteratorMapsType iteratorMaps;
  StmtSubscriptExprMapType stmtSubscripts;
  auto astBuild = isl::ast_build(schedule.get_ctx());
  auto t = std::tie(iteratorMaps, scop, stmtSubscripts);
  astBuild = isl::manage(
      isl_ast_build_set_at_each_domain(astBuild.release(), collect, &t));
  astBuild = astBuild.set_iterators(makeLoopIterators(ctx, maxDepth));
  auto astNode = astBuild.node_from_schedule(schedule);
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
  cg.createSignature(scop.halide.inputs, scop.halide.outputs, specializedName);
  cg.CodeGen(islCg.astNode);
  cg.halide_cg.optimize_module();
  if (FLAGS_llvm_dump_ir) {
    std::cout << cg.str() << std::endl;
  }
  return cg.halide_cg.move_module();
}

} // namespace polyhedral
} // namespace tc
