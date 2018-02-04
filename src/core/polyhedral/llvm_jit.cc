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
#include "tc/core/polyhedral/llvm_jit.h"

#include "llvm/ExecutionEngine/ExecutionEngine.h"
#include "llvm/ExecutionEngine/Orc/LambdaResolver.h"
#include "llvm/ExecutionEngine/RuntimeDyld.h"
#include "llvm/ExecutionEngine/SectionMemoryManager.h"
#include "llvm/IR/Mangler.h"
#include "llvm/IR/Verifier.h"
#include "llvm/Passes/PassBuilder.h"
#include "llvm/Support/DynamicLibrary.h"

#include "tc/core/flags.h"
#include "tc/core/polyhedral/codegen_llvm.h"

using namespace llvm;

namespace tc {

Jit::Jit()
    : TM_(EngineBuilder().selectTarget()),
      DL_(TM_->createDataLayout()),
      objectLayer_([]() { return std::make_shared<SectionMemoryManager>(); }),
      compileLayer_(objectLayer_, orc::SimpleCompiler(*TM_)) {
  sys::DynamicLibrary::LoadLibraryPermanently(nullptr);
}

void Jit::codegenScop(
    const std::string& specializedName,
    const polyhedral::Scop& scop) {
  addModule(emitLLVMKernel(
      specializedName, scop, getTargetMachine().createDataLayout()));
}

TargetMachine& Jit::getTargetMachine() {
  return *TM_;
}

Jit::ModuleHandle Jit::addModule(std::unique_ptr<Module> M) {
  M->setTargetTriple(TM_->getTargetTriple().str());
  auto Resolver = orc::createLambdaResolver(
      [&](const std::string& Name) {
        if (auto Sym = compileLayer_.findSymbol(Name, false))
          return Sym;
        return JITSymbol(nullptr);
      },
      [](const std::string& Name) {
        if (auto SymAddr = RTDyldMemoryManager::getSymbolAddressInProcess(Name))
          return JITSymbol(SymAddr, JITSymbolFlags::Exported);
        return JITSymbol(nullptr);
      });

  auto res = compileLayer_.addModule(std::move(M), std::move(Resolver));
  CHECK(res) << "Failed to jit compile.";
  return *res;
}

JITSymbol Jit::findSymbol(const std::string Name) {
  std::string MangledName;
  raw_string_ostream MangledNameStream(MangledName);
  Mangler::getNameWithPrefix(MangledNameStream, Name, DL_);
  return compileLayer_.findSymbol(MangledNameStream.str(), true);
}

JITTargetAddress Jit::getSymbolAddress(const std::string Name) {
  auto res = findSymbol(Name).getAddress();
  CHECK(res) << "Could not find jit-ed symbol";
  return *res;
}

DEFINE_bool(llvm_no_opt, false, "Disable LLVM optimizations");
DEFINE_bool(llvm_debug_passes, false, "Print pass debug info");
DEFINE_bool(llvm_dump_optimized_ir, false, "Print optimized IR");

std::shared_ptr<Module> Jit::optimizeModule(std::shared_ptr<Module> M) {
  if (FLAGS_llvm_no_opt) {
    return M;
  }

  PassBuilder PB(TM_.get());
  AAManager AA;
  CHECK(PB.parseAAPipeline(AA, "default"))
      << "Unable to parse AA pipeline description.";
  LoopAnalysisManager LAM(FLAGS_llvm_debug_passes);
  FunctionAnalysisManager FAM(FLAGS_llvm_debug_passes);
  CGSCCAnalysisManager CGAM(FLAGS_llvm_debug_passes);
  ModuleAnalysisManager MAM(FLAGS_llvm_debug_passes);
  FAM.registerPass([&] { return std::move(AA); });
  PB.registerModuleAnalyses(MAM);
  PB.registerCGSCCAnalyses(CGAM);
  PB.registerFunctionAnalyses(FAM);
  PB.registerLoopAnalyses(LAM);
  PB.crossRegisterProxies(LAM, FAM, CGAM, MAM);

  ModulePassManager MPM(FLAGS_llvm_debug_passes);
  MPM.addPass(VerifierPass());
  CHECK(PB.parsePassPipeline(MPM, "default<O3>", true, FLAGS_llvm_debug_passes))
      << "Unable to parse pass pipline description.";
  MPM.addPass(VerifierPass());

  MPM.run(*M, MAM);

  if (FLAGS_llvm_dump_optimized_ir) {
    // M->dump(); // does not link
    M->print(llvm::errs(), nullptr);
  }

  return M;
}
