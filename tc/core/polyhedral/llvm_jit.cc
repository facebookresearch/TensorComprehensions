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
#include <stdexcept>

#include "tc/core/polyhedral/llvm_jit.h"

#include "llvm/ExecutionEngine/ExecutionEngine.h"
#include "llvm/ExecutionEngine/Orc/LambdaResolver.h"
#include "llvm/ExecutionEngine/RuntimeDyld.h"
#include "llvm/ExecutionEngine/SectionMemoryManager.h"
#include "llvm/IR/Mangler.h"
#include "llvm/IR/Verifier.h"
#include "llvm/Passes/PassBuilder.h"
#include "llvm/Support/DynamicLibrary.h"
#include "llvm/Transforms/Utils/Cloning.h"

#include "tc/core/flags.h"
#include "tc/core/polyhedral/codegen_llvm.h"

using namespace llvm;

// Parse through ldconfig to find the path of a particular
// shared library. This is an unfortunate way to have to
// find it, but I couldn't immediately find something in
// imported libraries that would resolve this for us.
std::string find_library_path(std::string library) {
  std::string command = "ldconfig -p | grep " + library + " | grep x86-64";

  FILE* fpipe = popen(command.c_str(), "r");

  if (fpipe == nullptr) {
    throw std::runtime_error("Failed to popen()");
  }

  std::string output;
  char buffer[512];

  while (1) {
    int charactersRead = fread(buffer, 1, sizeof(buffer), fpipe);
    if (charactersRead == 0)
      break;
    output += std::string(buffer, charactersRead);
  }
  pclose(fpipe);

  int idx = output.rfind("=> ");
  if (idx == std::string::npos) {
    throw std::runtime_error("Failed locate library: " + library);
  }
  output = output.substr(idx + 3);
  if (output.length() > 0 && output[output.length() - 1] == '\n') {
    output = output.substr(0, output.length() - 1);
  }
  return output;
}

namespace tc {

#if LLVM_VERSION_MAJOR <= 6
Jit::Jit()
    : TM_(EngineBuilder().selectTarget()),
      DL_(TM_->createDataLayout()),
      objectLayer_([]() { return std::make_shared<SectionMemoryManager>(); }),
      compileLayer_(objectLayer_, orc::SimpleCompiler(*TM_)) {
  std::string err;

  auto path = find_library_path("libcilkrts.so");
  sys::DynamicLibrary::LoadLibraryPermanently(path.c_str(), &err);
  if (err != "") {
    throw std::runtime_error("Failed to find cilkrts: " + err);
  }
}

void Jit::addModule(std::shared_ptr<Module> M) {
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

  auto res = compileLayer_.addModule(M, std::move(Resolver));
  CHECK(res) << "Failed to jit compile.";
}
#else
Jit::Jit()
    : Resolver(createLegacyLookupResolver(
          [this](const std::string& Name) -> JITSymbol {
            if (auto Sym = compileLayer_.findSymbol(Name, false))
              return Sym;
            else if (auto Err = Sym.takeError())
              return std::move(Err);
            if (auto SymAddr =
                    RTDyldMemoryManager::getSymbolAddressInProcess(Name))
              return JITSymbol(SymAddr, JITSymbolFlags::Exported);
            return nullptr;
          },
          [](Error err) {
            throw std::runtime_error("Lookup failed: " + err);
          })),
      TM_(EngineBuilder().selectTarget()),
      DL_(TM_->createDataLayout()),
      objectLayer_(
          ES,
          [this](llvm::orc::VModuleKey) {
            return llvm::orc::RTDyldObjectLinkingLayer::Resources{
                std::make_shared<SectionMemoryManager>(), Resolver};
          }),
      compileLayer_(objectLayer_, orc::SimpleCompiler(*TM_)) {
  std::string err;

  auto path = find_library_path("libcilkrts.so");
  sys::DynamicLibrary::LoadLibraryPermanently(path.c_str(), &err);
  if (err != "") {
    throw std::runtime_error("Failed to find cilkrts: " + err);
  }
}

// Note that this copy may cause tapir tests to fail
// However, this code will never use tapir code
// and once the LLVM API churn stops, will be modified
// to be properly compatable.
void Jit::addModule(std::shared_ptr<Module> M) {
  M->setTargetTriple(TM_->getTargetTriple().str());
  auto K = ES.allocateVModule();
  llvm::Error res = compileLayer_.addModule(K, CloneModule(*M));
  CHECK(!res) << "Failed to jit compile.";
}
#endif

std::shared_ptr<Module> Jit::codegenScop(
    const std::string& specializedName,
    const polyhedral::Scop& scop) {
  std::shared_ptr<Module> mod = emitLLVMKernel(
      specializedName, scop, getTargetMachine().createDataLayout());
  addModule(mod);
  return mod;
}

TargetMachine& Jit::getTargetMachine() {
  return *TM_;
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

} // namespace tc
