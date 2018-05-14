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

#include <memory>
#include <string>
#include <type_traits>

#include "llvm/IR/LLVMContext.h"
#include "llvm/IR/Module.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/Target/TargetMachine.h"

#include "Halide.h"

namespace tc {

template <
    typename T,
    typename std::enable_if<
        std::is_base_of<llvm::Value, T>::value ||
        std::is_base_of<llvm::Type, T>::value>::type* = nullptr>
static inline std::string toString(T* llvmObject) {
  std::string output;
  llvm::raw_string_ostream rso(output);
  llvmObject->print(rso);
  rso.str();
  return output;
}

static inline std::string toString(llvm::Module* llvmObject) {
  std::string output;
  llvm::raw_string_ostream rso(output);
  llvmObject->print(rso, nullptr, false, true);
  rso.str();
  return output;
}

namespace polyhedral {
struct Scop;

std::unique_ptr<llvm::Module> emitLLVMKernel(
    const std::string& specializedName,
    const Scop& scop,
    const llvm::DataLayout& dataLayout);

// TODO: I want to do something like the following, but compilation was unhappy
//  using initialize_llvm = Halide::Internal::CodeGen_LLVM::initialize_llvm;
static inline void initialize_llvm() {
  Halide::Internal::CodeGen_LLVM::initialize_llvm();
}

} // namespace polyhedral
} // namespace tc
