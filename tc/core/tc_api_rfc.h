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

#include <map>
#include <string>
#include <vector>

#include "tc/lang/tree.h"
#include "tc/core/mapping_options.h"
#include "tc/core/tensor.h"

namespace tc {
// Given a TC representation as a TC + TC function name, this function
// compiles a new TcExecutor for the specified Backend.
template <typename Backend>
std::unique_ptr<typename Backend::ExecutorType> compile(
  std::string tc,
  std::string tcFunctionName,
  const std::vector<const DLConstTensor*>& inputs,
  /* TODO: in the future also pass outputs for stride and alignment info */
  const typename Backend::MappingOptionsType& options);

namespace detail {
// Given a Tc and a list of input tensors that match the definition in the
// TC in positional order, this generates the output tensor infos issued
// from forward inference.
// The typical flow is to infer output sizes, allocate/resize them within
// you favorite ML framework / tensor library and then call compile.
std::vector<TensorInfo> inferOutputTensorInfo(
  lang::TreeRef tcDefinition,
  const std::vector<const DLConstTensor*> inputs);

// Given a TC representation, this parses the TC functions into a map of
// TreeRef indexed by TC function names.
std::map<std::string, lang::TreeRef> parse(const std::string& tc);

// Given a TC representation as a TreeRef, this function compiles a new
// TcExecutor for the specified Backend.
// For now, contiguous output sizes are inferred given input sizes.
// If you need another kernel for another TC or another inputs, options then
// just compile another TcExecutor; because atm we fully JIT specialize on all
// sizes.
template <typename Backend>
std::unique_ptr<typename Backend::ExecutorType> compile(
  lang::TreeRef tcDefinition,
  const std::vector<const DLConstTensor*>& inputs,
  /* TODO: in the future also pass outputs for stride and alignment info */
  const typename Backend::MappingOptionsType& options);
} // namespace detail
} // namespace tc

#include "tc/core/tc-inl.h"
