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
#include "tc/core/compiler.h"

#include <sstream>
#include <string>

#include "tc/core/check.h"
#include "tc/core/flags.h"
#include "tc/core/halide_utils.h"
#include "tc/core/tensor.h"
#include "tc/lang/canonicalize.h"

namespace tc {
namespace detail {
void checkInputsCompliant(
    const tc2halide::HalideComponents& halideComponents,
    const std::vector<const DLConstTensor*>& inputsInfo);
} // namespace detail

template <typename Backend>
std::unique_ptr<typename Backend::ExecutorType> compile(
    const std::string& tc,
    const std::string& entryPoint,
    const std::vector<const DLConstTensor*>& inputs,
    /* TODO: in the future also pass outputs for stride and alignment info */
    const typename Backend::MappingOptionsType& options) {
  auto parsedTcs = detail::parse(tc);
  TC_CHECK_EQ(parsedTcs.count(entryPoint), 1u)
      << "attempting to access undefined function " << entryPoint;
  return compile<Backend>(parsedTcs[entryPoint], inputs, options);
}

template <typename Backend>
std::unique_ptr<typename Backend::ExecutorType> compile(
    lang::TreeRef tcDefinition,
    const std::vector<const DLConstTensor*>& inputs,
    /* TODO: in the future also pass outputs for stride and alignment info */
    const typename Backend::MappingOptionsType& options) {
  using CompilationResultType = typename Backend::CompilationResultType;

  auto inputsInfo = makeTensorInfoVector(inputs);
  auto outputsInfo = detail::inferOutputTensorInfo(tcDefinition, inputs);
  auto halideComponents =
      tc2halide::translate(isl::with_exceptions::globalIslCtx(), tcDefinition);
  detail::checkInputsCompliant(halideComponents, inputs);

  auto tcName = lang::Def(tcDefinition).name().name();
  CompilationResultType compilationResult = Backend::compileWithTcMapper(
      tcName,
      halideComponents,
      inputs,
      /* TODO outputs, */
      options);
  return std::unique_ptr<typename Backend::ExecutorType>(
      new typename Backend::ExecutorType(
          inputsInfo, outputsInfo, halideComponents, compilationResult));
}
} // namespace tc
