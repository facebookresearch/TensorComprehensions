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

#include "tc/core/flags.h"
#include "tc/core/halide_utils.h"
#include "tc/core/tensor.h"
#include "tc/lang/canonicalize.h"

namespace tc {
std::vector<TensorInfo> inferOutputTensorInfo(
    const std::string& tc,
    const std::string& entryPoint,
    const std::vector<const DLConstTensor*> inputs) {
  auto parsedTcs = detail::parse(tc);
  CHECK_EQ(parsedTcs.count(entryPoint), 1u)
      << "attempting to access undefined function " << entryPoint;
  return tc::detail::inferOutputTensorInfo(parsedTcs[entryPoint], inputs);
}

namespace detail {
void checkInputsCompliant(
    const tc2halide::HalideComponents& halideComponents,
    const std::vector<const DLConstTensor*>& inputsInfo) {
  if (inputsInfo.size() != halideComponents.inputs.size()) {
    throw lang::ErrorReport(halideComponents.getDef())
        << "expected " << halideComponents.inputs.size() << " inputs but found "
        << inputsInfo.size();
  }
  for (size_t i = 0; i < inputsInfo.size(); ++i) {
    auto dlType = inputsInfo[i]->dtype;
    auto hType = halideComponents.inputs[i].type();
    // we have three type representations here: (1) halide Type (2) DLTensor
    // type, and (3) the token representing the type in the frontend (e.g.
    // TK_FLOAT) we need to translate to (3) to report user facing errors
    auto dlLangType =
        lang::TypeInfo(lang::TypeInfo::Code(dlType.code), dlType.bits)
            .toScalarToken();
    auto hLangType =
        lang::TypeInfo(lang::TypeInfo::Code(hType.code()), hType.bits())
            .toScalarToken();
    if (dlLangType != hLangType) {
      throw lang::ErrorReport(halideComponents.getDef().params()[i])
          << "expected type " << lang::kindToString(hLangType) << " but found "
          << lang::kindToString(dlLangType);
    }
    int hdim = halideComponents.inputs[i].dimensions();
    int dldim = inputsInfo[i]->ndim;
    if (dldim != hdim) {
      throw lang::ErrorReport(halideComponents.getDef().params()[i])
          << "expected a tensor with " << hdim << " dimensions but found "
          << dldim << " dimensions.";
    }
    auto dlstrides = inputsInfo[i]->strides;
    auto dlsizes = inputsInfo[i]->shape;
    std::ostringstream ss;
    for (size_t j = 0; j < dldim - 1; ++j){
      if (dlstrides[j] < dlstrides[j+1] * dlsizes[j+1]){
	throw lang::ErrorReport(halideComponents.getDef().params()[i])
          << "invalid stride "; 
      }
      if (dlstrides[dldim -1] < 1 ){
	throw lang::ErrorReport(halideComponents.getDef().params()[i])
          << "invalid stride "; 
     }
     }
  }
}

std::vector<TensorInfo> inferOutputTensorInfo(
    lang::TreeRef tcDefinition,
    const std::vector<const DLConstTensor*> inputs) {
  return tc::inferOutputTensorInfo(
      tc2halide::translate(isl::with_exceptions::globalIslCtx(), tcDefinition),
      inputs);
}

std::map<std::string, lang::TreeRef> parse(const std::string& tc) {
  lang::Parser parser(tc);
  std::map<std::string, lang::TreeRef> parsed;
  while (parser.L.cur().kind != lang::TK_EOF) {
    auto t = parser.parseFunction();
    auto name = lang::Def(t).name().name();
    parsed.emplace(std::make_pair(name, t));
  }
  return parsed;
}
} // namespace detail
} // namespace tc
