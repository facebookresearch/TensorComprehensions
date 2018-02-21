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
#include "tc/core/tc_executor.h"
#include "tc/core/utils/dlpack.h"
#include "tc/lang/parser.h"
#include "tc/lang/sema.h"

namespace tc {

using namespace dlutils;

const size_t TcExecutor::InvalidHandle;

namespace {
lang::TreeRef parseOneFunction(const std::string& def) {
  lang::Parser parser(def);
  auto r = parser.parseFunction();
  if (parser.L.cur().kind != lang::TK_EOF) {
    throw lang::ErrorReport(parser.L.cur().range)
        << "More than one TCs were passed to TcExecutor.";
  }
  return r;
}

int toTypeToken(DLDataType dtype) {
  return lang::TypeInfo(lang::TypeInfo::Code(dtype.code), dtype.bits)
      .toScalarToken();
}
} // namespace

TcExecutor::TcExecutor(
    const std::string& TcDefinition,
    const std::vector<const DLTensor*>& inputsInfo)
    : TcExecutor(parseOneFunction(TcDefinition), inputsInfo) {}

TcExecutor::TcExecutor(
    lang::TreeRef TcDefinition,
    const std::vector<const DLTensor*>& inputsInfo)
    : tcTree_(TcDefinition) {
  execInfo_.kernelName = lang::Def(tcTree_).name().name();
  halideComponents_ =
      tc2halide::translate(isl::with_exceptions::globalIslCtx(), tcTree_);
  checkInputsCompliant(inputsInfo);
  execInfo_.inputsInfo = makeDLTensorVector(inputsInfo);
  // TODO: check if this is wrong, packed tensors may  have 0 strides stored
  execInfo_.outputsInfo =
      tc::inferOutputTensorInfo(halideComponents_, inputsInfo);
}

TcExecutor::~TcExecutor() {}

// TODO: make sure that the empty stride arrays (in DLTensor) are not a problem
void TcExecutor::checkSizesAndStridesAreCompliant(
    const DLTensor* actual,
    const DLTensor* expected,
    const lang::Param& dbg) const {
  if (actual->ndim != expected->ndim) {
    throw lang::ErrorReport(dbg)
        << "expected " << expected->ndim << " dimensions but found tensor with "
        << actual->ndim << " dimensions";
  }
  auto atype = toTypeToken(actual->dtype);
  auto etype = toTypeToken(expected->dtype);
  if (atype != etype) {
    throw lang::ErrorReport(dbg) << "expected " << lang::kindToString(etype)
                                 << " but found " << lang::kindToString(atype);
  }
  std::vector<int64_t> shapeA(actual->shape, actual->shape + actual->ndim);
  std::vector<int64_t> shapeE(
      expected->shape, expected->shape + expected->ndim);
  for (int i = 0; i < shapeA.size(); ++i) {
    if (shapeA[i] != shapeE[i]) {
      throw lang::ErrorReport(dbg)
          << "expected size " << shapeE[i] << " for dim " << i << " but found "
          << shapeA[i];
    }
  }
}

void TcExecutor::checkInputsCompliant(
    const std::vector<const DLTensor*>& inputsInfo) const {
  if (inputsInfo.size() != halideComponents_.inputs.size()) {
    throw lang::ErrorReport(halideComponents_.getDef())
        << "expected " << halideComponents_.inputs.size()
        << " inputs but found " << inputsInfo.size();
  }
  for (size_t i = 0; i < inputsInfo.size(); ++i) {
    auto dltype_ = inputsInfo[i]->dtype;
    auto htype_ = halideComponents_.inputs[i].type();
    // we have three type representations here: (1) halide Type (2) DLTensor
    // type, and (3) the token representing the type in the frontend (e.g.
    // TK_FLOAT) we need to translate to (3) to report user facing errors
    auto dltype =
        lang::TypeInfo(lang::TypeInfo::Code(dltype_.code), dltype_.bits)
            .toScalarToken();
    auto htype =
        lang::TypeInfo(lang::TypeInfo::Code(htype_.code()), htype_.bits())
            .toScalarToken();
    if (dltype != htype) {
      throw lang::ErrorReport(halideComponents_.getDef().params()[i])
          << "expected type " << lang::kindToString(htype) << " but found "
          << lang::kindToString(dltype);
    }
    int edim = halideComponents_.inputs[i].dimensions();
    int adim = inputsInfo[i]->ndim;
    if (adim != edim) {
      throw lang::ErrorReport(halideComponents_.getDef().params()[i])
          << "expected a tensor with " << edim << " dimensions but found "
          << adim << " dimensions.";
    }
  }
}

std::vector<const DLTensor*> TcExecutor::inferOutputTensorInfo() {
  return extractRawPtrs(execInfo_.outputsInfo);
}

} // namespace tc
