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
#include "tc/aten/aten_compiler.h"

#include <iostream>
#include <string>
#include <vector>

#include "tc/aten/aten.h"
#include "tc/core/check.h"
#include "tc/core/compiler.h"
#include "tc/core/tc_executor.h"
#include "tc/core/tensor.h"

namespace tc {
namespace aten {
std::vector<tc::DLTensorUPtr> inferOutputTensorInfo(
    const std::string& tc,
    const std::string& entryPoint,
    const std::vector<at::Tensor>& inputs) {
  auto parsedTcs = tc::detail::parse(tc);
  if (parsedTcs.count(entryPoint) != 1u) {
    TC_CHECK_GE(parsedTcs.size(), 1u)
        << "No TC was parsed, should have thrown earlier";
    throw lang::ErrorReport(parsedTcs.begin()->second)
        << "\nattempting to access undefined entryPoint: " << entryPoint;
  }
  auto inputDLTensors = makeDLConstTensors(inputs);
  return makeDLTensorVector(detail::inferOutputTensorInfo(
      parsedTcs.at(entryPoint), extractRawPtrs(inputDLTensors)));
}

std::vector<at::Tensor> prepareOutputs(
    const std::string& tc,
    const std::string& entryPoint,
    const std::vector<at::Tensor>& inputs) {
  std::vector<at::Tensor> outputs;
  auto outTensorInfo = inferOutputTensorInfo(tc, entryPoint, inputs);
  if (outTensorInfo.size() == 0) {
    return outputs;
  }
  TC_CHECK_GE(inputs.size(), 1u)
      << "NYI: Need >= 1 input tensors to determine "
      << "backend and prepare ATen outputs. Add an overload with just an ATen "
      << "backend";
  auto atenBackend = inputs[0].type().backend();
  for (size_t i = 0; i < outTensorInfo.size(); ++i) {
    TensorInfo info(outTensorInfo[i]);
    if (info.dtype.lanes == 1) {
      auto stype = at::toScalarType(info.dtype);
      outputs.push_back(
          at::getType(atenBackend, stype).tensor(at::IntList(info.shape)));
    } else {
      // "Cast" to a larger strided tensor with 1 lane
      auto lanes = info.dtype.lanes;
      TC_CHECK(lanes == 2 || lanes == 4);
      info.dtype.lanes = 1;
      info.shape[info.shape.size() - 1] *= lanes;
      auto stype = at::toScalarType(info.dtype);
      auto T = at::getType(atenBackend, stype).tensor(at::IntList(info.shape));

      info.shape[info.shape.size() - 1] /= lanes;
      outputs.push_back(T.set_(*T.storage(), 0, info.shape));
    }
  }
  return outputs;
}
} // namespace aten
} // namespace tc
