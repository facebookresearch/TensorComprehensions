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

#include <limits>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

#include "tc/aten/aten.h"

struct PrecisionException : public std::runtime_error {
  PrecisionException(const std::string& s) : std::runtime_error(s) {}
};

// Given the difference of output vs expected tensor, check whether the
// difference is within a relative tolerance range.
// By default we use IEEE float precision , in the future we should pull it
// from the type of the at::Tensor.
// Also allow a factor to specify the total number of reductions involved
// in each result so we can properly compute the expected precision.
bool checkRtol(
    const at::Tensor& diff,
    const std::vector<at::Tensor> inputs,
    double nOperations = 1.0,
    double machinePrecision = std::numeric_limits<float>::epsilon()) {
  double maxValue = 0.0;
  for (auto& tensor : inputs) {
    maxValue = fmax(tensor.abs().max().toCFloat(), maxValue);
  }
  auto maxDiff = diff.abs().max().toCFloat();
  if (maxDiff >= nOperations * machinePrecision * maxValue) {
    std::stringstream ss;
    ss << "Error at relative precision: " << machinePrecision
       << ", #operations: " << nOperations << ", maxValue: " << maxValue
       << ", maxDiff: " << maxDiff << ", random seed: " << tc::randomSeed();
    throw PrecisionException(ss.str());
  }
  return true;
}
