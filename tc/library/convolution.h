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

#include "tc/library/common.h"

namespace tc {

constexpr static auto CONVOLUTION2D_TC_NAME = "convolution";

namespace {
constexpr static auto CONVOLUTION2D_TC = R"TC(
  def convolution(float(N,C,H,W) I, float(M,C,KH,KW) W1, float(M) B) -> (O) {
    O(n, m, h, w) +=! I(n, c, ${sh} * h + kh, ${sw} * w + kw) * W1(m, c, kh, kw)
    O(n, m, h, w) = O(n, m, h, w) + B(m)
  }
)TC";
} // namespace

std::string makeConvolution2DTc(int strideH, int strideW) {
  CHECK(strideH > 0 && strideW > 0) << "Stride must be greater than 0";
  std::string tcStr;
  tcStr = CONVOLUTION2D_TC;
  tcStr = replaceString(tcStr, "${sh}", std::to_string(strideH));
  tcStr = replaceString(tcStr, "${sw}", std::to_string(strideW));
  return tcStr;
}
} // namespace tc
