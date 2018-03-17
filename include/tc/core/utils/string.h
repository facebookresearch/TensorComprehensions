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

#include <iostream>
#include <string>
#include <vector>

namespace tc {

// Sets the std::boolalpha flags of the given std::ostream and resets it to
// the previous value on scope exit.
class OstreamBoolalphaScope {
 public:
  OstreamBoolalphaScope(std::ostream& os)
      : os_(os), hasBoolalpha_(os.flags() & std::ios_base::boolalpha) {
    os << std::boolalpha;
  }
  ~OstreamBoolalphaScope() {
    if (!hasBoolalpha_) {
      os_ << std::noboolalpha;
    }
  }

 private:
  std::ostream& os_;
  bool hasBoolalpha_;
};

template <typename T>
inline std::vector<T> parseCommaSeparatedIntegers(const std::string& sizes);

} // namespace tc

#include "tc/core/utils/string-inl.h"
