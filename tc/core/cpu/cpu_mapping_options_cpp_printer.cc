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
#include "tc/core/cpu/cpu_mapping_options_cpp_printer.h"

#include <sstream>

namespace tc {

CpuMappingOptionsCppPrinter& operator<<(
    CpuMappingOptionsCppPrinter& prn,
    const CpuMappingOptions& options) {
  prn.print(options.generic);
  prn.endStmt();
  return prn;
}

std::ostream& operator<<(std::ostream& out, const CpuMappingOptionsAsCpp& mo) {
  auto prn = CpuMappingOptionsCppPrinter(out, mo.indent);
  prn << mo.options;
  return out;
}

} // namespace tc
