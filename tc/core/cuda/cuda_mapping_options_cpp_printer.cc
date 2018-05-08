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
#include "tc/core/cuda/cuda_mapping_options_cpp_printer.h"

#include <sstream>

namespace tc {

CudaMappingOptionsCppPrinter& operator<<(
    CudaMappingOptionsCppPrinter& prn,
    const CudaMappingOptions& cudaOptions) {
  prn.print(cudaOptions.generic);
  prn.printListOption("mapToThreads", cudaOptions.block.extractVector());
  prn.printListOption("mapToBlocks", cudaOptions.grid.extractVector());
  prn.printBooleanOption(
      "useSharedMemory", cudaOptions.proto().use_shared_memory());
  prn.printBooleanOption(
      "usePrivateMemory", cudaOptions.proto().use_private_memory());
  prn.printBooleanOption(
      "unrollCopyShared", cudaOptions.proto().unroll_copy_shared());
  if (cudaOptions.proto().has_max_shared_memory()) {
    prn.printValueOption(
        "maxSharedMemory", cudaOptions.proto().max_shared_memory());
  }
  prn.endStmt();
  return prn;
}

std::ostream& operator<<(std::ostream& out, const CudaMappingOptionsAsCpp& mo) {
  auto prn = CudaMappingOptionsCppPrinter(out, mo.indent);
  prn << mo.options;
  return out;
}

} // namespace tc
