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
#include "tc/core/mapping_options_cpp_printer.h"

#include <sstream>

namespace tc {

MappingOptionsCppPrinter& MappingOptionsCppPrinter::printSchedulerOptions(
    const SchedulerOptionsView& schedulerOptions,
    const std::string& prefix) {
  const SchedulerOptionsProto& proto = schedulerOptions.proto;
  printValueOption(
      prefix + "FusionStrategy",
      "tc::FusionStrategy::" + FusionStrategy_Name(proto.fusion_strategy()));
  printBooleanOption(prefix + "AllowSkewing", proto.allow_skewing());
  printBooleanOption(prefix + "PositiveOrthant", proto.positive_orthant());

  return *this;
}

MappingOptionsCppPrinter& operator<<(
    MappingOptionsCppPrinter& prn,
    const std::string& str) {
  prn.printString(str);
  return prn;
}

MappingOptionsCppPrinter& operator<<(
    MappingOptionsCppPrinter& prn,
    const MappingOptions& options) {
  prn.printString("tc::MappingOptions::makeNaiveMappingOptions()")
      .printSchedulerOptions(options.outerScheduleOptions, "outerSchedule");
  if (options.proto.has_intra_tile_schedule_options()) {
    prn.printSchedulerOptions(
        options.intraTileScheduleOptions, "intraTileSchedule");
  }
  if (options.proto.has_tiling()) {
    prn.printListOption("tile", options.tiling.extractVector());
  }
  prn.printListOption("mapToThreads", options.block.extractVector());
  prn.printListOption("mapToBlocks", options.grid.extractVector());
  if (options.proto.has_unroll()) {
    prn.printValueOption("unroll", options.proto.unroll());
  }
  prn.printBooleanOption(
      "tileImperfectlyNested", options.proto.tile_imperfectly_nested());
  prn.printBooleanOption("useSharedMemory", options.proto.use_shared_memory());
  prn.printBooleanOption(
      "usePrivateMemory", options.proto.use_private_memory());
  prn.printBooleanOption(
      "unrollCopyShared", options.proto.unroll_copy_shared());
  if (options.proto.has_max_shared_memory()) {
    prn.printValueOption("maxSharedMemory", options.proto.max_shared_memory());
  }
  prn.printBooleanOption(
      "matchLibraryCalls", options.proto.match_library_calls());
  prn.endStmt();
  return prn;
}
} // namespace tc
