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
      .printSchedulerOptions(
          options.view.outerScheduleOptions, "outerSchedule");
  if (options.view.proto.has_intra_tile_schedule_options()) {
    prn.printSchedulerOptions(
        options.view.intraTileScheduleOptions, "intraTileSchedule");
  }
  if (options.view.proto.has_tiling()) {
    prn.printListOption("tile", options.view.tiling.extractVector());
  }
  if (options.view.proto.has_unroll()) {
    prn.printValueOption("unroll", options.view.proto.unroll());
  }
  prn.printBooleanOption(
      "tileImperfectlyNested", options.view.proto.tile_imperfectly_nested());
  prn.printBooleanOption(
      "matchLibraryCalls", options.view.proto.match_library_calls());
  prn.endStmt();
  return prn;
}
} // namespace tc
