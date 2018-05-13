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

MappingOptionsCppPrinter::~MappingOptionsCppPrinter() = default;

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

MappingOptionsCppPrinter& MappingOptionsCppPrinter::print(
    const MappingOptions& options) {
  printString("tc::MappingOptions::makeNaiveMappingOptions()")
      .printSchedulerOptions(
          options.view.outerScheduleOptions, "outerSchedule");
  if (options.view.proto.has_intra_tile_schedule_options()) {
    printSchedulerOptions(
        options.view.intraTileScheduleOptions, "intraTileSchedule");
  }
  printBooleanOption(
      "fixParametersBeforeScheduling",
      options.view.proto.fix_parameters_before_scheduling());
  if (options.view.proto.has_tiling()) {
    printListOption("tile", options.view.tiling.extractVector());
  }
  if (options.view.proto.has_unroll()) {
    printValueOption("unroll", options.view.proto.unroll());
  }
  printBooleanOption(
      "tileImperfectlyNested", options.view.proto.tile_imperfectly_nested());
  printBooleanOption(
      "matchLibraryCalls", options.view.proto.match_library_calls());
  return *this;
}
} // namespace tc
