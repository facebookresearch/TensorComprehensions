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

#include "tc/core/utils/vararg.h"

namespace tc {

//
// TilingView & Tiling
//

Tiling::Tiling(const std::vector<uint64_t>& sizes) : TilingView(ownedProto_) {
  proto.clear_sizes();
  std::copy(
      sizes.begin(),
      sizes.end(),
      google::protobuf::RepeatedFieldBackInserter(proto.mutable_sizes()));
}

Tiling::Tiling(std::initializer_list<uint64_t> il)
    : Tiling(std::vector<uint64_t>(il)) {}

std::vector<uint64_t> TilingView::extractVector() const {
  std::vector<uint64_t> result(proto.sizes().begin(), proto.sizes().end());
  return result;
}

size_t TilingView::size() const {
  return proto.sizes_size();
}

ValueAccessor<uint64_t> TilingView::operator[](size_t i) {
  CHECK_LT(i, proto.sizes_size()) << "index overflow";
  return ValueAccessor<uint64_t>(
      [this, i](uint64_t u) { this->proto.set_sizes(i, u); },
      [this, i]() { return this->proto.sizes(i); });
}

uint64_t TilingView::operator[](size_t i) const {
  CHECK_LT(i, proto.sizes_size()) << "index overflow";
  return proto.sizes(i);
}

TilingView& TilingView::operator=(const TilingView& view) {
  proto = view.proto;
  return *this;
}

bool TilingView::operator==(const TilingView& view) const {
  return proto.SerializeAsString() == view.proto.SerializeAsString();
}

bool TilingView::operator!=(const TilingView& view) const {
  return !(*this == view);
}

//

//
// SchedulerOptionsView & SchedulerOptions
//
SchedulerOptionsView& SchedulerOptionsView::operator=(
    const SchedulerOptionsView& view) {
  proto = view.proto;
  return *this;
}

bool SchedulerOptionsView::operator==(const SchedulerOptionsView& view) const {
  return proto.SerializeAsString() == view.proto.SerializeAsString();
}

bool SchedulerOptionsView::operator!=(const SchedulerOptionsView& view) const {
  return !(*this == view);
}

//
// MappingOptionsView
//
MappingOptionsView::MappingOptionsView(const MappingOptionsView& options)
    : proto(options.proto),
      tiling(*proto.mutable_tiling()),
      outerScheduleOptions(*proto.mutable_outer_schedule_options()),
      intraTileScheduleOptions(*proto.mutable_intra_tile_schedule_options()) {}

MappingOptionsView::MappingOptionsView(MappingOptionsProto& buf)
    : proto(buf),
      tiling(*proto.mutable_tiling()),
      outerScheduleOptions(*proto.mutable_outer_schedule_options()),
      intraTileScheduleOptions(*proto.mutable_intra_tile_schedule_options()) {}

// MappingOptionsView::MappingOptionsView(const std::string& str) :
// MappingOptionsView() {
//   bool parsed = proto.ParseFromString(str);
//   CHECK(parsed) << "could not parse protobuf string";
// }

MappingOptionsView& MappingOptionsView::operator=(
    const MappingOptionsView& view) {
  proto = view.proto;
  return *this;
}

bool MappingOptionsView::operator==(const MappingOptionsView& options) const {
  return proto.SerializeAsString() == options.proto.SerializeAsString();
}

bool MappingOptionsView::operator!=(const MappingOptionsView& options) const {
  return !(*this == options);
}

//
// MappingOptionsView chainable builders.
//

MappingOptionsView& MappingOptionsView::tile(
    const std::vector<uint64_t>& sizes) {
  tiling = Tiling(sizes);
  return *this;
}

MappingOptionsView& MappingOptionsView::tile(
    std::initializer_list<uint64_t> sizes) {
  tiling = Tiling(sizes);
  return *this;
}

MappingOptionsView& MappingOptionsView::tile(const char* str) {
  return tile(std::string(str));
}

template <typename... Args>
MappingOptionsView& MappingOptionsView::tile(Args... args) {
  static_assert(
      TemplArgsAll<std::is_integral, Args...>::value,
      "arguments of tile() must be integers");
  return tile(vectorFromCastedArgs<uint64_t, Args...>(args...));
}

MappingOptionsView& MappingOptionsView::unroll(uint64_t size) {
  proto.set_unroll(size);
  return *this;
}

MappingOptionsView& MappingOptionsView::fixParametersBeforeScheduling(bool b) {
  proto.set_fix_parameters_before_scheduling(b);
  return *this;
}

MappingOptionsView& MappingOptionsView::tileImperfectlyNested(bool b) {
  proto.set_tile_imperfectly_nested(b);
  return *this;
}

MappingOptionsView& MappingOptionsView::matchLibraryCalls(bool b) {
  proto.set_match_library_calls(b);
  return *this;
}

MappingOptionsView& MappingOptionsView::scheduleFusionStrategy(
    FusionStrategy fs) {
  outerScheduleFusionStrategy(fs);
  intraTileScheduleFusionStrategy(fs);
  return *this;
}

MappingOptionsView& MappingOptionsView::scheduleFusionStrategy(
    const std::string& str) {
  FusionStrategy fs;
  bool couldParse = FusionStrategy_Parse(str, &fs);
  CHECK(couldParse) << "unknown FusionStrategy " << str;
  return scheduleFusionStrategy(fs);
}

MappingOptionsView& MappingOptionsView::outerScheduleFusionStrategy(
    FusionStrategy fs) {
  outerScheduleOptions.proto.set_fusion_strategy(fs);
  return *this;
}

MappingOptionsView& MappingOptionsView::outerScheduleFusionStrategy(
    const std::string& str) {
  FusionStrategy fs;
  bool couldParse = FusionStrategy_Parse(str, &fs);
  CHECK(couldParse) << "unknown FusionStrategy " << str;
  return outerScheduleFusionStrategy(fs);
}

MappingOptionsView& MappingOptionsView::outerScheduleAllowSkewing(bool b) {
  outerScheduleOptions.proto.set_allow_skewing(b);
  return *this;
}

MappingOptionsView& MappingOptionsView::outerSchedulePositiveOrthant(bool b) {
  outerScheduleOptions.proto.set_positive_orthant(b);
  return *this;
}

MappingOptionsView& MappingOptionsView::intraTileScheduleFusionStrategy(
    FusionStrategy fs) {
  intraTileScheduleOptions.proto.set_fusion_strategy(fs);
  return *this;
}

MappingOptionsView& MappingOptionsView::intraTileScheduleFusionStrategy(
    const std::string& str) {
  FusionStrategy fs;
  bool couldParse = FusionStrategy_Parse(str, &fs);
  CHECK(couldParse) << "unknown FusionStrategy " << str;
  return intraTileScheduleFusionStrategy(fs);
}

MappingOptionsView& MappingOptionsView::intraTileScheduleAllowSkewing(bool b) {
  intraTileScheduleOptions.proto.set_allow_skewing(b);
  return *this;
}

MappingOptionsView& MappingOptionsView::intraTileSchedulePositiveOrthant(
    bool b) {
  intraTileScheduleOptions.proto.set_positive_orthant(b);
  return *this;
}

//
// Predefined stratgies
//
MappingOptions MappingOptions::makeUnmappedMappingOptions() {
  MappingOptions mo;
  mo.outerScheduleFusionStrategy(FusionStrategy::Preserve3Coincident)
      .outerScheduleAllowSkewing(false)
      .outerSchedulePositiveOrthant(true)
      .intraTileScheduleFusionStrategy(FusionStrategy::Preserve3Coincident)
      .intraTileScheduleAllowSkewing(false)
      .intraTileSchedulePositiveOrthant(true)
      .tile(1)
      .fixParametersBeforeScheduling(false)
      .matchLibraryCalls(false)
      .tileImperfectlyNested(false);
#if 1
  // @ftynse The following is extremely confusing: in the current nesting of
  // views, one seems to need to to the following so that the underlying proto
  // "sees" that mo.ownedProto.has_outer_schedule_options. Without that, all
  // hell breaks loose and this inversion of control is extremely confusing. I
  // wonder if this is a general problem with deeper nesting of Views than 1
  // level.
  //
  // With this:
  // test/test_mapper_memory_promotion segfaults
  // test/test_mapper_memory_promotion segfaults
  *mo.ownedProto_.mutable_outer_schedule_options() =
      mo.outerScheduleOptions.proto;
  *mo.ownedProto_.mutable_intra_tile_schedule_options() =
      mo.intraTileScheduleOptions.proto;
  *mo.ownedProto_.mutable_tiling() = mo.tiling.proto;
  LOG(ERROR) << mo;
#endif
  return mo;
}

MappingOptions MappingOptions::makeNaiveMappingOptions() {
  return MappingOptions(makeUnmappedMappingOptions().tile({32, 32, 32}));
}

MappingOptions MappingOptions::makeSingleThreadMappingOptions() {
  return MappingOptions(makeUnmappedMappingOptions().tile({1}).unroll(1));
}

MappingOptions MappingOptions::makePointwiseMappingOptions() {
  return MappingOptions(
      makeUnmappedMappingOptions().tile({32, 32, 32}).unroll(128));
}

MappingOptions MappingOptions::makeMlpMappingOptions() {
  return MappingOptions(makeUnmappedMappingOptions()
                            .outerScheduleFusionStrategy(FusionStrategy::Max)
                            .tile({1})
                            .unroll(1));
}

MappingOptions MappingOptions::makeConvolutionMappingOptions() {
  return MappingOptions(
      makeUnmappedMappingOptions().tile({4, 8, 8, 8}).unroll(1));
}

MappingOptions MappingOptions::makeGroupConvolutionMappingOptions() {
  return MappingOptions(makeUnmappedMappingOptions().tile({1, 1}).unroll(1));
}
} // namespace tc
