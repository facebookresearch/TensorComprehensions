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

#include "tc/core/check.h"
#include "tc/core/utils/vararg.h"

namespace tc {

//
// TilingView & Tiling
//
Tiling::Tiling(const std::vector<uint64_t>& sizes)
    : ownedProto_(), view(ownedProto_) {
  ownedProto_.clear_sizes();
  std::copy(
      sizes.begin(),
      sizes.end(),
      google::protobuf::RepeatedFieldBackInserter(ownedProto_.mutable_sizes()));
}

std::vector<uint64_t> TilingView::extractVector() const {
  std::vector<uint64_t> result(proto.sizes().begin(), proto.sizes().end());
  return result;
}

size_t TilingView::size() const {
  return proto.sizes_size();
}

ValueAccessor<uint64_t> TilingView::operator[](size_t i) {
  TC_CHECK_LT(i, static_cast<size_t>(proto.sizes_size())) << "index overflow";
  return ValueAccessor<uint64_t>(
      [this, i](uint64_t u) { this->proto.set_sizes(i, u); },
      [this, i]() { return this->proto.sizes(i); });
}

uint64_t TilingView::operator[](size_t i) const {
  TC_CHECK_LT(i, static_cast<size_t>(proto.sizes_size())) << "index overflow";
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
  tiling = Tiling(sizes).view; // tmp Tiling, copy, delete
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
  FusionStrategy fs(FusionStrategy::Max);
  bool couldParse = FusionStrategy_Parse(str, &fs);
  TC_CHECK(couldParse) << "unknown FusionStrategy " << str;
  return scheduleFusionStrategy(fs);
}

MappingOptionsView& MappingOptionsView::outerScheduleFusionStrategy(
    FusionStrategy fs) {
  outerScheduleOptions.proto.set_fusion_strategy(fs);
  return *this;
}

MappingOptionsView& MappingOptionsView::outerScheduleFusionStrategy(
    const std::string& str) {
  FusionStrategy fs(FusionStrategy::Max);
  bool couldParse = FusionStrategy_Parse(str, &fs);
  TC_CHECK(couldParse) << "unknown FusionStrategy " << str;
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
  FusionStrategy fs(FusionStrategy::Max);
  bool couldParse = FusionStrategy_Parse(str, &fs);
  TC_CHECK(couldParse) << "unknown FusionStrategy " << str;
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
// Predefined strategies
//
MappingOptions MappingOptions::makeUnmappedMappingOptions() {
  MappingOptions mo;
  mo.view.outerScheduleFusionStrategy(FusionStrategy::Preserve3Coincident)
      .outerScheduleAllowSkewing(false)
      .outerSchedulePositiveOrthant(true)
      .intraTileScheduleFusionStrategy(FusionStrategy::Preserve3Coincident)
      .intraTileScheduleAllowSkewing(false)
      .intraTileSchedulePositiveOrthant(true)
      .fixParametersBeforeScheduling(false)
      .matchLibraryCalls(false)
      .tileImperfectlyNested(false);
  return mo;
}

MappingOptions MappingOptions::makeNaiveMappingOptions() {
  return makeUnmappedMappingOptions().tile(32, 32, 32).unroll(1);
}

MappingOptions MappingOptions::makeSingleThreadMappingOptions() {
  return makeUnmappedMappingOptions().tile(1).unroll(1);
}

MappingOptions MappingOptions::makePointwiseMappingOptions() {
  return makeUnmappedMappingOptions().tile(32, 32, 32).unroll(128);
}

MappingOptions MappingOptions::makeMlpMappingOptions() {
  return makeUnmappedMappingOptions()
      .view.outerScheduleFusionStrategy(FusionStrategy::Max)
      .tile(1)
      .unroll(1);
}

MappingOptions MappingOptions::makeConvolutionMappingOptions() {
  return makeUnmappedMappingOptions().tile(4, 8, 8, 8).unroll(1);
}

MappingOptions MappingOptions::makeGroupConvolutionMappingOptions() {
  return makeUnmappedMappingOptions().tile(1, 1).unroll(1);
}
} // namespace tc
