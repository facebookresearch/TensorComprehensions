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
#include "tc/autotuner/parameters.h"
#include <algorithm>
#include <cassert>
#include <cmath>
#include <iterator>
#include <numeric>
#include <ostream>
#include <sstream>
#include <typeinfo>

namespace tc {
namespace autotune {

BoolParameter& BoolParameter::operator=(const BoolParameter& other) {
  value_ = other.value_;
  fixedValue_ = other.fixedValue_;
  name = other.name;
  return *this;
}

BoolParameter::BoolParameter(const std::string& name) : name(name) {}
BoolParameter::BoolParameter(const BoolParameter& other)
    : name(other.name), value_(other.value_), fixedValue_(other.fixedValue_) {}

size_t BoolParameter::numberOptions() const {
  return 2;
}

void BoolParameter::apply(const std::function<void(ParameterView&)>& f) {
  ParameterView pv(*this);
  f(pv);
}

bool BoolParameter::value() const {
  if (fixedValue_) {
    return *fixedValue_;
  }
  return value_;
}

void BoolParameter::fixValue(bool val) {
  fixedValue_ = val;
}

void RangeParameter::fixValue(size_t val) {
  fixedValue_ = val;
}

void RangeParameter::apply(const std::function<void(ParameterView&)>& f) {
  ParameterView pv(*this);
  f(pv);
}

size_t RangeParameter::numberOptions() const {
  return values_.size();
}

size_t RangeParameter::value() const {
  if (fixedValue_) {
    return *fixedValue_;
  }
  return values_.at(selected_);
}

RangeParameter::RangeParameter(
    std::vector<size_t> values,
    const std::string& name)
    : name(name), values_(std::move(values)) {}

RangeParameter::RangeParameter(const RangeParameter& other)
    : name(other.name),
      selected_(other.selected_),
      values_(other.values_),
      fixedValue_(other.fixedValue_) {}

RangeParameter& RangeParameter::operator=(const RangeParameter& other) {
  fixedValue_ = other.fixedValue_;
  selected_ = other.selected_;
  values_ = other.values_;
  name = other.name;
  return *this;
}

void BoolParameter::selectOption(size_t idx) {
  CHECK_LE(idx, 1u);
  selectValue(idx);
}

void BoolParameter::selectValue(bool val) {
  value_ = val;
}

void RangeParameter::selectOption(size_t idx) {
  CHECK_LE(idx, values_.size());
  selected_ = idx;
}

void RangeParameter::selectFromValue(size_t value) {
  auto it = std::find(values_.begin(), values_.end(), value);
  if (it == values_.end()) {
    std::stringstream ss;
    ss << "RangeParameter::selectFromValue: value " << value
       << " not in range. The valid values are: ";
    std::copy(
        values_.begin(), values_.end(), std::ostream_iterator<size_t>(ss, " "));
    throw std::invalid_argument(ss.str());
  }
  selected_ = std::distance(values_.begin(), it);
}

void ParameterView::overwrite(const ParameterView& pv) {
  CHECK_EQ(rangePtr == nullptr, pv.rangePtr == nullptr);
  CHECK_EQ(boolPtr == nullptr, pv.boolPtr == nullptr);
  if (rangePtr) {
    *rangePtr = *pv.rangePtr;
  } else {
    *boolPtr = *pv.boolPtr;
  }
}

bool ParameterView::isForced() const {
  CHECK((rangePtr == nullptr) xor (boolPtr == nullptr));
  if (rangePtr) {
    return rangePtr->fixedValue_.hasValue();
  } else {
    return boolPtr->fixedValue_.hasValue();
  }
}

size_t ParameterView::numberOptions() const {
  CHECK((rangePtr == nullptr) xor (boolPtr == nullptr));
  if (rangePtr) {
    return rangePtr->numberOptions();
  } else {
    return boolPtr->numberOptions();
  }
}

void ParameterView::selectOption(size_t idx) {
  CHECK((rangePtr == nullptr) xor (boolPtr == nullptr));
  if (rangePtr) {
    return rangePtr->selectOption(idx);
  } else {
    return boolPtr->selectOption(idx);
  }
}

ParameterView::ParameterView(BoolParameter& p)
    : rangePtr(nullptr), boolPtr(&p) {}
ParameterView::ParameterView(RangeParameter& p)
    : rangePtr(&p), boolPtr(nullptr) {}

void TuningConfiguration::addValidator(
    std::function<bool(const TuningConfiguration&)> v) {
  validators_.push_back(v);
}

SchedulerOptionsParameters::SchedulerOptionsParameters()
    : fusionStrategy({0, 1, 2}, "fusion strategy") {}

void SchedulerOptionsParameters::apply(
    const std::function<void(ParameterView&)>& f) {
  fusionStrategy.apply(f);
}

std::vector<ParameterView> SchedulerOptionsParameters::collectParameters() {
  std::vector<ParameterView> params;
  params.reserve(1);
  params.emplace_back(fusionStrategy);

  return params;
}

void SchedulerOptionsParameters::applyToMappingOptions(
    SchedulerOptionsView& options) const {
  switch (fusionStrategy.value()) {
    case 0:
      options.proto.set_fusion_strategy(FusionStrategy::Max);
      break;
    case 1:
      options.proto.set_fusion_strategy(FusionStrategy::Preserve3Coincident);
      break;
    case 2:
      options.proto.set_fusion_strategy(FusionStrategy::Min);
      break;
    default:
      throw std::invalid_argument("Unknown fusion strategy.");
  }
}

namespace {
size_t toInt(const FusionStrategy& fs) {
  switch (fs) {
    case Max:
      return 0;
    case Preserve3Coincident:
      return 1;
    case Min:
      return 2;
    default:
      throw std::invalid_argument("Unknown fusion strategy");
  }
}
} // namespace

void SchedulerOptionsParameters::fromMappingOptions(
    const SchedulerOptionsView& options) {
  fusionStrategy.selectOption(toInt(options.proto.fusion_strategy()));
}

void TuningConfiguration::applyToParameters(
    const std::function<void(ParameterView&)>& f) {
  outerScheduleOptions.apply(f);
  intraTileScheduleOptions.apply(f);
  fixParametersBeforeScheduling.apply(f);
  tilingParams.apply(f);
  blockParams.apply(f);
  gridParams.apply(f);
  unrollFactor.apply(f);
  useSharedMemory.apply(f);
  usePrivateMemory.apply(f);
  unrollCopyShared.apply(f);
  matchLibraryCalls.apply(f);
}

bool TuningConfiguration::isValid() const {
  for (const auto& v : validators_) {
    if (!v(*this)) {
      return false;
    }
  }
  return true;
}

std::vector<ParameterView> TuningConfiguration::collectParameters() {
  std::vector<ParameterView> params;
  params.reserve(26);
  auto collect = [&](std::vector<ParameterView>&& newParams) {
    params.reserve(params.size() + newParams.size());
    std::move(
        std::make_move_iterator(newParams.begin()),
        std::make_move_iterator(newParams.end()),
        std::back_inserter(params));
  };
  collect(outerScheduleOptions.collectParameters());
  collect(intraTileScheduleOptions.collectParameters());
  params.emplace_back(fixParametersBeforeScheduling);
  collect(tilingParams.collectParameters());
  collect(blockParams.collectParameters());
  collect(gridParams.collectParameters());

  params.emplace_back(unrollFactor);
  params.emplace_back(useSharedMemory);
  params.emplace_back(usePrivateMemory);
  params.emplace_back(unrollCopyShared);
  params.emplace_back(matchLibraryCalls);

  return params;
}

void TuningConfiguration::fromMappingOptions(
    const MappingOptionsView& options) {
  outerScheduleOptions.fromMappingOptions(options.outerScheduleOptions);
  intraTileScheduleOptions.fromMappingOptions(options.intraTileScheduleOptions);
  fixParametersBeforeScheduling.selectValue(
      options.proto.fix_parameters_before_scheduling());
  tilingParams.fromMappingOptions(options.tiling);
  unrollFactor.selectFromValue(
      (options.proto.has_unroll() ? options.proto.unroll() : 1));
  tileImperfectlyNested.selectValue(options.proto.tile_imperfectly_nested());
  matchLibraryCalls.selectValue(options.proto.match_library_calls());
}

void TuningConfiguration::fromCudaMappingOptions(
    const CudaMappingOptions& options) {
  fromMappingOptions(options.generic);
  blockParams.fromMappingOptions(options.block);
  gridParams.fromMappingOptions(options.grid);
  useSharedMemory.selectValue(options.proto().use_shared_memory());
  usePrivateMemory.selectValue(options.proto().use_private_memory());
  unrollCopyShared.selectValue(options.proto().unroll_copy_shared());
}

void TuningConfiguration::fromCpuMappingOptions(
    const CpuMappingOptions& options) {
  fromMappingOptions(options.generic);
}

void TuningConfiguration::applyToMappingOptions(
    MappingOptionsView& options) const {
  outerScheduleOptions.applyToMappingOptions(options.outerScheduleOptions);
  intraTileScheduleOptions.applyToMappingOptions(
      options.intraTileScheduleOptions);
  options.fixParametersBeforeScheduling(fixParametersBeforeScheduling.value());
  tilingParams.applyToMappingOptions(options.tiling);
  options.unroll(unrollFactor.value());
  options.tileImperfectlyNested(tileImperfectlyNested.value());
  options.matchLibraryCalls(matchLibraryCalls.value());
}

void TuningConfiguration::applyToCudaMappingOptions(
    CudaMappingOptions& options) const {
  applyToMappingOptions(options.generic);
  blockParams.applyToMappingOptions(options.block);
  gridParams.applyToMappingOptions(options.grid);
  options.useSharedMemory(useSharedMemory.value());
  options.usePrivateMemory(usePrivateMemory.value());
  options.unrollCopyShared(unrollCopyShared.value());
}

void TuningConfiguration::applyToCpuMappingOptions(
    CpuMappingOptions& options) const {
  applyToMappingOptions(options.generic);
}

TuningConfiguration::TuningConfiguration()
    : fixParametersBeforeScheduling("fix parameters before scheduling"),
      tileImperfectlyNested("tile imperfectly nested"),
      useSharedMemory("use shared memory"),
      usePrivateMemory("use private memory"),
      unrollCopyShared("unroll copy shared"),
      matchLibraryCalls("match library calls") {
  addValidator([](const TuningConfiguration& conf) {
    auto b0v = conf.blockParams.dims.at(0).value();
    auto b1v = conf.blockParams.dims.at(1).value();
    auto b2v = conf.blockParams.dims.at(2).value();
    if (b0v <= 0 or b0v > 1024 or b1v <= 0 or b1v > 1024 or b2v <= 0 or
        b2v > 64) {
      return false;
    }
    auto blockProduct = [&]() {
      switch (conf.blockParams.numberDims.value()) {
        case 3:
          return b0v * b1v * b2v;
        case 2:
          return b0v * b1v;
        case 1:
          return b0v;
        default:
          CHECK(false) << "Must have (1-3) block dims, got: "
                       << conf.blockParams.numberDims.value();
      }
      return b0v;
    }();
    if (blockProduct < 32 or blockProduct > 512) {
      return false;
    }
    return true;
  });
}

namespace {

template <typename T, typename Param>
void maybeFixScalar(const llvm::Optional<T>& maybeFixed, Param& param) {
  if (maybeFixed) {
    param.fixValue(*maybeFixed);
  }
}

void maybeFixFusionStrategy(
    const llvm::Optional<FusionStrategy>& maybeFixed,
    RangeParameter& param) {
  if (maybeFixed) {
    param.fixValue(toInt(*maybeFixed));
  }
}

void maybeFixVector(
    const llvm::Optional<std::vector<size_t>>& maybeFixed,
    MultiRangeParams& param) {
  if (not maybeFixed) {
    return;
  }
  const auto& values = *maybeFixed;
  param.numberDims.fixValue(values.size());
  param.dims.resize(values.size());
  for (size_t i = 0; i < values.size(); ++i) {
    param.dims.at(i).fixValue(values.at(i));
  }
}

} // namespace

void TuningConfiguration::fixParameters(
    const TuningParameterFixer& fixedParams) {
  maybeFixFusionStrategy(
      fixedParams.outerScheduleFusionStrategy,
      outerScheduleOptions.fusionStrategy);
  maybeFixFusionStrategy(
      fixedParams.intraTileScheduleFusionStrategy,
      intraTileScheduleOptions.fusionStrategy);
  maybeFixScalar(
      fixedParams.fixParametersBeforeScheduling, fixParametersBeforeScheduling);
  maybeFixScalar(fixedParams.unrollFactor, unrollFactor);
  maybeFixVector(fixedParams.tilingParameters, tilingParams);
  maybeFixVector(fixedParams.blockParameters, blockParams);
  maybeFixVector(fixedParams.gridParameters, gridParams);
  maybeFixScalar(fixedParams.tileImperfectlyNested, tileImperfectlyNested);
  maybeFixScalar(fixedParams.useSharedMemory, useSharedMemory);
  maybeFixScalar(fixedParams.usePrivateMemory, usePrivateMemory);
  maybeFixScalar(fixedParams.unrollCopyShared, unrollCopyShared);
  maybeFixScalar(fixedParams.matchLibraryCalls, matchLibraryCalls);
}

void MultiRangeParams::setRange(
    size_t minDims,
    size_t maxDims,
    const std::string& name,
    std::vector<size_t>& values,
    const std::string& dimBaseName) {
  std::vector<size_t> dimValues;
  dimValues.resize(maxDims - minDims + 1);
  std::iota(dimValues.begin(), dimValues.end(), minDims);
  numberDims = RangeParameter(dimValues, name);
  dims.reserve(maxDims);
  for (size_t i = 0; i < maxDims; ++i) {
    dims.emplace_back(values, dimBaseName + std::to_string(i));
  }
}

void TilingParameters::setRange(size_t maxDims, std::vector<size_t>& values) {
  MultiRangeParams::setRange(
      1, maxDims, "number of tiling dimensions", values, "t");
}

void CudaDimParameters::setRange(
    std::vector<size_t>& values,
    const std::string& dimBaseName) {
  MultiRangeParams::setRange(
      1, 3, "number of cuda dimensions", values, dimBaseName);
}

namespace {
template <typename Params, typename View>
void fromMappingOptions(Params& params, const View& options) {
  CHECK_LE(options.size(), params.dims.size());
  params.numberDims.selectFromValue(options.size());
  for (size_t i = 0; i < options.size(); ++i) {
    params.dims[i].selectFromValue(options[i]);
  }
}
} // namespace

void TilingParameters::fromMappingOptions(const TilingView& options) {
  ::tc::autotune::fromMappingOptions(*this, options);
}

void CudaDimParameters::fromMappingOptions(const CudaDimView& options) {
  ::tc::autotune::fromMappingOptions(*this, options);
}

std::vector<ParameterView> MultiRangeParams::collectParameters() {
  std::vector<ParameterView> params;
  params.reserve(1 + dims.size());
  params.emplace_back(numberDims);
  std::copy(dims.begin(), dims.end(), std::back_inserter(params));
  return params;
}

void MultiRangeParams::apply(const std::function<void(ParameterView&)>& f) {
  numberDims.apply(f);
  for (auto& p : dims) {
    p.apply(f);
  }
}

void TilingParameters::applyToMappingOptions(TilingView& options) const {
  options.proto.clear_sizes();
  for (size_t i = 0; i < numberDims.value(); ++i) {
    options.proto.add_sizes(dims.at(i).value());
  }
}

void CudaDimParameters::applyToMappingOptions(CudaDimView& options) const {
  auto proto = options.proto.default_instance();
  switch (numberDims.value()) {
    case 3:
      proto.set_z(dims.at(2).value());
    case 2:
      proto.set_y(dims.at(1).value());
    case 1:
      proto.set_x(dims.at(0).value());
      break;
    default:
      throw std::invalid_argument(
          "The number of cuda dimensions must belong to [1,3]");
  }
  options.proto = proto;
}

TuningParameterFixer& TuningParameterFixer::fixOuterScheduleFusionStrategy(
    const FusionStrategy& fs) {
  outerScheduleFusionStrategy = fs;
  return *this;
}

TuningParameterFixer& TuningParameterFixer::fixIntraTileScheduleFusionStrategy(
    const FusionStrategy& fs) {
  intraTileScheduleFusionStrategy = fs;
  return *this;
}

TuningParameterFixer& TuningParameterFixer::fixFixParametersBeforeScheduling(
    bool val) {
  fixParametersBeforeScheduling = val;
  return *this;
}

TuningParameterFixer& TuningParameterFixer::fixUnrollFactor(size_t val) {
  unrollFactor = val;
  return *this;
}

TuningParameterFixer& TuningParameterFixer::fixTilingParameters(
    std::vector<size_t> vals) {
  tilingParameters = vals;
  return *this;
}

TuningParameterFixer& TuningParameterFixer::fixBlockParameters(
    std::vector<size_t> vals) {
  blockParameters = vals;
  return *this;
}

TuningParameterFixer& TuningParameterFixer::fixGridParameters(
    std::vector<size_t> vals) {
  gridParameters = vals;
  return *this;
}

TuningParameterFixer& TuningParameterFixer::fixTileImperfectlyNested(bool val) {
  tileImperfectlyNested = val;
  return *this;
}
TuningParameterFixer& TuningParameterFixer::fixUseSharedMemory(bool val) {
  useSharedMemory = val;
  return *this;
}

TuningParameterFixer& TuningParameterFixer::fixUsePrivateMemory(bool val) {
  usePrivateMemory = val;
  return *this;
}

TuningParameterFixer& TuningParameterFixer::fixUnrollCopyShared(bool val) {
  unrollCopyShared = val;
  return *this;
}

TuningParameterFixer& TuningParameterFixer::fixMatchLibraryCalls(bool val) {
  matchLibraryCalls = val;
  return *this;
}

} // namespace autotune
} // namespace tc
