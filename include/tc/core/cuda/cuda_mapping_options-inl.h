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

namespace tc {

// CudaDimView & CudaDim
//
CudaDim::CudaDim(std::vector<uint64_t> il) : CudaDimView(ownedProto_) {
  CHECK_GT(il.size(), 0) << "list of values in CudaDimView must be non-empty";
  CHECK_LE(il.size(), 3) << "at most 3 values allowed in CudaDimView";

  switch (il.size()) {
    case 3:
      proto.set_z(*(il.begin() + 2));
    case 2:
      proto.set_y(*(il.begin() + 1));
    case 1:
      proto.set_x(*il.begin());
      break;
    default:
      CHECK(false) << "unreachable";
  }
}

CudaDim::CudaDim(std::initializer_list<uint64_t> il)
    : CudaDim(std::vector<uint64_t>(il)) {}

CudaDim::CudaDim(uint64_t x, uint64_t y, uint64_t z)
    : CudaDimView(ownedProto_) {
  proto.set_x(x);
  if (y != defaultDim || z != defaultDim) {
    proto.set_y(y);
  }
  if (z != defaultDim) {
    proto.set_z(z);
  }
}

size_t CudaDimView::size() const {
  CHECK(!(!proto.has_y() && proto.has_z())) << "CudaDimView has z but not y";

  if (proto.has_z() && proto.has_y()) {
    return 3;
  } else if (proto.has_y()) {
    return 2;
  }
  return 1;
}

std::vector<uint64_t> CudaDimView::extractVector() const {
  CHECK(!(!proto.has_y() && proto.has_z())) << "CudaDimView has z but not y";

  std::vector<uint64_t> result;
  result.push_back(proto.x());
  if (proto.has_y()) {
    result.push_back(proto.y());
  }
  if (proto.has_z()) {
    result.push_back(proto.z());
  }
  return result;
}

std::array<uint64_t, 3> CudaDimView::extractDefaultedArray() const {
  std::array<uint64_t, 3> arr{CudaDimView::defaultDim,
                              CudaDimView::defaultDim,
                              CudaDimView::defaultDim};
  auto v = extractVector();
  CHECK_LE(v.size(), 3);
  std::copy(v.begin(), v.end(), arr.begin());
  return arr;
}

ValueAccessor<uint64_t> CudaDimView::operator[](size_t i) {
  CHECK_LT(i, 3) << "index overflow";
  if (i == 0) {
    return ValueAccessor<uint64_t>(
        [this](uint64_t u) { this->proto.set_x(u); },
        [this]() { return this->proto.x(); });
  } else if (i == 1) {
    return ValueAccessor<uint64_t>(
        [this](uint64_t u) { this->proto.set_y(u); },
        [this]() {
          return this->proto.has_y() ? this->proto.y()
                                     : CudaDimView::defaultDim;
        });
  } else {
    return ValueAccessor<uint64_t>(
        [this](uint64_t u) { this->proto.set_z(u); },
        [this]() {
          return this->proto.has_z() ? this->proto.z()
                                     : CudaDimView::defaultDim;
        });
  }
}

uint64_t CudaDimView::operator[](size_t i) const {
  CHECK_LT(i, 3) << "index overflow";
  if (i == 0) {
    return proto.x();
  } else if (i == 1) {
    return proto.has_y() ? proto.y() : CudaDimView::defaultDim;
  } else {
    return proto.has_z() ? proto.z() : CudaDimView::defaultDim;
  }
}

CudaDimView& CudaDimView::operator=(const CudaDimView& view) {
  proto = view.proto;
  return *this;
}

bool CudaDimView::operator==(const CudaDimView& view) const {
  return proto.SerializeAsString() == view.proto.SerializeAsString();
}

bool CudaDimView::operator!=(const CudaDimView& view) const {
  return !(*this == view);
}

//
// CudaMappingOptions
//
CudaMappingOptions::CudaMappingOptions()
    : genericMappingOptionsView(*proto.mutable_generic_mapping_options()),
      block(*proto.mutable_block()),
      grid(*proto.mutable_grid()) {}

CudaMappingOptions::CudaMappingOptions(const CudaMappingOptions& options)
    : proto(options.proto),
      genericMappingOptionsView(options.genericMappingOptionsView),
      block(*proto.mutable_block()),
      grid(*proto.mutable_grid()) {}

CudaMappingOptions::CudaMappingOptions(const CudaMappingOptionsProto& buf)
    : proto(buf),
      genericMappingOptionsView(*proto.mutable_generic_mapping_options()),
      block(*proto.mutable_block()),
      grid(*proto.mutable_grid()) {}

bool CudaMappingOptions::operator==(const CudaMappingOptions& options) const {
  return proto.SerializeAsString() == options.proto.SerializeAsString();
}

bool CudaMappingOptions::operator!=(const CudaMappingOptions& options) const {
  return proto.SerializeAsString() != options.proto.SerializeAsString();
}

CudaMappingOptions::CudaMappingOptions(const std::string& str)
    : CudaMappingOptions() {
  bool parsed = proto.ParseFromString(str);
  genericMappingOptionsView =
      MappingOptionsView(*proto.mutable_generic_mapping_options());
  block = CudaDimView(*proto.mutable_block());
  grid = CudaDimView(*proto.mutable_grid());
  CHECK(parsed) << "could not parse protobuf string";
}

CudaMappingOptions& CudaMappingOptions::tile(
    const std::vector<uint64_t>& sizes) {
  genericMappingOptionsView.tile(sizes);
  return *this;
}

CudaMappingOptions& CudaMappingOptions::tile(
    std::initializer_list<uint64_t> sizes) {
  genericMappingOptionsView.tile(sizes);
  return *this;
}

CudaMappingOptions& CudaMappingOptions::tile(
    const std::string& commaSeparatedSizes) {
  genericMappingOptionsView.tile(commaSeparatedSizes);
  return *this;
}

CudaMappingOptions& CudaMappingOptions::tile(const char* commaSeparatedSizes) {
  genericMappingOptionsView.tile(commaSeparatedSizes);
  return *this;
}

template <typename... Args>
CudaMappingOptions& CudaMappingOptions::tile(Args... args) {
  genericMappingOptionsView.tile(args...);
  return *this;
}

CudaMappingOptions& CudaMappingOptions::unroll(uint64_t size) {
  genericMappingOptionsView.unroll(size);
  return *this;
}

CudaMappingOptions& CudaMappingOptions::fixParametersBeforeScheduling(bool b) {
  genericMappingOptionsView.fixParametersBeforeScheduling(b);
  return *this;
}

CudaMappingOptions& CudaMappingOptions::tileImperfectlyNested(bool b) {
  genericMappingOptionsView.tileImperfectlyNested(b);
  return *this;
}

CudaMappingOptions& CudaMappingOptions::matchLibraryCalls(bool b) {
  genericMappingOptionsView.matchLibraryCalls(b);
  return *this;
}

CudaMappingOptions& CudaMappingOptions::scheduleFusionStrategy(
    FusionStrategy fs) {
  genericMappingOptionsView.scheduleFusionStrategy(fs);
  return *this;
}

CudaMappingOptions& CudaMappingOptions::scheduleFusionStrategy(
    const std::string& str) {
  genericMappingOptionsView.scheduleFusionStrategy(str);
  return *this;
}

CudaMappingOptions& CudaMappingOptions::outerScheduleFusionStrategy(
    FusionStrategy fs) {
  genericMappingOptionsView.outerScheduleFusionStrategy(fs);
  return *this;
}

CudaMappingOptions& CudaMappingOptions::outerScheduleFusionStrategy(
    const std::string& str) {
  genericMappingOptionsView.outerScheduleFusionStrategy(str);
  return *this;
}

CudaMappingOptions& CudaMappingOptions::outerScheduleAllowSkewing(bool b) {
  genericMappingOptionsView.outerScheduleAllowSkewing(b);
  return *this;
}

CudaMappingOptions& CudaMappingOptions::outerSchedulePositiveOrthant(bool b) {
  genericMappingOptionsView.outerSchedulePositiveOrthant(b);
  return *this;
}

CudaMappingOptions& CudaMappingOptions::intraTileScheduleFusionStrategy(
    FusionStrategy fs) {
  genericMappingOptionsView.intraTileScheduleFusionStrategy(fs);
  return *this;
}

CudaMappingOptions& CudaMappingOptions::intraTileScheduleFusionStrategy(
    const std::string& str) {
  genericMappingOptionsView.intraTileScheduleFusionStrategy(str);
  return *this;
}

CudaMappingOptions& CudaMappingOptions::intraTileScheduleAllowSkewing(bool b) {
  genericMappingOptionsView.intraTileScheduleAllowSkewing(b);
  return *this;
}

CudaMappingOptions& CudaMappingOptions::intraTileSchedulePositiveOrthant(
    bool b) {
  genericMappingOptionsView.intraTileSchedulePositiveOrthant(b);
  return *this;
}

CudaMappingOptions& CudaMappingOptions::mapToThreads(
    std::initializer_list<uint64_t> threads) {
  block = CudaDim(threads);
  return *this;
}

CudaMappingOptions&
CudaMappingOptions::mapToThreads(uint64_t x, uint64_t y, uint64_t z) {
  block = CudaDim(x, y, z);
  return *this;
}

CudaMappingOptions& CudaMappingOptions::mapToThreads(
    const std::vector<uint64_t>& threads) {
  CHECK_GT(threads.size(), 0) << "expected at least one thread size";
  CHECK_LE(threads.size(), 3) << "expected at most three thread sizes";

  uint64_t x = threads[0];
  uint64_t y = threads.size() > 1 ? threads[1] : CudaDimView::defaultDim;
  uint64_t z = threads.size() > 2 ? threads[2] : CudaDimView::defaultDim;
  block = CudaDim(x, y, z);
  return *this;
}

CudaMappingOptions& CudaMappingOptions::mapToBlocks(
    std::initializer_list<uint64_t> blocks) {
  grid = CudaDim(blocks);
  return *this;
}

CudaMappingOptions&
CudaMappingOptions::mapToBlocks(uint64_t x, uint64_t y, uint64_t z) {
  grid = CudaDim(x, y, z);
  return *this;
}

CudaMappingOptions& CudaMappingOptions::mapToBlocks(
    const std::vector<uint64_t>& blocks) {
  CHECK_GT(blocks.size(), 0) << "expected at least one thread size";
  CHECK_LE(blocks.size(), 3) << "expected at most three thread sizes";

  uint64_t x = blocks[0];
  uint64_t y = blocks.size() > 1 ? blocks[1] : CudaDimView::defaultDim;
  uint64_t z = blocks.size() > 2 ? blocks[2] : CudaDimView::defaultDim;
  grid = CudaDim(x, y, z);
  return *this;
}

CudaMappingOptions& CudaMappingOptions::genericMappingOptions(
    const MappingOptions& options) {
  *(proto.mutable_generic_mapping_options()) = options.proto;
  return *this;
}

CudaMappingOptions& CudaMappingOptions::useSharedMemory(bool b) {
  proto.set_use_shared_memory(b);
  return *this;
}

CudaMappingOptions& CudaMappingOptions::usePrivateMemory(bool b) {
  proto.set_use_private_memory(b);
  return *this;
}

CudaMappingOptions& CudaMappingOptions::maxSharedMemory(uint64_t size) {
  proto.set_max_shared_memory(size);
  return *this;
}

CudaMappingOptions& CudaMappingOptions::unrollCopyShared(bool b) {
  proto.set_unroll_copy_shared(b);
  return *this;
}

} // namespace tc
