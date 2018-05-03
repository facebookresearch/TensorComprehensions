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
#include "tc/core/cuda/cuda_mapping_options.h"

#include <fstream>
#include <iomanip>
#include <iostream>
#include <tuple>
#include <type_traits>

#include "tc/proto/mapping_options.pb.h"

#include "tc/core/cuda/cuda_mapping_options_cpp_printer.h"
#include "tc/core/flags.h"
#include "tc/core/utils/string.h"
#include "tc/external/isl.h"

namespace tc {

const uint64_t CudaDimView::defaultDim;

//
// Output operators and string conversion
//
std::string CudaDimView::toCommaSeparatedString() const {
  std::stringstream ss;
  ss << proto.x();
  if (proto.has_y()) {
    ss << ", " << proto.y();
  }
  if (proto.has_z()) {
    ss << ", " << proto.z();
  }
  return ss.str();
}

std::ostream& operator<<(std::ostream& os, const CudaDimView& view) {
  os << "CudaDim(" << view.toCommaSeparatedString() << ") @" << &view.proto;
  return os;
}

std::ostream& operator<<(std::ostream& os, const CudaDim& dim) {
  os << dim.view;
  return os;
}

std::ostream& operator<<(std::ostream& os, const Grid& dim) {
  os << dim.view;
  return os;
}

std::ostream& operator<<(std::ostream& os, const Block& dim) {
  os << dim.view;
  return os;
}

std::ostream& operator<<(
    std::ostream& os,
    const CudaMappingOptions& cudaOptions) {
  OstreamBoolalphaScope scope(os);
  tc::CudaMappingOptionsAsCpp cpp(cudaOptions);
  os << cpp;
  return os;
}

CudaMappingOptions& CudaMappingOptions::mapToThreads(
    const std::string& commaSeparatedSizes) {
  auto sizes = parseCommaSeparatedIntegers<uint64_t>(commaSeparatedSizes);
  CHECK_GT(sizes.size(), 0u)
      << "expected at least one block size in " << commaSeparatedSizes;
  CHECK_LE(sizes.size(), 3u)
      << "expected at most three block sizes in " << commaSeparatedSizes;
  sizes.resize(3, CudaDimView::defaultDim);
  return mapToThreads(sizes[0], sizes[1], sizes[2]);
}

CudaMappingOptions& CudaMappingOptions::mapToBlocks(
    const std::string& commaSeparatedSizes) {
  auto sizes = parseCommaSeparatedIntegers<uint64_t>(commaSeparatedSizes);
  CHECK_GT(sizes.size(), 0u)
      << "expected at least one grid size in " << commaSeparatedSizes;
  CHECK_LE(sizes.size(), 3u)
      << "expected at most three grid sizes in " << commaSeparatedSizes;
  sizes.resize(3, CudaDimView::defaultDim);
  return mapToBlocks(sizes[0], sizes[1], sizes[2]);
}

//
// Predefined strategies
//
CudaMappingOptions CudaMappingOptions::makeUnmappedMappingOptions() {
  CudaMappingOptions mo;
  mo.genericMappingOptions(MappingOptions::makeUnmappedMappingOptions())
      .useSharedMemory(false)
      .usePrivateMemory(false)
      .unrollCopyShared(false);
  return mo;
}

CudaMappingOptions CudaMappingOptions::makeNaiveMappingOptions() {
  return makeUnmappedMappingOptions()
      .tile(32, 32, 32)
      .mapToThreads(32, 8)
      .mapToBlocks(256, 256)
      .unroll(1);
}

CudaMappingOptions CudaMappingOptions::makeSingleThreadMappingOptions() {
  return makeUnmappedMappingOptions()
      .tile(1)
      .mapToThreads(1)
      .mapToBlocks(1)
      .unroll(1);
}

CudaMappingOptions CudaMappingOptions::makePointwiseMappingOptions() {
  return makeUnmappedMappingOptions()
      .tile(32, 32, 32)
      .mapToThreads(32, 4, 4)
      .mapToBlocks(100, 100, 100)
      .unroll(128);
}

CudaMappingOptions CudaMappingOptions::makeMlpMappingOptions() {
  return makeUnmappedMappingOptions()
      .outerScheduleFusionStrategy(FusionStrategy::Max)
      .tile(1)
      .mapToThreads(128)
      .mapToBlocks(128)
      .unroll(1);
}

CudaMappingOptions CudaMappingOptions::makeConvolutionMappingOptions() {
  return makeUnmappedMappingOptions()
      .tile(4, 8, 8, 8)
      .mapToThreads(4, 16, 4)
      .mapToBlocks(256, 256, 256)
      .unroll(1);
}

CudaMappingOptions CudaMappingOptions::makeGroupConvolutionMappingOptions() {
  return makeUnmappedMappingOptions()
      .tile(1, 1)
      .mapToThreads(4, 16, 4)
      .mapToBlocks(256, 256)
      .unroll(1);
}

} // namespace tc
