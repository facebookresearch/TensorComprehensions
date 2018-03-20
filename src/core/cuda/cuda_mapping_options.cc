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

#include <mapping_options.pb.h>

#include "tc/core/flags.h"
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

namespace {
// Sets the std::boolalpha flags of the given std::ostream and resets it to the
// previous value on scope exit.
class OstreamBoolalphaScope {
 public:
  OstreamBoolalphaScope(std::ostream& os)
      : os_(os), hasBoolalpha_(os.flags() & std::ios_base::boolalpha) {
    os << std::boolalpha;
  }
  ~OstreamBoolalphaScope() {
    if (!hasBoolalpha_) {
      os_ << std::noboolalpha;
    }
  }

 private:
  std::ostream& os_;
  bool hasBoolalpha_;
};
} // namespace

std::ostream& operator<<(
    std::ostream& os,
    const CudaMappingOptions& cudaOptions) {
  OstreamBoolalphaScope scope(os);

  os << "CudaMappingOptions(";
  os << static_cast<CudaMappingOptions>(cudaOptions);
  os << "," << std::endl
     << "block: " << cudaOptions.block << "," << std::endl
     << "grid: " << cudaOptions.grid << "," << std::endl
     << "use_shared_memory: " << cudaOptions.proto.use_shared_memory() << ","
     << std::endl
     << "use_private_memory: " << cudaOptions.proto.use_private_memory() << ","
     << std::endl
     << "unroll_copy_shared: " << cudaOptions.proto.unroll_copy_shared() << ","
     << std::endl
     << "max_shared_memory: "
     << (cudaOptions.proto.has_max_shared_memory()
             ? std::to_string(cudaOptions.proto.max_shared_memory())
             : "#none")
     << "," << std::endl
     << "@" << &cudaOptions.proto;
  return os;
}

//
// String-based chainable constructors.
//
namespace {
template <typename T>
std::vector<T> parseCommaSeparatedIntegers(const std::string& sizes) {
  std::stringstream ss(sizes);
  T size;
  std::vector<T> res;
  while (ss >> size) {
    res.push_back(size);
    if (ss.peek() == ',') {
      ss.ignore();
    }
  }
  return res;
}
} // namespace

CudaMappingOptions& CudaMappingOptions::mapToThreads(
    const std::string& commaSeparatedSizes) {
  auto sizes = parseCommaSeparatedIntegers<uint64_t>(commaSeparatedSizes);
  CHECK_GT(sizes.size(), 0)
      << "expected at least one block size in " << commaSeparatedSizes;
  CHECK_LE(sizes.size(), 3)
      << "expected at most three block sizes in " << commaSeparatedSizes;
  sizes.resize(3, CudaDimView::defaultDim);
  return mapToThreads(sizes[0], sizes[1], sizes[2]);
}

CudaMappingOptions& CudaMappingOptions::mapToBlocks(
    const std::string& commaSeparatedSizes) {
  auto sizes = parseCommaSeparatedIntegers<uint64_t>(commaSeparatedSizes);
  CHECK_GT(sizes.size(), 0)
      << "expected at least one grid size in " << commaSeparatedSizes;
  CHECK_LE(sizes.size(), 3)
      << "expected at most three grid sizes in " << commaSeparatedSizes;
  sizes.resize(3, CudaDimView::defaultDim);
  return mapToBlocks(sizes[0], sizes[1], sizes[2]);
}

//
// Predefined stratgies
//
CudaMappingOptions CudaMappingOptions::makeUnmappedCudaMappingOptions() {
  auto options = MappingOptions::makeUnmappedMappingOptions();
  LOG(INFO) << options.toProtobufSerializedString();

  CudaMappingOptions mo;
  mo.genericMappingOptions(options)
      .useSharedMemory(false)
      .usePrivateMemory(false)
      .unrollCopyShared(false);

  return mo;
}

CudaMappingOptions CudaMappingOptions::makeNaiveCudaMappingOptions() {
  auto mo = makeUnmappedCudaMappingOptions();
  mo =
      mo.tile({32, 32, 32}).mapToThreads(32, 8).mapToBlocks(256, 256).unroll(1);
  return mo;
}

CudaMappingOptions CudaMappingOptions::makeSingleThreadCudaMappingOptions() {
  return makeUnmappedCudaMappingOptions()
      .tile({1})
      .mapToThreads(1)
      .mapToBlocks(1)
      .unroll(1);
}

CudaMappingOptions CudaMappingOptions::makePointwiseCudaMappingOptions() {
  return makeUnmappedCudaMappingOptions()
      .tile({32, 32, 32})
      .mapToThreads(32, 4, 4)
      .mapToBlocks(100, 100, 100)
      .unroll(128);
}

CudaMappingOptions CudaMappingOptions::makeMlpCudaMappingOptions() {
  return makeUnmappedCudaMappingOptions()
      .outerScheduleFusionStrategy(FusionStrategy::Max)
      .tile({1})
      .mapToThreads(128)
      .mapToBlocks(128)
      .unroll(1);
}

CudaMappingOptions CudaMappingOptions::makeConvolutionCudaMappingOptions() {
  return makeUnmappedCudaMappingOptions()
      .tile({4, 8, 8, 8})
      .mapToThreads(4, 16, 4)
      .mapToBlocks(256, 256, 256)
      .unroll(1);
}

CudaMappingOptions
CudaMappingOptions::makeGroupConvolutionCudaMappingOptions() {
  return makeUnmappedCudaMappingOptions()
      .tile({1, 1})
      .mapToThreads(4, 16, 4)
      .mapToBlocks(256, 256)
      .unroll(1);
}

} // namespace tc
