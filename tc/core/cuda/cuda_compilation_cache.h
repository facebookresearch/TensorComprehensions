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

#include <cstdint>
#include <memory>
#include <mutex>
#include <stdexcept>
#include <string>
#include <vector>

#include <dlpack/dlpack.h>

#include "tc/proto/compcache.pb.h"

#include "tc/core/compilation_cache.h"
#include "tc/core/cuda/cuda.h"
#include "tc/core/cuda/cuda_mapping_options.h"
#include "tc/core/cuda/cuda_rtc.h"
#include "tc/core/utils/time.h"

namespace tc {

////////////////////////////////////////////////////////////////////////////////
// OptionsCache
////////////////////////////////////////////////////////////////////////////////
/**
 * An OptionsCache holds multiple OptionsCachedEntry's.
 * Each OptionsCachedEntry is split to two conceptual parts the key and the
 * values. The key is: the kernel/op's unique id (string), the specialized input
 * dimensions, the target architecture (string), tc's version
 * (string), The values are a vector of: the isl options used
 * when the kernel was optimized, profiling information
 */
struct OptionsCachedEntry {
  OptionsCachedEntry(
      const std::string& id,
      const std::vector<const DLTensor*>& inputs,
      const std::vector<const DLTensor*>& outputs,
      const std::string& deviceStr,
      const CudaMappingOptions& options,
      Duration runtime);
  OptionsCachedEntry(const OptionsCacheEntryProto& buf);
  OptionsCacheEntryProto toProtobuf() const;

  struct Key {
    Key(const std::string& id,
        const std::vector<const DLTensor*>& inputs,
        const std::vector<const DLTensor*>& outputs,
        const std::string& deviceStr,
        const std::string& gitVersion);

    Key(const std::string& id,
        std::vector<detail::TensorInfo>&& inputs,
        std::vector<detail::TensorInfo>&& outputs,
        const std::string& deviceStr,
        const std::string& gitVersion);

    std::string id;
    std::vector<detail::TensorInfo> inputs;
    std::vector<detail::TensorInfo> outputs;
    std::string deviceStr;
    std::string gitVersion;
  };

  struct Values {
    Values(const CudaMappingOptions& options, Duration runtime);
    Values(const CudaMappingOptions& options, std::vector<Duration>&& runtimes);
    CudaMappingOptions mappingOptions;
    std::vector<Duration> recordedRuntimes;
  };
  Key key;
  std::vector<Values> values;
};

struct OptionsCacheRetrievalResult {
  CudaMappingOptions options;
  std::vector<Duration> recordedRuntimes;
};

class OptionsCache : public Cache<OptionsCache, OptionsCachedEntry> {
 public:
  using ProtobufType = OptionsCacheProto;
  using CachedEntry = OptionsCachedEntry;
  using RetrievalResult = OptionsCacheRetrievalResult;
  static std::shared_ptr<OptionsCache>& getGlobalSharedCache();

  OptionsCache() = default;
  OptionsCache(const OptionsCacheProto& buf);

  OptionsCacheProto toProtobuf() const;

  // returns the sum of cache entry sizes (that is a single cache entry can have
  // multiple options and profiling information associated with it)
  size_t totalSize() const;

  void recordRuntime(
      const std::string& id,
      const CudaMappingOptions& options,
      const std::vector<const DLTensor*>& inputs,
      const std::vector<const DLTensor*>& outputs,
      Duration runtime);

  std::vector<OptionsCacheRetrievalResult> retrieveOptionsAndRuntimes(
      const std::string& id,
      const std::vector<const DLTensor*>& inputs,
      const std::vector<const DLTensor*>& outputs) const;

  std::unique_ptr<CudaMappingOptions> retrieveBestOptions(
      const std::string& id,
      const std::vector<const DLTensor*>& inputs,
      const std::vector<const DLTensor*>& outputs) const;

  std::vector<CudaMappingOptions> retrieveTopKOptions(
      const std::string& id,
      const std::vector<const DLTensor*>& inputs,
      const std::vector<const DLTensor*>& outputs,
      size_t k) const;

  // Only (up to) numberToKeep entries per operation (combination of id and
  // input info) are kept in the cache. The best performing versions are kept
  void keepOnlyBestCandidates(size_t numberToKeep);
};
} // namespace tc

#include "tc/core/cuda/cuda_compilation_cache-inl.h"
