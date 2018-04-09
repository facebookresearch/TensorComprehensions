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

#include <compcache.pb.h>

#include "tc/core/compilation_cache.h"
#include "tc/core/cuda/cuda.h"
#include "tc/core/cuda/cuda_mapping_options.h"
#include "tc/core/cuda/cuda_rtc.h"
#include "tc/core/utils/time.h"

namespace tc {

class OptionsCache;

struct CudaCachedEntry {
  CudaCachedEntry(
      const std::string& id,
      const std::string& kernelSpecializedName,
      const std::vector<int>& kernelParameters,
      const Grid& grid,
      const Block& block,
      const CudaMappingOptions& mappingOptions,
      const std::vector<const DLTensor*>& inputs,
      const std::vector<const DLTensor*>& outputs,
      const std::string& cudaSource,
      const std::string& deviceStr);

  CudaCachedEntry(const CudaCacheEntryProto& buf);
  CudaCacheEntryProto toProtobuf() const;

  struct Key {
    std::string id;
    CudaMappingOptions mappingOptions;
    std::vector<detail::TensorInfo> inputs;
    std::vector<detail::TensorInfo> outputs;
    std::string deviceStr;
    std::string gitVersion;
  };

  struct Values {
    std::string cudaSource;
    std::string kernelSpecializedName;
    std::vector<int> kernelParameters;
    Grid grid;
    Block block;
  };
  Key key;
  Values values;
};

struct CudaCacheRetrievalResult {
  std::string source;
  std::string specializedName;
  std::vector<int> parameters;
  Grid grid;
  Block block;
};

/**
 * CudaCache stores the Cuda source of optimized kernels
 * A CudaCache holds multiple CudaCachedEntry's.
 * Each CudaCachedEntry is split to two conceptual parts the key and the values.
 * The values are:
 *                  the specialized (wrt inputs) Cuda source code,
 *                  the kernel's specialized name,
 *                  the kernel parameters,
 *                  the Cuda block and grid dimensions
 * The key is:
 *                  the kernel/op's unique id (string),
 *                  the specialized input dimensions,
 *                  the isl options when the kernel was optimized,
 *                  the target architecture (string),
 *                  tc's version (string),
 */
class CudaCache : public Cache<CudaCache, CudaCachedEntry> {
 public:
  using ProtobufType = CudaCacheProto;
  using CachedEntry = CudaCachedEntry;
  using RetrievalResult = CudaCacheRetrievalResult;
  static std::shared_ptr<CudaCache>& getGlobalSharedCache();

 private:
  /**
   * SearchKernel (through SearchKernelImpl) searches op in the cache
   * if a cached entry that corresponds to the op's configuration
   * (CudaMappingOptions and TargetDevice) and the shape of inputs matches it is
   * returned
   */

  CachedEntry* searchKernel(
      const std::string& id,
      const CudaMappingOptions& options,
      const std::vector<detail::TensorInfo>& inputs,
      const std::vector<detail::TensorInfo>& outputs);
  CachedEntry* searchKernel(
      const std::string& id,
      const CudaMappingOptions& options,
      const std::vector<const DLTensor*>& inputs,
      const std::vector<const DLTensor*>& outputs);
  const CachedEntry* searchKernel(
      const std::string& id,
      const CudaMappingOptions& options,
      const std::vector<const DLTensor*>& inputs,
      const std::vector<const DLTensor*>& outputs) const;

  // deduces whether C is const or non-const
  template <typename C, typename TensorTy>
  static auto searchKernelImpl(
      C& c,
      const std::string& id,
      const CudaMappingOptions& options,
      const std::vector<TensorTy>& inputs,
      const std::vector<TensorTy>& outputs)
      -> decltype(c.searchKernel(id, options, inputs, outputs));

 public:
  CudaCache() = default;
  CudaCache(const CudaCacheProto& buf);
  CudaCacheProto toProtobuf() const;

  /**
   * If op was previously cached and the inputs' shape, isl options, and the
   * target device are the same then this is a noop
   * Else (cudaSource, grid, block) is stored in the cache
   */
  void cacheKernel(
      const std::string& id,
      const CudaMappingOptions& options,
      const std::vector<const DLTensor*>& inputs,
      const std::vector<const DLTensor*>& outputs,
      const std::string& kernelSpecializedName,
      const std::vector<int>& kernelParameters,
      const std::string& cudaSource,
      const Grid& grid,
      const Block& block);

  /**
   * Returns the cache entry that matches op (id, isl options, target device)
   * and inputs' shapes.
   */
  std::unique_ptr<CudaCacheRetrievalResult> retrieveKernel(
      const std::string& id,
      const CudaMappingOptions& options,
      const std::vector<const DLTensor*>& inputs,
      const std::vector<const DLTensor*>& outputs) const;

  void removeEntriesNotInOptionsCache(const OptionsCache& oc);
};

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

 private:
  /**
   * SearchKernel (through SearchKernelImpl) searches op in the cache
   * if a cached entry that corresponds to the op's configuration
   * (CudaMappingOptions and TargetDevice) and the shape of inputs matches it is
   * returned
   */

  CachedEntry* searchKernel(
      const std::string& id,
      const std::vector<const DLTensor*>& inputs,
      const std::vector<const DLTensor*>& outputs);
  const CachedEntry* searchKernel(
      const std::string& id,
      const std::vector<const DLTensor*>& input,
      const std::vector<const DLTensor*>& outputs) const;

  // deduces whether C is const or non-const
  template <typename C>
  static auto searchKernelImpl(
      C& c,
      const std::string& id,
      const std::vector<const DLTensor*>& inputs,
      const std::vector<const DLTensor*>& outputs)
      -> decltype(c.searchKernel(id, inputs, outputs));

 public:
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

/*
 *A CudaCache holds multiple CachedEntry's.
 *Each CachedEntry is split to two conceptual parts the key and the values.
 *The values are:
 *                 the specialized (wrt inputs) Cuda source code,
 *                 the Cuda block and grid dimensions
 *The key is:
 *                 the kernel/op's unique id (string),
 *                 the specialized input dimensions,
 *                 the target architecture (string),
 *                 tc's version (string),
 */
struct ManualCudaCachedEntry {
  ManualCudaCachedEntry(
      const std::string& id,
      const std::string& kernelSpecializedName,
      const std::vector<int>& kernelParameters,
      const Grid& grid,
      const Block& block,
      const std::vector<const DLTensor*>& inputs,
      const std::vector<const DLTensor*>& outputs,
      const std::string& cudaSource,
      const std::string& deviceStr);

  ManualCudaCachedEntry(const ManualCudaCacheEntryProto& buf);
  ManualCudaCacheEntryProto toProtobuf() const;

  struct Key {
    std::string id;
    std::vector<detail::TensorInfo> inputs;
    std::vector<detail::TensorInfo> outputs;
    std::string deviceStr;
    std::string gitVersion;
  };

  struct Values {
    std::string cudaSource;
    std::string kernelSpecializedName;
    std::vector<int> kernelParameters;
    Grid grid;
    Block block;
  };
  Key key;
  Values values;
};

typedef CudaCacheRetrievalResult ManualCudaCacheRetrievalResult;

/*
 * ManualCudaCache stores the manually injected source of Cuda kernels
 */
class ManualCudaCache : public Cache<ManualCudaCache, ManualCudaCachedEntry> {
 public:
  using ProtobufType = ManualCudaCacheProto;
  using CachedEntry = ManualCudaCachedEntry;
  using RetrievalResult = ManualCudaCacheRetrievalResult;
  static std::shared_ptr<ManualCudaCache>& getGlobalSharedCache();

 private:
  /*
   *SearchKernel (through SearchKernelImpl) searches op in the cache
   *if a cached entry that corresponds to the op's TargetDevice and the
   *shape of inputs matches it is returned
   */
  CachedEntry* searchKernel(
      const std::string& id,
      const std::vector<detail::TensorInfo>& inputs,
      const std::vector<detail::TensorInfo>& outputs);
  CachedEntry* searchKernel(
      const std::string& id,
      const std::vector<const DLTensor*>& inputs,
      const std::vector<const DLTensor*>& outputs);
  const CachedEntry* searchKernel(
      const std::string& id,
      const std::vector<const DLTensor*>& inputs,
      const std::vector<const DLTensor*>& outputs) const;

  // deduces whether C is const or non-const
  template <typename C, typename InputTy>
  static auto searchKernelImpl(
      C& c,
      const std::string& id,
      const std::vector<InputTy>& inputs,
      const std::vector<InputTy>& outputs)
      -> decltype(c.searchKernel(id, inputs, outputs));

 public:
  ManualCudaCache() = default;
  ManualCudaCache(const ManualCudaCacheProto& buf);
  ManualCudaCacheProto toProtobuf() const;

  /*
   *Stores (cudaSource, grid, block, specializedName, parameters)
   *in the cache with key (id, input shapes, output shapes,
   *target device). If the key already exist in the cache,
   *the values are replaced.
   */
  void cacheKernel(
      const std::string& id,
      const std::vector<const DLTensor*>& inputs,
      const std::vector<const DLTensor*>& outputs,
      const std::string& kernelSpecializedName,
      const std::vector<int>& kernelParameters,
      const std::string& cudaSource,
      const Grid& grid,
      const Block& block);

  /*
   *Returns the cache entry that matches
   *op(id, target device) and inputs' shapes.
   */
  std::unique_ptr<ManualCudaCacheRetrievalResult> retrieveKernel(
      const std::string& id,
      const std::vector<const DLTensor*>& inputs,
      const std::vector<const DLTensor*>& outputs) const;
};

void removeFromCudaCacheEntriesNotInOptionsCache(
    CudaCache& cc,
    const OptionsCache& oc);

} // namespace tc

#include "tc/core/cuda/cuda_compilation_cache-inl.h"
