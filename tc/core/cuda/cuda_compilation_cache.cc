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
#include "tc/core/cuda/cuda_compilation_cache.h"

#include <version.h>

#include <cstdint>
#include <fstream>
#include <numeric>
#include <tuple>

#include "tc/core/cuda/cuda_mapping_options.h"
#include "tc/core/utils/math.h"

namespace tc {

namespace {
std::vector<detail::TensorInfo> DLTensorToTensorInfoVector(
    const std::vector<const DLTensor*>& ts) {
  std::vector<detail::TensorInfo> iis;
  iis.reserve(ts.size());
  std::transform(
      ts.begin(), ts.end(), std::back_inserter(iis), [](const DLTensor* t) {
        return detail::TensorInfo{t};
      });
  return iis;
}
std::vector<detail::TensorInfo> ProtoToTensorInfoVector(
    const google::protobuf::RepeatedPtrField<TensorInfoProto>& buf) {
  std::vector<detail::TensorInfo> iis;
  iis.reserve(buf.size());
  std::transform(
      buf.begin(),
      buf.end(),
      std::back_inserter(iis),
      [](const TensorInfoProto& iip) { return detail::TensorInfo{iip}; });
  return iis;
}
template <typename Array, typename Buf>
void WriteProtobufArray(const Array& arr, Buf* buf) {
  google::protobuf::RepeatedField<typename Array::value_type> data(
      arr.begin(), arr.end());
  buf->Swap(&data);
}

template <typename CachedEntryType, typename TensorType>
const CachedEntryType* searchKernel(
    const std::vector<CachedEntryType>& entries,
    const std::string& id,
    const std::vector<TensorType>& inputs,
    const std::vector<TensorType>& outputs) {
  auto gpuStr = CudaGPUInfo::GPUInfo().GetCudaDeviceStr();
  auto it = std::find_if(
      entries.begin(), entries.end(), [&](const CachedEntryType& c) {
        using tc::operator==;
        return id == c.key.id && inputs == c.key.inputs &&
            outputs == c.key.outputs && gpuStr == c.key.deviceStr;
      });
  if (it != entries.end()) {
    if (it->key.gitVersion != tc::git_version) {
      std::cerr << "[WARNING] Proto version doesn't match. TC git version is: "
                << tc::git_version
                << " and Proto version is: " << it->key.gitVersion
                << " .This proto might be incompatible"
                << " with your TC binary and can break. Please autotune"
                << " against the correct TC version." << std::endl;
    }
    return &*it;
  }
  return nullptr;
}

template <typename CachedEntryType, typename TensorType>
CachedEntryType* searchKernel(
    std::vector<CachedEntryType>& entries,
    const std::string& id,
    const std::vector<TensorType>& inputs,
    const std::vector<TensorType>& outputs) {
  return const_cast<CachedEntryType*>(searchKernel(
      static_cast<const std::vector<CachedEntryType>&>(entries),
      id,
      inputs,
      outputs));
}

template <typename CachedEntryType, typename TensorType>
const CachedEntryType* searchKernel(
    const std::vector<CachedEntryType>& entries,
    const std::string& id,
    const CudaMappingOptions& options,
    const std::vector<TensorType>& inputs,
    const std::vector<TensorType>& outputs) {
  auto gpuStr = CudaGPUInfo::GPUInfo().GetCudaDeviceStr();
  auto it = std::find_if(
      entries.begin(), entries.end(), [&](const CachedEntryType& c) {
        using tc::operator==;
        return id == c.key.id && options == c.key.mappingOptions &&
            inputs == c.key.inputs && outputs == c.key.outputs &&
            gpuStr == c.key.deviceStr;
      });
  if (it != entries.end()) {
    if (it->key.gitVersion != tc::git_version) {
      std::cerr << "[WARNING] Proto version doesn't match. TC git version is: "
                << tc::git_version
                << " and Proto version is: " << it->key.gitVersion
                << " .This proto might be incompatible"
                << " with your TC binary and can break. Please autotune"
                << " against the correct TC version." << std::endl;
    }
    return &*it;
  }
  return nullptr;
}

template <typename CachedEntryType, typename TensorType>
CachedEntryType* searchKernel(
    std::vector<CachedEntryType>& entries,
    const std::string& id,
    const CudaMappingOptions& options,
    const std::vector<TensorType>& inputs,
    const std::vector<TensorType>& outputs) {
  return const_cast<CachedEntryType*>(searchKernel(
      static_cast<const std::vector<CachedEntryType>&>(entries),
      id,
      options,
      inputs,
      outputs));
}
} // namespace

////////////////////////////////////////////////////////////////////////////////
// CudaCache
////////////////////////////////////////////////////////////////////////////////
std::shared_ptr<CudaCache>& CudaCache::getGlobalSharedCache() {
  static std::shared_ptr<CudaCache> cudaCache_;
  return cudaCache_;
}

CudaCachedEntry::CudaCachedEntry(
    const std::string& id,
    const std::string& kernelSpecializedName,
    const std::vector<int>& kernelParameters,
    const Grid& grid,
    const Block& block,
    const CudaMappingOptions& mappingOptions,
    const std::vector<const DLTensor*>& inputs,
    const std::vector<const DLTensor*>& outputs,
    const std::string& cudaSource,
    const std::string& deviceStr)
    : key{id,
          mappingOptions,
          DLTensorToTensorInfoVector(inputs),
          DLTensorToTensorInfoVector(outputs),
          deviceStr,
          git_version},
      values{cudaSource, kernelSpecializedName, kernelParameters, grid, block} {
}

CudaCachedEntry::CudaCachedEntry(const CudaCacheEntryProto& buf)
    : key{buf.id(),
          CudaMappingOptions{buf.kernel_options()},
          ProtoToTensorInfoVector(buf.inputs()),
          ProtoToTensorInfoVector(buf.outputs()),
          buf.device_str(),
          buf.git_version()},
      values{buf.cuda_source(),
             buf.specialized_name(),
             std::vector<int>{buf.parameters().begin(), buf.parameters().end()},
             Grid(buf.grid_dims()),
             Block(buf.block_dims())} {}

CudaCache::CudaCache(const CudaCacheProto& buf) {
  entries_.reserve(buf.entries_size());
  for (const auto& entry_buf : buf.entries())
    entries_.emplace_back(entry_buf);
}

void CudaCache::cacheKernel(CudaCachedEntry&& entry) {
  std::lock_guard<std::mutex> lock(mtx_);
  ++numberCacheAttemps;
  auto retrievedEntry = searchKernel(
      entries_,
      entry.key.id,
      entry.key.mappingOptions,
      entry.key.inputs,
      entry.key.outputs);
  if (retrievedEntry) {
    if (retrievedEntry->values.cudaSource != entry.values.cudaSource or
        retrievedEntry->values.grid != entry.values.grid or
        retrievedEntry->values.block != entry.values.block) {
      throw CacheEntrySameKeyDifferentValue(
          "CudaCache::CacheKernel: a kernel matching the id, options and "
          "inputs was previously cached with different cuda source or block "
          "or grid dimensions.");
    }
    return;
  }
  entries_.emplace_back(std::move(entry));
}

std::unique_ptr<CudaCacheRetrievalResult> CudaCache::retrieveKernel(
    const std::string& id,
    const CudaMappingOptions& options,
    const std::vector<const DLTensor*>& inputs,
    const std::vector<const DLTensor*>& outputs) const {
  std::lock_guard<std::mutex> lock(mtx_);
  ++numberAttemptedRetrievals;
  auto entry = searchKernel(entries_, id, options, inputs, outputs);
  if (not entry) {
    return nullptr;
  }
  ++numberSuccessfulRetrievals;
  return std::unique_ptr<CudaCacheRetrievalResult>(
      new CudaCacheRetrievalResult{entry->values.cudaSource,
                                   entry->values.kernelSpecializedName,
                                   entry->values.kernelParameters,
                                   entry->values.grid,
                                   entry->values.block});
}

void CudaCache::removeEntriesNotInOptionsCache(const OptionsCache& oc) {
  std::vector<CudaCachedEntry> newEntries;
  for (const auto& entry : oc) {
    for (const auto& options : entry.values) {
      auto cudaEntry = searchKernel(
          entries_,
          entry.key.id,
          options.mappingOptions,
          entry.key.inputs,
          entry.key.outputs);
      if (cudaEntry) {
        newEntries.push_back(std::move(*cudaEntry));
      }
    }
  }
  entries_ = std::move(newEntries);
}

CudaCacheProto CudaCache::toProtobuf() const {
  CudaCacheProto buf;
  auto* entriesBuf = buf.mutable_entries();
  entriesBuf->Reserve(entries_.size());
  std::transform(
      entries_.begin(),
      entries_.end(),
      google::protobuf::RepeatedPtrFieldBackInserter(entriesBuf),
      [](const CudaCachedEntry& entry) { return entry.toProtobuf(); });
  return buf;
}

CudaCacheEntryProto CudaCachedEntry::toProtobuf() const {
  CudaCacheEntryProto buf;
  buf.set_id(key.id);
  *buf.mutable_kernel_options() = key.mappingOptions.proto();
  std::transform(
      key.inputs.begin(),
      key.inputs.end(),
      google::protobuf::RepeatedPtrFieldBackInserter(buf.mutable_inputs()),
      [](const detail::TensorInfo& input) { return input.toProtobuf(); });
  std::transform(
      key.outputs.begin(),
      key.outputs.end(),
      google::protobuf::RepeatedPtrFieldBackInserter(buf.mutable_outputs()),
      [](const detail::TensorInfo& output) { return output.toProtobuf(); });
  buf.set_device_str(key.deviceStr);
  buf.set_git_version(key.gitVersion);

  buf.set_cuda_source(values.cudaSource);
  *buf.mutable_grid_dims() = values.grid.view.proto;
  *buf.mutable_block_dims() = values.block.view.proto;
  buf.set_specialized_name(values.kernelSpecializedName);
  WriteProtobufArray(values.kernelParameters, buf.mutable_parameters());

  return buf;
}

////////////////////////////////////////////////////////////////////////////////
// OptionsCache
////////////////////////////////////////////////////////////////////////////////

OptionsCachedEntry::OptionsCachedEntry(
    const std::string& id,
    const std::vector<const DLTensor*>& inputs,
    const std::vector<const DLTensor*>& outputs,
    const std::string& deviceStr,
    const CudaMappingOptions& options,
    Duration runtime)
    : key(id, inputs, outputs, deviceStr, git_version) {
  values.emplace_back(options, runtime);
}

OptionsCachedEntry::OptionsCachedEntry(
    const std::string& id,
    const std::vector<const DLTensor*>& inputs,
    const std::vector<const DLTensor*>& outputs,
    const std::string& deviceStr,
    const CudaMappingOptions& options,
    const CudaProfilingInfo& pInfo)
    : key(id, inputs, outputs, deviceStr, git_version) {
  values.emplace_back(options, pInfo);
}

OptionsCachedEntry::OptionsCachedEntry(
    const std::string& id,
    const std::vector<const DLTensor*>& inputs,
    const std::vector<const DLTensor*>& outputs,
    const std::string& deviceStr,
    const CudaMappingOptions& options,
    Duration runtime,
    const CudaProfilingInfo& pInfo)
    : key(id, inputs, outputs, deviceStr, git_version) {
  values.emplace_back(options, runtime, pInfo);
}

OptionsCachedEntry::Key::Key(
    const std::string& id,
    const std::vector<const DLTensor*>& inputs_,
    const std::vector<const DLTensor*>& outputs_,
    const std::string& deviceStr,
    const std::string& gitVersion)
    : Key(id,
          DLTensorToTensorInfoVector(inputs_),
          DLTensorToTensorInfoVector(outputs_),
          deviceStr,
          gitVersion) {}

OptionsCachedEntry::Key::Key(
    const std::string& id,
    std::vector<detail::TensorInfo>&& inputs_,
    std::vector<detail::TensorInfo>&& outputs_,
    const std::string& deviceStr,
    const std::string& gitVersion)
    : id(id),
      inputs(std::move(inputs_)),
      outputs(std::move(outputs_)),
      deviceStr(deviceStr),
      gitVersion(gitVersion) {}

OptionsCachedEntry::Values::Values(
    const CudaMappingOptions& options,
    Duration runtime)
    : mappingOptions(options), recordedRuntimes{runtime} {}

OptionsCachedEntry::Values::Values(
    const CudaMappingOptions& options,
    const CudaProfilingInfo& pInfo)
    : mappingOptions(options), profiles{pInfo} {}

OptionsCachedEntry::Values::Values(
    const CudaMappingOptions& options,
    Duration runtime,
    const CudaProfilingInfo& pInfo)
    : mappingOptions(options), recordedRuntimes{runtime}, profiles{pInfo} {}

OptionsCachedEntry::Values::Values(
    const CudaMappingOptions& options,
    std::vector<Duration>&& runtimes,
    std::vector<CudaProfilingInfo>&& pInfos)
    : mappingOptions(options),
      recordedRuntimes(std::move(runtimes)),
      profiles(std::move(pInfos)) {}

namespace {
tc::CudaProfilingInfo fromProto(const tc::CudaProfilingProto& buf) {
  tc::CudaProfilingInfo pInfo;
  pInfo.runtime = std::chrono::microseconds(buf.runtime());
  pInfo.ipc = buf.ipc();
  pInfo.globalLoadEfficiency = buf.globalloadefficiency();
  pInfo.globalStoreEfficiency = buf.globalstoreefficiency();
  pInfo.sharedMemoryEfficiency = buf.sharedmemoryefficiency();
  pInfo.localMemoryOverhead = buf.localmemoryoverhead();
  pInfo.achievedOccupancy = buf.achievedoccupancy();
  pInfo.warpExecutionEfficiency = buf.warpexecutionefficiency();
  return pInfo;
}
} // namespace

OptionsCachedEntry::OptionsCachedEntry(const OptionsCacheEntryProto& buf)
    : key(buf.id(),
          ProtoToTensorInfoVector(buf.inputs()),
          ProtoToTensorInfoVector(buf.outputs()),
          buf.device_str(),
          buf.git_version()) {
  if (buf.values_size() == 0) {
    throw std::invalid_argument(
        "OptionsCachedEntry invalid protobuf: each entry should have at least one value field.");
  }

  for (const auto& value : buf.values()) {
    if (value.recorded_runtimes_size() == 0 and value.profiles_size() == 0) {
      throw std::invalid_argument(
          "OptionsCachedEntry invalid protobuf: each entry value should have at least one recorded runtime.");
    }
    std::vector<Duration> runtimes;
    runtimes.reserve(value.recorded_runtimes_size());
    std::transform(
        value.recorded_runtimes().begin(),
        value.recorded_runtimes().end(),
        std::back_inserter(runtimes),
        [](int64_t us) { return std::chrono::microseconds(us); });
    std::vector<CudaProfilingInfo> profiles;
    profiles.reserve(value.profiles_size());
    std::transform(
        value.profiles().begin(),
        value.profiles().end(),
        std::back_inserter(profiles),
        [](const CudaProfilingProto& buf) { return fromProto(buf); });

    values.emplace_back(
        CudaMappingOptions(value.kernel_options()),
        std::move(runtimes),
        std::move(profiles));
  }
}

namespace {
tc::CudaProfilingProto toProto(const tc::CudaProfilingInfo& pInfo) {
  tc::CudaProfilingProto buf;
  buf.set_runtime(
      std::chrono::duration_cast<std::chrono::microseconds>(pInfo.runtime)
          .count());
  buf.set_ipc(pInfo.ipc);
  buf.set_globalloadefficiency(pInfo.globalLoadEfficiency);
  buf.set_globalstoreefficiency(pInfo.globalStoreEfficiency);
  buf.set_sharedmemoryefficiency(pInfo.sharedMemoryEfficiency);
  buf.set_localmemoryoverhead(pInfo.localMemoryOverhead);
  buf.set_achievedoccupancy(pInfo.achievedOccupancy);
  buf.set_warpexecutionefficiency(pInfo.warpExecutionEfficiency);

  return buf;
}
} // namespace

OptionsCacheEntryProto OptionsCachedEntry::toProtobuf() const {
  OptionsCacheEntryProto buf;
  buf.set_id(key.id);
  std::transform(
      key.inputs.begin(),
      key.inputs.end(),
      google::protobuf::RepeatedPtrFieldBackInserter(buf.mutable_inputs()),
      [](const detail::TensorInfo& input) { return input.toProtobuf(); });
  std::transform(
      key.outputs.begin(),
      key.outputs.end(),
      google::protobuf::RepeatedPtrFieldBackInserter(buf.mutable_outputs()),
      [](const detail::TensorInfo& output) { return output.toProtobuf(); });

  buf.set_device_str(key.deviceStr);
  buf.set_git_version(key.gitVersion);

  std::transform(
      values.begin(),
      values.end(),
      google::protobuf::RepeatedPtrFieldBackInserter(buf.mutable_values()),
      [](const Values& v) {
        OptionsCacheValuesProto buf;
        *buf.mutable_kernel_options() = v.mappingOptions.proto();
        for (const auto& r : v.recordedRuntimes) {
          buf.add_recorded_runtimes(
              std::chrono::duration_cast<std::chrono::microseconds>(r).count());
        }
        for (const auto& p : v.profiles) {
          *buf.add_profiles() = toProto(p);
        }
        return buf;
      });
  return buf;
}

std::shared_ptr<OptionsCache>& OptionsCache::getGlobalSharedCache() {
  static std::shared_ptr<OptionsCache> optionsCache_;
  return optionsCache_;
}

OptionsCache::OptionsCache(const OptionsCacheProto& buf) {
  entries_.reserve(buf.entries_size());
  for (const auto& entry_buf : buf.entries())
    entries_.emplace_back(entry_buf);
}

OptionsCacheProto OptionsCache::toProtobuf() const {
  OptionsCacheProto buf;
  auto* entriesBuf = buf.mutable_entries();
  entriesBuf->Reserve(entries_.size());
  std::transform(
      entries_.begin(),
      entries_.end(),
      google::protobuf::RepeatedPtrFieldBackInserter(entriesBuf),
      [](const OptionsCachedEntry& entry) { return entry.toProtobuf(); });
  return buf;
}

size_t OptionsCache::totalSize() const {
  std::lock_guard<std::mutex> lock(mtx_);
  return std::accumulate(
      entries_.begin(),
      entries_.end(),
      size_t(0),
      [](size_t sum, const OptionsCachedEntry& e) {
        return sum + e.values.size();
      });
}

void OptionsCache::recordRuntime(
    const std::string& id,
    const CudaMappingOptions& options,
    const std::vector<const DLTensor*>& inputs,
    const std::vector<const DLTensor*>& outputs,
    Duration runtime) {
  std::lock_guard<std::mutex> lock(mtx_);
  ++numberCacheAttemps;
  auto gpuStr = CudaGPUInfo::GPUInfo().GetCudaDeviceStr();

  auto kernel = searchKernel(entries_, id, inputs, outputs);
  if (not kernel) {
    entries_.emplace_back(id, inputs, outputs, gpuStr, options, runtime);
    return;
  }
  auto v = std::find_if(
      kernel->values.begin(),
      kernel->values.end(),
      [&options](const OptionsCachedEntry::Values& v) {
        return v.mappingOptions == options;
      });
  if (v == kernel->values.end()) {
    kernel->values.emplace_back(options, runtime);
    return;
  }

  v->recordedRuntimes.push_back(runtime);
}

void OptionsCache::recordProfilingInfo(
    const std::string& id,
    const CudaMappingOptions& options,
    const std::vector<const DLTensor*>& inputs,
    const std::vector<const DLTensor*>& outputs,
    const CudaProfilingInfo pInfo) {
  std::lock_guard<std::mutex> lock(mtx_);
  ++numberCacheAttemps;
  auto gpuStr = CudaGPUInfo::GPUInfo().GetCudaDeviceStr();

  auto kernel = searchKernel(entries_, id, inputs, outputs);
  if (not kernel) {
    entries_.emplace_back(
        id, inputs, outputs, gpuStr, options, pInfo.runtime, pInfo);
    return;
  }
  auto v = std::find_if(
      kernel->values.begin(),
      kernel->values.end(),
      [&options](const CachedEntry::Values& v) {
        return v.mappingOptions == options;
      });
  if (v == kernel->values.end()) {
    kernel->values.emplace_back(options, pInfo);
    return;
  }

  v->recordedRuntimes.push_back(pInfo.runtime);
  v->profiles.push_back(pInfo);
}

void OptionsCache::mergeWith(const OptionsCache& other) {
  std::lock(mtx_, other.mtx_);
  std::lock_guard<std::mutex> lock1(mtx_, std::adopt_lock);
  std::lock_guard<std::mutex> lock2(other.mtx_, std::adopt_lock);
  for (const auto& entry : other.entries_) {
    auto it = std::find_if(
        entries_.begin(),
        entries_.end(),
        [&entry](const OptionsCache::CachedEntry& e) {
          return entry.key == e.key;
        });
    if (it == entries_.end()) {
      entries_.push_back(entry);
      continue;
    }
    auto& values = it->values;
    for (const auto& val : entry.values) {
      auto it = std::find_if(
          values.begin(), values.end(), [&val](const CachedEntry::Values& v) {
            return v.mappingOptions == val.mappingOptions;
          });
      if (it == values.end()) {
        values.push_back(val);
      } else {
        it->recordedRuntimes.insert(
            it->recordedRuntimes.end(),
            val.recordedRuntimes.begin(),
            val.recordedRuntimes.end());
        it->profiles.insert(
            it->profiles.end(), val.profiles.begin(), val.profiles.end());
      }
    }
  }
}

bool OptionsCachedEntry::Key::operator==(const Key& other) const {
  return id == other.id and inputs == other.inputs and
      outputs == other.outputs and deviceStr == other.deviceStr and
      gitVersion == other.gitVersion;
}

std::vector<OptionsCacheRetrievalResult>
OptionsCache::retrieveOptionsAndProfilingInfo(
    const std::string& id,
    const std::vector<const DLTensor*>& inputs,
    const std::vector<const DLTensor*>& outputs) const {
  std::lock_guard<std::mutex> lock(mtx_);
  ++numberAttemptedRetrievals;
  auto ret = searchKernel(entries_, id, inputs, outputs);
  if (not ret) {
    return {};
  }
  ++numberSuccessfulRetrievals;
  std::vector<OptionsCacheRetrievalResult> res;
  res.reserve(ret->values.size());
  std::transform(
      ret->values.begin(),
      ret->values.end(),
      std::back_inserter(res),
      [](const CachedEntry::Values& v) -> OptionsCacheRetrievalResult {
        return {v.mappingOptions, v.recordedRuntimes, v.profiles};
      });
  res.erase(
      std::remove_if(
          res.begin(),
          res.end(),
          [](const OptionsCacheRetrievalResult& rr) {
            return rr.profilingInfo.empty();
          }),
      res.end());
  return res;
}

std::vector<OptionsCacheRetrievalResult>
OptionsCache::retrieveOptionsAndRuntimes(
    const std::string& id,
    const std::vector<const DLTensor*>& inputs,
    const std::vector<const DLTensor*>& outputs) const {
  std::lock_guard<std::mutex> lock(mtx_);
  ++numberAttemptedRetrievals;
  auto ret = searchKernel(entries_, id, inputs, outputs);
  if (not ret) {
    return {};
  }
  ++numberSuccessfulRetrievals;
  std::vector<OptionsCacheRetrievalResult> res;
  res.reserve(ret->values.size());
  std::transform(
      ret->values.begin(),
      ret->values.end(),
      std::back_inserter(res),
      [](const OptionsCachedEntry::Values& v) -> OptionsCacheRetrievalResult {
        return {v.mappingOptions, v.recordedRuntimes};
      });
  return res;
}

std::unique_ptr<CudaMappingOptions> OptionsCache::retrieveBestOptions(
    const std::string& id,
    const std::vector<const DLTensor*>& inputs,
    const std::vector<const DLTensor*>& outputs) const {
  auto ret = retrieveTopKOptions(id, inputs, outputs, 1);
  if (ret.empty()) {
    return nullptr;
  }
  return std::unique_ptr<CudaMappingOptions>(
      new CudaMappingOptions(ret.front()));
}

std::vector<CudaMappingOptions> OptionsCache::retrieveTopKOptions(
    const std::string& id,
    const std::vector<const DLTensor*>& inputs,
    const std::vector<const DLTensor*>& outputs,
    size_t k) const {
  auto candidates = searchKernel(entries_, id, inputs, outputs);
  std::lock_guard<std::mutex> lock(mtx_);
  ++numberAttemptedRetrievals;
  if (not candidates) {
    return {};
  }

  struct OptionsWithMedian {
    const CudaMappingOptions* options;
    Duration medianRuntime;
  };

  std::vector<OptionsWithMedian> candidatesMedian;
  candidatesMedian.reserve(candidates->values.size());
  std::transform(
      candidates->values.begin(),
      candidates->values.end(),
      std::back_inserter(candidatesMedian),
      [](const OptionsCachedEntry::Values& v) {
        if (v.recordedRuntimes.empty()) {
          throw std::runtime_error(
              "OptionsCache invariant violated: each cached option should have at least one associated recorded runtime.");
        }
        return OptionsWithMedian{&v.mappingOptions, median(v.recordedRuntimes)};
      });
  std::sort(
      candidatesMedian.begin(),
      candidatesMedian.end(),
      [](const OptionsWithMedian& a, const OptionsWithMedian& b) {
        return a.medianRuntime < b.medianRuntime;
      });
  if (k > candidatesMedian.size()) {
    k = candidatesMedian.size();
  }

  std::vector<CudaMappingOptions> res;
  res.reserve(k);
  std::transform(
      candidatesMedian.begin(),
      candidatesMedian.begin() + k,
      std::back_inserter(res),
      [](const OptionsWithMedian& c) { return *c.options; });

  ++numberSuccessfulRetrievals;
  return res;
}

void OptionsCache::keepOnlyBestCandidates(size_t numberToKeep) {
  std::lock_guard<std::mutex> lock(mtx_);

  for (auto& entry : entries_) {
    std::sort(
        entry.values.begin(),
        entry.values.end(),
        [](const OptionsCachedEntry::Values& a,
           const OptionsCachedEntry::Values& b) {
          // XXX:this is stupid, medians should be precomputed
          return median(a.recordedRuntimes) < median(b.recordedRuntimes);
        });
    if (entry.values.size() > numberToKeep) {
      entry.values.erase(
          entry.values.begin() + numberToKeep, entry.values.end());
    }
  }
}

////////////////////////////////////////////////////////////////////////////////
// ManualCudaCache
////////////////////////////////////////////////////////////////////////////////
std::shared_ptr<ManualCudaCache>& ManualCudaCache::getGlobalSharedCache() {
  static std::shared_ptr<ManualCudaCache> manualCudaCache_;
  return manualCudaCache_;
}

ManualCudaCachedEntry::ManualCudaCachedEntry(
    const std::string& id,
    const std::string& kernelSpecializedName,
    const std::vector<int>& kernelParameters,
    const Grid& grid,
    const Block& block,
    const std::vector<const DLTensor*>& inputs,
    const std::vector<const DLTensor*>& outputs,
    const std::string& cudaSource,
    const std::string& deviceStr)
    : key{id,
          DLTensorToTensorInfoVector(inputs),
          DLTensorToTensorInfoVector(outputs),
          deviceStr,
          git_version},
      values{cudaSource, kernelSpecializedName, kernelParameters, grid, block} {
}

void ManualCudaCache::cacheKernel(ManualCudaCachedEntry&& entry) {
  std::lock_guard<std::mutex> lock(mtx_);
  ++numberCacheAttemps;
  auto retrievedEntry =
      searchKernel(entries_, entry.key.id, entry.key.inputs, entry.key.outputs);
  if (retrievedEntry) {
    retrievedEntry->values.grid = entry.values.grid;
    retrievedEntry->values.block = entry.values.block;
    retrievedEntry->values.cudaSource = entry.values.cudaSource;
    retrievedEntry->values.kernelSpecializedName =
        entry.values.kernelSpecializedName;
    retrievedEntry->values.kernelParameters = entry.values.kernelParameters;
    return;
  }
  entries_.emplace_back(std::move(entry));
}

std::unique_ptr<ManualCudaCacheRetrievalResult> ManualCudaCache::retrieveKernel(
    const std::string& id,
    const std::vector<const DLTensor*>& inputs,
    const std::vector<const DLTensor*>& outputs) const {
  std::lock_guard<std::mutex> lock(mtx_);
  ++numberAttemptedRetrievals;
  auto entry = searchKernel(entries_, id, inputs, outputs);
  if (not entry) {
    return nullptr;
  }
  ++numberSuccessfulRetrievals;
  return std::unique_ptr<ManualCudaCacheRetrievalResult>(
      new ManualCudaCacheRetrievalResult{entry->values.cudaSource,
                                         entry->values.kernelSpecializedName,
                                         entry->values.kernelParameters,
                                         entry->values.grid,
                                         entry->values.block});
}
} // namespace tc
