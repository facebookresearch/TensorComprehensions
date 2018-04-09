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
} // namespace

std::shared_ptr<CudaCache>& CudaCache::getGlobalSharedCache() {
  static std::shared_ptr<CudaCache> cudaCache_;
  return cudaCache_;
}

std::shared_ptr<OptionsCache>& OptionsCache::getGlobalSharedCache() {
  static std::shared_ptr<OptionsCache> optionsCache_;
  return optionsCache_;
}

std::shared_ptr<ManualCudaCache>& ManualCudaCache::getGlobalSharedCache() {
  static std::shared_ptr<ManualCudaCache> manualCudaCache_;
  return manualCudaCache_;
}

CudaCache::CudaCache(const CudaCacheProto& buf) {
  entries_.reserve(buf.entries_size());
  for (const auto& entry_buf : buf.entries())
    entries_.emplace_back(entry_buf);
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

void CudaCache::cacheKernel(
    const std::string& id,
    const CudaMappingOptions& options,
    const std::vector<const DLTensor*>& inputs,
    const std::vector<const DLTensor*>& outputs,
    const std::string& kernelSpecializedName,
    const std::vector<int>& kernelParameters,
    const std::string& cudaSource,
    const Grid& grid,
    const Block& block) {
  std::lock_guard<std::mutex> lock(mtx_);
  ++numberCacheAttemps;
  auto entry = searchKernel(id, options, inputs, outputs);
  if (entry) {
    if (entry->values.cudaSource == cudaSource or entry->values.grid == grid or
        entry->values.block == block) {
      throw CacheEntrySameKeyDifferentValue(
          "CudaCache::CacheKernel: a kernel matching the id, options and inputs was previously cached with different cuda source or block or grid dimensions.");
    }
    return;
  }

  entries_.emplace_back(
      id,
      kernelSpecializedName,
      kernelParameters,
      grid,
      block,
      options,
      inputs,
      outputs,
      cudaSource,
      CudaGPUInfo::GPUInfo().GetCudaDeviceStr());
}

CudaCachedEntry* CudaCache::searchKernel(
    const std::string& id,
    const CudaMappingOptions& options,
    const std::vector<detail::TensorInfo>& inputs,
    const std::vector<detail::TensorInfo>& outputs) {
  return searchKernelImpl(*this, id, options, inputs, outputs);
}

CudaCachedEntry* CudaCache::searchKernel(
    const std::string& id,
    const CudaMappingOptions& options,
    const std::vector<const DLTensor*>& inputs,
    const std::vector<const DLTensor*>& outputs) {
  return searchKernelImpl(*this, id, options, inputs, outputs);
}

const CudaCachedEntry* CudaCache::searchKernel(
    const std::string& id,
    const CudaMappingOptions& options,
    const std::vector<const DLTensor*>& inputs,
    const std::vector<const DLTensor*>& outputs) const {
  return searchKernelImpl(*this, id, options, inputs, outputs);
}

std::unique_ptr<CudaCacheRetrievalResult> CudaCache::retrieveKernel(
    const std::string& id,
    const CudaMappingOptions& options,
    const std::vector<const DLTensor*>& inputs,
    const std::vector<const DLTensor*>& outputs) const {
  std::lock_guard<std::mutex> lock(mtx_);
  ++numberAttemptedRetrievals;
  auto entry = searchKernel(id, options, inputs, outputs);
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
  std::vector<CachedEntry> newEntries;
  for (const auto& entry : oc) {
    for (const auto& options : entry.values) {
      auto cudaEntry = searchKernel(
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

size_t OptionsCache::totalSize() const {
  std::lock_guard<std::mutex> lock(mtx_);
  return std::accumulate(
      entries_.begin(),
      entries_.end(),
      size_t(0),
      [](size_t sum, const CachedEntry& e) { return sum + e.values.size(); });
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

std::vector<OptionsCacheRetrievalResult>
OptionsCache::retrieveOptionsAndRuntimes(
    const std::string& id,
    const std::vector<const DLTensor*>& inputs,
    const std::vector<const DLTensor*>& outputs) const {
  std::lock_guard<std::mutex> lock(mtx_);
  ++numberAttemptedRetrievals;
  auto ret = searchKernel(id, inputs, outputs);
  if (not ret) {
    return {};
  }
  ++numberSuccessfulRetrievals;
  std::vector<RetrievalResult> res;
  res.reserve(ret->values.size());
  std::transform(
      ret->values.begin(),
      ret->values.end(),
      std::back_inserter(res),
      [](const CachedEntry::Values& v) -> RetrievalResult {
        return {v.mappingOptions, v.recordedRuntimes};
      });
  return res;
}

std::vector<CudaMappingOptions> OptionsCache::retrieveTopKOptions(
    const std::string& id,
    const std::vector<const DLTensor*>& inputs,
    const std::vector<const DLTensor*>& outputs,
    size_t k) const {
  auto candidates = searchKernel(id, inputs, outputs);
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
      [](const CachedEntry::Values& v) {
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
        [](const CachedEntry::Values& a, const CachedEntry::Values& b) {
          // XXX:this is stupid, medians should be precomputed
          return median(a.recordedRuntimes) < median(b.recordedRuntimes);
        });
    if (entry.values.size() > numberToKeep) {
      entry.values.erase(
          entry.values.begin() + numberToKeep, entry.values.end());
    }
  }
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

  auto kernel = searchKernel(id, inputs, outputs);
  if (not kernel) {
    entries_.emplace_back(id, inputs, outputs, gpuStr, options, runtime);
    return;
  }
  auto v = std::find_if(
      kernel->values.begin(),
      kernel->values.end(),
      [&options](const CachedEntry::Values& v) {
        return v.mappingOptions == options;
      });
  if (v == kernel->values.end()) {
    kernel->values.emplace_back(options, runtime);
    return;
  }

  v->recordedRuntimes.push_back(runtime);
}

OptionsCachedEntry* OptionsCache::searchKernel(
    const std::string& id,
    const std::vector<const DLTensor*>& inputs,
    const std::vector<const DLTensor*>& outputs) {
  return searchKernelImpl(*this, id, inputs, outputs);
}

const OptionsCachedEntry* OptionsCache::searchKernel(
    const std::string& id,
    const std::vector<const DLTensor*>& inputs,
    const std::vector<const DLTensor*>& outputs) const {
  return searchKernelImpl(*this, id, inputs, outputs);
}

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
    std::vector<Duration>&& runtimes)
    : mappingOptions(options), recordedRuntimes(std::move(runtimes)) {}

OptionsCache::OptionsCache(const OptionsCacheProto& buf) {
  entries_.reserve(buf.entries_size());
  for (const auto& entry_buf : buf.entries())
    entries_.emplace_back(entry_buf);
}

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
    if (value.recorded_runtimes_size() == 0) {
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
    values.emplace_back(
        CudaMappingOptions(value.kernel_options()), std::move(runtimes));
  }
}

OptionsCacheProto OptionsCache::toProtobuf() const {
  OptionsCacheProto buf;
  auto* entriesBuf = buf.mutable_entries();
  entriesBuf->Reserve(entries_.size());
  std::transform(
      entries_.begin(),
      entries_.end(),
      google::protobuf::RepeatedPtrFieldBackInserter(entriesBuf),
      [](const CachedEntry& entry) { return entry.toProtobuf(); });
  return buf;
}

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
        return buf;
      });
  return buf;
}

CudaCacheProto CudaCache::toProtobuf() const {
  CudaCacheProto buf;
  auto* entriesBuf = buf.mutable_entries();
  entriesBuf->Reserve(entries_.size());
  std::transform(
      entries_.begin(),
      entries_.end(),
      google::protobuf::RepeatedPtrFieldBackInserter(entriesBuf),
      [](const CachedEntry& entry) { return entry.toProtobuf(); });
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

void removeFromCudaCacheEntriesNotInOptionsCache(
    CudaCache& cc,
    const OptionsCache& oc) {
  cc.removeEntriesNotInOptionsCache(oc);
}

std::unique_ptr<CudaCacheRetrievalResult> ManualCudaCache::retrieveKernel(
    const std::string& id,
    const std::vector<const DLTensor*>& inputs,
    const std::vector<const DLTensor*>& outputs) const {
  std::lock_guard<std::mutex> lock(mtx_);
  ++numberAttemptedRetrievals;
  auto entry = searchKernel(id, inputs, outputs);
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

ManualCudaCachedEntry* ManualCudaCache::searchKernel(
    const std::string& id,
    const std::vector<const DLTensor*>& inputs,
    const std::vector<const DLTensor*>& outputs) {
  return searchKernelImpl(*this, id, inputs, outputs);
}

const ManualCudaCachedEntry* ManualCudaCache::searchKernel(
    const std::string& id,
    const std::vector<const DLTensor*>& inputs,
    const std::vector<const DLTensor*>& outputs) const {
  return searchKernelImpl(*this, id, inputs, outputs);
}

void ManualCudaCache::cacheKernel(
    const std::string& id,
    const std::vector<const DLTensor*>& inputs,
    const std::vector<const DLTensor*>& outputs,
    const std::string& kernelSpecializedName,
    const std::vector<int>& kernelParameters,
    const std::string& cudaSource,
    const Grid& grid,
    const Block& block) {
  std::lock_guard<std::mutex> lock(mtx_);
  ++numberCacheAttemps;
  auto entry = searchKernel(id, inputs, outputs);
  if (entry) {
    entry->values.grid = grid;
    entry->values.block = block;
    entry->values.cudaSource = cudaSource;
    entry->values.kernelSpecializedName = kernelSpecializedName;
    entry->values.kernelParameters = kernelParameters;
    return;
  }

  entries_.emplace_back(
      id,
      kernelSpecializedName,
      kernelParameters,
      grid,
      block,
      inputs,
      outputs,
      cudaSource,
      CudaGPUInfo::GPUInfo().GetCudaDeviceStr());
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

} // namespace tc
