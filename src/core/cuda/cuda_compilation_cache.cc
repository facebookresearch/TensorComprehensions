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

#include "tc/core/mapping_options.h"
#include "tc/core/utils/math.h"

namespace tc {

namespace {
uint64_t GetDLTensorAlignment(const DLTensor* t) {
  return (reinterpret_cast<std::uintptr_t>(t->data) + t->byte_offset) % 256;
}
} // namespace

detail::TensorInfo::TensorInfo(const DLTensor* t)
    : alignment{GetDLTensorAlignment(t)}, dType(t->dtype) {
  shape.reserve(t->ndim);
  std::copy(t->shape, t->shape + t->ndim, std::back_inserter(shape));
  if (not t->strides) {
    return;
  }
  strides.reserve(t->ndim);
  std::copy(t->strides, t->strides + t->ndim, std::back_inserter(strides));
}

detail::TensorInfo::TensorInfo(const TensorInfoProto& buf)
    : shape{buf.shape().begin(), buf.shape().end()},
      strides{buf.strides().begin(), buf.strides().end()},
      alignment{buf.alignment()},
      dType{static_cast<uint8_t>(buf.dtype().code()),
            static_cast<uint8_t>(buf.dtype().bits()),
            static_cast<uint16_t>(buf.dtype().lanes())} {}

TensorInfoProto detail::TensorInfo::toProtobuf() const {
  TensorInfoProto buf;
  buf.mutable_shape()->Reserve(shape.size());
  std::copy(
      shape.begin(),
      shape.end(),
      google::protobuf::RepeatedFieldBackInserter(buf.mutable_shape()));
  buf.mutable_strides()->Reserve(strides.size());
  std::copy(
      strides.begin(),
      strides.end(),
      google::protobuf::RepeatedFieldBackInserter(buf.mutable_strides()));
  buf.set_alignment(alignment);
  buf.mutable_dtype()->set_code(dType.code);
  buf.mutable_dtype()->set_bits(dType.bits);
  buf.mutable_dtype()->set_lanes(dType.lanes);
  return buf;
}

bool detail::TensorInfo::operator==(const DLTensor* t) const {
  if (t->ndim != static_cast<int>(shape.size())) {
    return false;
  }

  auto res = std::mismatch(shape.begin(), shape.end(), t->shape);
  if (res.first != shape.end() || res.second != t->shape + t->ndim) {
    return false;
  }

  if (t->strides == nullptr) {
    if (strides.size() > 0) {
      return false;
    }
  } else {
    if (t->ndim != static_cast<int>(strides.size())) {
      return false;
    }

    res = std::mismatch(strides.begin(), strides.end(), t->strides);
    if (res.first != strides.end() || res.second != t->strides + t->ndim) {
      return false;
    }
  }

  /*This should be enabled when/if tc starts using alignment information
   *if (GetDLTensorAlignment(t) != alignment) {
   *  return false;
   *}
   */
  return std::tie(t->dtype.code, t->dtype.bits, t->dtype.lanes) ==
      std::tie(dType.code, dType.bits, dType.lanes);
}

bool operator==(const DLDataType& a, const DLDataType& b) {
  return a.code == b.code and a.bits == b.bits and a.lanes == b.lanes;
}

bool operator<(const DLDataType& a, const DLDataType& b) {
  return a.code < b.code and a.bits < b.bits and a.lanes < b.lanes;
}

bool detail::TensorInfo::operator==(const TensorInfo& t) const {
  return alignment == t.alignment and dType == t.dType and shape == t.shape and
      strides == t.strides;
}

bool detail::TensorInfo::operator<(const TensorInfo& t) const {
  return alignment < t.alignment and dType < t.dType and shape < t.shape and
      strides < t.strides;
}

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
} // namespace

namespace {
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
} // namespace

namespace {
template <typename Array, typename Buf>
void WriteProtobufArray(const Array& arr, Buf* buf) {
  google::protobuf::RepeatedField<typename Array::value_type> data(
      arr.begin(), arr.end());
  buf->Swap(&data);
}
} // namespace

bool operator==(
    const std::vector<const DLTensor*>& inputsTensor,
    const std::vector<detail::TensorInfo>& inputsInfo) {
  if (inputsTensor.size() != inputsInfo.size()) {
    return false;
  }
  CHECK(inputsTensor.size() == inputsInfo.size());
  for (size_t i = 0, n = inputsInfo.size(); i < n; ++i) {
    if (!(inputsInfo[i] == inputsTensor[i])) {
      return false;
    }
  }
  return true;
}

namespace {
std::shared_ptr<CudaCache> cudaCache_;
std::shared_ptr<OptionsCache> optionsCache_;
std::shared_ptr<ManualCudaCache> manualCudaCache_;
} // namespace

std::shared_ptr<CudaCache>& CudaCache::getGlobalSharedCache() {
  return cudaCache_;
}

std::shared_ptr<OptionsCache>& OptionsCache::getGlobalSharedCache() {
  return optionsCache_;
}

std::shared_ptr<ManualCudaCache>& ManualCudaCache::getGlobalSharedCache() {
  return manualCudaCache_;
}

CudaCache::CudaCache(const CudaCacheProto& buf) {
  entries_.reserve(buf.entries_size());
  for (const auto& entry_buf : buf.entries())
    entries_.emplace_back(entry_buf);
}

CudaCache::CachedEntry::CachedEntry(
    const std::string& id,
    const std::string& kernelSpecializedName,
    const std::vector<int>& kernelParameters,
    const Grid& grid,
    const Block& block,
    const MappingOptions& mappingOptions,
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

CudaCache::CachedEntry::CachedEntry(const CudaCacheEntryProto& buf)
    : key{buf.id(),
          MappingOptions{buf.kernel_options()},
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
    const MappingOptions& options,
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

CudaCache::CachedEntry* CudaCache::searchKernel(
    const std::string& id,
    const MappingOptions& options,
    const std::vector<detail::TensorInfo>& inputs,
    const std::vector<detail::TensorInfo>& outputs) {
  return searchKernelImpl(*this, id, options, inputs, outputs);
}

CudaCache::CachedEntry* CudaCache::searchKernel(
    const std::string& id,
    const MappingOptions& options,
    const std::vector<const DLTensor*>& inputs,
    const std::vector<const DLTensor*>& outputs) {
  return searchKernelImpl(*this, id, options, inputs, outputs);
}

const CudaCache::CachedEntry* CudaCache::searchKernel(
    const std::string& id,
    const MappingOptions& options,
    const std::vector<const DLTensor*>& inputs,
    const std::vector<const DLTensor*>& outputs) const {
  return searchKernelImpl(*this, id, options, inputs, outputs);
}

std::unique_ptr<CudaCache::RetrievalResult> CudaCache::retrieveKernel(
    const std::string& id,
    const MappingOptions& options,
    const std::vector<const DLTensor*>& inputs,
    const std::vector<const DLTensor*>& outputs) const {
  std::lock_guard<std::mutex> lock(mtx_);
  ++numberAttemptedRetrievals;
  auto entry = searchKernel(id, options, inputs, outputs);
  if (not entry) {
    return nullptr;
  }
  ++numberSuccessfulRetrievals;
  return std::unique_ptr<CudaCache::RetrievalResult>(
      new CudaCache::RetrievalResult{entry->values.cudaSource,
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

std::unique_ptr<MappingOptions> OptionsCache::retrieveBestOptions(
    const std::string& id,
    const std::vector<const DLTensor*>& inputs,
    const std::vector<const DLTensor*>& outputs) const {
  auto ret = retrieveTopKOptions(id, inputs, outputs, 1);
  if (ret.empty()) {
    return nullptr;
  }
  return std::unique_ptr<MappingOptions>(new MappingOptions(ret.front()));
}

std::vector<OptionsCache::RetrievalResult>
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

std::vector<MappingOptions> OptionsCache::retrieveTopKOptions(
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
    const MappingOptions* options;
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

  std::vector<MappingOptions> res;
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
    const MappingOptions& options,
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

OptionsCache::CachedEntry* OptionsCache::searchKernel(
    const std::string& id,
    const std::vector<const DLTensor*>& inputs,
    const std::vector<const DLTensor*>& outputs) {
  return searchKernelImpl(*this, id, inputs, outputs);
}

const OptionsCache::CachedEntry* OptionsCache::searchKernel(
    const std::string& id,
    const std::vector<const DLTensor*>& inputs,
    const std::vector<const DLTensor*>& outputs) const {
  return searchKernelImpl(*this, id, inputs, outputs);
}

OptionsCache::CachedEntry::CachedEntry(
    const std::string& id,
    const std::vector<const DLTensor*>& inputs,
    const std::vector<const DLTensor*>& outputs,
    const std::string& deviceStr,
    const MappingOptions& options,
    Duration runtime)
    : key(id, inputs, outputs, deviceStr, git_version) {
  values.emplace_back(options, runtime);
}

OptionsCache::CachedEntry::Key::Key(
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

OptionsCache::CachedEntry::Key::Key(
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

OptionsCache::CachedEntry::Values::Values(
    const MappingOptions& options,
    Duration runtime)
    : mappingOptions(options), recordedRuntimes{runtime} {}

OptionsCache::CachedEntry::Values::Values(
    const MappingOptions& options,
    std::vector<Duration>&& runtimes)
    : mappingOptions(options), recordedRuntimes(std::move(runtimes)) {}

OptionsCache::OptionsCache(const OptionsCacheProto& buf) {
  entries_.reserve(buf.entries_size());
  for (const auto& entry_buf : buf.entries())
    entries_.emplace_back(entry_buf);
}

decltype(OptionsCache::entries_)::const_iterator OptionsCache::begin() const {
  return entries_.begin();
}

decltype(OptionsCache::entries_)::const_iterator OptionsCache::end() const {
  return entries_.end();
}

OptionsCache::CachedEntry::CachedEntry(const OptionsCacheEntryProto& buf)
    : key(buf.id(),
          ProtoToTensorInfoVector(buf.inputs()),
          ProtoToTensorInfoVector(buf.outputs()),
          buf.device_str(),
          buf.git_version()) {
  if (buf.values_size() == 0) {
    throw std::invalid_argument(
        "OptionsCache::CachedEntry invalid protobuf: each entry should have at least one value field.");
  }

  for (const auto& value : buf.values()) {
    if (value.recorded_runtimes_size() == 0) {
      throw std::invalid_argument(
          "OptionsCache::CachedEntry invalid protobuf: each entry value should have at least one recorded runtime.");
    }
    std::vector<Duration> runtimes;
    runtimes.reserve(value.recorded_runtimes_size());
    std::transform(
        value.recorded_runtimes().begin(),
        value.recorded_runtimes().end(),
        std::back_inserter(runtimes),
        [](int64_t us) { return std::chrono::microseconds(us); });
    values.emplace_back(
        MappingOptions(value.kernel_options()), std::move(runtimes));
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

OptionsCacheEntryProto OptionsCache::CachedEntry::toProtobuf() const {
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
        *buf.mutable_kernel_options() = v.mappingOptions.proto;
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

CudaCacheEntryProto CudaCache::CachedEntry::toProtobuf() const {
  CudaCacheEntryProto buf;
  buf.set_id(key.id);
  *buf.mutable_kernel_options() = key.mappingOptions.proto;
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
  *buf.mutable_grid_dims() = values.grid.proto;
  *buf.mutable_block_dims() = values.block.proto;
  buf.set_specialized_name(values.kernelSpecializedName);
  WriteProtobufArray(values.kernelParameters, buf.mutable_parameters());

  return buf;
}

void removeFromCudaCacheEntriesNotInOptionsCache(
    CudaCache& cc,
    const OptionsCache& oc) {
  cc.removeEntriesNotInOptionsCache(oc);
}

std::string makeOptionsFilename(const std::string& filename) {
  return filename + ".options";
}

std::string makeCudaFilename(const std::string& filename) {
  return filename + ".cuda";
}

std::unique_ptr<CudaCache::RetrievalResult> ManualCudaCache::retrieveKernel(
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
  return std::unique_ptr<CudaCache::RetrievalResult>(
      new CudaCache::RetrievalResult{entry->values.cudaSource,
                                     entry->values.kernelSpecializedName,
                                     entry->values.kernelParameters,
                                     entry->values.grid,
                                     entry->values.block});
}

ManualCudaCache::CachedEntry* ManualCudaCache::searchKernel(
    const std::string& id,
    const std::vector<const DLTensor*>& inputs,
    const std::vector<const DLTensor*>& outputs) {
  return searchKernelImpl(*this, id, inputs, outputs);
}

const ManualCudaCache::CachedEntry* ManualCudaCache::searchKernel(
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
ManualCudaCache::CachedEntry::CachedEntry(
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
