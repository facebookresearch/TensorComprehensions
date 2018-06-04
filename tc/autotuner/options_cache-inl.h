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
#include <fstream>
#include <memory>
#include <mutex>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <vector>

#include <llvm/ADT/Optional.h>

#include "tc/core/tensor.h"
#include "tc/core/utils/math.h"
#include "tc/core/utils/time.h"
#include "tc/lang/canonicalize.h"

#include "version.h"

namespace tc {
namespace autotune {

bool OptionsCacheKey::operator==(const OptionsCacheKey& other) const {
  return id == other.id && backendStr == other.backendStr &&
      inputs == other.inputs && outputs == other.outputs;
}

bool OptionsCacheKey::operator!=(const OptionsCacheKey& other) const {
  return !(*this == other);
}

OptionsCacheKeyProto OptionsCacheKey::toProtobuf() const {
  OptionsCacheKeyProto bufKey;
  bufKey.set_id(id);
  for (const auto& in : inputs) {
    auto pin = bufKey.add_inputs();
    *pin = in.toProtobuf();
  }
  for (const auto& out : outputs) {
    auto pout = bufKey.add_outputs();
    *pout = out.toProtobuf();
  }
  bufKey.set_backend_str(backendStr);
  bufKey.set_git_version(tc::git_version);
  return bufKey;
}

OptionsCacheKey OptionsCacheKey::fromProtobuf(
    const OptionsCacheKeyProto& proto) {
  OptionsCacheKey res;
  res.id = lang::CanonicalTcString(proto.id());
  for (int i = 0; i < proto.inputs().size(); ++i) {
    res.inputs.push_back(TensorInfo(proto.inputs().Get(i)));
  }
  for (int i = 0; i < proto.outputs().size(); ++i) {
    res.outputs.push_back(TensorInfo(proto.outputs().Get(i)));
  }
  res.backendStr = proto.backend_str();
  return res;
}

std::size_t OptionsCacheKeyHash::operator()(const OptionsCacheKey& k) const {
  using std::hash;
  // Just hash some string representation for now. When we measure
  // collisions are a problem; then deal with it. Before that it's
  // premature optimization.
  std::stringstream ss;
  ss << k.id;
  for (size_t i = 0; i < k.inputs.size(); ++i) {
    ss << k.inputs[i].toProtobuf().SerializeAsString();
  }
  for (size_t i = 0; i < k.outputs.size(); ++i) {
    ss << k.outputs[i].toProtobuf().SerializeAsString();
  }
  ss << k.backendStr;
  return std::hash<std::string>()(ss.str());
}

template <typename Backend>
typename Backend::OptionsCacheValueProtoType
OptionsCacheValue<Backend>::toProtobuf() const {
  typename Backend::OptionsCacheValueProtoType buf_value;
  *(buf_value.mutable_kernel_options()) = mappingOptions.proto();
  for (auto d : runtimes) {
    buf_value.add_recorded_runtimes(d.toMicroSeconds());
  }
  return buf_value;
}

template <typename Backend>
OptionsCacheValue<Backend> OptionsCacheValue<Backend>::fromProtobuf(
    const typename Backend::OptionsCacheValueProtoType& proto) {
  std::vector<Duration> runtimes;
  for (auto d : proto.recorded_runtimes()) {
    runtimes.push_back(Duration::fromMicroSeconds(d));
  }
  return OptionsCacheValue<Backend>{
      runtimes, typename Backend::MappingOptionsType(proto.kernel_options())};
}

template <typename Backend>
OptionsCache<Backend>::OptionsCache(const OptionsCache<Backend>& other) {
  std::lock_guard<std::mutex> lg(mutex);
  std::lock_guard<std::mutex> lg2(other.mutex);
  store_ = other.store_;
}

template <typename Backend>
void OptionsCache<Backend>::clear() {
  std::lock_guard<std::mutex> clear(mutex);
  store_.clear();
  numberCacheAttempts = 0;
  numberAttemptedRetrievals = 0;
  numberSuccessfulRetrievals = 0;
}

template <typename Backend>
size_t OptionsCache<Backend>::count(const OptionsCacheKey& key) const {
  std::lock_guard<std::mutex> lock(mutex);
  return store_.count(key);
}

template <typename Backend>
size_t OptionsCache<Backend>::size() const {
  std::lock_guard<std::mutex> lock(mutex);
  return store_.size();
}

template <typename Backend>
void OptionsCache<Backend>::loadCacheFromFile(const std::string& filename) {
  typename Backend::OptionsCacheProtoType buf;
  struct stat buffer = {0};
  if (stat(filename.c_str(), &buffer) == 0) {
    std::ifstream serialized(filename, std::ios::binary);
    buf.ParseFromIstream(&serialized);
  }
  fromProtobuf(buf);
}

template <typename Backend>
void OptionsCache<Backend>::storeCacheToFile(
    const std::string& filename) const {
  // toProtobuf() takes the lock too, get a copy of the result first
  auto proto = toProtobuf();
  {
    std::lock_guard<std::mutex> lock(mutex);
    std::fstream serialized(
        filename, std::ios::binary | std::ios::trunc | std::ios::out);
    if (!serialized.is_open()) {
      LOG(ERROR) << "Failed to open the output stream for dumping protobuf: "
                 << filename;
    } else {
      proto.SerializePartialToOstream(&serialized);
    }
  }
}

namespace detail {

template <typename Backend>
void recordRuntimeImpl(
    std::mutex& mutex,
    size_t& numberCacheAttempts,
    typename OptionsCache<Backend>::MultiMapType& store,
    const typename OptionsCache<Backend>::KeyType& key,
    const typename Backend::MappingOptionsType& options,
    Duration duration) {
  std::lock_guard<std::mutex> lock(mutex);
  ++numberCacheAttempts;
  auto range = store.equal_range(key);
  for (auto it = range.first; it != range.second; ++it) {
    if (it->second.mappingOptions == options) {
      // key exists, append to it and return
      it->second.runtimes.push_back(duration);
      return;
    }
  }
  // key does not exist, emplace a new key, value
  store.emplace(
      key,
      OptionsCacheValue<Backend>{std::vector<Duration>{duration}, options});
}

} // namespace detail

template <typename Backend>
void OptionsCache<Backend>::recordRuntime(
    const lang::CanonicalTcString& tc,
    const std::vector<TensorInfo>& inputs,
    const std::vector<TensorInfo>& outputs,
    const std::string& backendStr,
    const typename Backend::MappingOptionsType& options,
    Duration duration) {
  OptionsCacheKey key{tc, inputs, outputs, backendStr};
  detail::recordRuntimeImpl<Backend>(
      mutex, numberCacheAttempts, store_, key, options, duration);
}

template <typename Backend>
void OptionsCache<Backend>::recordRuntime(
    const typename Backend::ExecutorType& exec,
    Duration duration) {
  OptionsCacheKey key{canonicalTc(exec.tc()),
                      exec.inputsInfo(),
                      exec.outputsInfo(),
                      exec.deviceName()};

  detail::recordRuntimeImpl<Backend>(
      mutex, numberCacheAttempts, store_, key, exec.options(), duration);
}

namespace detail {
template <typename Backend>
struct OptionsWithMedianAndRuntimes {
  Duration median;
  std::vector<Duration> runtimes;
  typename Backend::MappingOptionsType mappingOptions;
};

template <typename Backend>
std::vector<OptionsWithMedianAndRuntimes<Backend>> sortedOptions(
    const OptionsCacheKey& key,
    const std::unordered_multimap<
        OptionsCacheKey,
        OptionsCacheValue<Backend>,
        OptionsCacheKeyHash>& store) {
  using Options = OptionsWithMedianAndRuntimes<Backend>;
  std::vector<Options> toSort;
  if (store.count(key) == 0) {
    return {};
  }
  auto range = store.equal_range(key);
  for (auto it = range.first; it != range.second; ++it) {
    if (it->second.runtimes.size() == 0) {
      throw std::runtime_error("No runtime for cache option");
    }
    toSort.push_back(Options{median(it->second.runtimes),
                             it->second.runtimes,
                             it->second.mappingOptions});
  }
  std::sort(
      toSort.begin(), toSort.end(), [](const Options& a, const Options& b) {
        // fun with C++, a.median < b.median does not mix with templates
        return operator<(a.median, b.median);
      });
  return toSort;
}

template <typename Backend>
std::vector<typename Backend::MappingOptionsType> getTopKOptionsImpl(
    std::mutex& mutex,
    size_t& numberAttemptedRetrievals,
    size_t& numberSuccessfulRetrievals,
    const typename OptionsCache<Backend>::MultiMapType& store,
    const typename OptionsCache<Backend>::KeyType& key,
    size_t K) {
  std::lock_guard<std::mutex> lock(mutex);
  ++numberAttemptedRetrievals;
  auto sorted = detail::sortedOptions<Backend>(key, store);
  if (sorted.size() == 0u) {
    return {};
  }
  std::vector<typename Backend::MappingOptionsType> res;
  res.reserve(K);
  for (size_t i = 0; i < std::min(K, sorted.size()); ++i) {
    res.push_back(sorted[i].mappingOptions);
  }
  ++numberSuccessfulRetrievals;
  return res;
}

} // namespace detail

template <typename Backend>
std::vector<typename Backend::MappingOptionsType>
OptionsCache<Backend>::getTopKOptions(
    const lang::CanonicalTcString& tc,
    const std::vector<TensorInfo>& inputs,
    const std::vector<TensorInfo>& outputs,
    const std::string& backendStr,
    size_t K) const {
  OptionsCacheKey key{tc, inputs, outputs, backendStr};
  return detail::getTopKOptionsImpl<Backend>(
      mutex,
      numberAttemptedRetrievals,
      numberSuccessfulRetrievals,
      store_,
      key,
      K);
}

template <typename Backend>
std::vector<typename Backend::MappingOptionsType>
OptionsCache<Backend>::getTopKOptions(
    const typename Backend::ExecutorType& exec,
    size_t K) const {
  OptionsCacheKey key{canonicalTc(exec.tc()),
                      exec.inputsInfo(),
                      exec.outputsInfo(),
                      exec.deviceName()};
  return detail::getTopKOptionsImpl<Backend>(
      mutex,
      numberAttemptedRetrievals,
      numberSuccessfulRetrievals,
      store_,
      key,
      K);
}

template <typename Backend>
std::unordered_set<OptionsCacheKey, OptionsCacheKeyHash>
OptionsCache<Backend>::getKeys() const {
  std::lock_guard<std::mutex> lock(mutex);
  std::unordered_set<OptionsCacheKey, OptionsCacheKeyHash> keys;
  for (auto kvp : store_) {
    keys.emplace(kvp.first);
  }
  return keys;
}

template <typename Backend>
void OptionsCache<Backend>::pruneKeepTopK(size_t K) {
  auto keys = getKeys();
  {
    std::lock_guard<std::mutex> lock(mutex);
    for (auto& k : keys) {
      // this is of course wasteful but real topk is premature optimization atm
      auto sorted = detail::sortedOptions<Backend>(k, store_);
      // erase all then reinsert makes it easier to reuse code
      store_.erase(k);
      for (size_t i = 0; i < std::min(sorted.size(), K); ++i) {
        auto option = sorted[i];
        store_.emplace(
            k,
            OptionsCacheValue<Backend>{option.runtimes, option.mappingOptions});
      }
    }
  }
}

template <typename Backend>
typename Backend::OptionsCacheProtoType OptionsCache<Backend>::toProtobuf()
    const {
  std::lock_guard<std::mutex> lock(mutex);
  typename Backend::OptionsCacheProtoType buf;
  for (const auto& kvp : store_) {
    auto pkey = buf.add_keys();
    *pkey = kvp.first.toProtobuf();
    auto pvalues = buf.add_values();
    *pvalues = kvp.second.toProtobuf();
  }
  return buf;
}

template <typename Backend>
void OptionsCache<Backend>::fromProtobuf(
    const typename Backend::OptionsCacheProtoType& proto) {
  std::lock_guard<std::mutex> lock(mutex);
  CHECK_EQ(proto.keys().size(), proto.values().size());
  for (int i = 0; i < proto.keys().size(); ++i) {
    OptionsCacheKey key(OptionsCacheKey::fromProtobuf(proto.keys().Get(i)));
    OptionsCacheValue<Backend> value(
        OptionsCacheValue<Backend>::fromProtobuf(proto.values().Get(i)));
    store_.emplace(key, value);
  }
}

} // namespace autotune

inline std::string makeOptionsFilename(const std::string& fn) {
  return fn + ".options";
}
} // namespace tc
