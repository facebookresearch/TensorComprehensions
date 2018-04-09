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

#include <version.h>

#include "tc/core/tensor.h"
#include "tc/core/utils/math.h"
#include "tc/core/utils/time.h"
#include "tc/lang/canonicalize.h"

namespace tc {
namespace autotune {

bool OptionsCacheKey::operator==(const OptionsCacheKey& other) const {
  if (id != other.id) {
    return false;
  }
  if (deviceStr != other.deviceStr) {
    return false;
  }
  for (size_t i = 0; i < inputs.size(); ++i) {
    if (not(inputs[i] == other.inputs[i])) {
      return false;
    }
  }
  for (size_t i = 0; i < outputs.size(); ++i) {
    if (not(outputs[i] == other.outputs[i])) {
      return false;
    }
  }
  return true;
}

OptionsCacheKeyProto OptionsCacheKey::toProtobuf() const {
  OptionsCacheKeyProto buf_key;
  buf_key.set_id(id);
  for (const auto& in : inputs) {
    auto pin = buf_key.add_inputs();
    *pin = in.toProtobuf();
  }
  for (const auto& out : outputs) {
    auto pout = buf_key.add_outputs();
    *pout = out.toProtobuf();
  }
  buf_key.set_device_str(deviceStr);
  buf_key.set_git_version("");
  return buf_key;
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
  res.deviceStr = proto.device_str();
  return res;
}

std::size_t OptionsCacheKeyHash::operator()(const OptionsCacheKey& k) const {
  using std::hash;
  // Just hash some string representation for now, when we measure
  // collisions are a problem then deal with it, before that it's
  // premature optimization.
  std::stringstream ss;
  ss << k.id;
  for (size_t i = 0; i < k.inputs.size(); ++i) {
    ss << k.inputs[i].toProtobuf().SerializeAsString();
  }
  for (size_t i = 0; i < k.outputs.size(); ++i) {
    ss << k.outputs[i].toProtobuf().SerializeAsString();
  }
  ss << k.deviceStr;
  return std::hash<std::string>()(ss.str());
}

template <typename Backend>
typename Backend::OptionsCacheValueProtoType
OptionsCacheValue<Backend>::toProtobuf() const {
  typename Backend::OptionsCacheValueProtoType buf_value;
  *(buf_value.mutable_kernel_options()) = mappingOptions.proto();
  for (auto d : runtimes) {
    buf_value.add_recorded_runtimes(
        std::chrono::duration_cast<std::chrono::microseconds>(d).count());
  }
  return buf_value;
}

template <typename Backend>
OptionsCacheValue<Backend> OptionsCacheValue<Backend>::fromProtobuf(
    const typename Backend::OptionsCacheValueProtoType& proto) {
  std::vector<Duration> runtimes;
  for (auto d : proto.recorded_runtimes()) {
    runtimes.push_back(Duration(d));
  }
  return OptionsCacheValue<Backend>{
      runtimes, typename Backend::MappingOptionsType(proto.kernel_options())};
}

template <typename Backend>
OptionsCache<Backend>::OptionsCache() {}

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
void OptionsCache<Backend>::clear() {
  std::lock_guard<std::mutex> clear(mutex);
  store_.clear();
  numberCacheAttempts = 0;
  numberAttemptedRetrievals = 0;
  numberSuccessfulRetrievals = 0;
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
  std::lock_guard<std::mutex> lock(mutex);
  std::fstream serialized(
      filename, std::ios::binary | std::ios::trunc | std::ios::out);
  if (!serialized.is_open()) {
    LOG(ERROR) << "Failed to open the output stream for dumping protobuf: "
               << filename;
  } else {
    toProtobuf().SerializePartialToOstream(&serialized);
  }
}

template <typename Backend>
void OptionsCache<Backend>::recordRuntime(
    const lang::CanonicalTcString& tc,
    const std::vector<TensorInfo>& inputs,
    const std::vector<TensorInfo>& outputs,
    const std::string& deviceStr,
    const typename Backend::MappingOptionsType& options,
    Duration duration) {
  std::lock_guard<std::mutex> lock(mutex);
  ++numberCacheAttempts;
  OptionsCacheKey key{tc, inputs, outputs, deviceStr};
  bool inserted = false;
  auto range = store_.equal_range(key);
  for (auto it = range.first; it != range.second; ++it) {
    if (it->second.mappingOptions == options) {
      it->second.runtimes.push_back(duration);
      inserted = true;
      break;
    }
  }
  if (!inserted) {
    store_.emplace(
        key,
        OptionsCacheValue<Backend>{std::vector<Duration>{duration}, options});
  }
}

template <typename Backend>
std::vector<typename Backend::MappingOptionsType>
OptionsCache<Backend>::getTopKOptions(
    const lang::CanonicalTcString& tc,
    const std::vector<TensorInfo>& inputs,
    const std::vector<TensorInfo>& outputs,
    const std::string& deviceStr,
    size_t K) const {
  std::lock_guard<std::mutex> lock(mutex);
  ++numberAttemptedRetrievals;
  struct WithTime {
    typename Backend::MappingOptionsType mappingOptions;
    Duration duration;
  };
  std::vector<WithTime> toSort;
  OptionsCacheKey key{tc, inputs, outputs, deviceStr};
  if (store_.count(key) == 0) {
    return {};
  }
  auto range = store_.equal_range(key);
  for (auto it = range.first; it != range.second; ++it) {
    if (it->second.runtimes.size() == 0) {
      throw std::runtime_error("No runtime for cache option");
    }
    toSort.push_back(
        WithTime{it->second.mappingOptions, median(it->second.runtimes)});
  }
  // this is of course wasteful but real topk is premature optimization atm
  std::sort(
      toSort.begin(), toSort.end(), [](const WithTime& a, const WithTime& b) {
        return a.duration < b.duration;
      });
  std::vector<typename Backend::MappingOptionsType> res;
  res.reserve(K);
  for (size_t i = 0; i < std::min(K, toSort.size()); ++i) {
    res.push_back(toSort[i].mappingOptions);
  }
  ++numberSuccessfulRetrievals;
  return res;
}

template <typename Backend>
std::unordered_set<OptionsCacheKey, OptionsCacheKeyHash>
OptionsCache<Backend>::getKeys() const {
  std::lock_guard<std::mutex> lock(mutex);
  std::unordered_set<OptionsCacheKey, OptionsCacheKeyHash> keys;
  for (auto kvp : store_) {
    if (keys.count(kvp.first) > 0) {
      continue;
    }
    keys.emplace(kvp.first);
  }
  return keys;
}

template <typename Backend>
void OptionsCache<Backend>::pruneKeepTopK(size_t K) {
  auto keys = getKeys();
  std::lock_guard<std::mutex> lock(mutex);
  for (auto& k : keys) {
    auto range = store_.equal_range(k);
    std::vector<typename MultiMapType::iterator> toSort;
    for (auto it = range.first; it != range.second; ++it) {
      toSort.push_back(it);
    }
    // this is of course wasteful but real topk is premature optimization atm
    std::sort(
        toSort.begin(),
        toSort.end(),
        [](const typename MultiMapType::iterator& a,
           const typename MultiMapType::iterator& b) {
          return median(a->second.runtimes) < median(b->second.runtimes);
        });
    size_t kept = 0;
    for (auto it : toSort) {
      if (kept >= K) {
        store_.erase(it);
      }
      kept++;
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
} // namespace tc
