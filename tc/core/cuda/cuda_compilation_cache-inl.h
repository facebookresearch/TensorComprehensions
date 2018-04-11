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

#include <sys/stat.h>
#include <algorithm>
#include <fstream>
#include <string>

#include <glog/logging.h>
#include <version.h>

namespace tc {

template <typename CC, typename CachedEntryType>
void Cache<CC, CachedEntryType>::enableCache() {
  CC::getGlobalSharedCache() = std::make_shared<CC>();
}

template <typename CC, typename CachedEntryType>
void Cache<CC, CachedEntryType>::disableCache() {
  CC::getGlobalSharedCache() = nullptr;
}

template <typename CC, typename CachedEntryType>
std::shared_ptr<CC> Cache<CC, CachedEntryType>::getCache() {
  if (not cacheEnabled()) {
    throw std::runtime_error(
        "EnableCache or LoadCacheFromProtobuf must be called before using the cache.");
  }
  return CC::getGlobalSharedCache();
}

template <typename CC, typename CachedEntryType>
void Cache<CC, CachedEntryType>::dumpCacheToProtobuf(
    const std::string& filename) {
  std::fstream serialized(
      filename, std::ios::binary | std::ios::trunc | std::ios::out);
  if (!serialized) {
    LOG(ERROR) << "Failed to open the output stream for dumping protobuf: "
               << filename;
  } else {
    getCache()->toProtobuf().SerializePartialToOstream(&serialized);
  }
}

template <typename CC, typename CachedEntryType>
void Cache<CC, CachedEntryType>::loadCacheFromProtobuf(
    const std::string& filename) {
  typename CC::ProtobufType buf;
  struct stat buffer = {0};
  if (stat(filename.c_str(), &buffer) == 0) {
    std::ifstream serialized(filename, std::ios::binary);
    buf.ParseFromIstream(&serialized);
  }
  loadCacheFromProtobuf(buf);
}

template <typename CC, typename CachedEntryType>
template <typename Protobuf>
void Cache<CC, CachedEntryType>::loadCacheFromProtobuf(const Protobuf& buf) {
  static_assert(
      std::is_same<Protobuf, typename CC::ProtobufType>::value,
      "LoadCacheFromProtobuf called with invalide protobuf type.");
  CC::getGlobalSharedCache() = std::make_shared<CC>(buf);
}

template <typename CC, typename CachedEntryType>
bool Cache<CC, CachedEntryType>::cacheEnabled() {
  return CC::getGlobalSharedCache() != nullptr;
}

template <typename CC, typename CachedEntryType>
size_t Cache<CC, CachedEntryType>::size() const {
  std::lock_guard<std::mutex> lock(mtx_);
  return static_cast<const CC*>(this)->entries_.size();
}

template <typename CC, typename CachedEntryType>
void Cache<CC, CachedEntryType>::clear() {
  std::lock_guard<std::mutex> lock(mtx_);
  numberAttemptedRetrievals = numberSuccessfulRetrievals = numberCacheAttemps =
      0;
  static_cast<CC*>(this)->entries_.clear();
}
} // namespace tc
