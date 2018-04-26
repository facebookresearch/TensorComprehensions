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
#include <unordered_map>
#include <vector>

#include <llvm/ADT/Optional.h>

#include <version.h>

#include "tc/core/tensor.h"
#include "tc/core/utils/time.h"
#include "tc/lang/canonicalize.h"

namespace tc {
namespace autotune {

/**
 * A key in the options cache is exactly the content of the underlying proto in
 * tc/proto/compcache.proto. It provides simple conversions and the equality
 * operator. Additionally we provide a hash function to allow it to be a key
 * in a hash map.
 */
struct OptionsCacheKey {
  lang::CanonicalTcString id;
  std::vector<TensorInfo> inputs;
  std::vector<TensorInfo> outputs;
  std::string deviceStr;

  inline bool operator==(const OptionsCacheKey& other) const;
  inline OptionsCacheKeyProto toProtobuf() const;
  static inline OptionsCacheKey fromProtobuf(const OptionsCacheKeyProto& proto);
};

struct OptionsCacheKeyHash {
  inline std::size_t operator()(const OptionsCacheKey& k) const;
};

/**
 * A value in the options cache is exactly the content of the underlying proto
 * in tc/proto/compcache.proto. An OptionsCacheValue is templated by the Backend
 * and contains an object of the proper MappingOptions type as well as the
 * runtimes of subsequent runs.
 */
template <typename Backend>
struct OptionsCacheValue {
  typename Backend::OptionsCacheValueProtoType toProtobuf() const;
  static OptionsCacheValue<Backend> fromProtobuf(
      const typename Backend::OptionsCacheValueProtoType& proto);

  std::vector<Duration> runtimes;
  typename Backend::MappingOptionsType mappingOptions;
};

/**
 * An Options cache is a simple abstraction around an unordered_multimap of
 * protobuf-backed key value pairs. It has an underlying store object and
 * provides simple functions to load/store from proto, record, prune and
 * extract topK values ordered by runtime.
 * An OptionsCache is templated by the backend type because the values stored
 * are backend-dependent.
 */
template <typename Backend>
struct OptionsCache {
 public:
  using KeyType = OptionsCacheKey;
  using ValueType = OptionsCacheValue<Backend>;
  using MultiMapType = std::unordered_multimap<
      OptionsCacheKey,
      OptionsCacheValue<Backend>,
      OptionsCacheKeyHash>;

  OptionsCache<Backend>();

  /// Clears the content of the cache and resets the counters
  void clear();

  /// \return the number of values for a particular key
  size_t count(const OptionsCacheKey& key) const;

  //// \return the number of elements in the cache
  size_t size() const;

  /// Collects the keys in the cache
  /// \return an unordered_set of keys
  std::unordered_set<OptionsCacheKey, OptionsCacheKeyHash> getKeys() const;

  /// Loads in place from proto file. Calls fromProto which can insert
  /// duplicates, so be sure your cache is cleared if you don't want those
  void loadCacheFromFile(const std::string& filename);

  /// Stores to a proto file at the specified location
  void storeCacheToFile(const std::string& filename) const;

  /// Saves a new runtime.
  /// If the key does not exist, a new entry is inserted.
  /// If the key exists, a search is performed on options.
  /// If the corresponding options are found, the duration is appended to the
  /// runtimes, otherwise a new entry is inserted in the multimap.
  void recordRuntime(
      const lang::CanonicalTcString& tc,
      const std::vector<TensorInfo>& inputs,
      const std::vector<TensorInfo>& outputs,
      const std::string& deviceStr,
      const typename Backend::MappingOptionsType& options,
      Duration duration);

  /// Returns the top-K best mapping options for a particular
  /// TC/inputs/outputs/device. Note that the result may be empty (in
  /// particular if problem size is small and pruning threshold is too high
  /// for the problem size).
  /// \returns a vector of mapping options
  std::vector<typename Backend::MappingOptionsType> getTopKOptions(
      const lang::CanonicalTcString& tc,
      const std::vector<TensorInfo>& inputs,
      const std::vector<TensorInfo>& outputs,
      const std::string& deviceStr,
      size_t K) const;

  /// Drops the (N - K) worst performing options
  void pruneKeepTopK(size_t K);

 protected:
  // Make protected and not private so we can derive and test the internals
  typename Backend::OptionsCacheProtoType toProtobuf() const;
  void fromProtobuf(const typename Backend::OptionsCacheProtoType& proto);

 public:
  mutable size_t numberCacheAttempts{0};
  mutable size_t numberAttemptedRetrievals{0};
  mutable size_t numberSuccessfulRetrievals{0};

 protected:
  // Make protected and not private so we can derive and test the internals
  mutable std::mutex mutex;

  std::unordered_multimap<
      OptionsCacheKey,
      OptionsCacheValue<Backend>,
      OptionsCacheKeyHash>
      store_;
};
} // namespace autotune

inline std::string makeOptionsFilename(const std::string& fn) {
  return fn + ".options";
}
} // namespace tc

#include "tc/autotuner/options_cache-inl.h"
