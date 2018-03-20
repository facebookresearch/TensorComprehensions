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

#include <mapping_options.pb.h>

#include <array>
#include <iostream>
#include <string>
#include <vector>

#include "tc/external/isl.h"

#include "tc/core/flags.h"
#include "tc/core/mapping_options.h"

namespace tc {

/// View of a CudaDimProto.
///
/// Provides sequence container-like access to a CudaDimProto, which holds at
/// least one (x) and at most three (x,y,z) values.
class CudaDimView {
 private:
  CudaDimView() = default;

 public:
  /// Construct a view that refers to a protocol buffers message.
  CudaDimView(const CudaDimView&) = default;
  explicit CudaDimView(CudaDimProto& buf) : proto(buf) {}

  /// Number of values held.
  inline size_t size() const;

  /// Return a copy of values as std::vector.
  inline std::vector<uint64_t> extractVector() const;

  /// Return a copy of values as std::array of size 3 padded with defaultDim.
  inline std::array<uint64_t, 3> extractDefaultedArray() const;

  /// Return a modifiable object which replicates assignments back to the
  /// underlying protocol buffers message.
  inline ValueAccessor<uint64_t> operator[](size_t i);

  /// Access the values positionally (x=0, y=1, z=2).
  inline uint64_t operator[](size_t i) const;

  /// Assign the values from another view.
  inline CudaDimView& operator=(const CudaDimView& view);

  /// Compare the values with those from another view.
  inline bool operator==(const CudaDimView& view) const;
  inline bool operator!=(const CudaDimView& view) const;

  /// Conversion to string and output operators.
  std::string toCommaSeparatedString() const;
  friend std::ostream& operator<<(std::ostream& os, const CudaDimView& view);

 public:
  CudaDimProto& proto;

  static const uint64_t defaultDim = 1;
};

/// "Materialized" CudaDimView.
///
/// When constructed from values, ignores trailing defaultDim, e.g.,
///
///   CudaDim(42, defaultDim);
///
/// will only set x, but
///
///   CudaDim(42, defaultDim, 32);
///
/// will x, y and z.
class CudaDim : public CudaDimView {
 public:
  CudaDim() : CudaDimView(ownedProto_) {}
  CudaDim(const CudaDim& cudaDim)
      : CudaDimView(ownedProto_), ownedProto_(cudaDim.proto) {}
  CudaDim(const CudaDimProto& proto)
      : CudaDimView(ownedProto_), ownedProto_(proto) {}
  CudaDim(const CudaDimView& view)
      : CudaDimView(ownedProto_), ownedProto_(view.proto) {}
  inline CudaDim(std::initializer_list<uint64_t> il);
  inline CudaDim(std::vector<uint64_t> il);
  inline CudaDim(
      uint64_t x,
      uint64_t y = CudaDimView::defaultDim,
      uint64_t z = CudaDimView::defaultDim);

  using CudaDimView::operator=;

 private:
  CudaDimProto ownedProto_;
};

/// Specializing CudaDim to differentiate between Block and Grid sizes.
class Block : public CudaDim {
 public:
  Block() = default;
  Block(const CudaDimView& view) : CudaDim(view.proto) {}
  Block(const CudaDimProto& proto) : CudaDim(proto) {}
  Block(std::initializer_list<uint64_t> il) : CudaDim(il) {}
  Block(std::vector<uint64_t> il) : CudaDim(il) {}

  using CudaDimView::operator=;
};

/// Specializing CudaDim to differentiate between Block and Grid sizes.
class Grid : public CudaDim {
 public:
  Grid() = default;
  Grid(const CudaDimView& view) : CudaDim(view.proto) {}
  Grid(const CudaDimProto& proto) : CudaDim(proto) {}
  Grid(std::initializer_list<uint64_t> il) : CudaDim(il) {}
  Grid(std::vector<uint64_t> il) : CudaDim(il) {}

  using CudaDimView::operator=;
};

class CudaMappingOptions {
 private:
  inline CudaMappingOptions();
  static CudaMappingOptions makeUnmappedCudaMappingOptions();

 public:
  /// Construct a deep copy of the options.
  inline CudaMappingOptions(const CudaMappingOptions& options);
  inline explicit CudaMappingOptions(const CudaMappingOptionsProto& buf);

  /// Compare with another message.
  inline bool operator==(const CudaMappingOptions& options) const;
  inline bool operator!=(const CudaMappingOptions& options) const;

  /// Construct from a serialized protocol buffer message.
  inline explicit CudaMappingOptions(const std::string& str);

  std::string toProtobufSerializedString() const {
    return proto.SerializeAsString();
  }

  /// The following only call the operation on CudaMappingOptions and return
  /// reference to self for the purpose of chaining
  /**
   * @name Chainable Modifiers
   * See protobuf for documentation on each option.
   * @{
   */
  inline CudaMappingOptions& tile(const std::vector<uint64_t>& sizes);
  inline CudaMappingOptions& tile(std::initializer_list<uint64_t> sizes);
  inline CudaMappingOptions& tile(const std::string& commaSeparatedSizes);
  inline CudaMappingOptions& tile(const char* commaSeparatedSizes);
  template <typename... Args>
  CudaMappingOptions& tile(Args...);

  inline CudaMappingOptions& unroll(uint64_t size);
  inline CudaMappingOptions& fixParametersBeforeScheduling(bool b);
  inline CudaMappingOptions& tileImperfectlyNested(bool b);
  inline CudaMappingOptions& matchLibraryCalls(bool b);
  ///@}

  /// Set single fusion strategy.
  ///@{
  inline CudaMappingOptions& scheduleFusionStrategy(FusionStrategy fs);
  inline CudaMappingOptions& scheduleFusionStrategy(const std::string& str);
  ///@}

  /// Set fusion strategy for outer scheduling.
  ///@{
  inline CudaMappingOptions& outerScheduleFusionStrategy(FusionStrategy fs);
  inline CudaMappingOptions& outerScheduleFusionStrategy(
      const std::string& str);
  inline CudaMappingOptions& outerScheduleAllowSkewing(bool b);
  inline CudaMappingOptions& outerSchedulePositiveOrthant(bool b);
  ///@}

  /// Set fusion strategy for intra-tile scheduling.
  ///@{
  inline CudaMappingOptions& intraTileScheduleFusionStrategy(FusionStrategy fs);
  inline CudaMappingOptions& intraTileScheduleFusionStrategy(
      const std::string& str);
  inline CudaMappingOptions& intraTileScheduleAllowSkewing(bool b);
  inline CudaMappingOptions& intraTileSchedulePositiveOrthant(bool b);
  ///@}

  /**
   * @name Chainable Modifiers specific to CudaMappingOptions
   * See protobuf for documentation on each option.
   * @{
   */
  /// Set mappings
  ///@{
  inline CudaMappingOptions& mapToThreads(
      std::initializer_list<uint64_t> threads);
  inline CudaMappingOptions& mapToThreads(
      uint64_t x,
      uint64_t y = CudaDimView::defaultDim,
      uint64_t z = CudaDimView::defaultDim);
  inline CudaMappingOptions& mapToThreads(const std::vector<uint64_t>& threads);
  CudaMappingOptions& mapToThreads(const std::string& commaSeparatedSizes);

  inline CudaMappingOptions& mapToBlocks(
      std::initializer_list<uint64_t> blocks);
  inline CudaMappingOptions& mapToBlocks(
      uint64_t x,
      uint64_t y = CudaDimView::defaultDim,
      uint64_t z = CudaDimView::defaultDim);
  inline CudaMappingOptions& mapToBlocks(const std::vector<uint64_t>& blocks);
  CudaMappingOptions& mapToBlocks(const std::string& commaSeparatedSizes);
  ///@}

  /// Set mappings
  inline CudaMappingOptions& genericMappingOptions(
      const MappingOptions& options);
  inline CudaMappingOptions& useSharedMemory(bool b);
  inline CudaMappingOptions& usePrivateMemory(bool b);
  inline CudaMappingOptions& maxSharedMemory(uint64_t size);
  inline CudaMappingOptions& unrollCopyShared(bool b);
  ///@}

  /// Static constructors for predefined strategies.
  ///@{
  static CudaMappingOptions makeNaiveCudaMappingOptions();
  static CudaMappingOptions makeSingleThreadCudaMappingOptions();
  static CudaMappingOptions makePointwiseCudaMappingOptions();
  static CudaMappingOptions makeMlpCudaMappingOptions();
  static CudaMappingOptions makeConvolutionCudaMappingOptions();
  static CudaMappingOptions makeGroupConvolutionCudaMappingOptions();
  ///@}

  CudaMappingOptionsProto proto;

  MappingOptionsView genericMappingOptionsView;
  CudaDimView block;
  CudaDimView grid;
};

std::ostream& operator<<(std::ostream& os, const CudaMappingOptions& view);

} // namespace tc

#include "tc/core/cuda/cuda_mapping_options-inl.h"
