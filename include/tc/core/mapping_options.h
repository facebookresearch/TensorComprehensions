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

/// \file mapping_options.h
/// A set of classes that act as the in-memory interface to control polyhedral
/// scheduling and mapping.  Storage is provided by protocol buffers.
///
/// The interface is based on a concept of "view" classes that provide a
/// different interface to (parts of) underlying protocol buffers.  Each view
/// has a mutable reference to a protocol buffer message, which may be a part of
/// a larger message.  It also provides a set of convenience functions to
/// inspect and modify the underlying message.  All modifications are
/// immediately stored in the protocol buffers, up to the top-level message.
/// Because views only represent a (part of a) message, which they do not own,
/// they cannot be constructed from values.  However, they can be
/// copy-constructed, with a copy referring to the same underlying object, and
/// assigned, with all fields of the underlying object assigned.  Views can be
/// constructed given a protocol buffer message, to which they hold a reference.
/// The caller is responsible for ensuring the actual message lives at least as
/// long as the view.
///
/// All views come with a "materialized" counterpart that owns the underlying
/// message.  They can be constructed from a set of values, from another view or
/// from a protocol buffer.  In the two latter cases, they make a deep copy of
/// the message.  "Materialized" view classes derive from views making it
/// possible to assign a view referring to a part of top-level message from a
/// "materialized" temporary.  For example,
///
///     CudaMappingOptions mo;
///     // Copy of a view refers to the same object.
///     CudaDimView view = mo.block;
///     // Ultimately assigns mo.proto.mutable_block.
///     view = CudaDim(42, 100, 2);
///
/// is equivalent to
///
///     CudaMappingOptions mo;
///     mo.proto.mutable_block()->set_x(42);
///     mo.proto.mutable_block()->set_y(100);
///     mo.proto.mutable_block()->set_z(2);
///
/// References to underlying protocol buffers message objects are exposed
/// publicly.  They can be changed directly, and changes are immediately visible
/// through all views referring to (a part of) the message.  For example,
///
///     CudaMappingOptions mo;
///     mo.proto.mutable_block()->set_x(42);
///     cout << mo.block[0];    // outputs 42;
///
/// "Materialized" views do not expose the message they own, only a modifiable
/// reference through the view interface.
///
/// The top-level interface (CudaMappingOptions) owns and publicly exposes the
/// top-level protocol buffer message along with views to its sub-messages.

namespace tc {

/// Simple template class to wrap getters and by-value setters.  Instances of
/// this class can be implicitly converted to the template parameter type by
/// calling the provided getter function.  They can be assigned from an instance
/// of the template parameter type by calling the setter function provided in
/// the constructor.
///
/// Note that this class does not in any sense extend the lifetime of the
/// accessed object.  Make sure that getters and setters actually change the
/// object, e.g., capture by-reference in lambdas.
template <typename T>
class ValueAccessor {
 public:
  using Setter = std::function<void(T)>;
  using Getter = std::function<T()>;

  ValueAccessor(const Setter& s, const Getter& g) : setter_(s), getter_(g) {}
  ValueAccessor(const ValueAccessor&) = default;

  operator T() const {
    return getter_();
  }

  ValueAccessor& operator=(const T& t) {
    setter_(t);
    return *this;
  }

 private:
  Setter setter_;
  Getter getter_;
};

/// View of a TilingProto.
///
/// Provides sequence container-like access to TilingProto.
class TilingView {
 private:
  TilingView() = default;

 public:
  /// Construct a view that refers to a protocol buffers message.
  TilingView(const TilingView&) = default;
  explicit TilingView(TilingProto& p) : proto(p) {}

  /// Return a copy of values as std::vector.
  inline std::vector<uint64_t> extractVector() const;

  /// Number of values held.
  inline size_t size() const;

  /// Return a modifiable object which replicates assignments back to the
  /// underlying protocol buffers message.
  inline ValueAccessor<uint64_t> operator[](size_t i);

  /// Access the values positionally (x=0, y=1, z=2).
  inline uint64_t operator[](size_t i) const;

  /// Assign the values from another view.
  inline TilingView& operator=(const TilingView& view);

  /// Compare the values with those from another view.
  inline bool operator==(const TilingView& view) const;
  inline bool operator!=(const TilingView& view) const;

  /// Conversion to string and output operators.
  std::string toCommaSeparatedString() const;
  friend std::ostream& operator<<(std::ostream& os, const TilingView& view);

 public:
  TilingProto& proto;
};

/// "Materialized" TilingView.
class Tiling {
 public:
  Tiling() : ownedProto_(), view(ownedProto_) {}
  Tiling(const Tiling& t) : ownedProto_(t.ownedProto_), view(ownedProto_) {}
  Tiling(const TilingProto& proto) : ownedProto_(proto), view(ownedProto_) {}
  Tiling(const TilingView& view) : ownedProto_(view.proto), view(ownedProto_) {}
  inline Tiling(std::initializer_list<uint64_t> il);
  inline Tiling(const std::vector<uint64_t>& sizes);

 private:
  TilingProto ownedProto_;

 public:
  TilingView view;
};

//// View of a SchedulerOptionsProto.
///
/// Provides isl callbacks based on the options.
class SchedulerOptionsView {
 public:
  /// isl scheduler callback types.
  using MergeCallback = std::function<
      isl_bool(isl_union_map*, isl_union_map*, int, int, int, void*)>;
  using ConstraintsCallback = std::function<isl_basic_set*(
      isl_basic_set*,
      int,
      int,
      isl_id_list*,
      int*,
      int*,
      void*)>;

 private:
  SchedulerOptionsView() = default;

 public:
  /// Construct a view that refers to a protocol buffers message.
  SchedulerOptionsView(const SchedulerOptionsView&) = default;
  SchedulerOptionsView(SchedulerOptionsProto& buf) : proto(buf) {}

  /// Assign the values from another view.
  inline SchedulerOptionsView& operator=(const SchedulerOptionsView&);

  /// Compare the values with those from another view.
  inline bool operator==(const SchedulerOptionsView& view) const;
  inline bool operator!=(const SchedulerOptionsView& view) const;

  /// Output operators.
  friend std::ostream& operator<<(
      std::ostream& os,
      const SchedulerOptionsView& options);

 public:
  SchedulerOptionsProto& proto;
};

/// "Materialized" SchedulerOptionsView.
class SchedulerOptions {
 public:
  SchedulerOptions() : ownedProto_(), view(ownedProto_) {}
  SchedulerOptions(const SchedulerOptions& options)
      : ownedProto_(options.ownedProto_), view(ownedProto_) {}
  explicit SchedulerOptions(const SchedulerOptionsProto& proto)
      : ownedProto_(proto), view(ownedProto_) {}
  explicit SchedulerOptions(const SchedulerOptionsView& view)
      : ownedProto_(view.proto), view(ownedProto_) {}

 private:
  SchedulerOptionsProto ownedProto_;

 public:
  SchedulerOptionsView view;
};

/// Top-level interface to MappingOptionsProto.
///
/// Contains views of the sub-messages (scheduler options, tiling, grid and
/// block sizes).  Provides static constructors for common operator options.
/// Provides fluent (chainable) API for progressively modifying the options.
class MappingOptionsView {
 private:
  MappingOptionsView() = delete;

 public:
  /// Construct a deep copy of the options.
  inline MappingOptionsView(const MappingOptionsView& options);
  inline explicit MappingOptionsView(MappingOptionsProto& buf);

  /// Assign from another view.
  inline MappingOptionsView& operator=(const MappingOptionsView&);

  /// Compare with another view.
  inline bool operator==(const MappingOptionsView& options) const;
  inline bool operator!=(const MappingOptionsView& options) const;

  /**
   * @name Chainable Modifiers
   * See protobuf for documentation on each option.
   * @{
   */
  inline MappingOptionsView& tile(const std::vector<uint64_t>& sizes);
  inline MappingOptionsView& tile(std::initializer_list<uint64_t> sizes);
  MappingOptionsView& tile(const std::string& commaSeparatedSizes);
  inline MappingOptionsView& tile(const char* commaSeparatedSizes);
  template <typename... Args>
  MappingOptionsView& tile(Args...);

  inline MappingOptionsView& unroll(uint64_t size);
  inline MappingOptionsView& fixParametersBeforeScheduling(bool b);
  inline MappingOptionsView& tileImperfectlyNested(bool b);
  inline MappingOptionsView& matchLibraryCalls(bool b);
  ///@}

  /// Set single fusion strategy.
  ///@{
  inline MappingOptionsView& scheduleFusionStrategy(FusionStrategy fs);
  inline MappingOptionsView& scheduleFusionStrategy(const std::string& str);
  ///@}

  /// Set fusion strategy for outer scheduling.
  ///@{
  inline MappingOptionsView& outerScheduleFusionStrategy(FusionStrategy fs);
  inline MappingOptionsView& outerScheduleFusionStrategy(
      const std::string& str);
  inline MappingOptionsView& outerScheduleAllowSkewing(bool b);
  inline MappingOptionsView& outerSchedulePositiveOrthant(bool b);
  ///@}

  /// Set fusion strategy for intra-tile scheduling.
  ///@{
  inline MappingOptionsView& intraTileScheduleFusionStrategy(FusionStrategy fs);
  inline MappingOptionsView& intraTileScheduleFusionStrategy(
      const std::string& str);
  inline MappingOptionsView& intraTileScheduleAllowSkewing(bool b);
  inline MappingOptionsView& intraTileSchedulePositiveOrthant(bool b);
  ///@}

  /// Output operator.
  friend std::ostream& operator<<(
      std::ostream& os,
      const MappingOptionsView& options);

 public:
  MappingOptionsProto& proto;

  // Views of sub-messages.
  TilingView tiling;
  SchedulerOptionsView outerScheduleOptions;
  SchedulerOptionsView intraTileScheduleOptions;
};

/// "Materialized" MappingOptionsView.
class MappingOptions {
 public:
  MappingOptions() : ownedProto_(), view(ownedProto_) {}
  MappingOptions(const MappingOptions& options)
      : ownedProto_(options.ownedProto_), view(ownedProto_) {}
  /// Performs an underlying copy of the proto ```proto```
  explicit MappingOptions(const MappingOptionsProto& proto)
      : ownedProto_(proto), view(ownedProto_) {}
  /// Performs an underlying copy of the proto viewed by ```view```
  /* implicit */ MappingOptions(const MappingOptionsView& view)
      : ownedProto_(view.proto), view(ownedProto_) {}

  std::string toProtobufSerializedString() const {
    return ownedProto_.SerializeAsString();
  }

  /// Static constructors for predefined strategies.
  ///@{
  inline static MappingOptions makeUnmappedMappingOptions();
  inline static MappingOptions makeNaiveMappingOptions();
  inline static MappingOptions makeSingleThreadMappingOptions();
  inline static MappingOptions makePointwiseMappingOptions();
  inline static MappingOptions makeMlpMappingOptions();
  inline static MappingOptions makeConvolutionMappingOptions();
  inline static MappingOptions makeGroupConvolutionMappingOptions();
  ///@}

  friend std::ostream& operator<<(
      std::ostream& os,
      const MappingOptions& options);

 private:
  MappingOptionsProto ownedProto_;

 public:
  MappingOptionsView view;
};

namespace callbacks {
__isl_give isl_basic_set* AddPositiveCoefficientConstraints(
    __isl_take isl_basic_set* lp,
    int n_param,
    int dim,
    __isl_keep isl_id_list* stmt_ids,
    int* node_n_params,
    int* node_n_dims,
    void*);

isl_bool FuseAllPreserve3Coincident(
    __isl_take isl_union_map* original_schedule,
    __isl_take isl_union_map* updated_schedule,
    int n_updated_coincident,
    int n_original_coincident,
    int is_along_edge,
    void*);

isl_bool FuseAll(
    __isl_take isl_union_map* original_schedule,
    __isl_take isl_union_map* updated_schedule,
    int n_updated_coincident,
    int n_original_coincident,
    int is_along_edge,
    void*);

isl_bool FuseNone(
    __isl_take isl_union_map* original_schedule,
    __isl_take isl_union_map* updated_schedule,
    int n_updated_coincident,
    int n_original_coincident,
    int is_along_edge,
    void*);
} // namespace callbacks

} // namespace tc

#include "tc/core/mapping_options-inl.h"
