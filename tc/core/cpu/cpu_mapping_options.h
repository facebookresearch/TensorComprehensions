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

#include "tc/proto/mapping_options.pb.h"

#include "tc/core/mapping_options.h"

namespace tc {

class CpuMappingOptions {
 private:
  CpuMappingOptions();
  static CpuMappingOptions makeUnmappedMappingOptions();

 public:
  /// Construct a deep copy of the options.
  CpuMappingOptions(const CpuMappingOptions& options);
  explicit CpuMappingOptions(const CpuMappingOptionsProto& buf);
  CpuMappingOptions& operator=(const CpuMappingOptions& options);

  /// Compare with another message.
  bool operator==(const CpuMappingOptions& options) const;
  bool operator!=(const CpuMappingOptions& options) const;

  /// Construct from a serialized protocol buffer message.
  explicit CpuMappingOptions(const std::string& str);

  std::string toProtobufSerializedString() const;

  /// Set mappings
  CpuMappingOptions& genericMappingOptions(const MappingOptions& options);

  /// Static constructors for predefined strategies.
  static CpuMappingOptions makeNaiveMappingOptions();

  const CpuMappingOptionsProto& proto() const {
    return ownedProto_;
  }

#define FORWARD_FUN(FUN_NAME)                        \
  template <typename... Args>                        \
  inline CpuMappingOptions& FUN_NAME(Args... args) { \
    generic.FUN_NAME(args...);                       \
    return *this;                                    \
  }

  FORWARD_FUN(tile);
  FORWARD_FUN(unroll);
  FORWARD_FUN(fixParametersBeforeScheduling);
  FORWARD_FUN(tileImperfectlyNested);
  FORWARD_FUN(matchLibraryCalls);
  FORWARD_FUN(scheduleFusionStrategy);
  FORWARD_FUN(outerScheduleFusionStrategy);
  FORWARD_FUN(outerScheduleAllowSkewing);
  FORWARD_FUN(outerSchedulePositiveOrthant);
  FORWARD_FUN(intraTileScheduleFusionStrategy);
  FORWARD_FUN(intraTileScheduleAllowSkewing);
  FORWARD_FUN(intraTileSchedulePositiveOrthant);

#undef FORWARD_FUN

 private:
  CpuMappingOptionsProto ownedProto_;

 public:
  MappingOptionsView generic;
};
} // namespace tc
