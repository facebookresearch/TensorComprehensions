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
 public:
  CpuMappingOptions();

  /// Construct from a serialized protocol buffer message.
  inline explicit CpuMappingOptions(const std::string& str);

  inline bool operator==(const CpuMappingOptions& options);

  inline std::string toProtobufSerializedString() const;

 private:
  CpuMappingOptionsProto ownedProto_;

 public:
  MappingOptionsView generic;
};
} // namespace tc

#include "tc/core/cpu/cpu_mapping_options-inl.h"
