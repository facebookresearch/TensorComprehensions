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

#include "tc/core/cpu/cpu_mapping_options.h"

namespace tc {

CpuMappingOptions::CpuMappingOptions()
    : ownedProto_(), generic(*ownedProto_.mutable_generic_mapping_options()) {}

/// Construct from a serialized protocol buffer message.
CpuMappingOptions::CpuMappingOptions(const std::string& str)
    : CpuMappingOptions() {
  bool parsed = ownedProto_.ParseFromString(str);
  CHECK(parsed) << "could not parse protobuf string";
}

bool CpuMappingOptions::operator==(const CpuMappingOptions& options) {
  return generic == options.generic;
}

std::string CpuMappingOptions::toProtobufSerializedString() const {
  return ownedProto_.SerializeAsString();
}

} // namespace tc
