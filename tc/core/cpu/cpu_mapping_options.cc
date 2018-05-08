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
#include "tc/core/cpu/cpu_mapping_options.h"

#include <fstream>
#include <iomanip>
#include <iostream>
#include <tuple>
#include <type_traits>

#include "tc/proto/mapping_options.pb.h"

#include "tc/core/cpu/cpu_mapping_options_cpp_printer.h"
#include "tc/core/flags.h"
#include "tc/core/utils/string.h"
#include "tc/external/isl.h"

namespace tc {

CpuMappingOptions::CpuMappingOptions()
    : ownedProto_(), generic(*ownedProto_.mutable_generic_mapping_options()) {}

/// Construct from a serialized protocol buffer message.
CpuMappingOptions::CpuMappingOptions(const std::string& str)
    : CpuMappingOptions() {
  bool parsed = ownedProto_.ParseFromString(str);
  CHECK(parsed) << "could not parse protobuf string";
}

CpuMappingOptions::CpuMappingOptions(const CpuMappingOptions& options)
    : ownedProto_(options.ownedProto_),
      generic(*ownedProto_.mutable_generic_mapping_options()) {}

CpuMappingOptions::CpuMappingOptions(const CpuMappingOptionsProto& buf)
    : ownedProto_(buf),
      generic(*ownedProto_.mutable_generic_mapping_options()) {}

CpuMappingOptions& CpuMappingOptions::operator=(
    const CpuMappingOptions& options) {
  ownedProto_ = options.ownedProto_; // views already point to the proper place
  return *this;
}

bool CpuMappingOptions::operator==(const CpuMappingOptions& options) const {
  return generic == options.generic;
}

std::string CpuMappingOptions::toProtobufSerializedString() const {
  return ownedProto_.SerializeAsString();
}

CpuMappingOptions& CpuMappingOptions::genericMappingOptions(
    const MappingOptions& options) {
  *(ownedProto_.mutable_generic_mapping_options()) = options.view.proto;
  return *this;
}

CpuMappingOptions CpuMappingOptions::makeUnmappedMappingOptions() {
  CpuMappingOptions mo;
  mo.genericMappingOptions(MappingOptions::makeUnmappedMappingOptions());
  return mo;
}

CpuMappingOptions CpuMappingOptions::makeNaiveMappingOptions() {
  return makeUnmappedMappingOptions().tile(32, 32, 32).unroll(1);
}

} // namespace tc
