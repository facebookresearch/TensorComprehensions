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
#include "tc/core/polyhedral/mapping_types.h"
#include "tc/core/polyhedral/cuda/mapping_types.h"

namespace tc {
namespace polyhedral {
namespace mapping {
bool MappingId::isBlockId() {
  return *this == BlockId::x() or *this == BlockId::y() or
      *this == BlockId::z();
}
BlockId* MappingId::asBlockId() {
  if (!isBlockId()) {
    return nullptr;
  }
  return static_cast<BlockId*>(this);
}
bool MappingId::isThreadId() {
  return *this == ThreadId::x() or *this == ThreadId::y() or
      *this == ThreadId::z();
}
ThreadId* MappingId::asThreadId() {
  if (!isThreadId()) {
    return nullptr;
  }
  return static_cast<ThreadId*>(this);
}
} // namespace mapping
} // namespace polyhedral
} // namespace tc
