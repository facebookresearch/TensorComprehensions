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

#include <cstddef>
#include <vector>

#include "tc/external/isl.h"

namespace tc {

class Block;

namespace polyhedral {
class MappedScop;
class Scop;

namespace detail {
class ScheduleTree;
}

// In the given mapped scop "mscop",
// promote to shared memory at "depth" until "sharedMemorySize" is used.
// Map copies between global and shared memory to threads and unroll those
// copies if "unrollCopies" is set, using the options in "mscop".
// "threadIdxXScheduleDepthState" contains the schedule depth at which the
// computation was mapped to thread x and is used to check whether the global
// memory is accessed in a coalesced way.
// Return the remaining memory.
size_t promoteGreedilyAtDepth(
    MappedScop& scop,
    std::vector<detail::ScheduleTree*> trees,
    std::vector<std::size_t> depths,
    std::size_t sharedMemorySize,
    bool unrollCopies);

void promoteToRegistersBelow(MappedScop& mscop, detail::ScheduleTree* scope);

void promoteToRegistersBelowThreads(MappedScop& scop, std::size_t nRegisters);
} // namespace polyhedral
} // namespace tc
