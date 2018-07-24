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
#include "tc/core/polyhedral/cpu/mapped_scop.h"

#include <algorithm>
#include <array>
#include <iostream>
#include <numeric>
#include <sstream>
#include <stdexcept>
#include <unordered_set>

#include "tc/core/check.h"
#include "tc/core/flags.h"
#include "tc/core/functional.h"
#include "tc/core/polyhedral/codegen_llvm.h"
#include "tc/core/polyhedral/exceptions.h"
#include "tc/core/polyhedral/llvm_jit.h"
#include "tc/core/polyhedral/schedule_transforms.h"
#include "tc/core/polyhedral/schedule_utils.h"
#include "tc/core/polyhedral/scop.h"

#include <glog/logging.h>

namespace tc {
namespace polyhedral {

std::unique_ptr<Jit> MappedScop::codegen(
    const std::string& specializedName) const {
  std::unique_ptr<Jit> jit(new Jit());
  jit->codegenScop(specializedName, *scop_);
  return jit;
}

std::unique_ptr<MappedScop> MappedScop::makeSequential(
    std::unique_ptr<Scop>&& scopUPtr,
    const CpuMappingOptions& cpuOptions) {
  using namespace polyhedral::detail;

  const auto& generic = cpuOptions.generic;
  auto mappedScop = std::unique_ptr<MappedScop>(
      new MappedScop(std::move(scopUPtr), generic.proto.unroll()));
  auto& scop = mappedScop->scop_;

  // 1a. Optionally specialize before scheduling...
  if (generic.proto.fix_parameters_before_scheduling()) {
    scop->specializeToContext();
  }

  // 2. Schedule
  scop = Scop::makeScheduled(*scop, generic.outerScheduleOptions);

  // 3. Tile
  TC_CHECK_LT(0u, generic.tiling.size())
      << "Must pass tile vector with >= 1 tile sizes";
  auto outerBand = scop->tileOuterBand(generic.tiling);

  // 4. Optionally reschedule if point loops need a different strategy than
  // tile loops
  if (generic.outerScheduleOptions != generic.intraTileScheduleOptions) {
    scop->reschedule(outerBand->child({0}), generic.intraTileScheduleOptions);
    LOG_IF(INFO, FLAGS_debug_tc_mapper)
        << "After intra-tile rescheduling:" << std::endl
        << *mappedScop->schedule();
  }

  // 1b. ...or after rescheduling
  if (!generic.proto.fix_parameters_before_scheduling()) {
    scop->specializeToContext();
  }

  LOG_IF(INFO, FLAGS_debug_tc_mapper)
      << "After sequential strategy:" << std::endl
      << *mappedScop->schedule();

  return mappedScop;
}

} // namespace polyhedral
} // namespace tc
