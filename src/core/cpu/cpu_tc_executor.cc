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
#include "tc/core/cpu/cpu_tc_executor.h"

#include "tc/core/halide_utils.h"
#include "tc/core/mapping_options_cpp_printer.h"
#include "tc/core/polyhedral/mapped_scop.h"
#include "tc/core/tc2halide.h"
#include "tc/core/utils/dlpack.h"

#include "tc/lang/parser.h"
#include "tc/lang/sema.h"

#include <version.h>
#include <utility>

namespace tc {

using namespace dlutils;

void CpuTcExecutor::compile(const tc::MappingOptions& options) {}

void CpuTcExecutor::compileWithTcMapper() {}

Duration CpuTcExecutor::run(
    const std::vector<const DLTensor*>& inputs,
    const std::vector<DLTensor*>& outputs,
    bool profile) const {
  CHECK(rtcFun) << "Can't launch uncompiled: " << executionInfo_.kernelName;
  CHECK_NE(executionInfo_.options, "");
  checkSizesAndStridesAreCompliant(
      inputs, executionInfo_.inputsInfo, halideComponents_.getDef().params());
  checkSizesAndStridesAreCompliant(
      outputs,
      executionInfo_.outputsInfo,
      halideComponents_.getDef().returns());

  return Duration();
}

void CpuTcExecutor::uncheckedRun(
    const std::vector<const void*>& inputs,
    const std::vector<void*>& outputs) const {}

} // namespace tc
