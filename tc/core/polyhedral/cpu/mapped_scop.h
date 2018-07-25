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

#include <map>
#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

#include "tc/core/cpu/cpu_mapping_options.h"
#include "tc/core/polyhedral/llvm_jit.h"
#include "tc/core/polyhedral/scop.h"
#include "tc/core/tensor.h"
#include "tc/external/isl.h"

namespace tc {
namespace polyhedral {
namespace cpu {

class MappedScop {
 private:
  MappedScop(std::unique_ptr<Scop>&& scop, uint64_t unroll_)
      : scop_(std::move(scop)), unroll(unroll_) {}

 public:
  static std::unique_ptr<MappedScop> makeSequential(
      std::unique_ptr<Scop>&& scopUPtr,
      const CpuMappingOptions& mappingOptions);

  // Fix the values of the specified parameters in the context
  // to the corresponding specified values.
  template <typename T>
  void fixParameters(const std::unordered_map<std::string, T>& sizes) {
    scop_->fixParameters(sizes);
  }

  // Generate code at the current state of transformation provided a
  // name for the generated function.
  std::unique_ptr<Jit> codegen(const std::string& specializedName) const;

  // Accessors..
  // Const accessor to schedule of underlying Scop.
  inline const detail::ScheduleTree* schedule() const {
    return scop_->scheduleRoot();
  }
  // Reference to underlying scop, no ownership transfer intended.
  inline const Scop& scop() const {
    return *scop_;
  }
  inline Scop& scop() {
    return *scop_;
  }

 private:
  std::unique_ptr<Scop> scop_;

 public:
  const uint64_t unroll;
};

} // namespace cpu
} // namespace polyhedral
} // namespace tc
