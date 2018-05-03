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

#include <iostream>
#include <string>

#include "tc/core/cuda/cuda_mapping_options.h"
#include "tc/core/mapping_options_cpp_printer.h"

namespace tc {

class CudaMappingOptionsAsCpp {
 public:
  explicit CudaMappingOptionsAsCpp(
      const CudaMappingOptions& options_,
      size_t indent_ = 0)
      : options(options_), indent(indent_) {}
  const CudaMappingOptions& options;
  size_t indent;
};

class CudaMappingOptionsCppPrinter : public MappingOptionsCppPrinter {
 public:
  CudaMappingOptionsCppPrinter(std::ostream& out, size_t ws = 0)
      : MappingOptionsCppPrinter(out, ws) {}

  ~CudaMappingOptionsCppPrinter() = default;

  friend CudaMappingOptionsCppPrinter& operator<<(
      CudaMappingOptionsCppPrinter& prn,
      const CudaMappingOptions& options);
};

CudaMappingOptionsCppPrinter& operator<<(
    CudaMappingOptionsCppPrinter& prn,
    const CudaMappingOptions& cudaOptions);

std::ostream& operator<<(std::ostream& out, const CudaMappingOptionsAsCpp& mo);

} // namespace tc
