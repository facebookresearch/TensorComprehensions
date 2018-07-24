/**
 * Copyright (c) 2018-present, Facebook, Inc.
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

namespace tc {

/// Container class for TC compiler options.
/// Unlike MappingOptions, these do not affect the behavior of the polyhedral
/// mapper but the general flow of the TC compilation, for example whether
/// syntax warnings should be generated when parsing TC definitions.
///
/// This class intends to replace the uses of flags (aka global variables)
/// scattered around the codebae.
class CompilerOptions {
 public:
  /// Explicitly-default constructor.  All member variables must have a default
  /// assigned value.
  CompilerOptions() = default;

  /// Print syntactic warnings.
  bool emitWarnings = true;
};

} // namespace tc
