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

#include "tc/lang/tree.h"

#include <ostream>

namespace lang {

/// \file tc_format.h
/// A pretty printer that turns a Def (TC AST) into a valid TC string that
/// could be e.g. re-parsed.

void tcFormat(std::ostream& s, TreeRef def);

} // namespace lang
