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
#include "tc/core/mapping_options.h"

#include <fstream>
#include <iomanip>
#include <iostream>
#include <tuple>
#include <type_traits>

#include "tc/proto/mapping_options.pb.h"

#include "tc/core/flags.h"
#include "tc/core/mapping_options_cpp_printer.h"
#include "tc/core/utils/string.h"
#include "tc/external/isl.h"

namespace tc {

std::string TilingView::toCommaSeparatedString() const {
  std::stringstream ss;
  for (size_t i = 0, e = size(); i < e; ++i) {
    if (i != 0) {
      ss << ", ";
    }
    ss << operator[](i);
  }
  return ss.str();
}

std::ostream& operator<<(std::ostream& os, const TilingView& view) {
  os << "Tiling(" << view.toCommaSeparatedString() << ") @" << &view.proto;
  return os;
}

std::ostream& operator<<(
    std::ostream& os,
    const SchedulerOptionsView& options) {
  OstreamBoolalphaScope scope(os);

  os << "SchedulerOptions("
     << "fusion_strategy: "
     << FusionStrategy_Name(options.proto.fusion_strategy()) << ", "
     << "allow_skewing: " << options.proto.allow_skewing() << ", "
     << "positive_orthant: " << options.proto.positive_orthant() << ") @"
     << &options.proto;
  return os;
}

MappingOptionsView& MappingOptionsView::tile(
    const std::string& commaSeparatedSizes) {
  return tile(parseCommaSeparatedIntegers<uint64_t>(commaSeparatedSizes));
}

//
// Callbacks
//

namespace callbacks {
// Fusion decisions.
// These callbacks are called from the cluster merging process inside isl
// scheduler.  Current version of this process is two-stage.  First, it
// considers pairs of clusters (initially, individual dependence graph SCCs)
// between which exists a proximity edge (dependence).  After traversing all
// edges, it considers all remaining pairs of SCCs.  The fifth parameter of the
// callback is "1" during the first stage and "0" during the second stage.
// In any case, at least one level of loop fusion is required for the callback
// to be called.  Two first arguments contain the statement-wise schedule maps
// before and after clustering (which may modify them).  The following two
// values contain the number of leading coincident (parallel) dimensions after
// and before the scheduling.
// The callback should return "isl_bool_true" to merge clusters,
// "isl_bool_false" to keep them unmerged and "isl_bool_error" to abort the
// scheduling process completely.
// Note that the default Isl heuristics are NOT applied if a callback is
// provided.

isl_bool FuseAllPreserve3Coincident(
    __isl_take isl_union_map* original_schedule,
    __isl_take isl_union_map* updated_schedule,
    int n_updated_coincident,
    int n_original_coincident,
    int,
    void*) {
  isl_union_map_free(original_schedule);
  isl_union_map_free(updated_schedule);

  if (n_updated_coincident >= n_original_coincident ||
      n_updated_coincident >= 3) {
    return isl_bool_true;
  }
  return isl_bool_false;
}

isl_bool FuseAll(
    __isl_take isl_union_map* original_schedule,
    __isl_take isl_union_map* updated_schedule,
    int,
    int,
    int,
    void*) {
  isl_union_map_free(original_schedule);
  isl_union_map_free(updated_schedule);

  return isl_bool_true;
}

isl_bool FuseNone(
    __isl_take isl_union_map* original_schedule,
    __isl_take isl_union_map* updated_schedule,
    int,
    int,
    int,
    void*) {
  isl_union_map_free(original_schedule);
  isl_union_map_free(updated_schedule);

  return isl_bool_false;
}
} // namespace callbacks

} // namespace tc
