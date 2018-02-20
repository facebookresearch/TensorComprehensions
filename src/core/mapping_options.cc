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

#include <mapping_options.pb.h>

#include "tc/core/flags.h"
#include "tc/core/rtc.h"
#include "tc/external/isl.h"

namespace tc {

const uint64_t CudaDimView::defaultDim;

//
// Output operators and string conversion
//
std::string CudaDimView::toCommaSeparatedString() const {
  std::stringstream ss;
  ss << proto.x();
  if (proto.has_y()) {
    ss << ", " << proto.y();
  }
  if (proto.has_z()) {
    ss << ", " << proto.z();
  }
  return ss.str();
}

std::ostream& operator<<(std::ostream& os, const CudaDimView& view) {
  os << "CudaDim(" << view.toCommaSeparatedString() << ") @" << &view.proto;
  return os;
}

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

namespace {
// Sets the std::boolalpha flags of the given std::ostream and resets it to the
// previous value on scope exit.
class OstreamBoolalphaScope {
 public:
  OstreamBoolalphaScope(std::ostream& os)
      : os_(os), hasBoolalpha_(os.flags() & std::ios_base::boolalpha) {
    os << std::boolalpha;
  }
  ~OstreamBoolalphaScope() {
    if (!hasBoolalpha_) {
      os_ << std::noboolalpha;
    }
  }

 private:
  std::ostream& os_;
  bool hasBoolalpha_;
};
} // namespace

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

std::ostream& operator<<(std::ostream& os, const MappingOptions& options) {
  OstreamBoolalphaScope scope(os);

  os << "MappingOptions("
     << "outer_schedule_options: " << options.outerScheduleOptions << ","
     << std::endl;
  if (options.proto.has_intra_tile_schedule_options()) {
    os << "intra_tile_schedule_options: " << options.intraTileScheduleOptions;
  } else {
    os << "#none";
  }
  os << ","
     << "fix_parameters_before_scheduling: "
     << options.proto.fix_parameters_before_scheduling() << "," << std::endl
     << "tiling: " << options.tiling << "," << std::endl
     << "block: " << options.block << "," << std::endl
     << "grid: " << options.grid << "," << std::endl
     << "unroll: "
     << (options.proto.has_unroll() ? std::to_string(options.proto.unroll())
                                    : "#none")
     << "," << std::endl
     << "tile_imperfectly_nested: " << options.proto.tile_imperfectly_nested()
     << "," << std::endl
     << "use_shared_memory: " << options.proto.use_shared_memory() << ","
     << std::endl
     << "use_private_memory: " << options.proto.use_private_memory() << ","
     << std::endl
     << "unroll_copy_shared: " << options.proto.unroll_copy_shared() << ","
     << std::endl
     << "max_shared_memory: "
     << (options.proto.has_max_shared_memory()
             ? std::to_string(options.proto.max_shared_memory())
             : "#none")
     << "," << std::endl
     << "match_library_calls: " << options.proto.match_library_calls() << ") @"
     << &options.proto;
  return os;
}

//
// String-based chainable constructors.
//
namespace {
template <typename T>
std::vector<T> parseCommaSeparatedIntegers(const std::string& sizes) {
  std::stringstream ss(sizes);
  T size;
  std::vector<T> res;
  while (ss >> size) {
    res.push_back(size);
    if (ss.peek() == ',') {
      ss.ignore();
    }
  }
  return res;
}
} // namespace

MappingOptions& MappingOptions::tile(const std::string& commaSeparatedSizes) {
  return tile(parseCommaSeparatedIntegers<uint64_t>(commaSeparatedSizes));
}

MappingOptions& MappingOptions::mapToThreads(
    const std::string& commaSeparatedSizes) {
  auto sizes = parseCommaSeparatedIntegers<uint64_t>(commaSeparatedSizes);
  CHECK_GT(sizes.size(), 0)
      << "expected at least one block size in " << commaSeparatedSizes;
  CHECK_LE(sizes.size(), 3)
      << "expected at most three block sizes in " << commaSeparatedSizes;
  sizes.resize(3, CudaDimView::defaultDim);
  return mapToThreads(sizes[0], sizes[1], sizes[2]);
}

MappingOptions& MappingOptions::mapToBlocks(
    const std::string& commaSeparatedSizes) {
  auto sizes = parseCommaSeparatedIntegers<uint64_t>(commaSeparatedSizes);
  CHECK_GT(sizes.size(), 0)
      << "expected at least one grid size in " << commaSeparatedSizes;
  CHECK_LE(sizes.size(), 3)
      << "expected at most three grid sizes in " << commaSeparatedSizes;
  sizes.resize(3, CudaDimView::defaultDim);
  return mapToBlocks(sizes[0], sizes[1], sizes[2]);
}

//
// Predefined stratgies
//

MappingOptions MappingOptions::makeUnmappedMappingOptions() {
  MappingOptions mo;
  mo.useSharedMemory(false)
      .usePrivateMemory(false)
      .unrollCopyShared(false)
      .outerScheduleFusionStrategy(FusionStrategy::Preserve3Coincident)
      .outerScheduleAllowSkewing(false)
      .outerSchedulePositiveOrthant(true)
      .intraTileScheduleFusionStrategy(FusionStrategy::Preserve3Coincident)
      .intraTileScheduleAllowSkewing(false)
      .intraTileSchedulePositiveOrthant(true)
      .fixParametersBeforeScheduling(false)
      .matchLibraryCalls(false)
      .tileImperfectlyNested(false);
  return mo;
}

MappingOptions MappingOptions::makeNaiveMappingOptions() {
  auto mo = makeUnmappedMappingOptions();
  mo =
      mo.tile({32, 32, 32}).mapToThreads(32, 8).mapToBlocks(256, 256).unroll(1);
  return mo;
}

MappingOptions MappingOptions::makeSingleThreadMappingOptions() {
  return makeUnmappedMappingOptions()
      .tile({1})
      .mapToThreads(1)
      .mapToBlocks(1)
      .unroll(1);
}

MappingOptions MappingOptions::makePointwiseMappingOptions() {
  return makeUnmappedMappingOptions()
      .tile({32, 32, 32})
      .mapToThreads(32, 4, 4)
      .mapToBlocks(100, 100, 100)
      .unroll(128);
}

MappingOptions MappingOptions::makeMlpMappingOptions() {
  return makeUnmappedMappingOptions()
      .outerScheduleFusionStrategy(FusionStrategy::Max)
      .tile({1})
      .mapToThreads(128)
      .mapToBlocks(128)
      .unroll(1);
}

MappingOptions MappingOptions::makeConvolutionMappingOptions() {
  return makeUnmappedMappingOptions()
      .tile({4, 8, 8, 8})
      .mapToThreads(4, 16, 4)
      .mapToBlocks(256, 256, 256)
      .unroll(1);
}

MappingOptions MappingOptions::makeGroupConvolutionMappingOptions() {
  return makeUnmappedMappingOptions()
      .tile({1, 1})
      .mapToThreads(4, 16, 4)
      .mapToBlocks(256, 256)
      .unroll(1);
}

//
// Callbacks
//

namespace callbacks {
// The ILP in basic set has the following dimensions:
// - sum of positive and negative parts of the dependence distance bound
// - constant term of the dependence distance bound
// - sum of parameter coefficients
// - sum of positive and negative parts of schedule coefficients
// - for each statement (in the order of id_list)
//   - pairs of values representing schedule coefficients, in the order
//     opposite to their order in the respective domain
//   - parameter coefficients
//   - constant term of the schedule
// Schedule coefficients are represented as a pair of non-negative
// dimensions (c_1 = c_1^+ - c_1^-), where the negative part comes first.
// XXX: FRAGILE! the order depends on the internal operation of the isl
// scheduler, which may change.  This description is based on git-708721f.

__isl_give isl_basic_set* AddPositiveCoefficientConstraints(
    __isl_take isl_basic_set* lp,
    int n_param,
    int,
    __isl_keep isl_id_list* stmt_ids,
    int* node_n_params,
    int* node_n_dims,
    void*) {
  int offset = 4 + 2 * n_param;
  int n_node = isl_id_list_n_id(stmt_ids);
  auto ilp = isl::manage(lp);
  auto space = ilp.get_local_space();
  auto c = isl::constraint::alloc_equality(space);
  for (int i = 0; i < n_node; ++i) {
    for (int j = 0; j < 2 * node_n_dims[i]; j += 2) {
      c = c.set_coefficient_si(isl::dim_type::set, offset + j, -1);
    }
    offset += 2 * node_n_dims[i] + node_n_params[i] + 1;
  }
  c = c.set_constant_si(0);
  ilp = ilp.add_constraint(c);

  return ilp.release();
}

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
