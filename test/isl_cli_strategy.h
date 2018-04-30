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

#include <gflags/gflags.h>
#include <vector>

#include "tc/core/cuda/cuda_mapping_options.h"

#define DEFAULT_FUSION_STRATEGY "Preserve3Coincident"
#define DEFAULT_ALLOW_SKEWING false
#define DEFAULT_POSITIVE_ORTHANT true
#define DEFAULT_FIX_PARAMETERS_BEFORE_SCHEDULING false
#define DEFAULT_TILE "1"
#define DEFAULT_TILE_IMPERFECTLY_NESTED false
#define DEFAULT_BLOCK "1"
#define DEFAULT_GRID "1"
#define DEFAULT_UNROLL_FACTOR 1
#define DEFAULT_USE_SHARED_MEMORY true
#define DEFAULT_USE_PRIVATE_MEMORY true
#define DEFAULT_UNROLL_COPY_SHARED false

DEFINE_string(
    fusion_strategy,
    DEFAULT_FUSION_STRATEGY,
    "Choose fusion strategy (atm: Max, Preserve3Coincident, Min)");
DEFINE_bool(
    allow_skewing,
    DEFAULT_ALLOW_SKEWING,
    "Allow skewing in generated schedules");
DEFINE_bool(
    positive_orthant,
    DEFAULT_POSITIVE_ORTHANT,
    "Request schedules with positive coefficients only, i.e. no loop reversal");
DEFINE_bool(
    fix_parameters_before_scheduling,
    DEFAULT_FIX_PARAMETERS_BEFORE_SCHEDULING,
    "Propagate parametric context before calling the scheduler");
DEFINE_bool(use_shared_memory, DEFAULT_USE_SHARED_MEMORY, "Use shared memory");
DEFINE_bool(
    use_private_memory,
    DEFAULT_USE_PRIVATE_MEMORY,
    "Use private memory");
DEFINE_bool(
    unroll_copy_shared,
    DEFAULT_UNROLL_COPY_SHARED,
    "Unroll copy to/from shared");
DEFINE_string(tile, DEFAULT_TILE, "Tile sizes (comma-separated list)");
DEFINE_bool(
    tile_imperfectly_nested,
    DEFAULT_TILE_IMPERFECTLY_NESTED,
    "Use cross-band tiling");
DEFINE_string(block, DEFAULT_BLOCK, "Block sizes (comma-separated list)");
DEFINE_string(grid, DEFAULT_GRID, "Grid sizes (comma-separated list)");
DEFINE_uint32(unroll, DEFAULT_UNROLL_FACTOR, "self explanatory");

namespace tc {

// The proper way to use this CLI strategy is to:
// 1. call makeBaseCliStrategy
// 2. override relevant options
//    (at a minimum: tile, mapToThreads and mapToBlocks)
// 3. call makeCliStrategy with the overridden options
tc::CudaMappingOptions makeBaseCliStrategy() {
  tc::FusionStrategy fs;
  CHECK(tc::FusionStrategy_Parse(DEFAULT_FUSION_STRATEGY, &fs));
  CudaMappingOptions options =
      CudaMappingOptions::makeNaiveMappingOptions()
          .mapToThreads(DEFAULT_BLOCK)
          .mapToBlocks(DEFAULT_GRID)
          .useSharedMemory(DEFAULT_USE_SHARED_MEMORY)
          .usePrivateMemory(DEFAULT_USE_PRIVATE_MEMORY)
          .unrollCopyShared(DEFAULT_UNROLL_COPY_SHARED);
  options.scheduleFusionStrategy(fs)
      .fixParametersBeforeScheduling(DEFAULT_FIX_PARAMETERS_BEFORE_SCHEDULING)
      .tile(DEFAULT_TILE)
      .tileImperfectlyNested(DEFAULT_TILE_IMPERFECTLY_NESTED)
      .unroll(DEFAULT_UNROLL_FACTOR);
  options.generic.outerScheduleOptions.proto.set_allow_skewing(
      DEFAULT_ALLOW_SKEWING);
  options.generic.outerScheduleOptions.proto.set_positive_orthant(
      DEFAULT_POSITIVE_ORTHANT);
  return options;
}

tc::CudaMappingOptions makeCliStrategy(tc::CudaMappingOptions options) {
  if (FLAGS_fusion_strategy != std::string(DEFAULT_FUSION_STRATEGY)) {
    tc::FusionStrategy fs;
    if (tc::FusionStrategy_Parse(FLAGS_fusion_strategy, &fs)) {
      options.scheduleFusionStrategy(fs);
    } else {
      CHECK(false) << "Unknown fusion_strategy: " << FLAGS_fusion_strategy;
    }
  }
  options.generic.outerScheduleOptions.proto.set_allow_skewing(
      FLAGS_allow_skewing);
  options.generic.outerScheduleOptions.proto.set_positive_orthant(
      FLAGS_positive_orthant);

  if (FLAGS_fix_parameters_before_scheduling !=
      DEFAULT_FIX_PARAMETERS_BEFORE_SCHEDULING) {
    options.fixParametersBeforeScheduling(
        FLAGS_fix_parameters_before_scheduling);
  }
  if (FLAGS_tile != DEFAULT_TILE) {
    options.tile(FLAGS_tile);
  }
  if (FLAGS_tile_imperfectly_nested != DEFAULT_TILE_IMPERFECTLY_NESTED) {
    options.tileImperfectlyNested(FLAGS_tile_imperfectly_nested);
  }
  if (FLAGS_block != DEFAULT_BLOCK) {
    options.mapToThreads(FLAGS_block);
  }
  if (FLAGS_grid != DEFAULT_GRID) {
    options.mapToBlocks(FLAGS_grid);
  }
  if (FLAGS_use_shared_memory != DEFAULT_USE_SHARED_MEMORY) {
    options.useSharedMemory(FLAGS_use_shared_memory);
  }
  if (FLAGS_use_private_memory != DEFAULT_USE_PRIVATE_MEMORY) {
    options.usePrivateMemory(FLAGS_use_private_memory);
  }
  if (FLAGS_unroll_copy_shared != DEFAULT_UNROLL_COPY_SHARED) {
    options.unrollCopyShared(FLAGS_unroll_copy_shared);
  }
  if (FLAGS_unroll != DEFAULT_UNROLL_FACTOR) {
    options.unroll(FLAGS_unroll);
  }
  return options;
}

} // namespace tc
