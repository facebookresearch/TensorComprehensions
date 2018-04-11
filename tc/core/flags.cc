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
#include <ctime>
#include <iostream>
#include <mutex>

#include "tc/core/flags.h"

namespace tc {

DEFINE_bool(debug_lang, false, "Dump TC lang information.");
DEFINE_bool(debug_halide, false, "Dump Halide information.");
DEFINE_bool(
    debug_cuda,
    false,
    "Compile with debug flags on to run in cuda-gdb");
DEFINE_bool(
    debug_tuner,
    false,
    "Print debug spew for the tuner multithreading behavior");
DEFINE_bool(
    debug_tc_mapper,
    false,
    "Print debug spew for the tc_mapper like cuda code, mapping options etc");
DEFINE_bool(dump_cuda, false, "Print the generated cudaSource");

// CPU codegen options
DEFINE_bool(llvm_dump_before_opt, false, "Print IR before optimization");
DEFINE_bool(llvm_dump_after_opt, false, "Print IR after optimization");

DEFINE_uint32(
    benchmark_warmup,
    10,
    "Number of runs to use for warming up benchmarking (also for autotuning)");
DEFINE_uint32(
    benchmark_iterations,
    100,
    "Number of runs to use for collecting benchmarks (also for autotuning)");
DEFINE_bool(
    schedule_tree_verbose_validation,
    false,
    "Print debug spew for experimental schedule_tree");

// Autotuner flags
DEFINE_uint32(
    tuner_gen_pop_size,
    100,
    "Population size for genetic autotuning");
DEFINE_uint32(
    tuner_gen_crossover_rate,
    80,
    "Crossover rate for genetic autotuning");
DEFINE_uint32(
    tuner_gen_mutation_rate,
    7,
    "Mutation rate for genetic autotuning");
DEFINE_uint32(
    tuner_gen_generations,
    25,
    "How many generations to run genetic tuning for");
DEFINE_uint32(
    tuner_gen_number_elites,
    10,
    "The number of best candidates that are preserved intact between generations");
DEFINE_uint32(tuner_threads, 1, "Number of CPU threads to use when autotuning");
DEFINE_string(
    tuner_gpus,
    "0",
    "Comma separated list of GPUs to use for autotuning");
DEFINE_bool(
    tuner_print_best,
    false,
    "Print to INFO the best tuning options after each generation");
DEFINE_string(tuner_rng_restore, "", "Rng state to restore");
DEFINE_bool(
    tuner_gen_restore_from_proto,
    true,
    "Restore the population from proto cache");
DEFINE_uint32(
    tuner_gen_restore_number,
    10,
    "The number of best candidates to restore from the proto cache");
DEFINE_bool(
    tuner_gen_log_generations,
    false,
    "Log each generation's runtimes.");
DEFINE_uint64(
    tuner_min_launch_total_threads,
    64,
    "Prune out kernels mapped to fewer than this many threads and block");
DEFINE_int64(
    random_seed,
    -1,
    "The number of best candidates to restore from the proto cache");
DEFINE_uint32(
    tuner_save_best_candidates_count,
    10,
    "Number of best candidates to save from autotuning");

uint64_t initRandomSeed() {
  static std::mutex mut;
  static bool inited = false;
  std::lock_guard<std::mutex> lg(mut);
  auto& seed = const_cast<uint64_t&>(randomSeed());
  if (!inited) {
    inited = true;
    seed = (FLAGS_random_seed >= 0) ? static_cast<uint64_t>(FLAGS_random_seed)
                                    : static_cast<uint64_t>(time(0));
    seed = seed % 15485863; // 1-millionth prime
  }
  return seed;
}

const uint64_t& randomSeed() {
  static uint64_t seed = static_cast<uint64_t>(-1);
  return seed;
}

} // namespace tc
