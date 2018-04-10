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

#include "tc/autotuner/genetic_autotuner.h"

#include <chrono>
#include <csignal>
#include <thread>

#include "tc/core/cuda/cuda_compilation_cache.h"
#include "tc/core/cuda/cuda_tc_executor.h"
#include "tc/core/flags.h"
#include "tc/core/scope_guard.h"
#include "tc/lang/parser.h"

namespace tc {
namespace autotune {
namespace detail {

GeneticAutotuner::GeneticAutotuner(const std::string& tc) : tc_(tc) {
  lang::Parser parser(tc);
  while (parser.L.cur().kind != lang::TK_EOF) {
    auto treeRef = parser.parseFunction();
    auto name = lang::Def(treeRef).name().name();
    tcNameMap_.emplace(std::make_pair(name, treeRef));
  }
}

namespace {

void enableOrLoadCache(const std::string& filename) {
  tc::OptionsCache::enableCache();
  tc::CudaCache::enableCache();
  if (!filename.empty()) {
    tc::OptionsCache::loadCacheFromProtobuf(tc::makeOptionsFilename(filename));
    tc::CudaCache::loadCacheFromProtobuf(tc::makeCudaFilename(filename));
  }
}
} // namespace

void GeneticAutotuner::storeCaches(const std::string& filename) {
  if (filename.empty()) {
    std::cout << "No filepath provided, not saving cache" << std::endl;
  } else {
    std::cout << "Dumping cache to " << filename << ".cuda/options"
              << std::endl;
    tc::OptionsCache::getCache()->keepOnlyBestCandidates(
        tc::FLAGS_tuner_save_best_candidates_count);
    tc::OptionsCache::dumpCacheToProtobuf(tc::makeOptionsFilename(filename));

    tc::OptionsCache::getCache()->keepOnlyBestCandidates(1);
    tc::removeFromCudaCacheEntriesNotInOptionsCache(
        *tc::CudaCache::getCache(), *tc::OptionsCache::getCache());
    tc::CudaCache::dumpCacheToProtobuf(tc::makeCudaFilename(filename));
  }
}

std::vector<CudaMappingOptions> GeneticAutotuner::load(
    const std::string& cacheFileName,
    const std::string& tcName,
    const std::vector<const DLTensor*>& inputs,
    const size_t numCandidates) {
  std::cout << "Loading proto from: " << tc::makeOptionsFilename(cacheFileName)
            << " and " << tc::makeCudaFilename(cacheFileName) << std::endl;
  enableOrLoadCache(cacheFileName);
  tc::FLAGS_tuner_gen_restore_number =
      std::min(numCandidates, size_t(FLAGS_tuner_gen_pop_size) - 1);
  ExecutionEngine<CudaTcExecutor> ee;
  ee.define(tc_);
  auto outputs = ee.inferOutputTensorInfo(tcName, inputs);
  return tc::autotune::restoreCandidates(
      canonicalTc(tcNameMap_.at(tcName)), inputs, outputs);
}

namespace {
volatile std::sig_atomic_t sigint_ = 0;
volatile std::sig_atomic_t sigterm_ = 0;
} // namespace

llvm::Optional<CudaMappingOptions> GeneticAutotuner::tune(
    const std::string& cacheFileName,
    const std::string& tcName,
    const std::unordered_map<size_t, std::vector<const DLTensor*>>& inputs,
    std::unordered_map<size_t, std::vector<DLTensor*>>& outputs,
    CudaMappingOptions baseMapping,
    std::vector<CudaMappingOptions> startingPoints,
    const TuningParameterFixer& fixedParams) {
  CHECK_EQ(1, tcNameMap_.count(tcName)) << "Error looking up " << tcName;
  enableOrLoadCache(cacheFileName);

  if (FLAGS_tuner_gen_restore_from_proto && !(cacheFileName.empty())) {
    CHECK_GT(inputs.size(), 0);

    auto restoredCandidates = load(
        cacheFileName,
        tcName,
        inputs.begin()->second,
        FLAGS_tuner_gen_restore_number);
    startingPoints.reserve(startingPoints.size() + restoredCandidates.size());
    std::move(
        restoredCandidates.begin(),
        restoredCandidates.end(),
        std::back_inserter(startingPoints));
  }

  GeneticTunerHarness tuner(
      FLAGS_tuner_gen_pop_size,
      FLAGS_tuner_gen_crossover_rate,
      FLAGS_tuner_gen_mutation_rate,
      FLAGS_tuner_gen_number_elites,
      tcNameMap_.at(tcName),
      tcName,
      inputs,
      outputs,
      baseMapping,
      startingPoints,
      fixedParams);

  sigterm_ = 0;
  sigint_ = 0;
  auto sigtermOrigHandler = std::signal(SIGTERM, [](int) { sigterm_ = 1; });
  auto sigintOrigHandler = std::signal(SIGINT, [](int) { sigint_ = 1; });
  ScopeGuard handlersGuard([&]() {
    std::signal(SIGTERM, sigtermOrigHandler);
    std::signal(SIGINT, sigintOrigHandler);
  });

  std::atomic_bool tunerFinished(false);
  std::exception_ptr tunerThreadEx = nullptr;

  std::thread tunerThread([&]() {
    try {
      tuner.run(FLAGS_tuner_gen_generations);
    } catch (const std::exception& e) {
      std::cerr << "Exception during autotuning: " << e.what()
                << "\n dumping cache to " << cacheFileName << ".cuda/options"
                << std::endl;
      storeCaches(cacheFileName);
      tunerThreadEx = std::current_exception();
    }
    tunerFinished = true;
  });
  while (not tunerFinished) {
    std::this_thread::sleep_for(std::chrono::milliseconds(100));
    if (sigint_) {
      std::cerr
          << "Autotuning will stop after the current generation has finished."
          << std::endl;
      tuner.stopAfterCurrentGeneration();
      tunerThread.join();
      storeCaches(cacheFileName);
    }
    if (sigterm_) {
      std::cerr << "Autotuning aborted." << std::endl;
      storeCaches(cacheFileName);
      std::abort();
    }
  }

  tunerThread.join();

  if (tunerThreadEx) {
    std::rethrow_exception(tunerThreadEx);
  }

  // only store cache if the file path is provided
  if (!cacheFileName.empty()) {
    storeCaches(cacheFileName);
  }

  ExecutionEngine<CudaTcExecutor> ee;
  ee.define(tc_);
  auto outputPtrs = ee.inferOutputTensorInfo(tcName, inputs.begin()->second);

  CHECK_GT(inputs.size(), 0);
  return tc::autotune::getBestOptions(
      canonicalTc(tcNameMap_.at(tcName)), inputs.begin()->second, outputPtrs);
}

} // namespace detail
} // namespace autotune
} // namespace tc
