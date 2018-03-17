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
    tc::OptionsCache::getCache()->keepOnlyBestCandidates(10);
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
  return tc::autotune::restoreCandidates(tcName, inputs, outputs);
}

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

  std::signal(SIGTERM, [](int sig) {
    signal_ = sig;
    killRequested_ = 1;
  });
  std::signal(SIGINT, [](int sig) {
    signal_ = sig;
    killRequested_ = 1;
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
    if (killRequested_) {
      std::cerr << "Autotuning aborted." << std::endl;
      storeCaches(cacheFileName);
      tunerThread.join();
      killRequested_ = 0;
      throw std::runtime_error("Abort requested");
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
      tcName, inputs.begin()->second, outputPtrs);
}

} // namespace detail
} // namespace autotune
} // namespace tc
