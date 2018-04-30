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

#include "tc/autotuner/autotuner.h"

#include <atomic>
#include <chrono>
#include <numeric>
#include <thread>

#include <glog/stl_logging.h>

#include "tc/autotuner/utils.h"
#include "tc/core/compiler.h"
#include "tc/core/flags.h"
#include "tc/core/scope_guard.h"
#include "tc/core/tensor.h"
#include "tc/core/utils/math.h"
#include "tc/lang/canonicalize.h"

namespace tc {
namespace autotune {
namespace detail {
template <typename Backend>
TuningHarness<Backend>::TuningHarness(
    size_t maxPopulationSize,
    lang::TreeRef tcTree,
    const std::unordered_map<size_t, std::vector<const DLConstTensor*>>& inputs,
    std::unordered_map<size_t, std::vector<const DLTensor*>>& outputs,
    const typename Backend::MappingOptionsType& baseMapping,
    const TuningParameterFixer& fixedParams,
    std::shared_ptr<OptionsCache<Backend>> optionsCache)
    : stopRequested_(false),
      currentCompilationJob_(0),
      numEvaluations_(0),
      kTcTree_(tcTree),
      kBaseMapping_(baseMapping),
      kInputs_(inputs),
      outputs_(outputs),
      bestTime_(std::numeric_limits<size_t>::max()),
      bestMappingOptions_(baseMapping),
      optionsCache_(optionsCache) {}

template <typename Backend>
template <typename SearchStrategy>
void TuningHarness<Backend>::run(SearchStrategy& searchStrategy) {
  // TODO: kNumGenerations -> iterations
  for (size_t i = 0; i < searchStrategy.kNumGenerations; ++i) {
    if (not stopRequested_) {
      runOneIteration(searchStrategy, i);
    }
  }
}

template <typename Backend>
void TuningHarness<Backend>::stopAfterCurrentIteration() {
  stopRequested_ = true;
}

#define LOG_LINE_BY_LINE(GSTREAM, ISTREAM)               \
  for (std::string line; std::getline(ISTREAM, line);) { \
    LOG(GSTREAM) << line;                                \
  }

template <typename Backend>
template <typename SearchStrategy>
void TuningHarness<Backend>::doCompile(SearchStrategy& searchStrategy) {
  // Atomically fetch and add the next job until there are no jobs left
  while (true) {
    auto current = currentCompilationJob_.fetch_add(1);
    if (current >= searchStrategy.population.size()) {
      break;
    }
    std::unique_ptr<typename Backend::ExecutorType> pExecutor(nullptr);
    auto pConf = searchStrategy.population.at(current).get();
    auto options = makeOptions<Backend>(kBaseMapping_, *pConf);
    try {
      if (FLAGS_debug_tuner) {
        std::stringstream ssInfo;
        typename Backend::MappingOptionsCppPrinter infoPrinter(ssInfo);
        infoPrinter << options;
        LOG(INFO) << "[COMPILE] Start compilation @:" << current;
        LOG_LINE_BY_LINE(INFO, ssInfo);
      }
      pExecutor =
          tc::compile<Backend>(kTcTree_, kInputs_.begin()->second, options);
      LOG_IF(INFO, FLAGS_debug_tuner) << "[COMPILE] Done compilation";
    } catch (const std::exception& e) {
      LOG(WARNING) << "[TUNER][COMPILE] failed compilation: " << e.what();
      std::stringstream ssWarning;
      typename Backend::MappingOptionsCppPrinter warningPrinter(ssWarning);
      warningPrinter << options;
      LOG_LINE_BY_LINE(WARNING, ssWarning);
      pConf->invalid = true;
    }

    // Emplace pExecutor (nullptr if compilation failed)
    std::lock_guard<std::mutex> lock(executorsMutex_);
    executors_.push(std::move(pExecutor));
    configurations_.push(pConf);
  }
}

template <typename Backend>
void TuningHarness<Backend>::doEvaluate(
    size_t device,
    size_t populationSize,
    Printer& printer) {
  typename Backend::WithDevice wd(device);
  CHECK_EQ(1, kInputs_.count(device));
  auto& inputs = kInputs_.at(device);
  CHECK_EQ(1, outputs_.count(device));
  auto& outputs = outputs_.at(device);

  while (true) {
    auto current = numEvaluations_.load();
    if (current >= populationSize) {
      // We have seen enough, exit
      break;
    }

    CandidateConfiguration* pConf;
    std::unique_ptr<typename Backend::ExecutorType> pExecutor;
    {
      std::lock_guard<std::mutex> lock(executorsMutex_);
      if (executors_.size() == 0) {
        // No executors atm, wait for the queue to populate
        continue;
      }
      pExecutor = std::move(executors_.front());
      executors_.pop();
      pConf = configurations_.front();
      configurations_.pop();
    }

    // Properly keep track of count, RAII way
    ScopeGuard sg([this]() { this->numEvaluations_.fetch_add(1); });
    if (!pExecutor.get()) {
      // If I popped an empty executor then compilation didn't go as
      // planned, skip it.
      CHECK(pConf->invalid);
      continue;
    }

    auto options = makeOptions<Backend>(kBaseMapping_, *pConf);
    if (FLAGS_debug_tuner) {
      // Always log option to INFO so we can correlate to timing offling
      std::stringstream ssInfo;
      ssInfo << "Launch device kernel on device: " << std::to_string(device)
             << " options:\n"
             << typename Backend::MappingOptionsAsCpp(options);
      LOG_LINE_BY_LINE(INFO, ssInfo);
    }

    std::vector<Duration> runtimes;
    try {
      size_t bestTimeSoFar;
      {
        std::lock_guard<std::mutex> lock(bestTimeMutex_);
        bestTimeSoFar = bestTime_;
      }
      auto prune = detail::skipExecutionOrWarmup<Backend>(
          *pExecutor, outputs, inputs, bestTimeSoFar);
      if (prune) {
        pConf->invalid = true;
        continue;
      } else {
        /// We don't want the autotuner to take too long evaluating, we just
        /// need to *rank* the results.
        constexpr int kReducedBenchmarkIterations = 10;
        runtimes.reserve(kReducedBenchmarkIterations);
        for (size_t i = 0; i < kReducedBenchmarkIterations; ++i) {
          auto timings = pExecutor->profile(inputs, outputs);
          runtimes.push_back(timings.kernelRuntime);
        }
      }
    } catch (std::exception& e) {
      LOG(WARNING) << "Runtime error device " << device << ": " << e.what();
      std::stringstream ssWarning;
      typename Backend::MappingOptionsCppPrinter warningPrinter(ssWarning);
      warningPrinter << options;
      LOG(WARNING) << "Aborted execution on device " << device;
      LOG_LINE_BY_LINE(WARNING, ssWarning);
      handleDeviceRuntimeError<Backend>(device, options);
      pConf->invalid = true;
      continue;
    }

    auto prof = median(runtimes);
    size_t prof_us =
        std::chrono::duration_cast<std::chrono::microseconds>(prof).count();

    LOG_IF(INFO, tc::FLAGS_debug_tuner)
        << "Run on device " << device << " took: " << prof_us << "us";
    printer.record(prof);
    pConf->runtime = prof;

    optionsCache_->recordRuntime(
        lang::canonicalTc(kTcTree_),
        makeTensorInfoVector(inputs),
        makeTensorInfoVector(outputs),
        Backend::deviceString(),
        options,
        prof);

    // Save best time under lock
    {
      std::lock_guard<std::mutex> lock(bestTimeMutex_);
      if (prof_us < bestTime_) {
        bestTime_ = prof_us;
        bestMappingOptions_ = options;
      }
    }

    // Trailing sanity check: this looks like a spurious case, fail very hard
    if (prof_us == 0) {
      std::stringstream ss;
      ss << "Runtimes: ";
      for (auto r : runtimes) {
        std::cout
            << std::chrono::duration_cast<std::chrono::microseconds>(r).count()
            << " ";
      }
      LOG(FATAL) << "The measured runtime is 0, marking as invalid: "
                 << ss.str() << "\n"
                 << typename Backend::MappingOptionsAsCpp(options);
    }
  } // end while
}

template <typename Backend>
template <typename SearchStrategy>
void TuningHarness<Backend>::runOneIteration(
    SearchStrategy& searchStrategy,
    size_t iteration) {
  // Define tensors per device once globally
  auto devices = detail::parseDevices<Backend>(FLAGS_tuner_devices);
  CHECK(executors_.empty());
  CHECK(configurations_.empty());

  {
    // Initialize for this round
    currentCompilationJob_.store(0);
    numEvaluations_.store(0);
    Printer printer(
        iteration,
        searchStrategy.population.size(),
        currentCompilationJob_,
        numEvaluations_);
    auto logIterations = FLAGS_tuner_gen_log_generations;
    ScopeGuard sgPrinter([logIterations, &printer]() {
      printer.stop();
      if (logIterations) {
        printer.printAll();
      }
    });

    // Just spawn and join new threads for each iteration
    std::vector<std::thread> cpuCompilationThreads;
    cpuCompilationThreads.reserve(FLAGS_tuner_threads);
    ScopeGuard sgCompilationThreads([&cpuCompilationThreads]() {
      for (auto& cpuCompilationThread : cpuCompilationThreads) {
        cpuCompilationThread.join();
      }
    });
    for (size_t i = 0; i < FLAGS_tuner_threads; ++i) {
      cpuCompilationThreads.emplace_back(
          [this, &searchStrategy]() { this->doCompile(searchStrategy); });
    }

    // Just spawn and join new threads for each iteration
    std::vector<std::thread> workerThreads;
    workerThreads.reserve(devices.size());
    LOG_IF(INFO, tc::FLAGS_debug_tuner)
        << "Start evaluation: " << devices.size() << " " << executors_.size()
        << " " << configurations_.size();
    ScopeGuard sgDeviceWorkerThreads([&workerThreads]() {
      for (auto& workerThread : workerThreads) {
        workerThread.join();
      }
    });
    auto populationSize = searchStrategy.population.size();
    for (auto device : devices) {
      workerThreads.emplace_back([this, device, populationSize, &printer]() {
        this->doEvaluate(device, populationSize, printer);
      });
    }
  }

  // At this point everything is synchronized because out of scope, done
  if (FLAGS_debug_tuner) {
    LOG(INFO) << "[TUNER][ITERATION LOG] best option so far:";
    std::stringstream ssInfo;
    typename Backend::MappingOptionsCppPrinter infoPrinter(ssInfo);
    infoPrinter << bestMappingOptions();
    LOG_LINE_BY_LINE(INFO, ssInfo);
  }
  searchStrategy.updateParameters();
}
} // namespace detail

namespace {
volatile std::sig_atomic_t sigint_ = 0;
volatile std::sig_atomic_t sigterm_ = 0;

template <typename Backend>
std::vector<typename Backend::MappingOptionsType> loadThroughCaches(
    lang::TreeRef tree,
    std::shared_ptr<OptionsCache<Backend>> optionsCache,
    const std::string& cacheFileName,
    const std::vector<const DLConstTensor*>& inputs,
    const size_t numCandidates) {
  std::cout << "Loading proto from: " << tc::makeOptionsFilename(cacheFileName)
            << " and " << Backend::makeDeviceFilename(cacheFileName)
            << std::endl;
  if (!cacheFileName.empty()) {
    optionsCache->loadCacheFromFile(tc::makeOptionsFilename(cacheFileName));
  }
  tc::FLAGS_tuner_gen_restore_number =
      std::min(numCandidates, size_t(FLAGS_tuner_gen_pop_size) - 1);
  auto outputs = tc::detail::inferOutputTensorInfo(tree, inputs);
  return optionsCache->getTopKOptions(
      canonicalTc(tree),
      makeTensorInfoVector(inputs),
      outputs,
      Backend::deviceString(),
      FLAGS_tuner_gen_restore_number);
}

template <typename Backend>
void storeCaches(
    const std::shared_ptr<OptionsCache<Backend>> optionsCache,
    const std::string& cacheFilename) {
  if (cacheFilename.empty()) {
    std::cout << "No filepath provided, not saving cache" << std::endl;
  } else {
    std::cout << "Dumping cache to " << tc::makeOptionsFilename(cacheFilename)
              << "/" << Backend::makeDeviceFilename(cacheFilename) << std::endl;
    optionsCache->pruneKeepTopK(10);
    optionsCache->storeCacheToFile(tc::makeOptionsFilename(cacheFilename));
    optionsCache->pruneKeepTopK(1);
  }
}

void removeDuplicates(std::vector<size_t>& v) {
  std::sort(v.begin(), v.end());
  v.erase(std::unique(v.begin(), v.end()), v.end());
}

std::vector<size_t> inputDivisorsAndPowers2(
    const std::vector<const DLConstTensor*>& inputs) {
  std::vector<size_t> sizes;
  for (const auto& input : inputs) {
    for (int i = 0; i < input->ndim; i++) {
      sizes.push_back(input->shape[i]);
    }
  }
  removeDuplicates(sizes);
  std::vector<size_t> divsAndPows;
  for (auto size : sizes) {
    divsAndPows =
        mergeVectors(std::move(divsAndPows), powers2andCeilDivisors(size));
  }
  divsAndPows =
      mergeVectors(std::move(divsAndPows), powers2andCeilDivisors(256));
  return divsAndPows;
}

size_t largestDim(const std::vector<const DLConstTensor*>& inputs) {
  CHECK_GE(inputs.size(), 0);
  auto maxElement = std::max_element(
      inputs.begin(),
      inputs.end(),
      [](const DLConstTensor* a, const DLConstTensor* b) {
        return a->ndim < b->ndim;
      });
  return (*maxElement)->ndim;
}

// Creates well-chosen parameter sizes to match the input shapes.
void setupTuningParameters(
    const std::vector<const DLConstTensor*>& inputs,
    TuningConfiguration& configuration) {
  CHECK_GT(inputs.size(), 0);
  auto range = inputDivisorsAndPowers2(inputs);
  // 0 is a valid tiling annotation and signals no tiling of that dimension
  // 0 is not a valid block / grid annotation
  auto nTilesDim = largestDim(inputs) + 1;
  auto tileRange = range;
  tileRange.push_back(0);
  configuration.tilingParams.setRange(nTilesDim, range);
  configuration.blockParams.setRange(range, "b");
  configuration.gridParams.setRange(range, "g");
  configuration.unrollFactor = RangeParameter({1, 2, 4, 8, 16, 32}, "unroll");
}
} // namespace

template <typename Backend, typename SearchStrategy>
Autotuner<Backend, SearchStrategy>::Autotuner(const std::string& tc)
    : tcEntryPointMap_(tc::detail::parse(tc)),
      optionsCache_(new OptionsCache<Backend>()) {}

template <typename Backend, typename SearchStrategy>
std::vector<typename Backend::MappingOptionsType>
Autotuner<Backend, SearchStrategy>::tune(
    const std::string& tcEntryPoint,
    const std::unordered_map<size_t, std::vector<const DLConstTensor*>>& inputs,
    std::unordered_map<size_t, std::vector<const DLTensor*>>& outputs,
    const typename Backend::MappingOptionsType& baseMapping,
    const std::string& cacheFileName,
    const TuningParameterFixer& fixedParams) {
  CHECK_EQ(1, tcEntryPointMap_.count(tcEntryPoint))
      << "Error looking up " << tcEntryPoint;

  // Initialize a model configuration
  TuningConfiguration modelConfiguration;
  setupTuningParameters(inputs.begin()->second, modelConfiguration);
  modelConfiguration.fixParameters(fixedParams);

  // Build starting points from baseMapping + whatever we recover from cache
  std::vector<typename Backend::MappingOptionsType> startingPoints{baseMapping};
  if (FLAGS_tuner_gen_restore_from_proto && !(cacheFileName.empty())) {
    CHECK_GT(inputs.size(), 0);
    auto restoredCandidates = loadThroughCaches<Backend>(
        tcEntryPointMap_.at(tcEntryPoint),
        optionsCache_,
        cacheFileName,
        inputs.begin()->second,
        FLAGS_tuner_gen_restore_number);
    startingPoints.reserve(1 + restoredCandidates.size());
    std::move(
        restoredCandidates.begin(),
        restoredCandidates.end(),
        std::back_inserter(startingPoints));
  }

  // Create initial configs based on options + model configuration
  std::vector<TuningConfiguration> configs;
  configs.reserve(startingPoints.size());
  std::transform(
      startingPoints.begin(),
      startingPoints.end(),
      std::back_inserter(configs),
      [this, &fixedParams, &modelConfiguration](
          const typename Backend::MappingOptionsType& options) {
        auto config = detail::makeTuningConfiguration<Backend>(
            options, modelConfiguration);
        config.fixParameters(fixedParams);
        return config;
      });

  // searchStrategy is passed to tuningHarness.run()
  SearchStrategy searchStrategy(
      configs,
      FLAGS_tuner_gen_generations,
      FLAGS_tuner_gen_pop_size,
      FLAGS_tuner_gen_crossover_rate,
      FLAGS_tuner_gen_mutation_rate,
      FLAGS_tuner_gen_number_elites);

  // Create a tuning harness
  detail::TuningHarness<Backend> tuningHarness(
      FLAGS_tuner_gen_pop_size,
      tcEntryPointMap_.at(tcEntryPoint),
      inputs,
      outputs,
      baseMapping,
      fixedParams,
      optionsCache_);

  // Setup handlers
  sigterm_ = 0;
  sigint_ = 0;
  auto sigtermOrigHandler = std::signal(SIGTERM, [](int) { sigterm_ = 1; });
  auto sigintOrigHandler = std::signal(SIGINT, [](int) { sigint_ = 1; });
  ScopeGuard handlersGuard([&]() {
    std::signal(SIGTERM, sigtermOrigHandler);
    std::signal(SIGINT, sigintOrigHandler);
  });

  // Run harness in a separate thread
  std::atomic_bool tuningHarnessFinished(false);
  std::exception_ptr tuningHarnessThreadEx = nullptr;
  std::thread tuningHarnessThread([&]() {
    try {
      tuningHarness.run(searchStrategy);
    } catch (const std::exception& e) {
      std::cerr << "Exception during autotuning: " << e.what()
                << "\n dumping cache to "
                << tc::makeOptionsFilename(cacheFileName) << "/"
                << Backend::makeDeviceFilename(cacheFileName) << std::endl;
      storeCaches<Backend>(optionsCache_, cacheFileName);
      tuningHarnessThreadEx = std::current_exception();
    }
    tuningHarnessFinished = true;
  });
  while (not tuningHarnessFinished) {
    std::this_thread::sleep_for(std::chrono::milliseconds(100));
    if (sigint_) {
      std::cerr
          << "Autotuning will stop after the current iteration has finished."
          << std::endl;
      tuningHarness.stopAfterCurrentIteration();
      tuningHarnessThread.join();
      storeCaches<Backend>(optionsCache_, cacheFileName);
    }
    if (sigterm_) {
      std::cerr << "Autotuning aborted." << std::endl;
      storeCaches<Backend>(optionsCache_, cacheFileName);
      std::abort();
    }
  }

  tuningHarnessThread.join();

  if (tuningHarnessThreadEx) {
    std::rethrow_exception(tuningHarnessThreadEx);
  }

  // only store cache if the file path is provided
  if (!cacheFileName.empty()) {
    storeCaches<Backend>(optionsCache_, cacheFileName);
  }

  CHECK_GT(inputs.size(), 0);
  auto inputInfos = makeTensorInfoVector(inputs.begin()->second);
  auto outputInfos = makeTensorInfoVector(outputs.begin()->second);
  return optionsCache_->getTopKOptions(
      canonicalTc(tcEntryPointMap_.at(tcEntryPoint)),
      inputInfos,
      outputInfos,
      Backend::deviceString(),
      1);
}
} // namespace autotune
} // namespace tc
