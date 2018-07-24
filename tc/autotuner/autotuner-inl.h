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
#include <atomic>
#include <chrono>
#include <functional>
#include <numeric>
#include <thread>

#include <glog/stl_logging.h>

#include "tc/autotuner/utils.h"
#include "tc/core/check.h"
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
      tcTree_(tcTree),
      baseMapping_(baseMapping),
      inputs_(inputs),
      outputs_(outputs),
      optionsCache_(optionsCache) {}

template <typename Backend>
template <typename SearchStrategy>
void TuningHarness<Backend>::run(SearchStrategy& searchStrategy) {
  for (size_t i = 0; i < searchStrategy.numGenerations; ++i) {
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
    if (not stopRequested_) {
      auto options = makeOptions<Backend>(baseMapping_, *pConf);
      try {
        if (FLAGS_debug_tuner) {
          std::stringstream ssInfo;
          typename Backend::MappingOptionsCppPrinter infoPrinter(ssInfo);
          infoPrinter << options;
          LOG(INFO) << "[COMPILE] Start compilation @:" << current;
          LOG_LINE_BY_LINE(INFO, ssInfo);
        }
        pExecutor =
            tc::compile<Backend>(tcTree_, inputs_.begin()->second, options);
        LOG_IF(INFO, FLAGS_debug_tuner) << "[COMPILE] Done compilation";
      } catch (const std::exception& e) {
        LOG(WARNING) << "[TUNER][COMPILE] failed compilation: " << e.what();
        std::stringstream ssWarning;
        typename Backend::MappingOptionsCppPrinter warningPrinter(ssWarning);
        warningPrinter << options;
        LOG_LINE_BY_LINE(WARNING, ssWarning);
        pConf->invalid = true;
      }
    } else {
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
  TC_CHECK_EQ(inputs_.count(device), 1u);
  auto& inputs = inputs_.at(device);
  TC_CHECK_EQ(outputs_.count(device), 1u);
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
      TC_CHECK(pConf->invalid);
      continue;
    }

    if (stopRequested_) {
      pConf->invalid = true;
      continue;
    }

    auto options = makeOptions<Backend>(baseMapping_, *pConf);
    if (FLAGS_debug_tuner) {
      // Always log option to INFO so we can correlate to timing offline
      std::stringstream ssInfo;
      ssInfo << "Launch device kernel on device: " << std::to_string(device)
             << " options:\n"
             << typename Backend::MappingOptionsAsCpp(options);
      LOG_LINE_BY_LINE(INFO, ssInfo);
    }

    std::vector<Duration> runtimes{Duration::max()};
    try {
      auto vBest = optionsCache_->getTopKEntries(
          lang::canonicalTc(tcTree_),
          makeTensorInfoVector(inputs),
          makeTensorInfoVector(outputs),
          Backend::backendString(),
          1);
      Duration bestTimeSoFar =
          (vBest.size() > 0) ? vBest[0].second : Duration::max();
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
          if (timings.kernelRuntime.toMicroSeconds() > 0) {
            runtimes.push_back(timings.kernelRuntime);
          }
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

    if (runtimes.size() == 0u) {
      pConf->invalid = true;
      return;
    }

    auto prof = median(runtimes);

    LOG_IF(INFO, tc::FLAGS_debug_tuner)
        << "Run on device " << device << " took: " << prof.toMicroSeconds()
        << "us";
    printer.record(prof);
    pConf->runtime = prof;

    optionsCache_->recordRuntime(
        lang::canonicalTc(tcTree_),
        makeTensorInfoVector(inputs),
        makeTensorInfoVector(outputs),
        Backend::backendString(),
        options,
        prof);
  } // end while
}

template <typename Backend>
template <typename SearchStrategy>
void TuningHarness<Backend>::runOneIteration(
    SearchStrategy& searchStrategy,
    size_t iteration) {
  // Define tensors per device once globally
  auto devices = detail::parseDevices<Backend>(FLAGS_tuner_devices);
  TC_CHECK(executors_.empty());
  TC_CHECK(configurations_.empty());

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

    // Just spawn and join new threads for each device
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
  if (FLAGS_debug_tuner || FLAGS_tuner_print_best) {
    LOG(INFO) << "[TUNER][ITERATION LOG] best option so far:";
    std::stringstream ssInfo;
    typename Backend::MappingOptionsCppPrinter infoPrinter(ssInfo);
    auto vBest = optionsCache_->getTopKOptions(
        lang::canonicalTc(tcTree_),
        makeTensorInfoVector(inputs_.begin()->second),
        makeTensorInfoVector(outputs_.begin()->second),
        Backend::backendString(),
        1);
    TC_CHECK_GT(vBest.size(), 0u);
    infoPrinter << vBest[0];
    LOG_LINE_BY_LINE(INFO, ssInfo);
  }
  searchStrategy.updateParameters();
}
} // namespace detail

namespace {
volatile std::sig_atomic_t sigint_ = 0;
volatile std::sig_atomic_t sigterm_ = 0;

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
  TC_CHECK_GE(inputs.size(), 1u);
  auto maxElement = std::max_element(
      inputs.begin(),
      inputs.end(),
      [](const DLConstTensor* a, const DLConstTensor* b) {
        return a->ndim < b->ndim;
      });
  return (*maxElement)->ndim;
}

// Creates well-chosen generic parameter sizes to match the input shapes.
template <typename MappingOptionsType>
inline std::pair<TuningConfiguration, std::vector<size_t>>
setupGenericTuningParametersAndGetRange(
    const std::vector<const DLConstTensor*>& inputs,
    const std::vector<MappingOptionsType>& baseMappings) {
  TC_CHECK_GE(inputs.size(), 1u);
  auto range = inputDivisorsAndPowers2(inputs);
  // 0 is a valid tiling annotation and signals no tiling of that dimension
  // 0 is not a valid block / grid annotation
  auto nTilesDim = largestDim(inputs) + 1;
  auto tileRange = range;
  tileRange.push_back(0);

  TuningConfiguration configuration;
  configuration.tilingParams.setRange(nTilesDim, tileRange);
  configuration.unrollFactor =
      RangeParameter(powers2(FLAGS_tuner_max_unroll_size), "unroll");

  return {configuration, range};
}

// Creates well-chosen parameter sizes to match the input shapes.
inline TuningConfiguration setupTuningParameters(
    const std::vector<const DLConstTensor*>& inputs,
    const std::vector<CudaMappingOptions>& baseMappings) {
  std::vector<size_t> range;
  TuningConfiguration configuration;
  std::tie(configuration, range) =
      setupGenericTuningParametersAndGetRange(inputs, baseMappings);
  auto blockRange = range;
  auto gridRange = range;

  for (const auto& baseMapping : baseMappings) {
    blockRange =
        mergeVectors(std::move(blockRange), baseMapping.block.extractVector());
    gridRange =
        mergeVectors(std::move(gridRange), baseMapping.grid.extractVector());
  }

  configuration.blockParams.setRange(blockRange, "b");
  configuration.gridParams.setRange(gridRange, "g");
  configuration.privateDepth =
      RangeParameter({0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10}, "pdepth");
  configuration.sharedDepth =
      RangeParameter({0, 1, 2, 3, 4, 5, 6, 7}, "sdepth");

  return configuration;
}

// Creates well-chosen parameter sizes to match the input shapes.
inline TuningConfiguration setupTuningParameters(
    const std::vector<const DLConstTensor*>& inputs,
    const std::vector<CpuMappingOptions>& baseMappings) {
  return setupGenericTuningParametersAndGetRange(inputs, baseMappings).first;
}
} // namespace

template <typename Backend, typename SearchStrategy>
Autotuner<Backend, SearchStrategy>::Autotuner()
    : optionsCache(new OptionsCache<Backend>()) {}

template <typename Backend, typename SearchStrategy>
std::vector<typename Backend::MappingOptionsType>
Autotuner<Backend, SearchStrategy>::tune(
    const std::string& tc,
    const std::string& tcEntryPoint,
    const std::unordered_map<size_t, std::vector<const DLConstTensor*>>& inputs,
    std::unordered_map<size_t, std::vector<const DLTensor*>>& outputs,
    const std::vector<typename Backend::MappingOptionsType>& baseMappings,
    size_t topK,
    const TuningParameterFixer& fixedParams) {
  std::map<std::string, lang::TreeRef> tcEntryPointMap(tc::detail::parse(tc));
  TC_CHECK_EQ(tcEntryPointMap.count(tcEntryPoint), 1u)
      << "Error looking up " << tcEntryPoint;

  // Do not emit lang warnings during the tuning run
  auto enableWarnings = lang::EnableWarnings(false);

  // Initialize a model configuration
  TC_CHECK_GE(inputs.size(), 1u);
  auto modelConfiguration =
      setupTuningParameters(inputs.begin()->second, baseMappings);
  modelConfiguration.fixParameters(fixedParams);

  // Create initial configs based on options + model configuration
  const std::vector<typename Backend::MappingOptionsType> options{baseMappings};
  std::vector<TuningConfiguration> configs;
  configs.reserve(options.size());
  std::transform(
      options.begin(),
      options.end(),
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
      tcEntryPointMap.at(tcEntryPoint),
      inputs,
      outputs,
      options[0],
      fixedParams,
      optionsCache);

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
      tuningHarnessThreadEx = std::current_exception();
    }
    tuningHarnessFinished = true;
  });
  while (not tuningHarnessFinished) {
    std::this_thread::sleep_for(std::chrono::milliseconds(100));
    if (sigint_) {
      tuningHarness.stopAfterCurrentIteration();
      break;
    }
    if (sigterm_) {
      std::cerr << "Autotuning aborted." << std::endl;
      std::abort();
    }
  }

  tuningHarnessThread.join();

  if (tuningHarnessThreadEx) {
    std::rethrow_exception(tuningHarnessThreadEx);
  }

  return optionsCache->getTopKOptions(
      lang::canonicalTc(tcEntryPointMap.at(tcEntryPoint)),
      makeTensorInfoVector(inputs.begin()->second),
      makeTensorInfoVector(outputs.begin()->second),
      Backend::backendString(),
      topK);
}
} // namespace autotune
} // namespace tc
