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

#include "tc/autotuner/genetic_tuning_harness.h"

#include <atomic>
#include <chrono>
#include <numeric>
#include <thread>

#include <cuda_runtime_api.h>
#include <glog/stl_logging.h>

#include "tc/autotuner/utils/printer.h"
#include "tc/autotuner/utils/utils.h"
#include "tc/core/cuda/cuda.h"
#include "tc/core/cuda/cuda_compilation_cache.h"
#include "tc/core/cuda/cuda_mapping_options_cpp_printer.h"
#include "tc/core/cuda/cuda_tc_executor.h"
#include "tc/core/execution_engine.h"
#include "tc/core/flags.h"
#include "tc/core/polyhedral/cuda/mapping_types.h"
#include "tc/core/scope_guard.h"
#include "tc/core/utils/math.h"

namespace tc {
namespace autotune {
namespace detail {

GeneticTunerHarness::GeneticTunerHarness(
    size_t n,
    uint8_t crossoverRate,
    uint8_t mutationRate,
    size_t matingPoolSize,
    size_t selectionPoolSize,
    lang::TreeRef tc,
    std::string kernelName,
    const std::unordered_map<size_t, std::vector<const DLTensor*>>& inputs,
    std::unordered_map<size_t, std::vector<DLTensor*>>& outputs,
    CudaMappingOptions baseMapping,
    std::vector<CudaMappingOptions> startingPoints,
    const TuningParameterFixer& fixedParams)
    : kMaxPopulationSize(n),
      kCrossOverRate(crossoverRate),
      kMutationRate(mutationRate),
      kMatingPoolSize(matingPoolSize),
      kSelectionPoolSize(selectionPoolSize),
      bestCudaMappingOptions_(baseMapping),
      kTc_(std::move(tc)),
      kKernelName_(std::move(kernelName)),
      currentCompilationJob_(0),
      readyToEvaluate_(),
      numEvaluations_(0),
      kInputs_(std::move(inputs)),
      outputs_(std::move(outputs)),
      kBaseMapping_(std::move(baseMapping)),
      kStartingPoints_(std::move(startingPoints)) {
  setupTuningParameters();
  configuration.fixParameters(fixedParams);
  if (not kStartingPoints_.empty()) {
    std::vector<TuningConfiguration> configs;
    configs.reserve(kStartingPoints_.size());
    std::transform(
        kStartingPoints_.begin(),
        kStartingPoints_.end(),
        std::back_inserter(configs),
        [this, &fixedParams](const CudaMappingOptions& options) {
          auto config = makeTuningConfiguration(options);
          config.fixParameters(fixedParams);
          return config;
        });
    tuner_ = make_unique<GeneticSearch>(
        configs,
        kMaxPopulationSize,
        kCrossOverRate,
        kMutationRate,
        kMatingPoolSize,
        kSelectionPoolSize);
  } else {
    tuner_ = make_unique<GeneticSearch>(
        configuration,
        kMaxPopulationSize,
        kCrossOverRate,
        kMutationRate,
        kMatingPoolSize,
        kSelectionPoolSize);
  }
}

void GeneticTunerHarness::run(size_t numGenerations) {
  for (size_t i = 0; i < numGenerations; ++i) {
    if (not stopRequested_) {
      runOneGeneration(i);
    }
  }
}

void GeneticTunerHarness::stopAfterCurrentGeneration() {
  stopRequested_ = true;
}

namespace {

std::vector<size_t> filterHigherThan(
    const std::vector<size_t>& v,
    size_t limit) {
  std::vector<size_t> newV;
  std::copy_if(
      v.begin(), v.end(), std::back_inserter(newV), [limit](size_t val) {
        return val <= limit;
      });
  return newV;
}

void removeDuplicates(std::vector<size_t>& v) {
  std::sort(v.begin(), v.end());
  v.erase(std::unique(v.begin(), v.end()), v.end());
}

std::vector<size_t> inputDivisorsAndPowers2(
    const std::vector<const DLTensor*>& inputs) {
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

size_t largestDim(const std::vector<const DLTensor*>& inputs) {
  CHECK_GE(inputs.size(), 0);
  auto maxElement = std::max_element(
      inputs.begin(), inputs.end(), [](const DLTensor* a, const DLTensor* b) {
        return a->ndim < b->ndim;
      });
  return (*maxElement)->ndim;
}

} // namespace

void GeneticTunerHarness::setupTuningParameters() {
  CHECK_GT(kInputs_.size(), 0);
  auto range = inputDivisorsAndPowers2(kInputs_.begin()->second);
  auto rangeUpTo64 = filterHigherThan(range, 64);

  // 0 is a valid tiling annotation and signals no tiling of that dimension
  // 0 is not a valid block / grid annotation
  auto nTilesDim =
      largestDim(kInputs_.begin()->second) + 1; // TODO [ntv]: change me
  auto tileRange = range;
  tileRange.push_back(0);
  configuration.tilingParams.setRange(nTilesDim, range);

  configuration.blockParams.setRange(range, "b");
  configuration.gridParams.setRange(range, "g");

  configuration.unrollFactor =
      RangeParameter({1, 2, 4, 8, 16, 32, 64, 128, 256}, "unroll");
}

CudaMappingOptions GeneticTunerHarness::makeOptions(
    const CandidateConfiguration& c) {
  auto options = kBaseMapping_;
  c.configuration.applyToCudaMappingOptions(options);
  return options;
}

TuningConfiguration GeneticTunerHarness::makeTuningConfiguration(
    const CudaMappingOptions& options) {
  TuningConfiguration conf = configuration;
  conf.fromCudaMappingOptions(options);
  return conf;
}

std::vector<size_t> parseGpus() {
  std::stringstream ss(FLAGS_tuner_gpus);
  size_t gpu;
  std::vector<size_t> res;
  while (ss >> gpu) {
    res.push_back(gpu);
    if (ss.peek() == ',') {
      ss.ignore();
    }
  }
  return res;
}

#define LOG_LINE_BY_LINE(GSTREAM, ISTREAM)               \
  for (std::string line; std::getline(ISTREAM, line);) { \
    LOG(GSTREAM) << line;                                \
  }

// This function is ran on a single pre-determined GPU, in a single thread
// It takes the input/output DLTensor objects that reside on that GPU
//
// We pass the bestTimeSoFar as an option to avoid taking locks in this
// function. This trades off a bit of conservativeness for code sanity.
//
// The function returns true if purning is possible and we can skip poorly
// performing versions early.
template <typename ExecutorType>
bool GeneticTunerHarness::warmupOrPrune(
    ExecutorType& engine,
    const std::vector<DLTensor*>& outputs,
    const std::vector<const DLTensor*>& inputs,
    size_t handle,
    size_t bestTimeSoFar) {
  // Pruning based on number of threads: if you don't hit at least k warps
  // (default k = 8; 256 total threads, controlled by
  // FLAGS_tuner_min_launch_total_threads) then it's likely the kernel is not
  // performing great.
  // This may be completely off but is a good first initial rule of thumb
  // for stress-testing autotuning.
  //
  // The launch bounds information is only available after compilation and is
  // task-local. We pass a callback to determine whether to prune or not.
  auto debugTuner = FLAGS_debug_tuner;
  auto minThreads = FLAGS_tuner_min_launch_total_threads;
  auto threadPruningFunction = std::function<bool(const TcExecutor*)>(
      [debugTuner, minThreads](const TcExecutor* exec) {
        CHECK(exec);
        USING_MAPPING_SHORT_NAMES(BX, BY, BZ, TX, TY, TZ);
        auto block = static_cast<const CudaTcExecutor*>(exec)->block;
        auto nThreads = TX.mappingSize(block) * TY.mappingSize(block) *
            TZ.mappingSize(block);
        auto grid = static_cast<const CudaTcExecutor*>(exec)->grid;
        auto nBlocks =
            BX.mappingSize(grid) * BY.mappingSize(grid) * BZ.mappingSize(grid);
        if (nBlocks * nThreads < minThreads) {
          if (debugTuner) {
            std::stringstream ssInfo;
            ssInfo << "Skip configuration with too few threads: " << block
                   << "\n"
                   << CudaMappingOptionsAsCpp(
                          CudaMappingOptions(exec->options));
            LOG_LINE_BY_LINE(INFO, ssInfo);
          }
          return true;
        } else {
          LOG_IF(INFO, debugTuner)
              << "Run configuration launch bounds blocks: " << grid
              << " and threads: " << block << "\n";
        }
        return false;
      });

  // 1. Perform a first run which may have one of 3 behaviors:
  //   1.a. return Duration::max(), which means that pruning should occur,
  //   1.b. return a very slow first execution time, we should stop
  //     early. This is akin to pruning but in this case we have run once,
  //   1.c. return a reasonable execution time, in which case we proceed with
  //     warmup.
  auto prof = engine.run(handle, inputs, outputs, true, threadPruningFunction);

  // 1.a.
  if (prof == Duration::max()) {
    return true;
  }

  // 1.b.
  constexpr size_t kCatastrophicPerfFactor = 100;
  if (bestTimeSoFar < std::numeric_limits<size_t>::max() and
      prof >= std::chrono::microseconds(
                  (kCatastrophicPerfFactor * bestTimeSoFar))) {
    return true;
  }

  // 1.c.
  for (size_t i = 1; i < kReducedWarmupIterations - 1; ++i) {
    engine.run(handle, inputs, outputs, true);
  }

  // 2. After reasonable warmup, look at the performance and prune with
  // kEarlyPruneFactor
  prof = engine.run(handle, inputs, outputs, true);
  if (bestTimeSoFar < std::numeric_limits<size_t>::max() and
      prof >= std::chrono::microseconds((kEarlyPruneFactor * bestTimeSoFar))) {
    return true;
  }

  // 3. If we get here then the kernel is good to be benchmarked
  return false;
}

template <typename ExecutorType, typename Population>
void GeneticTunerHarness::doCompile(
    ExecutorType& engine,
    Population& population) {
  // Atomically fetch and add the next job until there are no jobs left
  while (true) {
    auto current = currentCompilationJob_.fetch_add(1);
    if (current >= population.size()) {
      break;
    }

    auto& pConf = population.at(current);
    auto options = makeOptions(*pConf);
    try {
      if (FLAGS_debug_tuner) {
        std::stringstream ssInfo;
        CudaMappingOptionsCppPrinter infoPrinter(ssInfo);
        infoPrinter << options;
        LOG(INFO) << "[COMPILE] Start compilation @:" << current;
        LOG_LINE_BY_LINE(INFO, ssInfo);
      }
      auto handle = engine.compile(
          kKernelName_,
          kInputs_.begin()->second,
          options.toProtobufSerializedString());
      LOG_IF(INFO, FLAGS_debug_tuner)
          << "[COMPILE] Done compilation, got handle: " << handle;
      pConf->optionalCompilationHandle =
          std::unique_ptr<size_t>(new size_t(handle));
    } catch (const std::exception& e) {
      LOG(WARNING) << "[TUNER][COMPILE] failed compilation: " << e.what();
      std::stringstream ssWarning;
      CudaMappingOptionsCppPrinter warningPrinter(ssWarning);
      warningPrinter << options;
      LOG_LINE_BY_LINE(WARNING, ssWarning);
      pConf->invalid = true;
    }
    CHECK(pConf->invalid || pConf->optionalCompilationHandle)
        << "GPU kernel not compiled";
    readyToEvaluate_[current].store(true);
  }
}

namespace {
std::vector<const DLTensor*> toConstDlpackTensors(
    const std::vector<DLTensor*>& v) {
  std::vector<const DLTensor*> out(v.begin(), v.end());
  return out;
}
} // namespace

template <typename ExecutorType>
std::vector<Duration> retrieveCachedRuntimes(
    ExecutorType& engine,
    const std::string& id,
    const std::vector<const DLTensor*>& inputs,
    const std::vector<DLTensor*>& outputs,
    const CudaMappingOptions& options) {
  if (not OptionsCache::cacheEnabled()) {
    return {};
  }
  auto cache = OptionsCache::getCache();
  auto allResults = cache->retrieveOptionsAndRuntimes(
      id, inputs, toConstDlpackTensors(outputs));
  auto wantedResult = std::find_if(
      allResults.begin(),
      allResults.end(),
      [&options](const OptionsCache::RetrievalResult& r) {
        return r.options == options;
      });
  if (wantedResult == allResults.end()) {
    return {};
  }
  return wantedResult->recordedRuntimes;
}

template <typename ExecutorType, typename Population>
void GeneticTunerHarness::doGpuWork(
    size_t gpu,
    ExecutorType& engine,
    Population& population,
    Printer& printer) {
  WithDevice wd(gpu);
  CHECK_EQ(1, kInputs_.count(gpu));
  auto& inputs = kInputs_.at(gpu);
  CHECK_EQ(1, outputs_.count(gpu));
  auto& outputs = outputs_.at(gpu);

  while (true) {
    bool found = false;
    size_t current = 0;
    // Lock-free traversal of readyToEvaluate, if I find one I take it
    for (size_t i = 0; i < readyToEvaluate_.size(); ++i) {
      bool expected = true;
      found = readyToEvaluate_[i].compare_exchange_strong(expected, false);
      if (found) {
        current = i;
        break;
      }
    }
    if (found) {
      // Found work to do, increment number of evaluations performed
      numEvaluations_.fetch_add(1);
    } else {
      if (numEvaluations_.load() >= population.size()) {
        // No more work can arrive, exit
        return;
      }
      // More work will arrive, loop.
      // TODO: Prob need some delaying mechanism to reduce contention
      continue;
    }

    auto& pConf = population.at(current);
    if (pConf->invalid) {
      continue;
    }
    CHECK(pConf->optionalCompilationHandle) << "GPU kernel not compiled";
    auto handle = *(pConf->optionalCompilationHandle);
    auto options = makeOptions(*pConf);
    if (FLAGS_debug_tuner) {
      // Always log option to INFO so we can correlate to timing offling
      std::stringstream ssInfo;
      ssInfo << "Launch GPU kernel on gpu: " << std::to_string(gpu)
             << " handle: " << std::to_string(handle) << " options:\n"
             << CudaMappingOptionsAsCpp(options);
      LOG_LINE_BY_LINE(INFO, ssInfo);
    }

    std::vector<Duration> runtimes =
        retrieveCachedRuntimes(engine, kKernelName_, inputs, outputs, options);
    if (runtimes.empty()) {
      try {
        size_t bestTimeSoFar;
        {
          std::lock_guard<std::mutex> lock(bestTimeMtx_);
          bestTimeSoFar = bestTime_;
        }
        auto prune =
            warmupOrPrune(engine, outputs, inputs, handle, bestTimeSoFar);
        if (prune) {
          pConf->invalid = true;
          continue;
        } else {
          runtimes.reserve(kReducedBenchmarkIterations);
          for (size_t i = 0; i < kReducedBenchmarkIterations; ++i) {
            runtimes.push_back(engine.run(handle, inputs, outputs, true));
          }
          engine.clear(handle);
        }
      } catch (std::exception& e) {
        if (FLAGS_debug_tuner) {
          LOG(WARNING) << "Runtime error gpu " << gpu << ": " << e.what();
          std::stringstream ssWarning;
          CudaMappingOptionsCppPrinter warningPrinter(ssWarning);
          warningPrinter << options;
          LOG(WARNING) << "Aborted execution on gpu " << gpu;
          LOG_LINE_BY_LINE(WARNING, ssWarning);
        }
        while (cudaGetLastError() != cudaSuccess) {
          // In case of errors in the generated, we cannot rely on deviceReset
          // to set the GPU in a clean state. So instead we just pop and discard
          // all the errors accumulated on the GPU until we get to a clean slate
          // (i.e. cudaSuccess).
          ;
        }
        try {
          // Some errors, such as illegal memory access, cannot be recovered
          // from without a cudaDeviceReset (i.e. because user protection) In
          // those cases we have no choice than to fail hard.
          TC_CUDA_RUNTIMEAPI_ENFORCE(cudaDeviceSynchronize());
        } catch (const std::exception& e) {
          LOG(FATAL) << "[CUDA][FATAL] cuda error on gpu " << gpu << ": "
                     << e.what() << "\n"
                     << CudaMappingOptionsAsCpp(options);
        }
        pConf->invalid = true;
        continue;
      }
    }

    auto prof = median(runtimes);
    auto prof_us =
        std::chrono::duration_cast<std::chrono::microseconds>(prof).count();

    LOG_IF(INFO, tc::FLAGS_debug_tuner)
        << "Run on gpu " << gpu << " took: " << prof_us << "us";
    printer.record(prof);
    pConf->runtime = prof;

    // Save best time under lock
    {
      std::lock_guard<std::mutex> lock(bestTimeMtx_);
      if (prof_us < bestTime_) {
        bestTime_ = prof_us;
        bestCudaMappingOptions_ = options;
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
                 << CudaMappingOptionsAsCpp(options);
    }
  } // end while
}

void GeneticTunerHarness::runOneGeneration(size_t generation) {
  // Define tensors per GPU once globally
  auto gpus = parseGpus();
  tc::ExecutionEngine<tc::CudaTcExecutor> engine;
  engine.define({kTc_});

  auto setUpJobsAndRun = [&](GeneticSearch::Population& population,
                             const std::string& printerText) {
    // Most candidates should have been evaluated during the previous
    // generation's selection phase.
    // There are two exceptions:
    // 1) the 1st generation
    // 2) too many invalid configurations were previously encounted and the
    //    valid ones were not enough to form a new generation.
    auto firstNew = std::partition(
        population.begin(),
        population.end(),
        [](const std::unique_ptr<CandidateConfiguration>& c) {
          return c->runtime != Duration::zero();
        });
    if (std::distance(firstNew, population.end()) == 0) {
      return;
    }
    GeneticSearch::Population newCandidates(
        std::distance(firstNew, population.end()));
    std::move(firstNew, population.end(), newCandidates.begin());
    {
      // Initialize for this round
      currentCompilationJob_.store(0);
      numEvaluations_.store(0);
      readyToEvaluate_.resize(0);
      for (size_t i = 0; i < newCandidates.size(); ++i) {
        readyToEvaluate_.emplace_back();
        readyToEvaluate_[i].store(false);
      }
      Printer printer(
          printerText,
          readyToEvaluate_.size(),
          currentCompilationJob_,
          numEvaluations_);
      auto logGenerations = FLAGS_tuner_gen_log_generations;
      ScopeGuard sgPrinter([logGenerations, &printer]() {
        printer.stop();
        if (logGenerations) {
          printer.printAll();
        }
      });

      // Just spawn and join new threads for each generation
      std::vector<std::thread> cpuCompilationThreads;
      cpuCompilationThreads.reserve(FLAGS_tuner_threads);
      ScopeGuard sgCompilationThreads([&cpuCompilationThreads]() {
        for (auto& cpuCompilationThread : cpuCompilationThreads) {
          cpuCompilationThread.join();
        }
      });
      for (int i = 0; i < FLAGS_tuner_threads; ++i) {
        cpuCompilationThreads.emplace_back([this, &engine, &newCandidates]() {
          this->doCompile(engine, newCandidates);
        });
      }

      // Just spawn and join new threads for each generation
      std::vector<std::thread> gpuWorkerThreads;
      gpuWorkerThreads.reserve(gpus.size());
      ScopeGuard sgGpuWorkerThreads([&gpuWorkerThreads]() {
        for (auto& gpuWorkerThread : gpuWorkerThreads) {
          gpuWorkerThread.join();
        }
      });
      for (auto gpu : gpus) {
        gpuWorkerThreads.emplace_back(
            [this, gpu, &engine, &newCandidates, &printer]() {
              this->doGpuWork(gpu, engine, newCandidates, printer);
            });
      }
    }
    // At this point everything is synchronized because out of scope, done
    std::move(newCandidates.begin(), newCandidates.end(), firstNew);
  };
  std::cout << "Generation " << generation << ':' << std::endl;
  setUpJobsAndRun(tuner_->population, "New Candidates");
  tuner_->generateSelectionPool();
  setUpJobsAndRun(tuner_->selectionPool, "Selection Pool");
  tuner_->selectSurvivors();

  if (FLAGS_debug_tuner) {
    LOG(INFO) << "[TUNER][GENERATION LOG] best option so far:";
    std::stringstream ssInfo;
    CudaMappingOptionsCppPrinter infoPrinter(ssInfo);
    infoPrinter << bestMappingOption();
    LOG_LINE_BY_LINE(INFO, ssInfo);
  }
}

} // namespace detail
} // namespace autotune
} // namespace tc
