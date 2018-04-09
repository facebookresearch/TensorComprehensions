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

#include <atomic>
#include <csignal>
#include <memory>
#include <queue>
#include <string>
#include <unordered_map>
#include <vector>

#include "tc/autotuner/genetic_search.h"
#include "tc/autotuner/options_cache.h"
#include "tc/autotuner/parameters.h"
#include "tc/autotuner/utils.h"
#include "tc/core/tensor.h"
#include "tc/core/utils/time.h"
#include "tc/lang/parser.h"

namespace tc {
namespace autotune {

namespace detail {
/**
 * Internal harness to support multithreaded compilation and evaluation over
 * various backends for a particular SearchStrategy.
 * The SearchStrategy is a template parameter passed to the run method.
 */
template <typename Backend>
class TuningHarness {
 public:
  using ExecutorType = typename Backend::ExecutorType;
  using MappingOptionsType = typename Backend::MappingOptionsType;
  using OptionsCacheType = OptionsCache<Backend>;

  TuningHarness(
      size_t maxPopulationSize,
      lang::TreeRef tcTree,
      const std::unordered_map<size_t, std::vector<const DLConstTensor*>>&
          inputs,
      std::unordered_map<size_t, std::vector<const DLTensor*>>& outputs,
      const MappingOptionsType& baseMapping,
      const TuningParameterFixer& fixedParams,
      std::shared_ptr<OptionsCache<Backend>> optionsCache);

  /// Runs a SearchStrategy
  template <typename SearchStrategy>
  void run(SearchStrategy& searchStrategy);

  /// Once a signal has been caught, a flag is set and we terminate after the
  /// current iteration is finished. This is apparently necessary for proper
  /// python termination.
  /// TODO: we should detect when we come from python and exit properly in C++.
  void stopAfterCurrentIteration();

 private:
  /// Traverse one iteration of candidates in parallel and evaluate their
  /// runtimes
  template <typename SearchStrategy>
  void runOneIteration(SearchStrategy& searchStrategy, size_t iteration);

  /// Helper function to delegate compiling on the cpu to different threads
  template <typename SearchStrategy>
  void doCompile(SearchStrategy& searchStrategy);

  /// Helper function to delegate running the compiled code to different
  /// threads.
  /// This function must be specialized per Backend
  /// TODO: virtual inheritance?
  void doEvaluate(size_t device, size_t populationSize, Printer& printer);

  const MappingOptionsType& bestMappingOptions() {
    std::lock_guard<std::mutex> lock(bestTimeMutex_);
    return bestMappingOptions_;
  }

 private:
  /// Synchronization related objects
  /// The main invariant is that we always try to compile and evaluate
  /// exactly searchStrategy->population.size() candidates.
  /// If a candidate fails compilation we still add a null Executor so that
  /// the invariant holds.
  /// This way it is easy to implement multi-threaded termination by just
  /// taking an atomic counter and pushing/popping the queues under lock until
  /// we have evaluated searchStrategy->population.size() compilation results.
  std::mutex bestTimeMutex_;
  std::mutex executorsMutex_;
  std::atomic_bool stopRequested_;
  std::atomic_size_t currentCompilationJob_;
  std::atomic_size_t numEvaluations_;
  std::queue<std::unique_ptr<ExecutorType>> executors_;
  std::queue<CandidateConfiguration*> configurations_;

  /// inputs
  lang::TreeRef kTcTree_;
  const MappingOptionsType kBaseMapping_;
  /// maps of inputs and outputs per each device (represented by a size_t)
  /// involved in autotuning. The client of the autotuner API must allocate
  /// these properly on each device where autotuning evaluation needs to run.
  /// In particular all the inputs and outputs must contain the same values
  /// across devices for the purpose of running correctness checks during
  /// tuning (future work).
  const std::unordered_map<size_t, std::vector<const DLConstTensor*>> kInputs_;
  std::unordered_map<size_t, std::vector<const DLTensor*>> outputs_;

  // results
  size_t bestTime_;
  MappingOptionsType bestMappingOptions_;

  // backing options cache
  std::shared_ptr<OptionsCache<Backend>> optionsCache_;
};
} // namespace detail

/**
 * An Autotuner provides the basic interface to run a SearchStrategy over a
 * particular Backend.
 * Under the hood it instantiates a TuningHarness<Backend> and a
 * SearchStrategy. The Autotuner then calls TuningHarness<Backend>::run()
 * on the SearchStrategy.
 */
template <typename Backend, typename SearchStrategy>
class Autotuner {
 public:
  using MappingOptionsType = typename Backend::MappingOptionsType;
  using OptionsCacheType = tc::autotune::OptionsCache<Backend>;

  /// Parse and store the underlying tree representation for all the TC
  /// functions provided.
  explicit Autotuner(const std::string& tc);

  /// Runs autotuning on the TC function tcEntryPoint.
  /// This assumes input and output tensors of proper sizes have been
  /// initialized on each device. For now this initialization is left to the
  /// client of the Autotuner.
  /// Tuning performs a search of template type SearchStrategy starting from
  /// baseMapping.
  ///
  /// Alternatively an OptionsCache cacheFileName serialized path
  /// can be specified from which the tuner recovers multiple starting points
  /// and appends to the baseMapping. This can be useful in a reinforcement
  /// situation where short tunings are run and their results cached
  /// iteratively.
  ///
  /// Lastly a TuningParameterFixer function can be specified to limit the
  /// search space (i.e. when certain parameters are known to be good/bad
  /// independently on a particular TC).
  ///
  /// \return a vector MappingOptions, if it is empty then tuning did not find
  /// a single good configuration. This should be a very rare occurrence but
  /// it is possible in particular if the skipExecutionOrWarmup functin is too
  /// aggressive and the problem size is too small.
  std::vector<MappingOptionsType> tune(
      const std::string& tcEntryPoint,
      const std::unordered_map<size_t, std::vector<const DLConstTensor*>>&
          inputs,
      std::unordered_map<size_t, std::vector<const DLTensor*>>& outputs,
      const MappingOptionsType& baseMapping,
      const std::string& cacheFileName = "",
      const TuningParameterFixer& fixedParams = TuningParameterFixer());

 private:
  std::map<std::string, lang::TreeRef> tcEntryPointMap_;
  std::shared_ptr<OptionsCache<Backend>> optionsCache_;
};

/// Helper functions that need specializing for various backends.
/// The implementation is not set in stone but just a first approximation to
/// break the CUDA dependence.
///
/// Always worth noting, "specializations don't overload", don't ever overload
/// but only specialize the following functions. An alternative is to turn
/// those functions into static methods of the specific Backend type but since
/// they are only used for autotuning, it is unclear we want them to live in
/// the backend at this point.
namespace detail {
template <typename Backend>
typename Backend::MappingOptionsType makeOptions(
    const typename Backend::MappingOptionsType& baseMapping,
    const CandidateConfiguration& c);

template <typename Backend>
TuningConfiguration makeTuningConfiguration(
    const typename Backend::MappingOptionsType& options,
    const TuningConfiguration& configuration);

template <typename Backend>
void handleDeviceRuntimeError(
    size_t device,
    typename Backend::MappingOptionsType& options);

/// Helper function to get a kernel into benchmark-able state.
/// Some compiled kernels may result in catastrophically bad performance
/// (e.g. when there are too few total threads in the case of CUDA).
/// Information from polyhedral-JIT time can help detect some cases early.
/// This function is an opportunity to take advantage of this information
/// and it must be specialized for each Backend.
///
/// \return true if the execution should be skipped. If the function returns
/// false then warmup has occured for the kernel and it is ready to be
/// benchmarked.
template <typename Backend>
bool skipExecutionOrWarmup(
    typename Backend::ExecutorType& executor,
    const std::vector<const DLTensor*>& outputs,
    const std::vector<const DLConstTensor*>& inputs,
    size_t bestTimeSoFar);

template <typename Backend>
std::vector<size_t> parseDevices(const std::string& devices);
} // namespace detail
} // namespace autotune
} // namespace tc

#include "tc/autotuner/autotuner-inl.h"
