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

#include "tc/autotuner/genetic_search.h"

#include <algorithm>
#include <numeric>
#include <random>
#include <sstream>

#include "tc/autotuner/utils/utils.h"

namespace tc {
namespace autotune {

namespace {

template <typename Parameter, typename RNG>
void randomizeParameter(Parameter& param, RNG& rng) {
  auto paramIndex = std::uniform_int_distribution<size_t>(
      size_t(0), param.numberOptions() - 1)(rng);
  param.selectOption(paramIndex);
}

template <typename RNG>
void randomizePopulation(
    GeneticSearch::Population::iterator begin,
    GeneticSearch::Population::iterator end,
    RNG& rng) {
  for (auto candidate = begin; candidate != end; ++candidate) {
    auto& conf = (*candidate)->configuration;
    do {
      conf.applyToParameters(
          [&](ParameterView& p) { randomizeParameter(p, rng); });
    } while (!conf.isValid());
  }
}

template <typename RNG>
void mutate(
    CandidateConfiguration& candidate,
    double rate,
    int iterations,
    RNG& rng) {
  auto shouldMutate = [&]() -> bool {
    return std::discrete_distribution<int>{static_cast<double>(100 - rate),
                                           static_cast<double>(rate)}(rng);
  };

  CandidateConfiguration res(candidate);
  for (size_t i = 0; i < iterations; ++i) {
    res.configuration.applyToParameters([&](ParameterView& p) {
      if (not p.isForced() and shouldMutate()) {
        randomizeParameter(p, rng);
      }
    });

    if (res.configuration.isValid()) {
      candidate.configuration = res.configuration;
      return;
    }
    res.configuration = candidate.configuration;
  }
}

std::vector<double> computeNormalizedFitness(
    const GeneticSearch::Population& population) {
  std::vector<double> fitness;
  fitness.reserve(population.size());
  std::transform(
      population.begin(),
      population.end(),
      std::back_inserter(fitness),
      [](const std::unique_ptr<CandidateConfiguration>& c) {
        return 1.0 /
            std::chrono::duration_cast<std::chrono::microseconds>(c->runtime)
                .count();
      });
  sigmaScale(fitness);
  normalizeVector(fitness);
  return fitness;
}

// return the accumulated fitness of the sorted individuals:
// fitness[0] = fitness[0]
// fitness[1] = fitness[0] + fitness[1];
// fitness[2] = fitness[0] + fitness[1] + fitness[2];
//...
std::vector<double> computeAccumulatedFitness(
    const GeneticSearch::Population& population) {
  auto fitness = computeNormalizedFitness(population);
  std::vector<double> accFitness;
  accFitness.reserve(fitness.size());
  std::partial_sum(
      fitness.begin(), fitness.end(), std::back_inserter(accFitness));
  return accFitness;
}

void checkRuntimeRecorded(const Duration& d) {
  if (d == Duration::zero()) {
    throw std::invalid_argument{
        "All candidates must have a recorded runtime before \
the new parameters can be computed."};
  }
}

void dropInvalidConfigurations(GeneticSearch::Population& population) {
  population.erase(
      std::remove_if(
          population.begin(),
          population.end(),
          [](const std::unique_ptr<CandidateConfiguration>& c) {
            return c->invalid;
          }),
      population.end());
}

} // namespace

#define VALIDATE()                                     \
  CHECK_LT(kMaxPopulationSize, kMatingPoolSize);       \
  CHECK_LT(kMaxPopulationSize, kSelectionPoolSize);    \
  CHECK(kMutationRate >= 0 and kMutationRate <= 100)   \
      << "the mutation rate (" << kMutationRate        \
      << ") should be in the [0,100] interval";        \
  CHECK(kCrossOverRate >= 0 and kCrossOverRate <= 100) \
      << "the crossover (" << kCrossOverRate           \
      << ") rate should be in the [0,100] interval";

namespace {

template <typename RNG>
void restoreRngState(RNG& rng) {
  if (FLAGS_tuner_rng_restore.empty()) {
    LOG_IF(INFO, FLAGS_debug_tuner) << "RNG state " << rng;
  } else {
    std::istringstream ss(FLAGS_tuner_rng_restore);
    ss >> rng;
    LOG_IF(INFO, FLAGS_debug_tuner) << "RNG restored state " << rng;
  }
}
} // namespace

GeneticSearch::GeneticSearch(
    const std::vector<TuningConfiguration>& confs,
    size_t n,
    uint8_t crossOverRate,
    uint8_t mutationRate,
    size_t matingPoolSize,
    size_t selectionPoolSize)
    : population(),
      lastBestConf(confs[0]),
      kMaxPopulationSize(n),
      kMatingPoolSize(matingPoolSize),
      kSelectionPoolSize(selectionPoolSize),
      kCrossOverRate(crossOverRate),
      kMutationRate(mutationRate),
      rng{std::random_device{}()} {
  restoreRngState(rng);
  VALIDATE();
  CHECK(not confs.empty()) << "empty set of predefined configurations";
  CHECK_LE(confs.size(), n) << "too many predefined configurations";

  population.reserve(confs.size());
  for (auto& c : confs) {
    population.push_back(make_unique<CandidateConfiguration>(c));
  }
  if (kMaxPopulationSize - population.size() > 0) {
    auto oldSize = population.size();
    for (int i = oldSize; i < kMaxPopulationSize; ++i) {
      population.emplace_back(
          make_unique<CandidateConfiguration>(*population.front()));
    }
    randomizePopulation(population.begin() + oldSize, population.end(), rng);
  }
}

GeneticSearch::GeneticSearch(
    const TuningConfiguration& conf,
    size_t n,
    uint8_t crossOverRate,
    uint8_t mutationRate,
    size_t matingPoolSize,
    size_t selectionPoolSize)
    : population(),
      lastBestConf(conf),
      kMaxPopulationSize(n),
      kMatingPoolSize(matingPoolSize),
      kSelectionPoolSize(selectionPoolSize),
      kCrossOverRate(crossOverRate),
      kMutationRate(mutationRate),
      rng{std::random_device{}()} {
  restoreRngState(rng);
  VALIDATE();
  for (int i = 0; i < kMaxPopulationSize; ++i) {
    population.emplace_back(make_unique<CandidateConfiguration>(conf));
  }
  randomizePopulation(population.begin(), population.end(), rng);
}

TuningConfiguration GeneticSearch::crossover(
    TuningConfiguration& a,
    TuningConfiguration& b,
    TuningConfiguration& c) const {
  auto aParams = a.collectParameters();
  auto bParams = b.collectParameters();
  auto cParams = c.collectParameters();
  auto selectParam = [&](const ParameterView& a,
                         const ParameterView& b,
                         const ParameterView& c) {
    switch (std::uniform_int_distribution<size_t>{0, 2}(rng)) {
      case 0:
        return a;
      case 1:
        return b;
      case 2:
        return c;
      default:
        throw std::runtime_error{"Unknown value."};
    }
  };

  for (size_t i = 0; i < kMutateIterations; ++i) {
    TuningConfiguration child{a};
    auto params = child.collectParameters();
    for (size_t i = 0; i < params.size(); ++i) {
      params.at(i).overwrite(
          selectParam(aParams.at(i), bParams.at(i), cParams.at(i)));
    }
    if (child.isValid()) {
      return child;
    }
  }

  // didn't manage to create a valid child so just return a
  return a;
}

std::vector<TuningConfiguration> GeneticSearch::stochasticUniversalSampling(
    const std::vector<double>& fitness) const {
  std::vector<TuningConfiguration> matingPool;
  matingPool.reserve(kMatingPoolSize);

  auto r =
      std::uniform_real_distribution<double>(0, 1.0 / kMatingPoolSize)(rng);
  size_t count = 0;
  size_t i = 0;
  while (count < kMatingPoolSize) {
    while (r <= fitness[i]) {
      matingPool.push_back(population[i]->configuration);
      r += 1.0 / kMatingPoolSize;
      ++count;
    }
    ++i;
  }
  return matingPool;
}

void GeneticSearch::breed() {
  auto matingPool =
      stochasticUniversalSampling(computeAccumulatedFitness(population));

  auto select = [&]() -> TuningConfiguration& {
    auto idx = std::uniform_int_distribution<size_t>{
        size_t(0), matingPool.size() - 1}(rng);
    return matingPool.at(idx);
  };
  auto shouldCrossOver = [&]() -> bool {
    /*
     *Crossover should occur with probability (kCrossOverRate)%
     */
    auto dist = std::discrete_distribution<int>{
        static_cast<double>(100 - kCrossOverRate),
        static_cast<double>(kCrossOverRate)};
    return dist(rng);
  };

  while (selectionPool.size() < kSelectionPoolSize) {
    if (shouldCrossOver()) {
      auto parent1 = select();
      auto parent2 = select();
      auto parent3 = select();
      selectionPool.emplace_back(make_unique<CandidateConfiguration>(
          crossover(parent1, parent2, parent3)));
    } else {
      selectionPool.emplace_back(make_unique<CandidateConfiguration>(select()));
    }
  }
}

void GeneticSearch::resetPopulationIfNotEnoughCandidates() {
  if (population.size() < kMinCandidatesForBreeding) {
    LOG_IF(ERROR, FLAGS_debug_tuner)
        << population.size() << " out of " << kMaxPopulationSize
        << " candidates were valid and are not enough to form a new "
           "generation. Likely, most of the tuning runs during this "
           "generation were pruned for lack of parallelism in the "
           "generated code. You can relax this constraints by setting "
           "--tuner_min_launch_total_threads=1. This is mostly relevant "
           "when autotuning a TC operating on small tensors. The next "
           "generation will be randomly initialized.";
    population.resize(0);
    for (int i = 0; i < kMaxPopulationSize; ++i) {
      population.emplace_back(
          make_unique<CandidateConfiguration>(lastBestConf));
    }
    // Don't lose the first one which was the best from before
    CHECK_LT(0, population.size());
    randomizePopulation(population.begin() + 1, population.end(), rng);
  }
}

namespace {
void sortByRuntime(GeneticSearch::Population& population) {
  std::sort(
      population.begin(),
      population.end(),
      [](const std::unique_ptr<CandidateConfiguration>& a,
         const std::unique_ptr<CandidateConfiguration>& b) {
        checkRuntimeRecorded(a->runtime);
        checkRuntimeRecorded(b->runtime);
        return a->runtime < b->runtime;
      });
}
} // namespace

void GeneticSearch::updateBestCandidate(const TuningConfiguration& c) {
  lastBestConf = c;
  if (FLAGS_tuner_print_best) {
    CudaMappingOptions options(
        CudaMappingOptions::makeSingleThreadCudaMappingOptions());
    lastBestConf.applyToCudaMappingOptions(options);
    LOG(INFO) << "Best so far:\n" << options;
  }
}

void GeneticSearch::generateSelectionPool() {
  dropInvalidConfigurations(population);
  sortByRuntime(population);
  updateBestCandidate(
      population.size() > 0 ? population.front()->configuration : lastBestConf);
  resetPopulationIfNotEnoughCandidates();
  breed();
  selectionPool.clear();
  selectionPool.emplace_back(make_unique<CandidateConfiguration>(lastBestConf));
  breed();
  for (size_t i = 1; i < selectionPool.size(); ++i) {
    mutate(*selectionPool[i], kMutationRate, kMutateIterations, rng);
  }
}

void GeneticSearch::selectSurvivors() {
  dropInvalidConfigurations(selectionPool);
  sortByRuntime(selectionPool);
  population.clear();
  std::transform(
      selectionPool.begin(),
      selectionPool.begin() +
          std::min(selectionPool.size(), kMaxPopulationSize),
      std::back_inserter(population),
      [](const std::unique_ptr<CandidateConfiguration>& c) {
        CHECK(c);
        return make_unique<CandidateConfiguration>(*c);
      });

  if (selectionPool.size() < kMaxPopulationSize) {
    auto numberMissing = kMaxPopulationSize - selectionPool.size();

    for (size_t i = 0; i < numberMissing; ++i) {
      selectionPool.emplace_back(
          make_unique<CandidateConfiguration>(lastBestConf));
    }
    randomizePopulation(
        selectionPool.end() - numberMissing, selectionPool.end(), rng);
  }
}

} // namespace autotune
} // namespace tc

#undef VALIDATE
