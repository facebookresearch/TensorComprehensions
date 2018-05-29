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

namespace tc {
namespace autotune {

namespace {

template <typename Parameter, typename RNG>
void randomizeParameter(Parameter& param, RNG& rng) {
  auto paramIndex = std::uniform_int_distribution<size_t>(
      size_t(0), param.numberOptions() - 1)(rng);
  param.selectOption(paramIndex);
}

template <typename RNG, typename Iterator>
void randomizePopulation(Iterator begin, Iterator end, RNG& rng) {
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
    size_t iterations,
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

double mean(std::vector<double>& v) {
  if (v.empty()) {
    throw std::invalid_argument("Cannot compute the mean of an empty vector.");
  }
  auto sum = std::accumulate(v.begin(), v.end(), 0.0);
  return sum / v.size();
}

double stdv(std::vector<double>& v, double mean) {
  std::vector<double> diffs(v.size());
  std::transform(v.begin(), v.end(), diffs.begin(), [mean](double val) {
    return val - mean;
  });

  auto squareSum =
      std::inner_product(diffs.begin(), diffs.end(), diffs.begin(), 0.0);
  return std::sqrt(squareSum / v.size());
}

void sigmaScale(std::vector<double>& v) {
  auto m = mean(v);
  auto s = stdv(v, m);
  std::transform(v.begin(), v.end(), v.begin(), [m, s](double val) {
    return std::max(val - (m - 2 * s), 0.0);
  });
}

void normalizeVector(std::vector<double>& v) {
  auto sum = std::accumulate(v.begin(), v.end(), 0.0);
  std::transform(
      v.begin(), v.end(), v.begin(), [sum](double v) { return v / sum; });
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
        return 1.0 / c->runtime.toMicroSeconds();
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

#define VALIDATE()                                   \
  CHECK_LT(maxPopulationSize, matingPoolSize);       \
  CHECK_LT(maxPopulationSize, selectionPoolSize);    \
  CHECK(mutationRate >= 0 and mutationRate <= 100)   \
      << "the mutation rate (" << mutationRate       \
      << ") should be in the [0,100] interval";      \
  CHECK(crossOverRate >= 0 and crossOverRate <= 100) \
      << "the crossover (" << crossOverRate          \
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
    size_t numGenerations,
    size_t populationSize,
    uint8_t crossOverRate,
    uint8_t mutationRate,
    size_t matingPoolSize,
    size_t selectionPoolSize)
    : population(),
      lastBestConf(confs[0]),
      numGenerations(numGenerations),
      maxPopulationSize(populationSize),
      matingPoolSize(matingPoolSize),
      selectionPoolSize(selectionPoolSize),
      crossOverRate(crossOverRate),
      mutationRate(mutationRate),
      rng{std::random_device{}()} {
  restoreRngState(rng);
  VALIDATE();
  CHECK(not confs.empty()) << "empty set of predefined configurations";

  population.reserve(populationSize);
  size_t size = 0;
  for (; size < confs.size() && size < maxPopulationSize; ++size) {
    population.push_back(make_unique<CandidateConfiguration>(confs[size]));
  }
  size_t oldSize = size;
  for (; size < maxPopulationSize; ++size) {
    population.emplace_back(
        make_unique<CandidateConfiguration>(*population.front()));
  }
  randomizePopulation(population.begin() + oldSize, population.end(), rng);
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

  for (size_t iter = 0; iter < mutateIterations; ++iter) {
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
  matingPool.reserve(matingPoolSize);

  auto r = std::uniform_real_distribution<double>(0, 1.0 / matingPoolSize)(rng);
  size_t count = 0;
  size_t i = 0;
  while (count < matingPoolSize) {
    while (r <= fitness[i]) {
      matingPool.push_back(population[i]->configuration);
      r += 1.0 / matingPoolSize;
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
     *Crossover should occur with probability (crossOverRate)%
     */
    auto dist = std::discrete_distribution<int>{
        static_cast<double>(100 - crossOverRate),
        static_cast<double>(crossOverRate)};
    return dist(rng);
  };

  while (selectionPool.size() < selectionPoolSize) {
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

bool GeneticSearch::resetPopulationIfNotEnoughCandidates() {
  if (population.size() < minCandidatesForBreeding) {
    LOG_IF(ERROR, FLAGS_debug_tuner)
        << population.size() << " out of " << maxPopulationSize
        << " candidates were valid and are not enough to form a new "
           "generation. Likely, most of the tuning runs during this "
           "generation were pruned for lack of parallelism in the "
           "generated code. You can relax this constraints by setting "
           "--tuner_min_launch_total_threads=1. This is mostly relevant "
           "when autotuning a TC operating on small tensors. The next "
           "generation will be randomly initialized.";
    selectionPool.clear();
    for (size_t i = 0; i < selectionPoolSize; ++i) {
      selectionPool.emplace_back(
          make_unique<CandidateConfiguration>(lastBestConf));
    }
    // Don't lose the first one which was the best from before
    randomizePopulation(selectionPool.begin() + 1, selectionPool.end(), rng);
    return true;
  }
  return false;
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

void GeneticSearch::generateSelectionPool() {
  dropInvalidConfigurations(population);
  sortByRuntime(population);
  lastBestConf =
      population.size() > 0 ? population.front()->configuration : lastBestConf;
  if (resetPopulationIfNotEnoughCandidates()) {
    return;
  }
  selectionPool.clear();
  selectionPool.emplace_back(make_unique<CandidateConfiguration>(lastBestConf));
  breed();
  for (size_t i = 1; i < selectionPool.size(); ++i) {
    mutate(*selectionPool[i], mutationRate, mutateIterations, rng);
  }
}

void GeneticSearch::selectSurvivors() {
  dropInvalidConfigurations(selectionPool);
  sortByRuntime(selectionPool);
  population.clear();
  std::transform(
      selectionPool.begin(),
      selectionPool.begin() + std::min(selectionPool.size(), maxPopulationSize),
      std::back_inserter(population),
      [](const std::unique_ptr<CandidateConfiguration>& c) {
        CHECK(c);
        return make_unique<CandidateConfiguration>(*c);
      });

  if (selectionPool.size() < maxPopulationSize) {
    auto numberMissing = maxPopulationSize - selectionPool.size();

    for (size_t i = 0; i < numberMissing; ++i) {
      selectionPool.emplace_back(
          make_unique<CandidateConfiguration>(lastBestConf));
    }
    randomizePopulation(
        selectionPool.rbegin(), selectionPool.rbegin() + numberMissing, rng);
  }
}

GeneticSearch::Population& GeneticSearch::candidatesOfStep(uint64_t step) {
  if (step > 1) {
    throw std::invalid_argument("GeneticSearch has only 2 steps.");
  }
  if (step == 0) {
    return population;
  } else {
    return selectionPool;
  }
}

void GeneticSearch::finishStep(uint64_t step) {
  if (step > 1) {
    throw std::invalid_argument("GeneticSearch has only 2 steps.");
  }
  if (step == 0) {
    generateSelectionPool();
  } else {
    selectSurvivors();
  }
}
} // namespace autotune
} // namespace tc

#undef VALIDATE
