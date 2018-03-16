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
#include <algorithm>
#include <cmath>
#include <numeric>

#include "tc/aten/aten_compiler.h"
#include "tc/autotuner/utils/utils.h"
#include "tc/core/cuda/cuda_compilation_cache.h"
#include "tc/core/utils/math.h"
#include "tc/lang/canonicalize.h"

namespace tc {
namespace autotune {

namespace {
std::vector<std::size_t> firstPowers2(std::size_t n) {
  std::vector<std::size_t> powers(n + 1);
  std::size_t p = 1;
  std::generate(powers.begin(), powers.end(), [p]() mutable {
    auto old_p = p;
    p *= 2;
    return old_p;
  });
  return powers;
}
} // namespace

std::vector<std::size_t> powers2andCeilDivisors(std::size_t val) {
  auto res = firstPowers2(static_cast<std::size_t>(std::ceil(std::log2(val))));
  res.reserve(res.size() * 2);
  for (std::size_t i = 0, s = res.size(); i < s; ++i) {
    if (res[i] > val) {
      continue;
    }
    res.push_back(std::ceil(static_cast<double>(val) / res[i]));
  }
  std::sort(res.begin(), res.end());
  res.erase(std::unique(res.begin(), res.end()), res.end());
  return res;
}

std::vector<OptionsWithMedianTime> getOptionsAndMedianRuntimes(
    const lang::CanonicalTcString& id,
    const std::vector<const DLTensor*>& inputs,
    const std::vector<const DLTensor*>& outputs) {
  auto candidates =
      OptionsCache::getCache()->retrieveOptionsAndRuntimes(id, inputs, outputs);

  std::vector<OptionsWithMedianTime> c;
  c.reserve(candidates.size());
  std::transform(
      candidates.begin(),
      candidates.end(),
      std::back_inserter(c),
      [](const OptionsCacheRetrievalResult& rr) -> OptionsWithMedianTime {
        return {std::move(rr.options), median(rr.recordedRuntimes)};
      });
  return c;
}

std::vector<CudaMappingOptions> restoreCandidates(
    const lang::CanonicalTcString& tc,
    const std::vector<const DLTensor*>& inputs,
    const std::vector<const DLTensor*>& outputs) {
  auto candidates = getOptionsAndMedianRuntimes(tc, inputs, outputs);
  LOG_IF(INFO, candidates.size() < FLAGS_tuner_gen_restore_number)
      << "Requested " << FLAGS_tuner_gen_restore_number
      << " candidates but there are only " << candidates.size() << " in cache.";
  auto restoreNumber =
      std::min(candidates.size(), size_t(FLAGS_tuner_gen_restore_number));
  std::sort(
      candidates.begin(),
      candidates.end(),
      [](const OptionsWithMedianTime& a, const OptionsWithMedianTime& b) {
        return a.medianRuntime < b.medianRuntime;
      });
  std::vector<CudaMappingOptions> res;
  res.reserve(restoreNumber);
  std::transform(
      candidates.begin(),
      candidates.begin() + restoreNumber,
      std::back_inserter(res),
      [](const OptionsWithMedianTime& rr) { return rr.options; });
  return res;
}

llvm::Optional<CudaMappingOptions> getBestOptions(
    const lang::CanonicalTcString& id,
    const std::vector<const DLTensor*>& inputs,
    const std::vector<const DLTensor*>& outputs) {
  auto bestOptions =
      OptionsCache::getCache()->retrieveBestOptions(id, inputs, outputs);
  if (bestOptions) {
    return *bestOptions;
  }
  return llvm::Optional<CudaMappingOptions>{};
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

} // namespace autotune
} // namespace tc
