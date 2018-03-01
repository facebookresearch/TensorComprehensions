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

#include "tc/aten/aten_compiler.h"
#include "tc/autotuner/utils/utils.h"
#include "tc/core/cuda/cuda_compilation_cache.h"
#include "tc/core/utils/math.h"

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
    const std::string& id,
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
      [](const OptionsCache::RetrievalResult& rr) -> OptionsWithMedianTime {
        return {std::move(rr.options), median(rr.recordedRuntimes)};
      });
  return c;
}

std::vector<MappingOptions> restoreCandidates(
    const std::string& id,
    const std::vector<const DLTensor*>& inputs,
    const std::vector<const DLTensor*>& outputs) {
  auto candidates = getOptionsAndMedianRuntimes(id, inputs, outputs);
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
  std::vector<MappingOptions> res;
  res.reserve(restoreNumber);
  std::transform(
      candidates.begin(),
      candidates.begin() + restoreNumber,
      std::back_inserter(res),
      [](const OptionsWithMedianTime& rr) { return rr.options; });
  return res;
}

llvm::Optional<MappingOptions> getBestOptions(
    const std::string& id,
    const std::vector<const DLTensor*>& inputs,
    const std::vector<const DLTensor*>& outputs) {
  auto bestOptions =
      OptionsCache::getCache()->retrieveBestOptions(id, inputs, outputs);
  if (bestOptions) {
    return *bestOptions;
  }
  return llvm::Optional<MappingOptions>{};
}

} // namespace autotune
} // namespace tc
