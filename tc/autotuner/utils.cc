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
#include "tc/autotuner/utils.h"

#include <algorithm>
#include <cmath>
#include <sstream>

#include <glog/stl_logging.h>

#include "tc/core/flags.h"
#include "tc/core/utils/time.h"

namespace tc {
namespace autotune {

std::vector<std::size_t> powers2andCeilDivisors(std::size_t val) {
  auto numPowers = static_cast<std::size_t>(std::ceil(std::log2(val)));
  // 1. generate `numPowers' powers of 2
  std::vector<std::size_t> res(numPowers + 1);
  std::size_t p = 1;
  std::generate(res.begin(), res.end(), [p]() mutable {
    auto old_p = p;
    p *= 2;
    return old_p;
  });
  // 2. additionally insert ceil(val / powers2)
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

namespace {
uint64_t toMicroseconds(const Duration& d) {
  return std::chrono::duration_cast<std::chrono::microseconds>(d).count();
}
} // namespace

void Printer::record(Duration runtime) {
  std::lock_guard<std::mutex> lock(runtimesMtx_);
  runtimes_.push_back(runtime);
}

void Printer::printLoop() {
  while (true) {
    std::this_thread::sleep_for(std::chrono::seconds(1));

    std::stringstream ss;
    ss << "Iteration " << iteration_;
    ss << "\tJobs(Compiled, Evaluated)/total  ("
       << std::min(total_, currentCompilationJob_.load()) << ", "
       << std::min(total_, numEvaluations_.load()) << ")/" << total_;

    {
      std::lock_guard<std::mutex> lock(runtimesMtx_);
      if (not runtimes_.empty()) {
        std::sort(runtimes_.begin(), runtimes_.end());
        auto best = toMicroseconds(runtimes_.front());
        auto median = toMicroseconds(runtimes_.at(runtimes_.size() / 2));
        auto worst = toMicroseconds(runtimes_.back());
        ss << "   (best/median/worst)us: " << best << '/' << median << '/'
           << worst;
      }
    }
    // XXX: platform specific erase current line and move cursor to begining
    // of line. Currently works with python/C++ both.
    std::cout << "\u001b[2K\r" << ss.str() << std::flush;
    LOG_IF(INFO, FLAGS_debug_tuner) << "\u001b[2K\r" << ss.str() << std::endl;

    if (stopPrinting_.load()) {
      // Print one more time to flush
      // XXX: platform specific erase current line and move cursor to begining
      // of line. Currently works with python/C++ both.
      std::cout << "\u001b[2K\r" << ss.str() << std::flush;
      LOG_IF(INFO, FLAGS_debug_tuner) << "\u001b[2K\r" << ss.str() << std::endl;
      // commit line so it does not get erased at the next iteration
      std::cerr << std::endl;
      return;
    }
  }
}

Printer::Printer(
    size_t iteration,
    size_t total,
    const std::atomic_size_t& currentCompilationJob,
    const std::atomic_size_t& numEvaluations)
    : iteration_(iteration),
      printerThread_([this]() { printLoop(); }),
      total_(total),
      currentCompilationJob_(currentCompilationJob),
      numEvaluations_(numEvaluations) {}

Printer::~Printer() {
  stop();
  printerThread_.join();
}

void Printer::stop() {
  stopPrinting_.store(true);
}

void Printer::printAll() {
  auto runtimes = [this]() {
    std::lock_guard<std::mutex> lock(runtimesMtx_);
    std::sort(runtimes_.begin(), runtimes_.end());
    std::vector<uint64_t> runtimes;
    runtimes.reserve(runtimes_.size());
    std::transform(
        runtimes_.begin(),
        runtimes_.end(),
        std::back_inserter(runtimes),
        toMicroseconds);
    return runtimes;
  }();
  LOG_IF(INFO, FLAGS_debug_tuner)
      << "\n [TUNER][ITERATION LOG] median times of each candidate (in us) "
      << runtimes << std::endl;
}
} // namespace autotune
} // namespace tc
