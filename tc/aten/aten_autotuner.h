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

#include <string>
#include <vector>

#include "tc/aten/aten.h"
#include "tc/autotuner/autotuner.h"

namespace tc {
namespace aten {
/**
 * An Autotuner provides the basic interface to run a SearchStrategy over a
 * particular Backend.
 *
 * Possible usage:
 *    using namespace tc::aten;
 *    std::string tc("...");
 *    ATenAutotuner<tc::CudaBackend, tc::autotune::GeneticSearch> tuner(tc);
 *    std::string cacheFn("/tmp/some_file");
 *    auto best = tuner.tune("tc_function_name", inputs, baseOption, cacheFn)
 *
 * The best options may then be used to compile an executor and run.
 *    CHECK_GT(best.size(), 0);
 *    auto pExecutor = compile(tc, "tc_function_name", inputs, best[0]);
 *    auto outputs = prepareOutputs(tc, "tc_function_name", inputs);
 *    // memoize the executor and outputs if needed
 *    run(*pExecutor, inputs, outputs);
 */
template <typename Backend, typename SearchStrategy>
class ATenAutotuner : public tc::autotune::Autotuner<Backend, SearchStrategy> {
 public:
  using BackendType = Backend;
  using MappingOptionsType = typename BackendType::MappingOptionsType;

  /// An ATenAutotuner is built from a TC string which contains multiple TC
  /// functions on which tuning can be run independently.
  ATenAutotuner(const std::string& tc);

  /// Runs autotuning on the TC function tcEntryPoint.
  /// Proper output shapes are inferred automatically from the input shapes.
  ///
  /// Optionally an OptionsCache cacheFileName serialized path
  /// can be specified to which the tuner will save the best options found for
  /// later offline reuse, in the proper protobuf format.
  ///
  /// Additionally, if such a cacheFileName is specified and if it contains a
  /// previously saved protobuf then the autotuner will load it. In that case
  /// the tuner recovers multiple starting points and appends them to the
  /// baseMapping. This can be useful in a reinforcement situation where short
  /// tunings are run and their results cached iteratively. The best options
  /// are still saved at the end of tuning, altering that previously saved
  /// protobuf file.
  ///
  /// Lastly a TuningParameterFixer function can be specified to limit the
  /// search space (i.e. when certain parameters are known to be good/bad
  /// independently on a particular TC).
  ///
  /// \return a vector MappingOptions, if it is empty then tuning did not find
  /// a single good configuration. This should be a very rare occurrence but
  /// it is possible in particular if the skipExecutionOrWarmup function is too
  /// aggressive and the problem size is too small. If the vector is not empty
  /// it contains the best performing options for the particular Backend,
  /// ranked by execution speed, where result[0] is the fastest.
  std::vector<MappingOptionsType> tune(
      const std::string& tcEntryPoint,
      const std::vector<at::Tensor>& inputs,
      const MappingOptionsType& baseMapping,
      const std::string& cacheFileName = "",
      const tc::autotune::TuningParameterFixer& fixedParams = {});

 private:
  /// The TC string is stored internally so we can tune independent TC
  /// functions on demand.
  const std::string tc_;
};
} // namespace aten
} // namespace tc

#include "tc/aten/aten_autotuner-inl.h"
