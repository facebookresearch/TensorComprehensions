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
#include <stdint.h>
#include <iostream>
#include <string>
#include <vector>

#include <glog/logging.h>

#include <Python.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <torch/torch.h>

#include "tc/aten/aten.h"
#include "tc/aten/aten_autotuner.h"
#include "tc/autotuner/genetic_search.h"
#include "tc/autotuner/options_cache.h"
#include "tc/core/cuda/cuda_backend.h"
#include "tc/core/cuda/cuda_tc_executor.h"
#include "tc/core/flags.h"
#include "tc/core/tensor.h"
#include "tc/lang/canonicalize.h"

namespace tc {
namespace python {

namespace py = pybind11;

namespace {
void initGlog() {
  static bool inited = false;
  if (!inited) {
    ::google::InitGoogleLogging("TC Python");
    inited = true;
  }
}

inline std::vector<tc::TensorInfo> getATenTensorsAsTensorInfo(
    const py::tuple& pyTensors) {
  std::vector<tc::TensorInfo> tensors;
  for (auto& inp : pyTensors) {
    tensors.push_back(tc::aten::toTensorInfo(inp.cast<at::Tensor>()));
  }
  return tensors;
}

inline std::vector<at::Tensor> getATenTensors(const py::tuple& pyTensors) {
  std::vector<at::Tensor> atTensors;
  for (auto& inp : pyTensors) {
    atTensors.push_back(inp.cast<at::Tensor>());
  }
  return atTensors;
}

template <typename VoidPtr>
inline std::vector<VoidPtr> getATenTensorsAsRawPtrs(
    const py::tuple& pyTensors) {
  std::vector<VoidPtr> res;
  for (auto& inp : pyTensors) {
    res.push_back(static_cast<VoidPtr>(inp.cast<at::Tensor>().data_ptr()));
  }
  return res;
}

inline py::list convertToPyObjects(const std::vector<at::Tensor>& tensors) {
  py::list outputs;
  for (auto& tensor : tensors) {
    outputs.append(py::cast(torch::autograd::make_variable(tensor)));
  }
  return outputs;
}
} // namespace

/**
 * This struct serves the purpose of memoizing the compiled TcExecutors
 * Since PyTorch's autograd is essentially stateless we cannot even store a
 * pointer that corresponds to our invariants (TC, def, input sizes + strides).
 * So we have to implement our own compilation cache.
 *
 * We want this to be lightweight so we cannot afford:
 *   1. python-side dictionary manipulations,
 *   2. parsing the TC string more than necessary (compilation and new
 *      allocations are acceptable)
 *   3. large string hashing
 *
 * Creating a key must be cheap, we cannot afford multiple conversions on the
 * way from ATen tensors to TensorInfo.
 */
struct CompilationCache {
  struct Key {
    Key(std::string entryPt, const py::tuple& inputTuple)
        : entryPoint(entryPt), inputs(getATenTensorsAsTensorInfo(inputTuple)) {}
    bool operator==(const Key& other) const {
      return entryPoint == other.entryPoint && inputs == other.inputs;
    }
    std::string entryPoint;
    std::vector<tc::TensorInfo> inputs;
  };

  struct KeyHasher {
    std::size_t operator()(const Key& k) const {
      size_t seed = 0x9e3779b9;
      for (const auto& t : k.inputs) {
        for (auto s : t.shape) {
          seed ^= std::hash<decltype(s)>()(s) + 0x9e3779b9 + (seed << 6) +
              (seed >> 2);
        }
        for (auto s : t.strides) {
          seed ^= std::hash<decltype(s)>()(s) + 0x9e3779b9 + (seed << 6) +
              (seed >> 2);
        }
      }
      return std::hash<std::string>()(k.entryPoint) + 0x9e3779b9 + (seed << 6) +
          (seed >> 2);
    }
  };

  CompilationCache(const std::string& tc) : tc(tc) {
    initGlog();
  }

  bool isCompiled(const std::string& entryPoint, const py::tuple& inputs) {
    return compiled.count(Key(entryPoint, inputs)) > 0;
  }

  /// This function infers the size of the outputs for each new compilation.
  /// This brings overhead, therefore we memoize the output sizes on-demand.
  /// The allocation itself is backed by ATen's caching allocator and is
  /// assumed acceptable (this is used everywhere in PyTorch).
  std::vector<at::Tensor> allocATenOutputTensors(
      const std::string& entryPoint,
      const py::tuple& inputs) {
    Key k(entryPoint, inputs);
    auto kvp = outputs.find(k);
    if (kvp == outputs.end()) {
      auto atInputs = getATenTensors(inputs);
      // Allocation for a key we haven't seen yet, we only want the metadata
      // to reuse for further allocations. Therefore we immediately release
      // the storage.
      // Then convertToPyObjects calls torch::autograd::make_variable.
      auto atOutputs = tc::aten::prepareOutputs(tc, entryPoint, atInputs);
      for (const auto& t : atOutputs) {
        t.storage().release();
      }
      outputs.emplace(k, atOutputs);
      kvp = outputs.find(k);
    }
    return kvp->second;
  }

  py::list allocOutputs(
      const std::string& entryPoint,
      const py::tuple& inputs) {
    return convertToPyObjects(allocATenOutputTensors(entryPoint, inputs));
  }

  /// This function forces recompilation and storage.
  /// This is because we do not want to own the decision of which options to
  /// build in the bindings but closer to the user level.
  /// Also we don't want to hash based on options so we just keep the last
  /// compiled version given an entryPoint and inputs.
  void compile(
      const std::string& entryPoint,
      const py::tuple& inputs,
      const tc::CudaMappingOptions& options) {
    Key k(entryPoint, inputs);
    compiled[k] = tc::aten::compile<tc::CudaBackend>(
        tc, entryPoint, getATenTensors(inputs), options);
  }

  py::list run(
      const std::string& entryPoint,
      const py::tuple& inputs,
      const py::tuple& outputs = py::tuple()) {
    if (outputs.size() > 0) {
      auto atOutputs = getATenTensors(outputs);
      auto atInputs = getATenTensors(inputs);
      tc::aten::run(*compiled.at(Key(entryPoint, inputs)), atInputs, atOutputs);
      return py::list(outputs);
    } else {
      auto atOutputs = allocATenOutputTensors(entryPoint, inputs);
      auto atInputs = getATenTensors(inputs);
      tc::aten::run(*compiled.at(Key(entryPoint, inputs)), atInputs, atOutputs);
      return convertToPyObjects(atOutputs);
    }
  }

  py::list uncheckedRun(
      const std::string& entryPoint,
      const py::tuple& inputs,
      const py::tuple& outputs = py::tuple()) {
    if (outputs.size() > 0) {
      compiled.at(Key(entryPoint, inputs))
          ->uncheckedRun(
              getATenTensorsAsRawPtrs<const void*>(inputs),
              getATenTensorsAsRawPtrs<void*>(outputs));
      return py::list(outputs);
    } else {
      auto outputs = allocOutputs(entryPoint, inputs);
      compiled.at(Key(entryPoint, inputs))
          ->uncheckedRun(
              getATenTensorsAsRawPtrs<const void*>(inputs),
              getATenTensorsAsRawPtrs<void*>(outputs));
      return outputs;
    }
  }

  std::string tc;
  std::unordered_map<Key, std::vector<at::Tensor>, KeyHasher> outputs;
  std::unordered_map<Key, std::unique_ptr<tc::CudaTcExecutor>, KeyHasher>
      compiled;
};

using ATenCudaGeneticTuner =
    tc::aten::ATenAutotuner<tc::CudaBackend, tc::autotune::GeneticSearch>;

class Tuner : public ATenCudaGeneticTuner {
 public:
  Tuner(const std::string& tc, const std::string& cacheFileName = "")
      : ATenCudaGeneticTuner(tc), cacheFileName(cacheFileName) {}

  std::string cacheFileName;
};

struct TcExecutor {
  py::list run(
      const py::tuple& inputs,
      const py::tuple& outputs = py::tuple()) {
    if (outputs.size() > 0) {
      auto atOutputs = getATenTensors(outputs);
      auto atInputs = getATenTensors(inputs);
      tc::aten::run(*executor, atInputs, atOutputs);
      return py::list(outputs);
    } else {
      auto atInputs = getATenTensors(inputs);
      auto atOutputs = tc::aten::prepareOutputs(tc, entryPoint, atInputs);
      tc::aten::run(*executor, atInputs, atOutputs);
      return convertToPyObjects(atOutputs);
    }
  }
  py::list uncheckedRun(
      const py::tuple& inputs,
      const py::tuple& outputs = py::tuple()) {
    if (outputs.size() > 0) {
      auto atOutputs = getATenTensors(outputs);
      auto atInputs = getATenTensors(inputs);
      tc::aten::uncheckedRun(*executor, atInputs, atOutputs);
      return py::list(outputs);
    } else {
      auto atInputs = getATenTensors(inputs);
      auto atOutputs = tc::aten::prepareOutputs(tc, entryPoint, atInputs);
      tc::aten::uncheckedRun(*executor, atInputs, atOutputs);
      return convertToPyObjects(atOutputs);
    }
  }
  std::string tc;
  std::string entryPoint;
  std::unique_ptr<tc::CudaBackend::ExecutorType> executor;
};

class TunerConfig {
 public:
  TunerConfig(
      uint32_t generations,
      uint32_t populationSize,
      uint32_t crossoverRate,
      uint32_t mutationRate,
      uint32_t numberElites,
      uint32_t tunerMinLaunchTotalThreads,
      uint32_t threads,
      std::string devices,
      bool logtostderr,
      uint32_t stderrthreshold) {
    generations_ = generations;
    populationSize_ = populationSize;
    crossoverRate_ = crossoverRate;
    mutationRate_ = mutationRate;
    numberElites_ = numberElites;
    tunerMinLaunchTotalThreads_ = tunerMinLaunchTotalThreads;
    threads_ = threads;
    devices_ = devices;
    logtostderr_ = logtostderr;
    stderrthreshold_ = stderrthreshold;
  }
  // __enter__ / __exit__ in case we want to use a ContextManager in Python in
  // the future. In any case, RAII and Python GC can just never work together.
  void __enter__() const {
    savedGenerations_ = tc::FLAGS_tuner_gen_generations;
    savedPopulationSize_ = tc::FLAGS_tuner_gen_pop_size;
    savedCrossoverRate_ = tc::FLAGS_tuner_gen_crossover_rate;
    savedMutationRate_ = tc::FLAGS_tuner_gen_mutation_rate;
    savedNumberElites_ = tc::FLAGS_tuner_gen_number_elites;
    savedTunerMinLaunchTotalThreads_ = tc::FLAGS_tuner_min_launch_total_threads;
    savedThreads_ = tc::FLAGS_tuner_threads;
    savedDevices_ = tc::FLAGS_tuner_devices;
    savedLogtostderr_ = FLAGS_logtostderr;
    savedStderrthreshold_ = FLAGS_stderrthreshold;

    tc::FLAGS_tuner_gen_generations = generations_;
    tc::FLAGS_tuner_gen_pop_size = populationSize_;
    tc::FLAGS_tuner_gen_crossover_rate = crossoverRate_;
    tc::FLAGS_tuner_gen_mutation_rate = mutationRate_;
    tc::FLAGS_tuner_gen_number_elites = numberElites_;
    tc::FLAGS_tuner_min_launch_total_threads = tunerMinLaunchTotalThreads_;
    tc::FLAGS_tuner_threads = threads_;
    tc::FLAGS_tuner_devices = devices_;
    FLAGS_logtostderr = logtostderr_;
    FLAGS_stderrthreshold = stderrthreshold_;
  }
  void __exit__() const {
    tc::FLAGS_tuner_gen_generations = savedGenerations_;
    tc::FLAGS_tuner_gen_pop_size = savedPopulationSize_;
    tc::FLAGS_tuner_gen_crossover_rate = savedCrossoverRate_;
    tc::FLAGS_tuner_gen_mutation_rate = savedMutationRate_;
    tc::FLAGS_tuner_gen_number_elites = savedNumberElites_;
    tc::FLAGS_tuner_min_launch_total_threads = savedTunerMinLaunchTotalThreads_;
    tc::FLAGS_tuner_threads = savedThreads_;
    tc::FLAGS_tuner_devices = savedDevices_;
    FLAGS_logtostderr = savedLogtostderr_;
    FLAGS_stderrthreshold = savedStderrthreshold_;
  }

 private:
  uint32_t generations_;
  uint32_t populationSize_;
  uint32_t crossoverRate_;
  uint32_t mutationRate_;
  uint32_t numberElites_;
  uint32_t tunerMinLaunchTotalThreads_;
  uint32_t threads_;
  std::string devices_;
  bool logtostderr_;
  uint32_t stderrthreshold_;
  mutable uint32_t savedGenerations_;
  mutable uint32_t savedPopulationSize_;
  mutable uint32_t savedCrossoverRate_;
  mutable uint32_t savedMutationRate_;
  mutable uint32_t savedNumberElites_;
  mutable uint32_t savedTunerMinLaunchTotalThreads_;
  mutable uint32_t savedThreads_;
  mutable std::string savedDevices_;
  mutable bool savedLogtostderr_;
  mutable uint32_t savedStderrthreshold_;
};

class MappingOptionsCache {
 public:
  MappingOptionsCache(const std::string& cacheFileName)
      : fileName_(cacheFileName) {}

  std::vector<tc::CudaMappingOptions> load(
      const std::string& tc,
      const std::string& entryPoint,
      const py::tuple& inputTuple,
      const size_t numCandidates) {
    tc::autotune::OptionsCache<tc::CudaBackend> cache;
    cache.loadCacheFromFile(fileName_);
    // This could be made more efficient but loading is premature optimization
    auto inputsDLTensors =
        tc::aten::makeDLConstTensors(getATenTensors(inputTuple));
    return cache.getTopKOptions(
        lang::canonicalTc(tc::detail::parse(tc).at(entryPoint)),
        getATenTensorsAsTensorInfo(inputTuple),
        tc::inferOutputTensorInfo(
            tc, entryPoint, extractRawPtrs(inputsDLTensors)),
        tc::CudaBackend::backendString(),
        numCandidates);
  }

 private:
  std::string fileName_;
};

PYBIND11_MODULE(tclib, m) {
  m.doc() = "Python bindings for Tensor Comprehensions";

  // Simple functions to set up debugging
  m.def(
      "logtostderr", [](bool logtostderr) { FLAGS_logtostderr = logtostderr; });
  m.def(
      "debug_lang", [](bool debug_lang) { tc::FLAGS_debug_lang = debug_lang; });
  m.def("debug_halide", [](bool debug_halide) {
    tc::FLAGS_debug_halide = debug_halide;
  });
  m.def("debug_tc_mapper", [](bool debug_tc_mapper) {
    tc::FLAGS_debug_tc_mapper = debug_tc_mapper;
  });
  m.def("debug_tuner", [](bool debug_tuner) {
    tc::FLAGS_debug_tuner = debug_tuner;
  });
  m.def("dump_cuda", [](bool dump_cuda) { tc::FLAGS_dump_cuda = dump_cuda; });

  // Low-level stateful API compile returns an executor on which run and
  // unchecked_run can be called.
  py::class_<TcExecutor>(m, "TcExecutor")
      .def("run", &TcExecutor::run)
      .def("unchecked_run", &TcExecutor::uncheckedRun);
  m.def(
      "compile",
      [](const std::string& tc,
         const std::string& entryPoint,
         const py::tuple& inputs,
         const tc::CudaMappingOptions& options) {
        auto execUPtr = tc::aten::compile<tc::CudaBackend>(
            tc, entryPoint, getATenTensors(inputs), options);
        return TcExecutor{tc, entryPoint, std::move(execUPtr)};
      });

  // A TunerConfig object can be passed to configure a tuning run
  py::class_<TunerConfig>(m, "TunerConfig")
      .def(
          py::init<
              uint32_t,
              uint32_t,
              uint32_t,
              uint32_t,
              uint32_t,
              uint32_t,
              uint32_t,
              std::string,
              bool,
              uint32_t>(),
          py::arg("generations") = tc::FLAGS_tuner_gen_generations,
          py::arg("pop_size") = tc::FLAGS_tuner_gen_pop_size,
          py::arg("crossover_rate") = tc::FLAGS_tuner_gen_crossover_rate,
          py::arg("mutation_rate") = tc::FLAGS_tuner_gen_mutation_rate,
          py::arg("number_elites") = tc::FLAGS_tuner_gen_number_elites,
          py::arg("tuner_min_launch_total_threads") =
              tc::FLAGS_tuner_min_launch_total_threads,
          py::arg("threads") = tc::FLAGS_tuner_threads,
          py::arg("devices") = tc::FLAGS_tuner_devices,
          py::arg("logtostderr") = false,
          // Suppress non-FATAL errors from the python user
          py::arg("stderrthreshold") = google::FATAL);

  //
  py::class_<Tuner>(m, "Tuner")
      .def(py::init<std::string>())
      .def(py::init<std::string, std::string>())
      .def(
          "tune",
          [](Tuner& instance,
             const std::string& entryPoint,
             const py::tuple& inputs,
             tc::CudaMappingOptions& baseMapping,
             const TunerConfig& config) {
            config.__enter__();
            ScopeGuard sg([&config]() { config.__exit__(); });
            std::vector<at::Tensor> atInputs = getATenTensors(inputs);
            auto bestOptions =
                instance.tune(entryPoint, atInputs, {baseMapping});
            if (bestOptions.size() > 0u) {
              if (not instance.cacheFileName.empty()) {
                tc::autotune::appendTopKToCacheFile(
                    *instance.optionsCache, instance.cacheFileName, 1);
              }
              return bestOptions[0];
            } else {
              std::cout << "Autotuner could not find options, returning base"
                        << std::endl;
              return baseMapping;
            }
          });

  py::class_<MappingOptionsCache>(m, "MappingOptionsCache")
      .def(py::init<std::string>())
      .def("load", &MappingOptionsCache::load);

  py::class_<CompilationCache>(m, "CompilationCache")
      .def(py::init<std::string>())
      .def("is_compiled", &CompilationCache::isCompiled)
      .def("alloc_outputs", &CompilationCache::allocOutputs)
      .def("compile", &CompilationCache::compile)
      .def("run", &CompilationCache::run)
      .def("unchecked_run", &CompilationCache::uncheckedRun);

  py::class_<tc::CudaMappingOptions>(
      m,
      "MappingOptions",
      "MappingOptions for a Tensor Comprehensions (TC)",
      py::module_local())
      .def(
          py::init([](const std::string& optionsName) {
            TC_CHECK_EQ(optionsName, "naive")
                << "Naive options are the only constructible user-facing options. We recommended using the tuner to get better options or, alternatively, retrieving some from a cache.";
            return tc::CudaMappingOptions::makeNaiveMappingOptions();
          }),
          "Initialize naive CudaMappingOption")
      .def(
          "__str__",
          [](tc::CudaMappingOptions& instance) {
            std::string str;
            google::protobuf::TextFormat::PrintToString(instance.proto(), &str);
            return str;
          },
          "Returns the CudaMappingOptions as a human-readable string")
      .def(
          "serialize",
          [](tc::CudaMappingOptions& instance) {
            std::string str = instance.toProtobufSerializedString();
            return py::bytes(str);
          },
          "Serialize the options to a protobuf string")
      .def(
          "maxSharedMemory",
          &tc::CudaMappingOptions::maxSharedMemory,
          "The amount of shared memory to use, in bytes. If not provided, "
          "TC will query the active GPU and use all available shared memory.")
      .def(
          "useSharedMemory",
          &tc::CudaMappingOptions::useSharedMemory,
          "Create block-local copies of data in shared memory when this can "
          "leverage data reuse or global memory access coalescing")
      .def(
          "usePrivateMemory",
          &tc::CudaMappingOptions::usePrivateMemory,
          "Create thread-local copies of data in private memory")
      .def(
          "unrollCopyShared",
          &tc::CudaMappingOptions::unrollCopyShared,
          "Also unroll the copies to and from shared memory. If an unroll "
          "value is not provided, has no effect")
      .def(
          "useReadOnlyCache",
          &tc::CudaMappingOptions::useReadOnlyCache,
          "Use the readonly cache (i.e. emit __ldg loads)")
      .def(
          "scheduleFusionStrategy",
          [](tc::CudaMappingOptions& instance, const std::string& type) {
            instance.scheduleFusionStrategy(type);
            return instance;
          },
          "Set up outerScheduleFusionStrategy and intraTileFusionStrategy "
          "to the given value")
      .def(
          "outerScheduleFusionStrategy",
          [](tc::CudaMappingOptions& instance, const std::string& type) {
            instance.outerScheduleFusionStrategy(type);
            return instance;
          },
          "Require TC to try and execute different TC expressions interleaved "
          "(Max), separately (Min)\n"
          "or interleaved as long as sufficient parallelism is exploited "
          "(Preserve3Coincident) by\n"
          "performing loop fusion and fission. "
          "Applies to inner loops created by tiling")
      .def(
          "intraTileScheduleFusionStrategy",
          [](tc::CudaMappingOptions& instance, const std::string& type) {
            instance.intraTileScheduleFusionStrategy(type);
            return instance;
          },
          "Require TC to try and execute different TC expressions interleaved "
          "(Max), separately (Min)\n"
          "or interleaved as long as sufficient parallelism is exploited "
          "(Preserve3Coincident) by\n"
          "performing loop fusion and fission. Applies before tiling")
      .def(
          "tile",
          // pybind11 has implicit conversion from tuple -> vector
          [](tc::CudaMappingOptions& instance,
             std::vector<uint64_t>& tileSizes) {
            instance.tile(tileSizes);
            return instance;
          },
          "Perform loop tiling on the generated code with the given sizes. "
          "Independent of mapping to a\n"
          "grid of thread blocks")
      .def(
          "mapToThreads",
          [](tc::CudaMappingOptions& instance,
             std::vector<uint64_t>& threadSizes) {
            instance.mapToThreads(threadSizes);
            return instance;
          },
          "The configuration of CUDA block, i.e. the number of CUDA threads "
          "in each block along three\n"
          "dimensions. Must be within the range allowed by CUDA (maximum 1024 "
          "for the first and second value,\n"
          "32 for the third, product below 1024)")
      .def(
          "mapToBlocks",
          [](tc::CudaMappingOptions& instance,
             std::vector<uint64_t>& blockSizes) {
            instance.mapToBlocks(blockSizes);
            return instance;
          },
          "The configuration of CUDA grid, i.e. the number of CUDA blocks "
          "along three dimensions. Must be\n"
          "within the range allowed by CUDA (maximum 2^31-1 for the first "
          "value and 65535 for the second and third)")
      .def(
          "matchLibraryCalls",
          [](tc::CudaMappingOptions& instance, bool match) {
            instance.matchLibraryCalls(match);
            return instance;
          },
          "Replace computation patterns with calls to highly optimized "
          "libraries (such as CUB, CUTLASS) when possible")
      .def(
          "fixParametersBeforeScheduling",
          [](tc::CudaMappingOptions& instance, bool fix) {
            instance.fixParametersBeforeScheduling(fix);
            return instance;
          },
          "Perform automatic loop scheduling taking into account specific "
          "tensor sizes.\n"
          "May produce faster kernels but significantly increases compilation "
          "time.\n"
          "Note that the mapping will be performed for specific tensor sizes "
          "anyway")
      .def(
          "unroll",
          [](tc::CudaMappingOptions& instance, uint64_t factor) {
            instance.unroll(factor);
            return instance;
          },
          "Perform loop unrolling on the generated code and produce at "
          "most the given number of statements");
}

} // namespace python
} // namespace tc
