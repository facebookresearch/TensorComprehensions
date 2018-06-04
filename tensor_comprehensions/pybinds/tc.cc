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

#include <Python.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "pybind_utils.h"
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

struct ATenCudaCompilationUnit {
  void define(const std::string& def) {
    tc = def;
  }
  std::string tc;
  std::unordered_map<std::string, std::unique_ptr<tc::CudaTcExecutor>> compiled;
};

using ATenCudaGeneticTuner =
    tc::aten::ATenAutotuner<tc::CudaBackend, tc::autotune::GeneticSearch>;

class ATenCudaTuner : public ATenCudaGeneticTuner {
 public:
  ATenCudaTuner(const std::string& tc)
      : ATenCudaGeneticTuner(tc), tcEntryPointMap_(tc::detail::parse(tc)) {}

  std::vector<tc::CudaMappingOptions> load(
      const std::string& cacheFileName,
      const std::string& entryPoint,
      const std::vector<at::Tensor>& inputs,
      const size_t numCandidates) {
    tc::autotune::OptionsCache<tc::CudaBackend> cache;
    cache.loadCacheFromFile(tc::makeOptionsFilename(cacheFileName));
    auto inputsDLTensors = tc::aten::makeDLConstTensors(inputs);
    return cache.getTopKOptions(
        lang::canonicalTc(tcEntryPointMap_.at(entryPoint)),
        makeTensorInfoVector(extractRawPtrs(inputsDLTensors)),
        tc::inferOutputTensorInfo(
            tc_, entryPoint, extractRawPtrs(inputsDLTensors)),
        tc::CudaBackend::backendString(),
        numCandidates);
  }

 private:
  std::map<std::string, lang::TreeRef> tcEntryPointMap_;
};

PYBIND11_MODULE(tc, m) {
  m.doc() = "Python bindings for Tensor Comprehensions";
  py::object dlpack;
  try {
    dlpack = py::module::import("torch.utils.dlpack");
  } catch (std::exception& e) {
    std::cerr << "\n PyTorch installation is missing, binary will be useless \n"
              << e.what() << std::endl;
  }

  m.def("set_logtostderr", [](bool logtostderr) {
    FLAGS_logtostderr = logtostderr;
  });
  m.def("set_debug_lang", [](bool debug_lang) {
    tc::FLAGS_debug_lang = debug_lang;
  });
  m.def("set_debug_halide", [](bool debug_halide) {
    tc::FLAGS_debug_halide = debug_halide;
  });
  m.def("set_debug_tc_mapper", [](bool debug_tc_mapper) {
    tc::FLAGS_debug_tc_mapper = debug_tc_mapper;
  });
  m.def("set_debug_cuda", [](bool debug_cuda) {
    tc::FLAGS_debug_cuda = debug_cuda;
  });
  m.def("set_debug_tuner", [](bool debug_tuner) {
    tc::FLAGS_debug_tuner = debug_tuner;
  });
  m.def(
      "set_dump_cuda", [](bool dump_cuda) { tc::FLAGS_dump_cuda = dump_cuda; });

  py::class_<ATenCudaTuner>(m, "ATenCudaTuner")
      .def(py::init<const std::string>())
      .def(
          "pop_size",
          [](ATenCudaTuner& instance, uint32_t& pop_size) {
            tc::FLAGS_tuner_gen_pop_size = pop_size;
          })
      .def(
          "crossover_rate",
          [](ATenCudaTuner& instance, uint32_t& crossover_rate) {
            tc::FLAGS_tuner_gen_crossover_rate = crossover_rate;
          })
      .def(
          "mutation_rate",
          [](ATenCudaTuner& instance, uint32_t& mutation_rate) {
            tc::FLAGS_tuner_gen_mutation_rate = mutation_rate;
          })
      .def(
          "generations",
          [](ATenCudaTuner& instance, uint32_t& generations) {
            tc::FLAGS_tuner_gen_generations = generations;
          })
      .def(
          "number_elites",
          [](ATenCudaTuner& instance, uint32_t& number_elites) {
            tc::FLAGS_tuner_gen_number_elites = number_elites;
          })
      .def(
          "threads",
          [](ATenCudaTuner& instance, uint32_t& threads) {
            tc::FLAGS_tuner_threads = threads;
          })
      .def(
          "gpus",
          [](ATenCudaTuner& instance, std::string& gpus) {
            tc::FLAGS_tuner_devices = gpus;
          })
      .def(
          "restore_from_proto",
          [](ATenCudaTuner& instance, bool restore_from_proto) {
            tc::FLAGS_tuner_gen_restore_from_proto = restore_from_proto;
          })
      .def(
          "restore_number",
          [](ATenCudaTuner& instance, uint32_t& restore_number) {
            tc::FLAGS_tuner_gen_restore_number = restore_number;
          })
      .def(
          "log_generations",
          [](ATenCudaTuner& instance, bool log_generations) {
            tc::FLAGS_tuner_gen_log_generations = log_generations;
          })
      .def(
          "tuner_min_launch_total_threads",
          [](ATenCudaTuner& instance, bool tuner_min_launch_total_threads) {
            tc::FLAGS_tuner_min_launch_total_threads =
                tuner_min_launch_total_threads;
          })
      .def(
          "save_best_candidates_count",
          [](ATenCudaTuner& instance, bool save_best_candidates_count) {
            tc::FLAGS_tuner_save_best_candidates_count =
                save_best_candidates_count;
          })
      .def(
          "tune",
          [dlpack](
              ATenCudaTuner& instance,
              const std::string& entryPoint,
              py::list& inputs,
              tc::CudaMappingOptions& baseMapping,
              const std::string& cacheFileName) {
            std::vector<at::Tensor> atInputs = getATenTensors(inputs, dlpack);
            auto bestOptions =
                instance.tune(entryPoint, atInputs, baseMapping, cacheFileName);
            if (bestOptions.size() > 0u) {
              return bestOptions[0];
            } else {
              std::cout << "Autotuner could not find options, returning base"
                        << std::endl;
              return baseMapping;
            }
          })
      .def(
          "load",
          [dlpack](
              ATenCudaTuner& instance,
              const std::string& cacheFileName,
              const std::string& entryPoint,
              py::list& inputs,
              const size_t& numCandidates) {
            return instance.load(
                cacheFileName,
                entryPoint,
                getATenTensors(inputs, dlpack),
                numCandidates);
          });

  py::class_<ATenCudaCompilationUnit>(m, "ATenCompilationUnit")
      .def(py::init<>())
      .def("define", &ATenCudaCompilationUnit::define, "Define the TC language")
      .def(
          "compile",
          [dlpack](
              ATenCudaCompilationUnit& instance,
              const std::string& entryPoint,
              py::list& inputs,
              const tc::CudaMappingOptions& options) {
            instance.compiled[entryPoint] = tc::aten::compile<tc::CudaBackend>(
                instance.tc,
                entryPoint,
                getATenTensors(inputs, dlpack),
                options);
          })
      .def(
          "run",
          [dlpack](
              ATenCudaCompilationUnit& instance,
              const std::string& entryPoint,
              py::list& inputs,
              py::list& outputs) {
            auto atInputs = getATenTensors(inputs, dlpack);
            auto atOutputs = (py::len(outputs) > 0)
                ? getATenTensors(outputs, dlpack)
                : tc::aten::prepareOutputs(instance.tc, entryPoint, atInputs);
            tc::aten::run(
                *instance.compiled.at(entryPoint), atInputs, atOutputs);
            if (py::len(outputs) == 0) {
              convertToPyObjects(atOutputs, dlpack, outputs);
            }
          })
      .def(
          "uncheckedRun",
          [dlpack](
              ATenCudaCompilationUnit& instance,
              const std::string& entryPoint,
              py::list& inputs,
              py::list& outputs) {
            CHECK_GE(outputs.size(), 1u);
            auto atOutputs = getATenTensors(outputs, dlpack);
            tc::aten::uncheckedRun(
                *instance.compiled.at(entryPoint),
                getATenTensors(inputs, dlpack),
                atOutputs);
          });

  py::class_<tc::CudaMappingOptions>(
      m,
      "CudaMappingOptions",
      "MappingCudaMappingOptions for a Tensor Comprehensions (TC)")
      .def(
          py::init([](std::string type) {
            if (type == "naive") {
              return tc::CudaMappingOptions::makeNaiveMappingOptions();
            }
            if (type == "single_thread") {
              return tc::CudaMappingOptions::makeSingleThreadMappingOptions();
            }
            if (type == "pointwise") {
              return tc::CudaMappingOptions::makePointwiseMappingOptions();
            }
            if (type == "mlp") {
              return tc::CudaMappingOptions::makeMlpMappingOptions();
            }
            if (type == "conv") {
              return tc::CudaMappingOptions::makeConvolutionMappingOptions();
            }
            if (type == "group_conv") {
              return tc::CudaMappingOptions::
                  makeGroupConvolutionMappingOptions();
            }
            throw std::runtime_error("Invalid option passed");
          }),
          "Initialize the mapping options from one of the following:\n"
          " 1. naive\n"
          " 2. pointwise\n"
          " 3. mlp\n"
          " 4. conv\n"
          " 5. group_conv\n"
          " 6. single_thread")
      .def(
          "__str__",
          [](tc::CudaMappingOptions& instance) {
            std::string str;
            google::protobuf::TextFormat::PrintToString(instance.proto(), &str);
            return str;
          },
          "Returns the CudaMappingOptions as a human-readable string")
      //
      // Generic options
      //
      .def(
          "scheduleFusionStrategy",
          [](tc::CudaMappingOptions& instance, const std::string& type) {
            instance.scheduleFusionStrategy(type);
          },
          "Set up outerScheduleFusionStrategy and intraTileFusionStrategy "
          "to the given value")
      .def(
          "outerScheduleFusionStrategy",
          [](tc::CudaMappingOptions& instance, const std::string& type) {
            instance.outerScheduleFusionStrategy(type);
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
          },
          "Require TC to try and execute different TC expressions interleaved "
          "(Max), separately (Min)\n"
          "or interleaved as long as sufficient parallelism is exploited "
          "(Preserve3Coincident) by\n"
          "performing loop fusion and fission. Applies before tiling")
      .def(
          "fixParametersBeforeScheduling",
          [](tc::CudaMappingOptions& instance, bool fix) {
            instance.fixParametersBeforeScheduling(fix);
          },
          "Perform automatic loop scheduling taking into account specific "
          "tensor sizes.\n"
          "May produce faster kernels but significantly increases compilation "
          "time.\n"
          "Note that the mapping will be performed for specific tensor sizes "
          "anyway")
      .def(
          "tile",
          // pybind11 has implicit conversion from list -> vector
          [](tc::CudaMappingOptions& instance,
             std::vector<uint64_t>& tileSizes) { instance.tile(tileSizes); },
          "Perform loop tiling on the generated code with the given sizes. "
          "Independent of mapping to a\n"
          "grid of thread blocks")
      .def(
          "tile_imperfectly_nested",
          [](tc::CudaMappingOptions& instance, bool tile) {
            instance.tileImperfectlyNested(tile);
          },
          "Allow imperfectly nested loop tiling")
      .def(
          "unroll",
          [](tc::CudaMappingOptions& instance, uint64_t factor) {
            instance.unroll(factor);
          },
          "Perform loop unrolling on the generated code and produce at "
          "most the given number of statements")
      .def(
          "matchLibraryCalls",
          [](tc::CudaMappingOptions& instance, bool match) {
            instance.matchLibraryCalls(match);
          },
          "Replace computation patterns with calls to highly optimized "
          "libraries (such as CUB, CUTLASS) when possible")
      //
      // CUDA-specific options
      //
      .def(
          "mapToThreads",
          [](tc::CudaMappingOptions& instance,
             std::vector<uint64_t>& threadSizes) {
            instance.mapToThreads(threadSizes);
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
          },
          "The configuration of CUDA grid, i.e. the number of CUDA blocks "
          "along three dimensions. Must be\n"
          "within the range allowed by CUDA (maximum 2^31-1 for the first "
          "value and 65535 for the second and third)")
      .def(
          "useSharedMemory",
          &tc::CudaMappingOptions::useSharedMemory,
          "Create block-local copies of data in shared memory when this can "
          "leverage data reuse or global memory access coalescing")
      .def(
          "usePrivateMemory",
          &tc::CudaMappingOptions::usePrivateMemory,
          "Use private memoery (registers) when possible")

      .def(
          "unrollCopyShared",
          &tc::CudaMappingOptions::unrollCopyShared,
          "Also unroll the copies to and from shared memory. If an unroll "
          "value is not provided, has no effect")
      .def(
          "maxSharedMemory",
          &tc::CudaMappingOptions::maxSharedMemory,
          "The amount of shared memory to use, in bytes. If not provided, "
          "TC will query the active GPU and use all available shared memory.")
      .def(
          "useReadOnlyCache",
          &tc::CudaMappingOptions::useReadOnlyCache,
          "Use the readonly cache (i.e. emit __ldg loads)");
}

} // namespace python
} // namespace tc
