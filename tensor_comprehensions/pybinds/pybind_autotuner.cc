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

#include <ATen/ATen.h>

#include "pybind_utils.h"
#include "tc/autotuner/genetic_autotuner_aten.h"
#include "tc/core/cuda/cuda_mapping_options.h"
#include "tc/core/flags.h"

namespace tc {
namespace python {

namespace py = pybind11;

PYBIND11_MODULE(autotuner, m) {
  m.doc() =
      "Python bindings for autotuning the kernels starting from some options";
  py::object dlpack;
  try {
    dlpack = py::module::import("torch.utils.dlpack");
  } catch (std::exception& e) {
    std::cerr << "\n PyTorch installation is missing, binary will be useless \n"
              << e.what() << std::endl;
  }
  py::class_<tc::autotune::GeneticAutotunerATen>(m, "ATenAutotuner")
      .def(py::init<const std::string>())
      .def(
          "pop_size",
          [](tc::autotune::GeneticAutotunerATen& instance, uint32_t& pop_size) {
            tc::FLAGS_tuner_gen_pop_size = pop_size;
          })
      .def(
          "crossover_rate",
          [](tc::autotune::GeneticAutotunerATen& instance,
             uint32_t& crossover_rate) {
            tc::FLAGS_tuner_gen_crossover_rate = crossover_rate;
          })
      .def(
          "mutation_rate",
          [](tc::autotune::GeneticAutotunerATen& instance,
             uint32_t& mutation_rate) {
            tc::FLAGS_tuner_gen_mutation_rate = mutation_rate;
          })
      .def(
          "generations",
          [](tc::autotune::GeneticAutotunerATen& instance,
             uint32_t& generations) {
            tc::FLAGS_tuner_gen_generations = generations;
          })
      .def(
          "number_elites",
          [](tc::autotune::GeneticAutotunerATen& instance,
             uint32_t& number_elites) {
            tc::FLAGS_tuner_gen_number_elites = number_elites;
          })
      .def(
          "threads",
          [](tc::autotune::GeneticAutotunerATen& instance, uint32_t& threads) {
            tc::FLAGS_tuner_threads = threads;
          })
      .def(
          "gpus",
          [](tc::autotune::GeneticAutotunerATen& instance, std::string& gpus) {
            tc::FLAGS_tuner_gpus = gpus;
          })
      .def(
          "restore_from_proto",
          [](tc::autotune::GeneticAutotunerATen& instance,
             bool restore_from_proto) {
            tc::FLAGS_tuner_gen_restore_from_proto = restore_from_proto;
          })
      .def(
          "restore_number",
          [](tc::autotune::GeneticAutotunerATen& instance,
             uint32_t& restore_number) {
            tc::FLAGS_tuner_gen_restore_number = restore_number;
          })
      .def(
          "log_generations",
          [](tc::autotune::GeneticAutotunerATen& instance,
             bool log_generations) {
            tc::FLAGS_tuner_gen_log_generations = log_generations;
          })
      .def(
          "tuner_min_launch_total_threads",
          [](tc::autotune::GeneticAutotunerATen& instance,
             bool tuner_min_launch_total_threads) {
            tc::FLAGS_tuner_min_launch_total_threads =
                tuner_min_launch_total_threads;
          })
      .def(
          "save_best_candidates_count",
          [](tc::autotune::GeneticAutotunerATen& instance,
             bool save_best_candidates_count) {
            tc::FLAGS_tuner_save_best_candidates_count =
                save_best_candidates_count;
          })
      .def(
          "tune",
          [dlpack](
              tc::autotune::GeneticAutotunerATen& instance,
              const std::string& cacheFileName,
              const std::string& tcName,
              py::list& inputs,
              tc::CudaMappingOptions& baseMapping,
              std::vector<tc::CudaMappingOptions>& startingOptions) {
            std::vector<at::Tensor> atInputs = getATenTensors(inputs, dlpack);
            auto bestOptions = instance.tune(
                cacheFileName, tcName, atInputs, baseMapping, startingOptions);
            if (bestOptions) {
              return *bestOptions;
            } else {
              std::cout << "Autotuner could not find options, returning base"
                        << std::endl;
              return baseMapping;
            }
          })
      .def(
          "load",
          [dlpack](
              tc::autotune::GeneticAutotunerATen& instance,
              const std::string& cacheFileName,
              const std::string& tcName,
              py::list& inputs,
              const size_t& numCandidates) {
            std::vector<at::Tensor> atInputs = getATenTensors(inputs, dlpack);
            std::vector<tc::CudaMappingOptions> mappingOptions =
                instance.load(cacheFileName, tcName, atInputs, numCandidates);
            return mappingOptions;
          });
}

} // namespace python
} // namespace tc
