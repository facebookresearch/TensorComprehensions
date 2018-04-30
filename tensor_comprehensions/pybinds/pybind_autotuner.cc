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
#include "tc/core/cuda/cuda_backend.h"
#include "tc/core/cuda/cuda_tc_executor.h"
#include "tc/core/flags.h"

namespace tc {
namespace python {

namespace py = pybind11;

using ATenCudaTuner =
    tc::aten::ATenAutotuner<tc::CudaBackend, tc::autotune::GeneticSearch>;

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
  py::class_<ATenCudaTuner>(m, "ATenAutotuner")
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
              const std::string& tcName,
              py::list& inputs,
              tc::CudaMappingOptions& baseMapping,
              const std::string& cacheFileName) {
            std::vector<at::Tensor> atInputs = getATenTensors(inputs, dlpack);
            auto bestOptions =
                instance.tune(tcName, atInputs, baseMapping, cacheFileName);
            if (bestOptions.size() == 0) {
              return bestOptions[0];
            } else {
              std::cout << "Autotuner could not find options, returning base"
                        << std::endl;
              return baseMapping;
            }
          });
}

} // namespace python
} // namespace tc
