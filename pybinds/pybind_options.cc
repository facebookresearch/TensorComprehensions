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
#include <iostream>
#include <string>
#include <vector>

#include <Python.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "tc/core/mapping_options.h"

namespace tc {
namespace python {

namespace py = pybind11;

PYBIND11_MODULE(mapping_options, m) {
  m.doc() = "Python bindings for setting the mapping options";
  py::class_<tc::MappingOptions>(m, "Options")
      .def(py::init([](std::string type) {
        if (type == "naive") {
          return tc::MappingOptions::makeNaiveMappingOptions();
        }
        if (type == "single_thread") {
          return tc::MappingOptions::makeSingleThreadMappingOptions();
        }
        if (type == "pointwise") {
          return tc::MappingOptions::makePointwiseMappingOptions();
        }
        if (type == "mlp") {
          return tc::MappingOptions::makeMlpMappingOptions();
        }
        if (type == "conv") {
          return tc::MappingOptions::makeConvolutionMappingOptions();
        }
        if (type == "group_conv") {
          return tc::MappingOptions::makeGroupConvolutionMappingOptions();
        }
        throw std::runtime_error("Invalid option passed");
      }))
      .def("maxSharedMemory", &tc::MappingOptions::maxSharedMemory)
      .def("useSharedMemory", &tc::MappingOptions::useSharedMemory)
      .def("usePrivateMemory", &tc::MappingOptions::usePrivateMemory)
      .def("unrollCopyShared", &tc::MappingOptions::unrollCopyShared)
      .def("tileImperfectlyNested", &tc::MappingOptions::tileImperfectlyNested)
      .def("matchLibraryCalls", &tc::MappingOptions::matchLibraryCalls)
      .def(
          "fixParametersBeforeScheduling",
          &tc::MappingOptions::fixParametersBeforeScheduling)
      .def("unroll", &tc::MappingOptions::unroll)
      .def(
          "scheduleFusionStrategy",
          [](tc::MappingOptions& instance, const std::string& type) {
            instance.scheduleFusionStrategy(type);
          })
      .def(
          "outerScheduleFusionStrategy",
          [](tc::MappingOptions& instance, const std::string& type) {
            instance.outerScheduleFusionStrategy(type);
          })
      .def(
          "intraTileScheduleFusionStrategy",
          [](tc::MappingOptions& instance, const std::string& type) {
            instance.intraTileScheduleFusionStrategy(type);
          })
      .def(
          "intraTileScheduleAllowSkewing",
          &tc::MappingOptions::intraTileScheduleAllowSkewing)
      .def(
          "intraTileSchedulePositiveOrthant",
          &tc::MappingOptions::intraTileSchedulePositiveOrthant)
      .def(
          "tile",
          // pybind11 has implicit conversion from list -> vector
          [](tc::MappingOptions& instance, std::vector<uint64_t>& tileSizes) {
            instance.tile(tileSizes);
          })
      .def(
          "mapToThreads",
          [](tc::MappingOptions& instance, std::vector<uint64_t>& threadSizes) {
            instance.mapToThreads(threadSizes);
          })
      .def(
          "mapToBlocks",
          [](tc::MappingOptions& instance, std::vector<uint64_t>& blockSizes) {
            instance.mapToBlocks(blockSizes);
          });
}

} // namespace python
} // namespace tc
