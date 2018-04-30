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

#include "tc/core/cuda/cuda_mapping_options.h"

namespace tc {
namespace python {

namespace py = pybind11;

PYBIND11_MODULE(mapping_options, m) {
  m.doc() = "Python bindings for setting the mapping options";
  py::class_<tc::CudaMappingOptions>(
      m, "Options", "Mapping Options for a Tensor Comprehensions (TC)")
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
          "Initialize the mapping options from one of the following:\n 1. naive\n 2. pointwise\n 3. mlp\n 4. conv\n 5. group_conv\n 6. single_thread")
      .def(
          "maxSharedMemory",
          &tc::CudaMappingOptions::maxSharedMemory,
          "The amount of shared memory to use, in bytes. If not provided, TC will query the active GPU and use all available shared memory.")
      .def(
          "useSharedMemory",
          &tc::CudaMappingOptions::useSharedMemory,
          "Create block-local copies of data in shared memory when this can leverage data reuse or global memory access coalescing")
      .def(
          "unrollCopyShared",
          &tc::CudaMappingOptions::unrollCopyShared,
          "Also unroll the copies to and from shared memory. If unroll value is not provided, has no effect")
      .def(
          "scheduleFusionStrategy",
          [](tc::CudaMappingOptions& instance, const std::string& type) {
            instance.scheduleFusionStrategy(type);
          },
          "Set up outerScheduleFusionStrategy and intraTileFusionStrategy to the given value")
      .def(
          "outerScheduleFusionStrategy",
          [](tc::CudaMappingOptions& instance, const std::string& type) {
            instance.outerScheduleFusionStrategy(type);
          },
          "Require TC to try and execute different TC expressions interleaved (Max), separately (Min)\nor interleaved as long as sufficient parallelism is exploited (Preserve3Coincident) by\nperforming loop fusion and fission. Applies to inner loops created by tiling")
      .def(
          "intraTileScheduleFusionStrategy",
          [](tc::CudaMappingOptions& instance, const std::string& type) {
            instance.intraTileScheduleFusionStrategy(type);
          },
          "Require TC to try and execute different TC expressions interleaved (Max), separately (Min)\nor interleaved as long as sufficient parallelism is exploited (Preserve3Coincident) by\nperforming loop fusion and fission. Applies before tiling")
      .def(
          "serialize",
          [](tc::CudaMappingOptions& instance) {
            std::string str = instance.toProtobufSerializedString();
            return py::bytes(str);
          },
          "Serialize the options to a protobuf string")
      .def(
          "tile",
          // pybind11 has implicit conversion from list -> vector
          [](tc::CudaMappingOptions& instance,
             std::vector<uint64_t>& tileSizes) { instance.tile(tileSizes); },
          "Perform loop tiling on the generated code with the given sizes. Independent of mapping to a\ngrid of thread blocks")
      .def(
          "mapToThreads",
          [](tc::CudaMappingOptions& instance,
             std::vector<uint64_t>& threadSizes) {
            instance.mapToThreads(threadSizes);
          },
          "The configuration of CUDA block, i.e. the number of CUDA threads in each block along three\ndimensions. Must be within the range allowed by CUDA (maximum 1024 for the first and second value,\n32 for the third, product below 1024)")
      .def(
          "mapToBlocks",
          [](tc::CudaMappingOptions& instance,
             std::vector<uint64_t>& blockSizes) {
            instance.mapToBlocks(blockSizes);
          },
          "The configuration of CUDA grid, i.e. the number of CUDA blocks along three dimensions. Must be\nwithin the range allowed by CUDA (maximum 2^31-1 for the first value and 65535 for the second and third)")
      .def(
          "matchLibraryCalls",
          [](tc::CudaMappingOptions& instance, bool match) {
            instance.matchLibraryCalls(match);
          },
          "Replace computation patterns with calls to highly optimized libraries (such as CUB, CUTLASS) when possible")
      .def(
          "fixParametersBeforeScheduling",
          [](tc::CudaMappingOptions& instance, bool fix) {
            instance.fixParametersBeforeScheduling(fix);
          },
          "Perform automatic loop scheduling taking into account specific tensor sizes.\nMay produce faster kernels but significantly increases compilation time.\n Note that the mapping will be performed for specific tensor sizes anyway")
      .def(
          "unroll",
          [](tc::CudaMappingOptions& instance, uint64_t factor) {
            instance.unroll(factor);
          },
          "Perform loop unrolling on the generated code and produce at most the given number of statements");
}

} // namespace python
} // namespace tc
