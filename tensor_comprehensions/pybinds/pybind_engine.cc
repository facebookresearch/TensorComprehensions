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
#include <sstream>
#include <string>
#include <vector>

#include <Python.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <ATen/ATen.h>

#include "pybind_utils.h"
#include "tc/aten/aten_compiler.h"
#include "tc/core/autodiff.h"
#include "tc/core/cuda/cuda_compilation_cache.h"
#include "tc/core/cuda/cuda_mapping_options.h"
#include "tc/core/cuda/cuda_tc_executor.h"
#include "tc/core/flags.h"
#include "tc/core/scope_guard.h"

namespace tc {
namespace python {

namespace py = pybind11;

using ATenCudaCompilationUnit = tc::ATenCompilationUnit<tc::CudaTcExecutor>;

PYBIND11_MODULE(tc, m) {
  m.def("_differentiate", differentiate);
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

  py::object dlpack;
  try {
    dlpack = py::module::import("torch.utils.dlpack");
  } catch (std::exception& e) {
    std::cerr << "\n PyTorch installation is missing, binary will be useless \n"
              << e.what() << std::endl;
  }
  py::class_<ATenCudaCompilationUnit>(m, "ATenCompilationUnit")
      .def(py::init<>())
      .def("define", &ATenCudaCompilationUnit::define, "Define the TC language")
      .def(
          "compile",
          [dlpack](
              ATenCudaCompilationUnit& instance,
              const std::string& name,
              py::list& inputs,
              const tc::CudaMappingOptions& options) {
            std::vector<at::Tensor> atInputs = getATenTensors(inputs, dlpack);
            return instance.compile(name, atInputs, options);
          })
      .def(
          "run",
          [dlpack](
              ATenCudaCompilationUnit& instance,
              const std::string& name,
              py::list& inputs,
              py::list& outputs,
              size_t handle) {
            std::vector<at::Tensor> atInputs = getATenTensors(inputs, dlpack);
            std::vector<at::Tensor> atOutputs = getATenTensors(outputs, dlpack);
            instance.run(name, atInputs, atOutputs, handle);
            if (py::len(outputs) == 0) {
              convertToPyObjects(atOutputs, dlpack, outputs);
            }
          })
      .def(
          "uncheckedRun",
          [dlpack](
              ATenCudaCompilationUnit& instance,
              py::list& inputs,
              py::list& outputs,
              size_t handle) {
            CHECK_LT(0, outputs.size());
            std::vector<at::Tensor> atInputs = getATenTensors(inputs, dlpack);
            std::vector<at::Tensor> atOutputs = getATenTensors(outputs, dlpack);
            instance.uncheckedRun(atInputs, atOutputs, handle);
          })
      .def(
          "inject_cuda",
          [dlpack](
              ATenCudaCompilationUnit& instance,
              const std::string& name,
              const std::string& injectedKernelName,
              const std::string& cudaSource,
              py::list& inputs,
              std::vector<uint64_t> grid,
              std::vector<uint64_t> block) {
            tc::ManualCudaCache::enableCache();
            tc::CudaMappingOptions options =
                tc::CudaMappingOptions::makeNaiveCudaMappingOptions();
            std::vector<at::Tensor> atInputs = getATenTensors(inputs, dlpack);
            auto tensorsPair = tc::toConstDlpackTensors(atInputs);
            tc::ScopeGuard g(
                [&]() { tc::deleteDlmTensors(tensorsPair.second); });
            auto outTensorInfo = instance.inferOutputTensorInfo(name, atInputs);
            tc::ManualCudaCache::getCache()->cacheKernel(
                name,
                tensorsPair.first,
                outTensorInfo,
                injectedKernelName,
                {},
                cudaSource,
                tc::Grid(grid),
                tc::Block(block));
          });
}

} // namespace python
} // namespace tc
