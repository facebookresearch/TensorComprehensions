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

#include <ATen/ATen.h>

#include "pybind_utils.h"
#include "tc/aten/aten_compiler.h"
#include "tc/core/cuda/cuda_compilation_cache.h"
#include "tc/core/cuda/cuda_tc_executor.h"
#include "tc/core/flags.h"
#include "tc/core/mapping_options.h"
#include "tc/core/scope_guard.h"

namespace tc {
namespace python {

namespace py = pybind11;

PYBIND11_MODULE(tc, m) {
  m.def(
      "global_debug_init", // exposing the debugging flags to people
      [](std::vector<std::string> args) {
        if (args.size() > 0) {
          args.insert(args.begin(), "tc");
        }
        int numArgs = args.size();
        // now we construct a char** argv type from args
        std::vector<char*> vargs; // char* vector args
        for (auto& arg : args) {
          vargs.push_back(const_cast<char*>(arg.data()));
        }
        char** argv = vargs.data();
        tc::python::globalDebugGflagsGlogInit(&numArgs, &argv);
      });

  py::object dlpack;
  try {
    dlpack = py::module::import("torch.utils.dlpack");
  } catch (std::exception& e) {
    std::cerr << "\n PyTorch installation is missing, binary will be useless \n"
              << e.what() << std::endl;
  }
  py::class_<tc::ATenCompilationUnit<tc::CudaTcExecutor>>(
      m, "ATenCompilationUnit")
      .def(py::init<>())
      .def(
          "define",
          &tc::ATenCompilationUnit<tc::CudaTcExecutor>::define,
          "Define the TC language")
      .def(
          "compile",
          [dlpack](
              tc::ATenCompilationUnit<tc::CudaTcExecutor>& instance,
              const std::string& name,
              py::list& inputs,
              const tc::MappingOptions& options) {
            std::vector<at::Tensor> atInputs = getATenTensors(inputs, dlpack);
            return instance.compile(name, atInputs, options);
          })
      .def(
          "run",
          [dlpack](
              tc::ATenCompilationUnit<tc::CudaTcExecutor>& instance,
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
              tc::ATenCompilationUnit<tc::CudaTcExecutor>& instance,
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
              tc::ATenCompilationUnit<tc::CudaTcExecutor>& instance,
              const std::string& name,
              const std::string& injectedKernelName,
              const std::string& cudaSource,
              py::list& inputs,
              std::vector<uint64_t> grid,
              std::vector<uint64_t> block) {
            tc::ManualCudaCache::enableCache();
            tc::MappingOptions options =
                tc::MappingOptions::makeNaiveMappingOptions();
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
