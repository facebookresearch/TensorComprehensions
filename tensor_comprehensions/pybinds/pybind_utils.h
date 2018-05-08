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

#include "tc/aten/aten.h"

namespace tc {
namespace python {

namespace py = pybind11;

std::vector<at::Tensor> getATenTensors(py::list& pyTensors, py::object dlpack) {
  std::vector<at::Tensor> atTensors;
  for (auto& inp : pyTensors) {
    py::object obj = dlpack.attr("to_dlpack")(inp);
    DLManagedTensor* dlMTensor =
        (DLManagedTensor*)PyCapsule_GetPointer(obj.ptr(), "dltensor");
    atTensors.push_back(at::fromDLPack(dlMTensor));
  }
  return atTensors;
}

void convertToPyObjects(
    std::vector<at::Tensor>& inputs,
    py::object dlpack,
    py::list& outputs) {
  for (auto& tensor : inputs) {
    auto obj = py::cast<py::object>(
        PyCapsule_New(at::toDLPack(tensor), "dltensor", NULL));
    outputs.append(dlpack.attr("from_dlpack")(obj));
  }
}

} // namespace python
} // namespace tc
