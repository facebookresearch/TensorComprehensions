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

#include <map>
#include <string>
#include <vector>

#include "tc/core/mapping_options.h"
#include "tc/core/tensor.h"
#include "tc/lang/tree.h"

/**
 * This provides a simple functional-style C++ API with multi-backend
 * capabilities to:
 *   1. compile a TC function and return an Executor for the specified Backend
 *      on which the run method can be called;
 *   2. infer actual tmp/output tensor shapes given input tensor shapes;
 *   3. parse a TC definition and retrieve the map of TC function to parsed TC
 *      trees.
 *
 * Compilation is backed by a compilation cache, its correspondence is:
 * 1 TcExecutor <-> 1 compiled tuple<TC function, input shapes, MappingOptions>
 *
 * The compile function is templated by the Backend type.
 * For each backend, the specific Backend type lives in
 *   backendname/backendname_backend.h and declares all the required dependent
 *   **derived** types.
 * For example:
 *   CudaBackend is declared in core/cuda/cuda_backend.h
 *
 * struct CudaBackend {
 *   using ExecutorType = CudaTcExecutor;
 *   using MappingOptionsType = CudaMappingOptions;
 *   using CompilationResultType = CudaCompilationResult;
 *   using RTCFunctionType = CudaRTCFunction;
 * };
 *
 * Sketching usage resembles:
 *   std::string someTc = "...";
 *   auto pExecutor = tc::compile<CudaBackend>(
 *     someTc, tcFunctionName, inputs, mappingOptions);
 *   auto profilingInfo = pExecutor->profile(handle, inputs, outputs, true);
 *   // alternatively:
 *   // auto kernelTiming = pExecutor->uncheckedRun(inputs, outputs, true);
 */
namespace tc {
/// Given a TC string containing multiple functions and a TC function name
/// "entryPoint", this function compiles a new TcExecutor for the specified
/// Backend. For now, contiguous output sizes are inferred given input sizes.
/// If you need another kernel for another entryPoint or other inputs or
//  other options then just compile another TcExecutor; because atm we fully
/// JIT specialize on all sizes.
/// \returns a new TcExecutor on which the run method can be called to run
/// entryPoint
template <typename Backend>
std::unique_ptr<typename Backend::ExecutorType> compile(
    const std::string& tc,
    const std::string& entryPoint,
    const std::vector<const DLConstTensor*>& inputs,
    /* TODO: in the future also pass outputs for stride and alignment info */
    const typename Backend::MappingOptionsType& options);

/// Given a TC representation as a TC + TC function name entryPoint and a list
/// of input tensors that match the definition in the TC function definition
/// (in positional order), this generates the output TensorInfo resulting from
/// running inference.
/// The typical flow is to infer output sizes, allocate/resize them within
/// you favorite ML framework/tensor library and then call compile and run.
/// \returns a vector of TensorInfo which can be used for allocating and
/// performing output shape validation.
std::vector<TensorInfo> inferOutputTensorInfo(
    const std::string& tc,
    const std::string& entryPoint,
    const std::vector<const DLConstTensor*> inputs);

namespace detail {
/// Given a TC representation, this parses the TC functions into a map of
/// TreeRef indexed by TC function names.
/// \returns an ordered map of TC function name to parsed TC tree
std::map<std::string, lang::TreeRef> parse(const std::string& tc);

/// Given a TC representation as a TreeRef, this function compiles a new
/// TcExecutor for the specified Backend.
/// For now, contiguous output sizes are inferred given input sizes.
/// If you need another kernel for another TC or other inputs or options then
/// just compile another TcExecutor; because atm we fully JIT specialize on all
/// sizes.
/// \returns a new TcExecutor on which the run method can be called
template <typename Backend>
std::unique_ptr<typename Backend::ExecutorType> compile(
    lang::TreeRef tcDefinition,
    const std::vector<const DLConstTensor*>& inputs,
    /* TODO: in the future also pass outputs for stride and alignment info */
    const typename Backend::MappingOptionsType& options);

/// Given a TC representation as a TreeRef and a list of input tensors that
/// match the definition in the TC function definition (in positional order),
/// this generates the output TensorInfo resulting from running inference.
/// The typical flow is to infer output sizes, allocate/resize them within
/// you favorite ML framework/tensor library and then call compile and run.
/// \returns a vector of TensorInfo which can be used for allocating and
/// performing output shape validation.
std::vector<TensorInfo> inferOutputTensorInfo(
    lang::TreeRef tcDefinition,
    const std::vector<const DLConstTensor*> inputs);

} // namespace detail
} // namespace tc

#include "tc/core/compiler-inl.h"
