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

#include <chrono>
#include <string>
#include <unordered_set>
#include <vector>

#include "tc/core/tc2halide.h"
#include "tc/core/utils/cuda_info.h"
#include "tc/core/utils/dlpack.h"

namespace tc {

struct SignatureSourcePair {
  std::string C99Signature;
  std::string CUDASignature;
  std::string Source;
};

// The main (only?) purpose of this class is to build a PENCIL string with
// fully specialized JIT parameters to pass down to PET.
// This will be scrapped in a near future
struct HalidePencilState {
  std::string kernelName;
  std::vector<std::string> outputNames;
  std::vector<std::string> inputNames;
  std::vector<int> parameters;
  std::string kernelSpecializedName;
  std::string parameterSignature;

  std::map<std::string, int> pvm;
  SignatureSourcePair signatureSourcePair;
  std::vector<dlutils::DLTensorUPtr> outputsDLT;
};

// Codegen cannot be static atm because of the way it is implemented.
// TODO: This should be broken down into 2 functions, one static and one
// with the object state.
//
// This function takes pieces of Halide IR and generates "PENCIL" (for now
// really just simple C). This code is the fed to applyPpcg and JIT compiled to
// HPC kernels.
// This function also builds the DLTensor objects and returns ownership.
// The DLTensor returned are pure metadata and the .data field is left unset.
// It updates the internal state
HalidePencilState toPencil(
    const tc2halide::HalideComponents& components,
    const std::vector<const DLTensor*>& inputsDLT,
    bool scheduleSpecialize = false,
    const std::string& kernelName = "anon");

std::string C99Signature(
    const HalidePencilState& state,
    const std::vector<const DLTensor*>& outputs,
    const std::vector<const DLTensor*>& inputs,
    const std::unordered_set<std::string>& indirectAccesses);

std::string C99Body(const HalidePencilState& state);

std::string CUDASignature(
    const HalidePencilState& state,
    const std::vector<const DLTensor*>& outputs,
    const std::vector<const DLTensor*>& inputs);

const std::string& parameterSignature(const HalidePencilState& state);

// These are *const* accessors because they tie runtime parameter values to
// JIT compiled code. Any uncontrolled change there may blow things up.
void SetKernelName(const HalidePencilState& state, const std::string& name);
std::string GetKernelName(const HalidePencilState& state);
std::vector<int> GetKernelParameters(const HalidePencilState& state);
std::string KernelSpecializedName(const HalidePencilState& state);

// TODO: Delete one of the two following functions
std::map<std::string, int> GetParameterValMap(const HalidePencilState& state);
// Get the mapping from symbolic sizes to their effective values.
std::unordered_map<std::string, long> GetParameterValues(
    const HalidePencilState& state);

// Just generates a function body from a Halide stmt. Exposed for testing.
std::string halide2Pencil(const Halide::Internal::Stmt& s);

namespace detail {

inline std::string makeExternC(const std::string& s) {
  return std::string("extern \"C\" {\n") + s + std::string("\n}\n");
}

inline std::string CUDADecl(
    const std::string& kernelSpecializedName,
    const std::string& CudaSignature) {
  // TODO: Inject __launch_bounds__
  return std::string("__global__ void ") + kernelSpecializedName +
      std::string("(") + CudaSignature + std::string(")");
}

} // namespace detail
} // namespace tc
