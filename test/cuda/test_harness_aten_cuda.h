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
#include <cmath>
#include <iostream>
#include <limits>

#include <gflags/gflags.h>
#include <gtest/gtest.h>

#include "tc/aten/aten.h"

#include <cuda_runtime_api.h>

#include "tc/aten/aten_compiler.h"
#include "tc/core/cuda/cuda.h"
#include "tc/core/cuda/cuda_tc_executor.h"
#include "tc/core/flags.h"

#include "../../test/test_harness_aten.h"

at::Tensor subtensor(at::Tensor& tensor, int dim, int groups, int g) {
  if (!tensor.defined()) {
    return at::Tensor();
  }
  int64_t n = tensor.sizes()[dim] / groups;
  return tensor.narrow(dim, n * g, n).contiguous();
}

void setAtenSeed(uint64_t seed, at::Backend backend) {
  at::Generator& gen = at::globalContext().defaultGenerator(backend);
  gen.manualSeed(seed);
}

uint64_t getAtenSeed(at::Backend backend) {
  at::Generator& gen = at::globalContext().defaultGenerator(backend);
  return gen.seed();
}

void benchmarkKernelOptions(
    const std::string& tc,
    const std::string& name,
    const std::vector<at::Tensor>& inputs,
    const tc::CudaMappingOptions mappingOptions) {
  auto pExecutor =
      tc::aten::compile<tc::CudaBackend>(tc, name, inputs, mappingOptions);
  std::vector<at::Tensor> outputs = tc::aten::prepareOutputs(tc, name, inputs);
  tc::aten::run(*pExecutor, inputs, outputs);
  for (size_t i = 1; i < tc::FLAGS_benchmark_warmup; ++i) {
    tc::aten::run(*pExecutor, inputs, outputs);
  }
  std::vector<tc::Duration> kernelTimes;
  kernelTimes.reserve(tc::FLAGS_benchmark_iterations);
  std::vector<tc::Duration> totalTimes;
  totalTimes.reserve(tc::FLAGS_benchmark_iterations);
  for (size_t i = 0; i < tc::FLAGS_benchmark_iterations; ++i) {
    auto timings = tc::aten::profile(*pExecutor, inputs, outputs);
    kernelTimes.push_back(timings.kernelRuntime);
    TC_CUDA_RUNTIMEAPI_ENFORCE(cudaDeviceSynchronize());
    auto time(std::chrono::system_clock::now());
    tc::aten::uncheckedRun(*pExecutor, inputs, outputs);
    TC_CUDA_RUNTIMEAPI_ENFORCE(cudaDeviceSynchronize());
    totalTimes.push_back(std::chrono::duration_cast<std::chrono::microseconds>(
        std::chrono::system_clock::now() - time));
  }

  auto p50idx = static_cast<int>(std::ceil(0.5 * kernelTimes.size()));
  auto p90idx = static_cast<int>(std::ceil(0.9 * kernelTimes.size()));
  auto p99idx = static_cast<int>(std::ceil(0.99 * kernelTimes.size()));

  std::sort(kernelTimes.begin(), kernelTimes.end());
#define GET_US(X) \
  (std::chrono::duration_cast<std::chrono::microseconds>((X)).count())

  std::cout << "\n---------------------------------------------------------";
  std::cout << "\n--------------------- KERNEL STATS ----------------------";
  std::cout << "\n------------------    " << tc::FLAGS_benchmark_iterations
            << " ITERATIONS    ----------------";
  std::cout << "\n---------------------------------------------------------";
  std::cout << "\n";
  std::cout
      << "Min: " << GET_US(kernelTimes.front()) << "us, "
      << "p50: "
      << GET_US(kernelTimes.at(std::min(p50idx, (int)kernelTimes.size() - 1)))
      << "us, "
      << "p90: "
      << GET_US(kernelTimes.at(std::min(p90idx, (int)kernelTimes.size() - 1)))
      << "us, "
      << "p99: "
      << GET_US(kernelTimes.at(std::min(p99idx, (int)kernelTimes.size() - 1)))
      << "us, "
      << "Max: " << GET_US(kernelTimes.back()) << "us";
  std::cout << "\n---------------------------------------------------------";
  std::cout << "\n\n";

#undef GET_US

  std::sort(totalTimes.begin(), totalTimes.end());
#define GET_US(X) \
  (std::chrono::duration_cast<std::chrono::microseconds>((X)).count())

  std::cout << "\n---------------------------------------------------------";
  std::cout << "\n-----------------------  TOTAL STATS --------------------";
  std::cout << "\n------------------    " << tc::FLAGS_benchmark_iterations
            << " ITERATIONS    ----------------";
  std::cout << "\n---------------------------------------------------------";
  std::cout << "\n";
  std::cout
      << "Min: " << GET_US(totalTimes.front()) << "us, "
      << "p50: "
      << GET_US(totalTimes.at(std::min(p50idx, (int)totalTimes.size() - 1)))
      << "us, "
      << "p90: "
      << GET_US(totalTimes.at(std::min(p90idx, (int)totalTimes.size() - 1)))
      << "us, "
      << "p99: "
      << GET_US(totalTimes.at(std::min(p99idx, (int)totalTimes.size() - 1)))
      << "us, "
      << "Max: " << GET_US(totalTimes.back()) << "us";
  std::cout << "\n---------------------------------------------------------";
  std::cout << "\n\n";

#undef GET_US
}
