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
#include <fstream>
#include <vector>

#include <gflags/gflags.h>
#include <glog/logging.h>
#include <gtest/gtest.h>

#include <version.h>
#include "tc/aten/aten.h"
#include "tc/aten/aten_autotuner.h"
#include "tc/aten/aten_compiler.h"
#include "tc/autotuner/genetic_search.h"
#include "tc/autotuner/utils.h"
#include "tc/core/check.h"
#include "tc/core/cuda/cuda.h"
#include "tc/core/cuda/cuda_mapping_options.h"
#include "tc/core/cuda/cuda_rtc.h"
#include "tc/core/cuda/cuda_tc_executor.h"
#include "tc/core/flags.h"
#include "tc/core/scope_guard.h"
#include "tc/core/tensor.h"
#include "tc/core/utils/time.h"
#include "tc/lang/canonicalize.h"

#include <cublas_v2.h> // Must be the same as Caffe2
#include <cuda_runtime_api.h>
#include <cudnn.h>
#include <nvrtc.h>

#define TC_CUDA_CUBLAS_ENFORCE(condition)                       \
  do {                                                          \
    if (condition != CUBLAS_STATUS_SUCCESS) {                   \
      std::stringstream ss;                                     \
      ss << "Error at: " << __FILE__ << ":" << __LINE__ << ": " \
         << "CUBLAS_STATUS_ERROR: " << (int)condition;          \
      throw std::runtime_error(ss.str().c_str());               \
    }                                                           \
  } while (0)

DEFINE_bool(
    disable_version_checks,
    false,
    "Test on other platforms than we claim perf results for");
DEFINE_bool(autotune, false, "Enable autotuning");

struct Benchmark : public ::testing::Test {
  void SetUp() {
    if (!FLAGS_disable_version_checks) {
      auto cudnnVersion = cudnnGetVersion();
      TC_CHECK_LE(6021u, cudnnVersion)
          << "[CUDNN][VERSION] Enforce version compatibility check";

      auto cudaRtVersion = cudnnGetCudartVersion();
      TC_CHECK_LE(8000u, cudaRtVersion)
          << "[CUDART][VERSION] Enforce version compatibility check";

      int cublasVersion;
      cublasHandle_t handle;
      TC_CUDA_CUBLAS_ENFORCE(cublasCreate_v2(&handle));
      TC_CUDA_CUBLAS_ENFORCE(cublasGetVersion_v2(handle, &cublasVersion));
      TC_CHECK_LE(8000, cublasVersion)
          << "[CUBLAS][VERSION] Enforce version compatibility check";
      tc::ScopeGuard sg(
          [&handle]() { TC_CUDA_CUBLAS_ENFORCE(cublasDestroy_v2(handle)); });

      int cudaRuntimeVersion;
      TC_CUDA_RUNTIMEAPI_ENFORCE(cudaRuntimeGetVersion(&cudaRuntimeVersion));
      TC_CHECK_LE(8000, cudaRuntimeVersion)
          << "[CUDA RUNTIME][VERSION] Enforce version compatibility check";

      int nvrtcVersionMajor;
      int nvrtcVersionMinor;
      TC_NVRTC_CHECK(nvrtcVersion(&nvrtcVersionMajor, &nvrtcVersionMinor));
      TC_CHECK_LE(8, nvrtcVersionMajor)
          << "[NVRTC][MAJOR][VERSION] Enforce version compatibility check";
      TC_CHECK_LE(0, nvrtcVersionMinor)
          << "[NVRTC][MINOR][VERSION] Enforce version compatibility check";
    }
  }

  using CheckFunction = std::function<bool(
      const std::vector<at::Tensor>& inputs,
      const std::vector<at::Tensor>& outputs)>;
  using PrologueFunction = std::function<tc::ProfilingInfo()>;
  std::vector<at::Tensor> Check(
      const std::string& tc,
      const std::string& name,
      const tc::CudaMappingOptions& mappingOptions,
      const std::vector<at::Tensor>& inputs,
      CheckFunction check_fun =
          [](const std::vector<at::Tensor>& inputs,
             const std::vector<at::Tensor>& outputs) { return true; },
      PrologueFunction prologue_fun =
          []() {
            return tc::ProfilingInfo{tc::Duration::zero(),
                                     tc::Duration::zero()};
          }) {
    // 1. Compile, run and check
    auto pExecutor =
        tc::aten::compile<tc::CudaBackend>(tc, name, inputs, mappingOptions);
    std::vector<at::Tensor> outputs =
        tc::aten::prepareOutputs(tc, name, inputs);
    tc::aten::run(*pExecutor, inputs, outputs);
    EXPECT_TRUE(check_fun(inputs, outputs));
    // 2. Run and report compiled kernel runtime
    std::vector<at::Tensor> outputs2 =
        tc::aten::prepareOutputs(tc, name, inputs);
    RunAndReport(
        [&pExecutor, &inputs, &outputs2, prologue_fun]() {
          prologue_fun();
          tc::aten::run(*pExecutor, inputs, outputs2);
        },
        [&pExecutor, &inputs, &outputs2, prologue_fun]() {
          TC_CUDA_RUNTIMEAPI_ENFORCE(cudaDeviceSynchronize());
          auto prologueTimings = prologue_fun();
          TC_CUDA_RUNTIMEAPI_ENFORCE(cudaDeviceSynchronize());
          auto timings = tc::aten::profile(*pExecutor, inputs, outputs2);
          TC_CUDA_RUNTIMEAPI_ENFORCE(cudaDeviceSynchronize());
          return prologueTimings.kernelRuntime + timings.kernelRuntime;
        },
        "COMPILED KERNEL");
    // 3. Run and report total compiled time (kernel runtime + CPU overhead)
    RunAndReport(
        [&pExecutor, &inputs, &outputs2]() {
          tc::aten::run(*pExecutor, inputs, outputs2);
        },
        [&pExecutor, &inputs, &outputs2, prologue_fun]() {
          TC_CUDA_RUNTIMEAPI_ENFORCE(cudaDeviceSynchronize());
          auto start(std::chrono::system_clock::now());
          prologue_fun();
          tc::aten::uncheckedRun(*pExecutor, inputs, outputs2);
          TC_CUDA_RUNTIMEAPI_ENFORCE(cudaDeviceSynchronize());
          return tc::Duration::since(start);
        },
        "COMPILED KERNEL + CPU");
    return outputs;
  }

  template <typename InitFunction, typename InplaceFunction>
  void Reference(InitFunction init, InplaceFunction compute) {
    // 1. Initialize1
    auto res = init();
    // 2. Run and report reference runtime
    RunAndReport(
        [&res, compute]() { compute(res); },
        [&res, compute]() {
          TC_CUDA_RUNTIMEAPI_ENFORCE(cudaDeviceSynchronize());
          auto start(std::chrono::system_clock::now());
          compute(res);
          TC_CUDA_RUNTIMEAPI_ENFORCE(cudaDeviceSynchronize());
          return tc::Duration::since(start);
        },
        "REFERENCE IMPL.");
  }

  std::vector<tc::CudaMappingOptions> autotune(
      std::string tc,
      std::string kernelName,
      std::vector<at::Tensor> inputs,
      tc::CudaMappingOptions baseMapping,
      CheckFunction check_fun =
          [](const std::vector<at::Tensor>&, const std::vector<at::Tensor>&) {
            return true;
          },
      const tc::autotune::TuningParameterFixer& fixedParams = {}) {
    if (!FLAGS_autotune) {
      return {};
    }
    tc::aten::ATenAutotuner<tc::CudaBackend, tc::autotune::GeneticSearch>
        geneticAutotuneATen(tc);
    auto bestOptions = [&]() {
      auto options = geneticAutotuneATen.tune(
          kernelName, inputs, {baseMapping}, 1, fixedParams);
      TC_CHECK_GE(options.size(), 1u) << "Benchmark mode: at least one "
                                      << "options expected";
      return options[0];
    }();
    Check(tc, kernelName, bestOptions, inputs, check_fun);
    return {bestOptions};
  }

 private:
  void RunAndReport(
      std::function<void(void)> warmupFn,
      std::function<tc::Duration(void)> runFn,
      const std::string& reportName) {
    for (size_t i = 1; i < tc::FLAGS_benchmark_warmup; ++i) {
    }
    std::vector<tc::Duration> durations;
    for (size_t i = 0; i < tc::FLAGS_benchmark_iterations; ++i) {
      durations.push_back(runFn());
    }

    auto p50idx = static_cast<int>(std::ceil(0.5 * durations.size()));
    auto p90idx = static_cast<int>(std::ceil(0.9 * durations.size()));
    auto p99idx = static_cast<int>(std::ceil(0.99 * durations.size()));

    std::sort(durations.begin(), durations.end());
#define GET_US(X) ((X)).toMicroSeconds()

    std::cout << "\n---------------------------------------------------------";
    std::cout << "\n---------------- " << reportName << " STATS --------------";
    std::cout << "\n------------------    " << tc::FLAGS_benchmark_iterations
              << " ITERATIONS    ----------------";
    std::cout << "\n---------------------------------------------------------";
    std::cout << "\n";
    std::cout
        << "Min: " << GET_US(durations.front()) << "us, "
        << "p50: "
        << GET_US(durations.at(std::min(p50idx, (int)durations.size() - 1)))
        << "us, "
        << "p90: "
        << GET_US(durations.at(std::min(p90idx, (int)durations.size() - 1)))
        << "us, "
        << "p99: "
        << GET_US(durations.at(std::min(p99idx, (int)durations.size() - 1)))
        << "us, "
        << "Max: " << GET_US(durations.back()) << "us";
    std::cout << "\n---------------------------------------------------------";
    std::cout << "\n\n";

#undef GET_US
  }
};
