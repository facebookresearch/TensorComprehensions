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

#include <ATen/ATen.h>
#include <version.h>

#include "tc/aten/aten_compiler.h"
#include "tc/autotuner/genetic_autotuner_aten.h"
#include "tc/autotuner/utils.h"
#include "tc/core/cuda/cuda.h"
#include "tc/core/cuda/cuda_compilation_cache.h"
#include "tc/core/cuda/cuda_mapping_options.h"
#include "tc/core/cuda/cuda_rtc.h"
#include "tc/core/cuda/cuda_tc_executor.h"
#include "tc/core/flags.h"
#include "tc/core/scope_guard.h"
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
DEFINE_string(save_tuner_proto_prefix, "/tmp", "Enable autotuning");
DEFINE_bool(validate_proto, false, "whether to load options from proto");

std::vector<const DLTensor*> inferOutputTensorInfo(
    const std::string& tc,
    const std::string& name,
    const std::vector<at::Tensor>& inputs) {
  tc::ATenCompilationUnit<tc::CudaTcExecutor> atCompl;
  atCompl.define(tc);
  return atCompl.inferOutputTensorInfo(name, inputs);
}

struct Benchmark : public ::testing::Test {
  void SetUp() {
    if (!FLAGS_disable_version_checks) {
      auto cudnnVersion = cudnnGetVersion();
      CHECK_LE(6021, cudnnVersion)
          << "[CUDNN][VERSION] Enforce version compatibility check";

      auto cudaRtVersion = cudnnGetCudartVersion();
      CHECK_LE(8000, cudaRtVersion)
          << "[CUDART][VERSION] Enforce version compatibility check";

      int cublasVersion;
      cublasHandle_t handle;
      TC_CUDA_CUBLAS_ENFORCE(cublasCreate_v2(&handle));
      TC_CUDA_CUBLAS_ENFORCE(cublasGetVersion_v2(handle, &cublasVersion));
      CHECK_LE(8000, cublasVersion)
          << "[CUBLAS][VERSION] Enforce version compatibility check";
      tc::ScopeGuard sg(
          [&handle]() { TC_CUDA_CUBLAS_ENFORCE(cublasDestroy_v2(handle)); });

      int cudaRuntimeVersion;
      TC_CUDA_RUNTIMEAPI_ENFORCE(cudaRuntimeGetVersion(&cudaRuntimeVersion));
      CHECK_LE(8000, cudaRuntimeVersion)
          << "[CUDA RUNTIME][VERSION] Enforce version compatibility check";

      int nvrtcVersionMajor;
      int nvrtcVersionMinor;
      TC_NVRTC_CHECK(nvrtcVersion(&nvrtcVersionMajor, &nvrtcVersionMinor));
      CHECK_LE(8, nvrtcVersionMajor)
          << "[NVRTC][MAJOR][VERSION] Enforce version compatibility check";
      CHECK_LE(0, nvrtcVersionMinor)
          << "[NVRTC][MINOR][VERSION] Enforce version compatibility check";
    }
  }

  template <typename CheckFunction>
  void Check(
      const std::string& tc,
      const std::string& name,
      const tc::CudaMappingOptions& mappingOptions,
      const std::vector<at::Tensor>& inputs,
      std::vector<at::Tensor>& outputs,
      CheckFunction checkFun = [](const std::vector<at::Tensor>& inputs,
                                  std::vector<at::Tensor>& outputs) {
        return true;
      }) {
    tc::ATenCompilationUnit<tc::CudaTcExecutor> atCompl;
    atCompl.define(tc);
    auto handle = atCompl.compile(name, inputs, mappingOptions);
    atCompl.run(name, inputs, outputs, handle);
    EXPECT_TRUE(checkFun(inputs, outputs));
    for (size_t i = 1; i < tc::FLAGS_benchmark_warmup; ++i) {
      atCompl.run(name, inputs, outputs, handle);
    }
    std::vector<tc::Duration> kernelTimes;
    kernelTimes.reserve(tc::FLAGS_benchmark_iterations);
    std::vector<tc::Duration> totalTimes;
    totalTimes.reserve(tc::FLAGS_benchmark_iterations);
    for (size_t i = 0; i < tc::FLAGS_benchmark_iterations; ++i) {
      kernelTimes.push_back(atCompl.run(name, inputs, outputs, handle, true));
      TC_CUDA_RUNTIMEAPI_ENFORCE(cudaDeviceSynchronize());
      auto time(std::chrono::system_clock::now());
      atCompl.uncheckedRun(inputs, outputs, handle);
      TC_CUDA_RUNTIMEAPI_ENFORCE(cudaDeviceSynchronize());
      totalTimes.push_back(
          std::chrono::duration_cast<std::chrono::microseconds>(
              std::chrono::system_clock::now() - time));
    }

    auto p50idx = static_cast<int>(std::ceil(0.5 * kernelTimes.size()));
    auto p90idx = static_cast<int>(std::ceil(0.9 * kernelTimes.size()));
    auto p99idx = static_cast<int>(std::ceil(0.99 * kernelTimes.size()));

    std::sort(kernelTimes.begin(), kernelTimes.end());
#define GET_US(X) \
  (std::chrono::duration_cast<std::chrono::microseconds>((X)).count())

    std::cout << "\n---------------------------------------------------------";
    std::cout << "\n------------------ COMPILED KERNEL STATS ----------------";
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
    std::cout << "\n------------------ COMPILED TOTAL STATS ----------------";
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

  template <typename InitFunction, typename InplaceFunction>
  void Reference(InitFunction init, InplaceFunction compute) {
    auto res = init();
    for (size_t i = 1; i < tc::FLAGS_benchmark_warmup; ++i) {
      compute(res);
    }
    std::vector<tc::Duration> times;
    times.reserve(tc::FLAGS_benchmark_iterations);
    for (size_t i = 0; i < tc::FLAGS_benchmark_iterations; ++i) {
      auto time(std::chrono::system_clock::now());
      compute(res);
      TC_CUDA_RUNTIMEAPI_ENFORCE(cudaDeviceSynchronize());
      times.push_back(std::chrono::duration_cast<std::chrono::microseconds>(
          std::chrono::system_clock::now() - time));
    }
    std::sort(times.begin(), times.end());
    auto p50idx = static_cast<int>(std::ceil(0.5 * times.size()));
    auto p90idx = static_cast<int>(std::ceil(0.9 * times.size()));
    auto p99idx = static_cast<int>(std::ceil(0.99 * times.size()));

#define GET_US(X) \
  (std::chrono::duration_cast<std::chrono::microseconds>((X)).count())

    std::cout << "\n---------------------------------------------------------";
    std::cout << "\n------------------ REFERENCE IMPL. STATS ----------------";
    std::cout << "\n------------------    " << tc::FLAGS_benchmark_iterations
              << " ITERATIONS    ----------------";
    std::cout << "\n---------------------------------------------------------";
    std::cout << "\n";
    std::cout << "Min: " << GET_US(times.front()) << "us, "
              << "p50: "
              << GET_US(times.at(std::min(p50idx, (int)times.size() - 1)))
              << "us, "
              << "p90: "
              << GET_US(times.at(std::min(p90idx, (int)times.size() - 1)))
              << "us, "
              << "p99: "
              << GET_US(times.at(std::min(p99idx, (int)times.size() - 1)))
              << "us, "
              << "Max: " << GET_US(times.back()) << "us";
    std::cout << "\n---------------------------------------------------------";
    std::cout << "\n\n";

#undef GET_US
  }

  template <typename CheckFunction>
  void validateProto(
      std::string cacheFilename,
      const std::string& tc,
      const std::string& name,
      const std::vector<at::Tensor>& inputs,
      CheckFunction checkFun = [](const std::vector<at::Tensor>&,
                                  const std::vector<at::Tensor>&) {
        return true;
      }) {
    std::cout << "Validating proto from: "
              << tc::makeOptionsFilename(cacheFilename) << std::endl;

    tc::OptionsCache::enableCache();
    tc::OptionsCache::loadCacheFromProtobuf(cacheFilename + ".options");
    tc::FLAGS_tuner_gen_restore_number = 1;

    tc::ATenCompilationUnit<tc::CudaTcExecutor> atCompl;
    atCompl.define(tc);

    auto mappingOptions = [&]() {
      auto inputsPair = tc::toConstDlpackTensors(inputs);
      auto outputs = atCompl.inferOutputTensorInfo(name, inputs);
      tc::ScopeGuard g([&]() { tc::deleteDlmTensors(inputsPair.second); });
      return tc::autotune::restoreCandidates(
          lang::canonicalTc(tc), inputsPair.first, outputs);
    }();
    auto handle = atCompl.compile(name, inputs, mappingOptions[0]);
    std::vector<at::Tensor> outputs;
    atCompl.run(name, inputs, outputs, handle);
    EXPECT_TRUE(checkFun(inputs, outputs));
    for (size_t i = 1; i < tc::FLAGS_benchmark_warmup; ++i) {
      atCompl.run(name, inputs, outputs, handle);
    }
    std::vector<tc::Duration> kernelTimes;
    kernelTimes.reserve(tc::FLAGS_benchmark_iterations);
    std::vector<tc::Duration> totalTimes;
    totalTimes.reserve(tc::FLAGS_benchmark_iterations);
    for (size_t i = 0; i < tc::FLAGS_benchmark_iterations; ++i) {
      kernelTimes.push_back(atCompl.run(name, inputs, outputs, handle, true));
      TC_CUDA_RUNTIMEAPI_ENFORCE(cudaDeviceSynchronize());
      auto time(std::chrono::system_clock::now());
      atCompl.uncheckedRun(inputs, outputs, handle);
      TC_CUDA_RUNTIMEAPI_ENFORCE(cudaDeviceSynchronize());
      totalTimes.push_back(
          std::chrono::duration_cast<std::chrono::microseconds>(
              std::chrono::system_clock::now() - time));
    }

    auto p50idx = static_cast<int>(std::ceil(0.5 * kernelTimes.size()));
    auto p90idx = static_cast<int>(std::ceil(0.9 * kernelTimes.size()));
    auto p99idx = static_cast<int>(std::ceil(0.99 * kernelTimes.size()));

    std::sort(kernelTimes.begin(), kernelTimes.end());
#define GET_US(X) \
  (std::chrono::duration_cast<std::chrono::microseconds>((X)).count())

    std::cout << "\n---------------------------------------------------------";
    std::cout << "\n------------- AUTOTUNED VALIDATED KERNEL STATS ----------";
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
    std::cout << "\n-------------- AUTOTUNED VALIDATED TOTAL STATS ----------";
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

  template <typename CheckFunction>
  void autotune(
      std::string cacheFilename,
      std::string resultsFilename,
      std::string TC,
      std::string kernelName,
      std::vector<at::Tensor> inputs,
      tc::CudaMappingOptions baseMapping,
      std::vector<tc::CudaMappingOptions> startingPoints,
      CheckFunction checkFun =
          [](const std::vector<at::Tensor>&, const std::vector<at::Tensor>&) {
            return true;
          },
      const tc::autotune::TuningParameterFixer& fixedParams = {}) {
    if (FLAGS_autotune) {
      tc::autotune::GeneticAutotunerATen geneticAutotuneATen(TC);
      auto bestOptions = [&]() {
        auto options = geneticAutotuneATen.tune(
            cacheFilename,
            kernelName,
            inputs,
            baseMapping,
            startingPoints,
            fixedParams);
        CHECK(options);
        return *options;
      }();

      tc::ATenCompilationUnit<tc::CudaTcExecutor> atCompl;
      atCompl.define(TC);
      auto handle = atCompl.compile(kernelName, inputs, bestOptions);
      std::vector<at::Tensor> outputs;
      atCompl.run(kernelName, inputs, outputs, handle);
      EXPECT_TRUE(checkFun(inputs, outputs));
      for (size_t i = 1; i < tc::FLAGS_benchmark_warmup; ++i) {
        atCompl.run(kernelName, inputs, outputs, handle);
      }
      std::vector<tc::Duration> kernelTimes;
      kernelTimes.reserve(tc::FLAGS_benchmark_iterations);
      std::vector<tc::Duration> totalTimes;
      totalTimes.reserve(tc::FLAGS_benchmark_iterations);
      for (size_t i = 0; i < tc::FLAGS_benchmark_iterations; ++i) {
        kernelTimes.push_back(
            atCompl.run(kernelName, inputs, outputs, handle, true));
        TC_CUDA_RUNTIMEAPI_ENFORCE(cudaDeviceSynchronize());
        auto time(std::chrono::system_clock::now());
        atCompl.uncheckedRun(inputs, outputs, handle);
        TC_CUDA_RUNTIMEAPI_ENFORCE(cudaDeviceSynchronize());
        totalTimes.push_back(
            std::chrono::duration_cast<std::chrono::microseconds>(
                std::chrono::system_clock::now() - time));
      }

      auto p50idx = static_cast<int>(std::ceil(0.5 * kernelTimes.size()));
      auto p90idx = static_cast<int>(std::ceil(0.9 * kernelTimes.size()));
      auto p99idx = static_cast<int>(std::ceil(0.99 * kernelTimes.size()));
      std::sort(kernelTimes.begin(), kernelTimes.end());

#define GET_US(X) \
  (std::chrono::duration_cast<std::chrono::microseconds>((X)).count())

      {
        std::ofstream out(resultsFilename);
        out << "tc version: " << tc::git_version << "\n"
            << bestOptions << "\n"
            << "Min: " << GET_US(kernelTimes.front()) << "us, "
            << "p50: "
            << GET_US(kernelTimes.at(
                   std::min(p50idx, (int)kernelTimes.size() - 1)))
            << "us, "
            << "p90: "
            << GET_US(kernelTimes.at(
                   std::min(p90idx, (int)kernelTimes.size() - 1)))
            << "us, "
            << "p99: "
            << GET_US(kernelTimes.at(
                   std::min(p99idx, (int)kernelTimes.size() - 1)))
            << "us, "
            << "Max: " << GET_US(kernelTimes.back()) << "us\n";
      }

      std::cout
          << "\n---------------------------------------------------------";
      std::cout
          << "\n------------------ AUTOTUNED KERNEL STATS ---------------";
      std::cout << "\n------------------    " << tc::FLAGS_benchmark_iterations
                << " ITERATIONS    ----------------";
      std::cout
          << "\n---------------------------------------------------------";
      std::cout << "\n";
      std::cout << "Min: " << GET_US(kernelTimes.front()) << "us, "
                << "p50: "
                << GET_US(kernelTimes.at(
                       std::min(p50idx, (int)kernelTimes.size() - 1)))
                << "us, "
                << "p90: "
                << GET_US(kernelTimes.at(
                       std::min(p90idx, (int)kernelTimes.size() - 1)))
                << "us, "
                << "p99: "
                << GET_US(kernelTimes.at(
                       std::min(p99idx, (int)kernelTimes.size() - 1)))
                << "us, "
                << "Max: " << GET_US(kernelTimes.back()) << "us";
      std::cout
          << "\n---------------------------------------------------------";
      std::cout << "\n\n";
#undef GET_US
    }
  }
};
