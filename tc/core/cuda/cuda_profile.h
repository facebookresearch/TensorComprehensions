/**
 * Copyright (c) 2018-present, Facebook, Inc.
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

#include <functional>
#include <vector>

#include <cuda.h>
#include <cupti.h>

#include "tc/core/utils/time.h"

namespace tc {
struct CudaProfilingInfo {
  Duration runtime;
  double ipc;
  double globalLoadEfficiency;
  double globalStoreEfficiency;
  double sharedMemoryEfficiency;
  double localMemoryOverhead;
  double achievedOccupancy;
  double warpExecutionEfficiency;

  friend bool operator==(
      const CudaProfilingInfo& a,
      const CudaProfilingInfo& b);
};

struct CudaMetric {
  CudaMetric(const char* name_, CUdevice device);

  std::string name;
  CUpti_MetricID id;
  uint32_t numberEvents;

  CUpti_MetricValue value;

  operator double() const;
  operator uint64_t() const;
  operator int64_t() const;
};

using KernelType = std::function<void(void)>;

class CudaCuptiProfiler {
 public:
  CudaCuptiProfiler(KernelType kernel, CUdevice device);
  CudaProfilingInfo Profile();

 private:
  void writeMetricValues(CudaProfilingInfo& pinfo) const;

  std::function<void(void)> kernel_;
  CUdevice device_;
  std::vector<CudaMetric> metrics;
};

} // namespace tc
