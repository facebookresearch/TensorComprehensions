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
#include "tc/core/cuda/cuda_profile.h"

#include <algorithm>
#include <mutex>
#include <numeric>
#include <sstream>

#include "tc/core/cuda/cuda.h"
#include "tc/core/flags.h"
#include "tc/core/scope_guard.h"

namespace tc {

class CudaDeviceLockableRuntimes {
 public:
  CudaDeviceLockableRuntimes()
      : runtimes_(CudaGPUInfo::GPUInfo().NumberGPUs()),
        mutexes_(CudaGPUInfo::GPUInfo().NumberGPUs()) {}

  std::mutex& getDeviceMutex(uint32_t deviceID) {
    return mutexes_.at(deviceID);
  }

  Duration getRuntime(uint32_t deviceID) {
    return runtimes_.at(deviceID);
  }

  void setRuntime(const Duration& d, uint32_t deviceID) {
    runtimes_.at(deviceID) = d;
  }

 private:
  std::vector<Duration> runtimes_;
  std::vector<std::mutex> mutexes_;
};

namespace {
CudaDeviceLockableRuntimes& runtimes() {
  static CudaDeviceLockableRuntimes runtimes;
  return runtimes;
}
} // namespace

bool operator==(const CudaProfilingInfo& a, const CudaProfilingInfo& b) {
  return std::tie(
             a.runtime,
             a.ipc,
             a.globalLoadEfficiency,
             a.globalStoreEfficiency,
             a.sharedMemoryEfficiency,
             a.localMemoryOverhead,
             a.achievedOccupancy,
             a.warpExecutionEfficiency) ==
      std::tie(
             b.runtime,
             b.ipc,
             b.globalLoadEfficiency,
             b.globalStoreEfficiency,
             b.sharedMemoryEfficiency,
             b.localMemoryOverhead,
             b.achievedOccupancy,
             b.warpExecutionEfficiency);
}

CudaCuptiProfiler::CudaCuptiProfiler(KernelType kernel, CUdevice device)
    : kernel_{std::move(kernel)},
      device_{device},
      metrics{{"ipc", device_},
              {"gld_efficiency", device_},
              {"gst_efficiency", device_},
              {"achieved_occupancy", device_},
              {"shared_efficiency", device_},
              {"warp_execution_efficiency", device_},
              {"local_memory_overhead", device_}} {}

namespace {
void CUPTIAPI
bufferRequested(uint8_t** buffer, size_t* size, size_t* maxNumRecords) {
  *size = 16 * 1024;
  *buffer = static_cast<uint8_t*>(aligned_alloc(8, *size));
  *maxNumRecords = 0;
  CHECK(*buffer != nullptr) << "Could not allocate memory.";
}

void CUPTIAPI bufferCompleted(
    CUcontext ctx,
    uint32_t streamId,
    uint8_t* buffer,
    size_t size,
    size_t validSize) {
  CUpti_Activity* record = nullptr;
  // since we launched only 1 kernel, we should have only 1 kernel record
  TC_CUPTI_CHECK(cuptiActivityGetNextRecord(buffer, validSize, &record));

#if (CUPTI_API_VERSION >= 10)
  using ActivityKernelType = CUpti_ActivityKernel4;
#else
  using ActivityKernelType = CUpti_ActivityKernel3;
#endif

  auto kernel = reinterpret_cast<ActivityKernelType*>(record);
  CHECK_EQ(kernel->kind, CUPTI_ACTIVITY_KIND_KERNEL)
      << "Expected kernel activity record, got " << kernel->kind;

  auto kernelDuration = kernel->end - kernel->start;
  auto device = kernel->deviceId;
  runtimes().setRuntime(std::chrono::nanoseconds(kernelDuration), device);
  free(buffer);
}

struct MetricData {
  CUdevice device;
  CUpti_EventGroupSet* eventGroups;
  uint32_t numEvents;
  std::vector<CUpti_EventID> eventIds;
  std::vector<uint64_t> eventValues;
};

std::pair<std::string, std::string> logEvent(
    CUpti_EventID eventId,
    uint64_t numInstances,
    uint64_t numTotalInstances,
    uint64_t sum,
    uint64_t normalized,
    uint64_t* values) {
  constexpr size_t keventNameArraySize = 128;
  char eventName[keventNameArraySize];
  size_t eventNameSize = keventNameArraySize - 1;
  CHECK_GT(eventNameSize, 0);
  TC_CUPTI_CHECK(cuptiEventGetAttribute(
      eventId, CUPTI_EVENT_ATTR_NAME, &eventNameSize, eventName));
  eventName[eventNameSize] = '\0';

  std::stringstream ss;
  ss << eventName << " = " << sum << " (";
  if (numInstances > 1) {
    for (uint64_t k = 0; k < numInstances; k++) {
      if (k != 0) {
        ss << ", ";
      }
      ss << values[k];
    }
  }
  ss << ')';
  auto line1 = ss.str();
  LOG(INFO) << ss.str();

  ss.str("");
  ss.clear();

  ss << eventName << " (normalized) (" << sum << " * " << numTotalInstances
     << ") / " << numInstances << " = " << normalized;
  auto line2 = ss.str();
  return std::make_pair(std::move(line1), std::move(line2));
}

void eventCollectionStart(
    MetricData& metricData,
    const CUpti_CallbackData* cbInfo) {
  cudaDeviceSynchronize();
  TC_CUPTI_CHECK(cuptiSetEventCollectionMode(
      cbInfo->context, CUPTI_EVENT_COLLECTION_MODE_KERNEL));
  for (uint64_t i = 0; i < metricData.eventGroups->numEventGroups; i++) {
    uint32_t all =
        1; // 1 means that all instances of the event will be collected
    TC_CUPTI_CHECK(cuptiEventGroupSetAttribute(
        metricData.eventGroups->eventGroups[i],
        CUPTI_EVENT_GROUP_ATTR_PROFILE_ALL_DOMAIN_INSTANCES,
        sizeof(all),
        &all));
    TC_CUPTI_CHECK(
        cuptiEventGroupEnable(metricData.eventGroups->eventGroups[i]));
  }
}

void eventCollectionEnd(MetricData& metricData) {
  cudaDeviceSynchronize();

  // for each group, read the event values from the group and record
  // in metricData
  for (uint64_t i = 0; i < metricData.eventGroups->numEventGroups; i++) {
    CUpti_EventGroup group = metricData.eventGroups->eventGroups[i];
    CUpti_EventDomainID groupDomain;
    uint32_t numEvents, numInstances, numTotalInstances;
    CUpti_EventID* eventIds;
    size_t groupDomainSize = sizeof(groupDomain);
    size_t numEventsSize = sizeof(numEvents);
    size_t numInstancesSize = sizeof(numInstances);
    size_t numTotalInstancesSize = sizeof(numTotalInstances);
    size_t valuesSize, eventIdsSize;

    TC_CUPTI_CHECK(cuptiEventGroupGetAttribute(
        group,
        CUPTI_EVENT_GROUP_ATTR_EVENT_DOMAIN_ID,
        &groupDomainSize,
        &groupDomain));

    TC_CUPTI_CHECK(cuptiDeviceGetEventDomainAttribute(
        metricData.device,
        groupDomain,
        CUPTI_EVENT_DOMAIN_ATTR_TOTAL_INSTANCE_COUNT,
        &numTotalInstancesSize,
        &numTotalInstances));
    TC_CUPTI_CHECK(cuptiEventGroupGetAttribute(
        group,
        CUPTI_EVENT_GROUP_ATTR_INSTANCE_COUNT,
        &numInstancesSize,
        &numInstances));
    TC_CUPTI_CHECK(cuptiEventGroupGetAttribute(
        group, CUPTI_EVENT_GROUP_ATTR_NUM_EVENTS, &numEventsSize, &numEvents));
    eventIdsSize = numEvents * sizeof(CUpti_EventID);
    eventIds = (CUpti_EventID*)malloc(eventIdsSize);
    TC_CUPTI_CHECK(cuptiEventGroupGetAttribute(
        group, CUPTI_EVENT_GROUP_ATTR_EVENTS, &eventIdsSize, eventIds));

    std::vector<uint64_t> values(numInstances);

    for (uint32_t j = 0; j < numEvents; j++) {
      values.clear();
      valuesSize = sizeof(uint64_t) * numInstances;
      TC_CUPTI_CHECK(cuptiEventGroupReadEvent(
          group,
          CUPTI_EVENT_READ_FLAG_NONE,
          eventIds[j],
          &valuesSize,
          values.data()));
      CHECK_EQ(numInstances, valuesSize / sizeof(uint64_t));
      CHECK_LE(metricData.eventValues.size(), metricData.numEvents)
          << "Too many events collected, metric expects only "
          << metricData.numEvents;

      // sum collect event values from all instances
      auto sum = std::accumulate(
          values.begin(), values.begin() + numInstances, uint64_t(0));

      // normalize the event value to represent the total number of
      // domain instances on the device
      auto normalized = (sum * numTotalInstances) / numInstances;

      metricData.eventIds.push_back(eventIds[j]);
      metricData.eventValues.push_back(normalized);

      if (FLAGS_cuda_profile_verbose_events) {
        std::string line1, line2;
        std::tie(line1, line2) = logEvent(
            eventIds[j],
            numInstances,
            numTotalInstances,
            sum,
            normalized,
            values.data());
        LOG(INFO) << line1;
        LOG(INFO) << line2;
      }
    }
  }
  for (uint64_t i = 0; i < metricData.eventGroups->numEventGroups; i++)
    TC_CUPTI_CHECK(
        cuptiEventGroupDisable(metricData.eventGroups->eventGroups[i]));
}

void CUPTIAPI getMetricValueCallback(
    void* userdata,
    CUpti_CallbackDomain domain,
    CUpti_CallbackId cbid,
    const CUpti_CallbackData* cbInfo) {
  MetricData& metricData = *static_cast<MetricData*>(userdata);

  // This callback is enabled only for launch so we shouldn't see
  // anything else.
  CHECK_EQ(cbid, CUPTI_DRIVER_TRACE_CBID_cuLaunchKernel)
      << "Unexpected cbid: " << cbid;

  // on entry, enable all the event groups being collected this pass,
  // for metrics we collect for all instances of the event
  if (cbInfo->callbackSite == CUPTI_API_ENTER) {
    eventCollectionStart(metricData, cbInfo);
  }

  // on exit, read and record event values
  if (cbInfo->callbackSite == CUPTI_API_EXIT) {
    eventCollectionEnd(metricData);
  }
}
CUpti_MetricValueKind getValueKind(const CudaMetric& metric) {
  CUpti_MetricValueKind valueKind;
  size_t valueKindSize = sizeof(valueKind);
  TC_CUPTI_CHECK(cuptiMetricGetAttribute(
      metric.id, CUPTI_METRIC_ATTR_VALUE_KIND, &valueKindSize, &valueKind));
  return valueKind;
}

void logMetric(const CudaMetric& metric) {
  auto valueKind = getValueKind(metric);
  std::stringstream ss;
  ss << "Metric " << metric.name << " = ";
  switch (valueKind) {
    case CUPTI_METRIC_VALUE_KIND_DOUBLE:
      ss << metric.value.metricValueDouble;
      break;
    case CUPTI_METRIC_VALUE_KIND_UINT64:
      ss << metric.value.metricValueUint64;
      break;
    case CUPTI_METRIC_VALUE_KIND_INT64:
      ss << metric.value.metricValueInt64;
      break;
    case CUPTI_METRIC_VALUE_KIND_PERCENT:
      ss << metric.value.metricValuePercent << '%';
      break;
    case CUPTI_METRIC_VALUE_KIND_THROUGHPUT:
      ss << metric.value.metricValueThroughput << " bytes/sec";
      break;
    case CUPTI_METRIC_VALUE_KIND_UTILIZATION_LEVEL:
      ss << "utilization level " << metric.value.metricValueUtilizationLevel;
      break;
    default:
      CHECK(false) << "Unknown metric value kind";
  }
  LOG(INFO) << ss.str();
}

template <typename T>
uint64_t vectorSizeBytes(const std::vector<T>& v) {
  return v.size() * sizeof(T);
}

} // namespace

CudaMetric::operator double() const {
  auto valueKind = getValueKind(*this);
  if (valueKind == CUPTI_METRIC_VALUE_KIND_DOUBLE) {
    return value.metricValueDouble;
  } else if (valueKind == CUPTI_METRIC_VALUE_KIND_PERCENT) {
    return value.metricValuePercent;
  } else {
    CHECK(false) << "Invalid metric value conversion.";
    return 0.0;
  }
}

CudaMetric::operator uint64_t() const {
  auto valueKind = getValueKind(*this);
  if (valueKind == CUPTI_METRIC_VALUE_KIND_UINT64) {
    return value.metricValueUint64;
  } else if (valueKind == CUPTI_METRIC_VALUE_KIND_THROUGHPUT) {
    return value.metricValueThroughput;
  } else {
    CHECK(false) << "Invalid metric value conversion.";
    return 0;
  }
}

CudaMetric::operator int64_t() const {
  auto valueKind = getValueKind(*this);
  if (valueKind == CUPTI_METRIC_VALUE_KIND_INT64) {
    return value.metricValueInt64;
  } else {
    CHECK(false) << "Invalid metric value conversion.";
    return 0;
  }
}

CudaMetric::CudaMetric(const char* name_, CUdevice device) : name{name_} {
  TC_CUPTI_CHECK(cuptiMetricGetIdFromName(device, name.c_str(), &id));
  TC_CUPTI_CHECK(cuptiMetricGetNumEvents(id, &numberEvents));
}

CudaProfilingInfo CudaCuptiProfiler::Profile() {
  std::lock_guard<std::mutex> lock(runtimes().getDeviceMutex(device_));

  TC_CUPTI_CHECK(cuptiActivityEnable(CUPTI_ACTIVITY_KIND_KERNEL));
  ScopeGuard activityGuard{[]() {
    TC_CUPTI_CHECK(cuptiActivityDisable(CUPTI_ACTIVITY_KIND_KERNEL));
  }};
  TC_CUPTI_CHECK(
      cuptiActivityRegisterCallbacks(bufferRequested, bufferCompleted));
  kernel_();
  cudaDeviceSynchronize();
  TC_CUPTI_CHECK(cuptiActivityFlushAll(0));

  CudaProfilingInfo pi;
  pi.runtime = runtimes().getRuntime(device_);
  CHECK_GT(pi.runtime.count(), 0);

  CUpti_SubscriberHandle subscriber;
  MetricData metricData;
  TC_CUPTI_CHECK(cuptiSubscribe(
      &subscriber, (CUpti_CallbackFunc)getMetricValueCallback, &metricData));
  ScopeGuard subscriberGuard{
      [&]() { TC_CUPTI_CHECK(cuptiUnsubscribe(subscriber)); }};
  TC_CUPTI_CHECK(cuptiEnableCallback(
      1,
      subscriber,
      CUPTI_CB_DOMAIN_DRIVER_API,
      CUPTI_DRIVER_TRACE_CBID_cuLaunchKernel));

  metricData.numEvents = std::accumulate(
      metrics.begin(),
      metrics.end(),
      uint32_t(0),
      [](uint32_t sum, const CudaMetric& metric) {
        return sum + metric.numberEvents;
      });
  metricData.device = device_;

  CUcontext context = 0;
  TC_CUDA_DRIVERAPI_ENFORCE(cuCtxGetCurrent(&context));

  CUpti_EventGroupSets* passData;
  {
    std::vector<CUpti_MetricID> ids(metrics.size());
    std::transform(
        metrics.begin(),
        metrics.end(),
        ids.begin(),
        [](const CudaMetric& metric) { return metric.id; });
    TC_CUPTI_CHECK(cuptiMetricCreateEventGroupSets(
        context, sizeof(CUpti_MetricID) * ids.size(), ids.data(), &passData));
  }

  ScopeGuard passDataGuard{
      [&]() { TC_CUPTI_CHECK(cuptiEventGroupSetsDestroy(passData)); }};

  for (uint32_t pass = 0; pass < passData->numSets; pass++) {
    LOG_IF(INFO, FLAGS_cuda_profile_verbose) << "Profiling Pass: " << pass;
    metricData.eventGroups = passData->sets + pass;
    kernel_();
  }

  auto getMetricValue = [&](const CudaMetric& metric) {
    CUpti_MetricValue value;
    TC_CUPTI_CHECK(cuptiMetricGetValue(
        device_,
        metric.id,
        vectorSizeBytes(metricData.eventIds),
        metricData.eventIds.data(),
        vectorSizeBytes(metricData.eventValues),
        metricData.eventValues.data(),
        std::chrono::duration_cast<std::chrono::nanoseconds>(pi.runtime)
            .count(),
        &value));
    return value;
  };

  for (auto& metric : metrics) {
    metric.value = getMetricValue(metric);
  }
  writeMetricValues(pi);

  if (FLAGS_cuda_profile_verbose) {
    for (const auto& metric : metrics) {
      logMetric(metric);
    }
  }

  return pi;
}

void CudaCuptiProfiler::writeMetricValues(CudaProfilingInfo& pinfo) const {
  for (const auto& metric : metrics) {
    if (metric.name == "ipc") {
      pinfo.ipc = metric;
      continue;
    }
    if (metric.name == "gld_efficiency") {
      pinfo.globalLoadEfficiency = metric;
      continue;
    }
    if (metric.name == "gst_efficiency") {
      pinfo.globalStoreEfficiency = metric;
      continue;
    }
    if (metric.name == "shared_efficiency") {
      pinfo.sharedMemoryEfficiency = metric;
      continue;
    }
    if (metric.name == "achieved_occupancy") {
      pinfo.achievedOccupancy = metric;
      continue;
    }
    if (metric.name == "warp_execution_efficiency") {
      pinfo.warpExecutionEfficiency = metric;
      continue;
    }
    if (metric.name == "local_memory_overhead") {
      pinfo.localMemoryOverhead = metric;
      continue;
    }

    CHECK(false) << "NYI: " << metric.name;
  }
}

} // namespace tc
