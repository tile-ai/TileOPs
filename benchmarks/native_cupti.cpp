#include <cuda_runtime_api.h>
#include <cupti.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <cstdint>
#include <cstdlib>
#include <mutex>
#include <stdexcept>
#include <string>
#include <vector>

namespace py = pybind11;

namespace {

struct KernelRecord {
  std::string name;
  uint64_t start;
  uint64_t end;
  uint32_t correlation;
  uint32_t device;
  uint32_t context;
  uint32_t stream;
};

std::mutex g_mutex;
bool g_active = false;
bool g_callbacks_registered = false;
std::vector<KernelRecord> g_kernels;
size_t g_dropped = 0;

#define CUPTI_CHECK(call)                                                       \
  do {                                                                         \
    CUptiResult _status = call;                                                \
    if (_status != CUPTI_SUCCESS) {                                            \
      const char* _err = nullptr;                                              \
      cuptiGetResultString(_status, &_err);                                    \
      throw std::runtime_error(std::string(#call) + " failed: " +              \
                               (_err ? _err : "unknown"));                    \
    }                                                                          \
  } while (0)

void buffer_requested(uint8_t** buffer, size_t* size, size_t* max_num_records) {
  constexpr size_t kBufferSize = 4 * 1024 * 1024;
  constexpr size_t kAlign = 8;
  void* ptr = nullptr;
  if (posix_memalign(&ptr, kAlign, kBufferSize) != 0) {
    *buffer = nullptr;
    *size = 0;
    *max_num_records = 0;
    return;
  }
  *buffer = reinterpret_cast<uint8_t*>(ptr);
  *size = kBufferSize;
  *max_num_records = 0;
}

void handle_record(CUpti_Activity* record) {
  if (record->kind != CUPTI_ACTIVITY_KIND_CONCURRENT_KERNEL &&
      record->kind != CUPTI_ACTIVITY_KIND_KERNEL) {
    return;
  }

  auto* k = reinterpret_cast<CUpti_ActivityKernel9*>(record);
  std::lock_guard<std::mutex> lock(g_mutex);
  g_kernels.push_back({
      k->name ? k->name : "",
      k->start,
      k->end,
      k->correlationId,
      k->deviceId,
      k->contextId,
      k->streamId,
  });
}

void buffer_completed(CUcontext ctx, uint32_t stream_id, uint8_t* buffer,
                      size_t size, size_t valid_size) {
  CUpti_Activity* record = nullptr;
  if (valid_size > 0) {
    while (cuptiActivityGetNextRecord(buffer, valid_size, &record) ==
           CUPTI_SUCCESS) {
      handle_record(record);
    }
  }

  size_t dropped = 0;
  cuptiActivityGetNumDroppedRecords(ctx, stream_id, &dropped);
  {
    std::lock_guard<std::mutex> lock(g_mutex);
    g_dropped += dropped;
  }
  free(buffer);
}

void reset_state() {
  std::lock_guard<std::mutex> lock(g_mutex);
  g_kernels.clear();
  g_dropped = 0;
}

uint64_t timestamp() {
  uint64_t value = 0;
  CUPTI_CHECK(cuptiGetTimestamp(&value));
  return value;
}

void start() {
  if (g_active) {
    throw std::runtime_error("native CUPTI collector is already active");
  }
  reset_state();
  if (!g_callbacks_registered) {
    CUPTI_CHECK(cuptiActivityRegisterCallbacks(buffer_requested, buffer_completed));
    g_callbacks_registered = true;
  }
  CUPTI_CHECK(cuptiActivityEnable(CUPTI_ACTIVITY_KIND_CONCURRENT_KERNEL));
  g_active = true;
}

void stop() {
  if (!g_active) {
    return;
  }
  cudaDeviceSynchronize();
  CUPTI_CHECK(cuptiActivityFlushAll(CUPTI_ACTIVITY_FLAG_FLUSH_FORCED));
  CUPTI_CHECK(cuptiActivityDisable(CUPTI_ACTIVITY_KIND_CONCURRENT_KERNEL));
  g_active = false;
}

py::dict results() {
  std::lock_guard<std::mutex> lock(g_mutex);

  py::list kernels;
  for (const auto& k : g_kernels) {
    py::dict d;
    d["name"] = k.name;
    d["start_ns"] = k.start;
    d["end_ns"] = k.end;
    d["correlation_id"] = k.correlation;
    d["device_id"] = k.device;
    d["context_id"] = k.context;
    d["stream_id"] = k.stream;
    kernels.append(d);
  }

  py::dict out;
  out["kernels"] = kernels;
  out["dropped"] = g_dropped;
  return out;
}

py::dict checkpoint() {
  if (!g_active) {
    throw std::runtime_error("native CUPTI collector is not active");
  }

  // Every benchmark iteration synchronizes before a checkpoint.  A forced
  // flush therefore only publishes completed activity records; it does not
  // stop/restart collection or introduce work into the measured GPU span.
  cudaError_t cuda_status = cudaDeviceSynchronize();
  if (cuda_status != cudaSuccess) {
    throw std::runtime_error(
        std::string("cudaDeviceSynchronize failed: ") +
        cudaGetErrorString(cuda_status));
  }
  CUPTI_CHECK(cuptiActivityFlushAll(CUPTI_ACTIVITY_FLAG_FLUSH_FORCED));

  std::lock_guard<std::mutex> lock(g_mutex);
  py::dict out;
  out["kernel_index"] = g_kernels.size();
  out["dropped"] = g_dropped;
  out["timestamp_ns"] = timestamp();
  return out;
}

py::dict results_range(size_t begin, size_t end, size_t dropped_begin) {
  std::lock_guard<std::mutex> lock(g_mutex);
  if (begin > end || end > g_kernels.size()) {
    throw std::runtime_error("invalid native CUPTI activity range");
  }

  py::list kernels;
  for (size_t i = begin; i < end; ++i) {
    const auto& k = g_kernels[i];
    py::dict d;
    d["name"] = k.name;
    d["start_ns"] = k.start;
    d["end_ns"] = k.end;
    d["correlation_id"] = k.correlation;
    d["device_id"] = k.device;
    d["context_id"] = k.context;
    d["stream_id"] = k.stream;
    kernels.append(d);
  }

  py::dict out;
  out["kernels"] = kernels;
  out["dropped"] = g_dropped >= dropped_begin ? g_dropped - dropped_begin : g_dropped;
  return out;
}

}  // namespace

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("start", &start);
  m.def("stop", &stop);
  m.def("timestamp", &timestamp);
  m.def("checkpoint", &checkpoint);
  m.def("results_range", &results_range);
  m.def("results", &results);
}
