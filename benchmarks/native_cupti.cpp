#include <cuda_runtime_api.h>
#include <cupti.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <cstdint>
#include <cstdlib>
#include <mutex>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace py = pybind11;

namespace {

// Only the fields the Python attribution layer reads: the activity identity
// (kind plus the parameters that distinguish two copies or fills) and the
// device timestamps that form a call's envelope.
struct ActivityRecord {
  std::string kind;
  std::string name;
  uint64_t start;
  uint64_t end;
  uint32_t copy_kind;
  uint64_t bytes;
  uint32_t value;
};

std::mutex g_mutex;
bool g_active = false;
bool g_callbacks_registered = false;
std::vector<ActivityRecord> g_activities;
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
  constexpr size_t kBufferSize = 8 * 1024 * 1024;
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
  ActivityRecord activity{};
  if (record->kind == CUPTI_ACTIVITY_KIND_CONCURRENT_KERNEL ||
      record->kind == CUPTI_ACTIVITY_KIND_KERNEL) {
    auto* kernel = reinterpret_cast<CUpti_ActivityKernel9*>(record);
    activity = {
        "kernel", kernel->name ? kernel->name : "", kernel->start, kernel->end,
        0,        0,                                0,
    };
  } else if (record->kind == CUPTI_ACTIVITY_KIND_MEMCPY) {
    auto* copy = reinterpret_cast<CUpti_ActivityMemcpy6*>(record);
    activity = {
        "memcpy",       "MEMCPY",     copy->start, copy->end,
        copy->copyKind, copy->bytes,  0,
    };
  } else if (record->kind == CUPTI_ACTIVITY_KIND_MEMSET) {
    auto* memset = reinterpret_cast<CUpti_ActivityMemset4*>(record);
    activity = {
        "memset", "MEMSET",      memset->start, memset->end,
        0,        memset->bytes, memset->value,
    };
  } else {
    return;
  }

  std::lock_guard<std::mutex> lock(g_mutex);
  g_activities.push_back(std::move(activity));
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
  g_activities.clear();
  g_dropped = 0;
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
  bool kernel_enabled = false;
  bool memcpy_enabled = false;
  try {
    CUPTI_CHECK(cuptiActivityEnable(CUPTI_ACTIVITY_KIND_CONCURRENT_KERNEL));
    kernel_enabled = true;
    CUPTI_CHECK(cuptiActivityEnable(CUPTI_ACTIVITY_KIND_MEMCPY));
    memcpy_enabled = true;
    CUPTI_CHECK(cuptiActivityEnable(CUPTI_ACTIVITY_KIND_MEMSET));
  } catch (...) {
    if (memcpy_enabled) {
      cuptiActivityDisable(CUPTI_ACTIVITY_KIND_MEMCPY);
    }
    if (kernel_enabled) {
      cuptiActivityDisable(CUPTI_ACTIVITY_KIND_CONCURRENT_KERNEL);
    }
    throw;
  }
  g_active = true;
}

void stop() {
  if (!g_active) {
    return;
  }
  // Clear the flag before tearing down. A failing flush or disable still
  // reports its error, but must not wedge the collector: the Python side
  // always drops its own flag, and start() re-enables every kind and resets
  // the buffer, so a later session has to be able to run.
  g_active = false;
  cudaDeviceSynchronize();
  CUPTI_CHECK(cuptiActivityFlushAll(CUPTI_ACTIVITY_FLAG_FLUSH_FORCED));
  CUPTI_CHECK(cuptiActivityDisable(CUPTI_ACTIVITY_KIND_MEMSET));
  CUPTI_CHECK(cuptiActivityDisable(CUPTI_ACTIVITY_KIND_MEMCPY));
  CUPTI_CHECK(cuptiActivityDisable(CUPTI_ACTIVITY_KIND_CONCURRENT_KERNEL));
}

py::dict activity_dict(const ActivityRecord& activity) {
  py::dict out;
  out["kind"] = activity.kind;
  out["name"] = activity.name;
  out["start_ns"] = activity.start;
  out["end_ns"] = activity.end;
  out["copy_kind"] = activity.copy_kind;
  out["bytes"] = activity.bytes;
  out["value"] = activity.value;
  return out;
}

py::dict checkpoint() {
  if (!g_active) {
    throw std::runtime_error("native CUPTI collector is not active");
  }

  // Synchronize before the forced flush so the checkpoint only publishes
  // completed activity records. This does not stop/restart collection or
  // introduce work into the measured GPU span.
  cudaError_t cuda_status = cudaDeviceSynchronize();
  if (cuda_status != cudaSuccess) {
    throw std::runtime_error(
        std::string("cudaDeviceSynchronize failed: ") +
        cudaGetErrorString(cuda_status));
  }
  CUPTI_CHECK(cuptiActivityFlushAll(CUPTI_ACTIVITY_FLAG_FLUSH_FORCED));

  std::lock_guard<std::mutex> lock(g_mutex);
  py::dict out;
  out["kernel_index"] = g_activities.size();
  out["dropped"] = g_dropped;
  return out;
}

py::dict results_range(size_t begin, size_t end, size_t dropped_begin) {
  std::lock_guard<std::mutex> lock(g_mutex);
  if (begin > end || end > g_activities.size()) {
    throw std::runtime_error("invalid native CUPTI activity range");
  }

  py::list activities;
  for (size_t i = begin; i < end; ++i) {
    activities.append(activity_dict(g_activities[i]));
  }

  py::dict out;
  out["kernels"] = activities;
  out["dropped"] = g_dropped >= dropped_begin ? g_dropped - dropped_begin : g_dropped;
  return out;
}

}  // namespace

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("start", &start);
  m.def("stop", &stop);
  m.def("checkpoint", &checkpoint);
  m.def("results_range", &results_range);
}
