// A PrivateUse1 backend for tests: two devices whose tensors live in host memory.
//
// It implements what torch asks of a device runtime -- the current device, streams,
// timing events, an allocator -- and runs every operator through torch's CPU fallback,
// so a test exercises TileOPs' accelerator path on a device that is not CUDA.

#include <ATen/ATen.h>
#include <ATen/EmptyTensor.h>
#include <ATen/detail/PrivateUse1HooksInterface.h>
#include <ATen/native/CPUFallback.h>
#include <ATen/ops/_reshape_alias_native.h>
#include <ATen/ops/as_strided_native.h>
#include <ATen/ops/resize_native.h>
#include <ATen/ops/set_native.h>
#include <ATen/ops/unfold_native.h>
#include <ATen/ops/view_as_complex_native.h>
#include <ATen/ops/view_as_real_native.h>
#include <ATen/ops/view_native.h>
#include <c10/core/CachingDeviceAllocator.h>
#include <c10/core/impl/DeviceGuardImplInterface.h>
#include <torch/library.h>

#include <chrono>
#include <cstdlib>
#include <cstring>

namespace {

constexpr c10::DeviceType kType = c10::DeviceType::PrivateUse1;
constexpr c10::DeviceIndex kDeviceCount = DEVICE_COUNT;

thread_local c10::DeviceIndex current_device = 0;

void check_index(c10::DeviceIndex index) {
  TORCH_CHECK(index >= 0 && index < kDeviceCount, "device index ", static_cast<int>(index),
              " is out of range; this backend has ", static_cast<int>(kDeviceCount),
              " devices");
}

// Work runs synchronously on the host, so an event records when the host reached it.
struct Event {
  std::chrono::steady_clock::time_point time;
};

struct GuardImpl final : c10::impl::DeviceGuardImplInterface {
  c10::DeviceType type() const override { return kType; }

  c10::Device exchangeDevice(c10::Device device) const override {
    c10::Device previous = getDevice();
    setDevice(device);
    return previous;
  }

  c10::Device getDevice() const override { return c10::Device(kType, current_device); }

  void setDevice(c10::Device device) const override {
    TORCH_CHECK(device.type() == kType, "expected a ", kType, " device, got ", device);
    check_index(device.index());
    current_device = device.index();
  }

  void uncheckedSetDevice(c10::Device device) const noexcept override {
    current_device = device.index();
  }

  // One stream per device: work is ordered by the host.
  c10::Stream getStream(c10::Device device) const override {
    return c10::Stream(c10::Stream::DEFAULT, device);
  }

  c10::Stream exchangeStream(c10::Stream stream) const override {
    return getStream(stream.device());
  }

  c10::DeviceIndex deviceCount() const noexcept override { return kDeviceCount; }

  void record(void** event, const c10::Stream& stream, const c10::DeviceIndex device_index,
              const c10::EventFlag flag) const override {
    TORCH_CHECK(device_index == -1 || device_index == stream.device_index(),
                "an event is recorded on its own device's stream");
    if (*event == nullptr) {
      *event = new Event();
    }
    static_cast<Event*>(*event)->time = std::chrono::steady_clock::now();
  }

  void block(void* event, const c10::Stream& stream) const override {}

  bool queryEvent(void* event) const override { return true; }

  void destroyEvent(void* event, const c10::DeviceIndex device_index) const noexcept override {
    delete static_cast<Event*>(event);
  }

  double elapsedTime(void* start, void* end, const c10::DeviceIndex device_index) const override {
    TORCH_CHECK(start != nullptr && end != nullptr,
                "both events must be recorded before reading the time between them");
    auto elapsed = static_cast<Event*>(end)->time - static_cast<Event*>(start)->time;
    return std::chrono::duration<double, std::milli>(elapsed).count();
  }

  void synchronizeEvent(void* event) const override {}

  void synchronizeDevice(const c10::DeviceIndex device_index) const override {}

  bool queryStream(const c10::Stream& stream) const override { return true; }

  void synchronizeStream(const c10::Stream& stream) const override {}

  void recordDataPtrOnStream(const c10::DataPtr& data_ptr,
                             const c10::Stream& stream) const override {}
};

C10_REGISTER_GUARD_IMPL(PrivateUse1, GuardImpl);

void release(void* data) { std::free(data); }

// Allocates on the current device; a factory function makes the requested device current.
struct Allocator final : c10::DeviceAllocator {
  c10::DataPtr allocate(size_t bytes) override {
    void* data = bytes == 0 ? nullptr : std::malloc(bytes);
    TORCH_CHECK(bytes == 0 || data != nullptr, "out of host memory allocating ", bytes,
                " bytes");
    return {data, data, &release, c10::Device(kType, current_device)};
  }

  c10::DeleterFnPtr raw_deleter() const override { return &release; }

  void copy_data(void* dest, const void* src, std::size_t count) const override {
    std::memcpy(dest, src, count);
  }

  bool initialized() override { return true; }

  // Nothing is cached, so the statistics stay at zero.
  void emptyCache(c10::MempoolId_t mempool_id) override {}

  void recordStream(const c10::DataPtr& ptr, c10::Stream stream) override {}

  c10::CachingDeviceAllocator::DeviceStats getDeviceStats(c10::DeviceIndex device) override {
    return {};
  }

  void resetAccumulatedStats(c10::DeviceIndex device) override {}

  void resetPeakStats(c10::DeviceIndex device) override {}
};

Allocator allocator;

REGISTER_ALLOCATOR(c10::DeviceType::PrivateUse1, &allocator);

struct Hooks final : at::PrivateUse1HooksInterface {
  bool isBuilt() const override { return true; }

  bool isAvailable() const override { return true; }

  bool hasPrimaryContext(c10::DeviceIndex device_index) const override { return true; }

  c10::DeviceIndex deviceCount() const override { return kDeviceCount; }

  void setCurrentDevice(c10::DeviceIndex device) const override {
    check_index(device);
    current_device = device;
  }

  c10::DeviceIndex getCurrentDevice() const override { return current_device; }

  c10::DeviceIndex exchangeDevice(c10::DeviceIndex device) const override {
    c10::DeviceIndex previous = current_device;
    setCurrentDevice(device);
    return previous;
  }

  c10::DeviceIndex maybeExchangeDevice(c10::DeviceIndex device) const override {
    return exchangeDevice(device);
  }

  bool isPinnedPtr(const void* data) const override { return false; }

  c10::Allocator* getPinnedMemoryAllocator() const override { return at::getCPUAllocator(); }

  c10::Device getDeviceFromPtr(void* data) const override {
    return c10::Device(kType, current_device);
  }
};

const bool hooks_registered = [] {
  at::RegisterPrivateUse1HooksInterface(new Hooks());
  return true;
}();

constexpr c10::DispatchKeySet kKeys(c10::DispatchKey::PrivateUse1);

at::Tensor empty_memory_format(c10::IntArrayRef size, std::optional<c10::ScalarType> dtype,
                               std::optional<c10::Layout> layout,
                               std::optional<c10::Device> device,
                               std::optional<bool> pin_memory,
                               std::optional<c10::MemoryFormat> memory_format) {
  const c10::DeviceGuard guard(device.value_or(c10::Device(kType)));
  return at::detail::empty_generic(size, &allocator, kKeys, c10::dtype_or_default(dtype),
                                   memory_format);
}

at::Tensor empty_strided(c10::IntArrayRef size, c10::IntArrayRef stride,
                         std::optional<c10::ScalarType> dtype,
                         std::optional<c10::Layout> layout, std::optional<c10::Device> device,
                         std::optional<bool> pin_memory) {
  const c10::DeviceGuard guard(device.value_or(c10::Device(kType)));
  return at::detail::empty_strided_generic(size, stride, &allocator, kKeys,
                                           c10::dtype_or_default(dtype));
}

// The same bytes seen as a CPU tensor, both living in host memory, with the same lazy
// conjugation and negation.
at::Tensor on_host(const at::Tensor& tensor) {
  if (tensor.is_cpu()) {
    return tensor;
  }
  at::Tensor host = at::from_blob(tensor.data_ptr(), tensor.sizes(), tensor.strides(),
                                  tensor.options().device(at::kCPU));
  if (tensor.is_conj()) {
    host = host._conj();
  }
  if (tensor.is_neg()) {
    host = host._neg_view();
  }
  return host;
}

// The CPU copy broadcasts and converts dtypes, as copy_ promises.
at::Tensor copy_from(const at::Tensor& self, const at::Tensor& dst, bool non_blocking) {
  if (dst.numel() > 0) {
    on_host(dst).copy_(on_host(self));
  }
  return dst;
}

// Storage grows on the tensor's own device, not the current one.
const at::Tensor& resize_on_own_device(const at::Tensor& self, c10::IntArrayRef size,
                                       std::optional<c10::MemoryFormat> memory_format) {
  const c10::DeviceGuard guard(self.device());
  return at::native::resize_(self, size, memory_format);
}

at::Tensor copy_from_and_resize(const at::Tensor& self, const at::Tensor& dst) {
  resize_on_own_device(dst, self.sizes(), std::nullopt);
  return copy_from(self, dst, false);
}

c10::Scalar local_scalar_dense(const at::Tensor& self) {
  return at::_local_scalar_dense(on_host(self));
}

void cpu_fallback(const c10::OperatorHandle& op, torch::jit::Stack* stack) {
  at::native::cpu_fallback(op, stack);
}

}  // namespace

// What the CPU fallback cannot run: allocation, copies to and from the host, and views,
// which share their source's storage where the fallback would return a copy.
TORCH_LIBRARY_IMPL(aten, PrivateUse1, m) {
  m.impl("empty.memory_format", empty_memory_format);
  m.impl("empty_strided", empty_strided);
  m.impl("as_strided", at::native::as_strided_tensorimpl);
  m.impl("view", at::native::view);
  m.impl("_reshape_alias", at::native::_reshape_alias);
  m.impl("unfold", at::native::unfold);
  m.impl("view_as_real", at::native::view_as_real);
  m.impl("view_as_complex", at::native::view_as_complex);
  m.impl("resize_", resize_on_own_device);
  m.impl("set_.source_Storage", at::native::set_);
  m.impl("set_.source_Storage_storage_offset", at::native::set_storage_cpu_);
  m.impl("set_.source_Tensor", at::native::set_tensor_);
  m.impl("_copy_from", copy_from);
  m.impl("_copy_from_and_resize", copy_from_and_resize);
  m.impl("_local_scalar_dense", local_scalar_dense);
}

// A copy reads and writes conjugated and negated views itself, through on_host; the math-bit
// fallbacks would materialize them with another copy, which reaches this one again.
TORCH_LIBRARY_IMPL(aten, Conjugate, m) {
  m.impl("_copy_from", torch::CppFunction::makeFallthrough());
  m.impl("_copy_from_and_resize", torch::CppFunction::makeFallthrough());
}

TORCH_LIBRARY_IMPL(aten, Negative, m) {
  m.impl("_copy_from", torch::CppFunction::makeFallthrough());
  m.impl("_copy_from_and_resize", torch::CppFunction::makeFallthrough());
}

TORCH_LIBRARY_IMPL(_, PrivateUse1, m) {
  m.fallback(torch::CppFunction::makeFromBoxedFunction<&cpu_fallback>());
}
