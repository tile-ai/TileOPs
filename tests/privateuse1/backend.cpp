// PrivateUse1 test devices backed by synchronous host memory.

#include <ATen/ATen.h>
#include <ATen/CPUGeneratorImpl.h>
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

#include <unistd.h>

#include <chrono>
#include <cstdlib>
#include <cstring>
#include <vector>

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

  void destroyEvent(void* event, const c10::DeviceIndex device_index) const noexcept override {
    delete static_cast<Event*>(event);
  }

  double elapsedTime(void* start, void* end, const c10::DeviceIndex device_index) const override {
    TORCH_CHECK(start != nullptr && end != nullptr,
                "both events must be recorded before reading the time between them");
    auto elapsed = static_cast<Event*>(end)->time - static_cast<Event*>(start)->time;
    return std::chrono::duration<double, std::milli>(elapsed).count();
  }

  bool queryEvent(void* event) const override { return true; }

  void synchronizeDevice(const c10::DeviceIndex device_index) const override {}
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

  // Every device allocates host memory, so each reports the host's.
  std::pair<size_t, size_t> getMemoryInfo(c10::DeviceIndex device) override {
    const auto page = static_cast<size_t>(sysconf(_SC_PAGE_SIZE));
    return {static_cast<size_t>(sysconf(_SC_AVPHYS_PAGES)) * page,
            static_cast<size_t>(sysconf(_SC_PHYS_PAGES)) * page};
  }
};

Allocator allocator;

REGISTER_ALLOCATOR(c10::DeviceType::PrivateUse1, &allocator);

// A device's generator. Draws run on the host, from the CPU generator it holds.
struct HostGenerator final : c10::GeneratorImpl {
  HostGenerator(c10::DeviceIndex index, at::Generator host)
      : c10::GeneratorImpl(c10::Device(kType, index),
                           c10::DispatchKeySet(c10::DispatchKey::PrivateUse1)),
        host(std::move(host)) {}

  void set_current_seed(uint64_t seed) override { host.set_current_seed(seed); }
  void set_offset(uint64_t offset) override { host.set_offset(offset); }
  uint64_t get_offset() const override { return host.get_offset(); }
  uint64_t current_seed() const override { return host.current_seed(); }
  uint64_t seed() override { return host.seed(); }

  void set_state(const c10::TensorImpl& state) override {
    host.unsafeGetGeneratorImpl()->set_state(state);
  }

  c10::intrusive_ptr<c10::TensorImpl> get_state() const override {
    return host.unsafeGetGeneratorImpl()->get_state();
  }

  HostGenerator* clone_impl() const override {
    return new HostGenerator(device_.index(), host.clone());
  }

  at::Generator host;
};

struct Hooks final : at::PrivateUse1HooksInterface {
  bool isAvailable() const override { return true; }

  bool hasPrimaryContext(c10::DeviceIndex device) const override { return true; }

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

  // A device's default generator draws from the CPU's, which torch.manual_seed seeds.
  const at::Generator& getDefaultGenerator(c10::DeviceIndex device) const override {
    static const std::vector<at::Generator> generators = [] {
      std::vector<at::Generator> made;
      for (c10::DeviceIndex index = 0; index < kDeviceCount; ++index) {
        made.push_back(
            at::make_generator<HostGenerator>(index, at::detail::getDefaultCPUGenerator()));
      }
      return made;
    }();
    const c10::DeviceIndex index = device < 0 ? current_device : device;
    check_index(index);
    return generators[index];
  }

  at::Generator getNewGenerator(c10::DeviceIndex device) const override {
    const c10::DeviceIndex index = device < 0 ? current_device : device;
    check_index(index);
    return at::make_generator<HostGenerator>(index, at::make_generator<at::CPUGeneratorImpl>());
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

// Share host storage while preserving lazy conjugation and negation.
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

// The CPU copy checks shapes, broadcasts and converts dtypes, as copy_ promises.
at::Tensor copy_from(const at::Tensor& self, const at::Tensor& dst, bool non_blocking) {
  on_host(dst).copy_(on_host(self));
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

// The host kernel draws from the CPU generator a device generator holds.
void cpu_fallback(const c10::OperatorHandle& op, torch::jit::Stack* stack) {
  const auto arguments = op.schema().arguments().size();
  for (auto it = stack->end() - arguments; it != stack->end(); ++it) {
    if (it->isGenerator()) {
      auto* generator = dynamic_cast<HostGenerator*>(it->toGenerator().unsafeGetGeneratorImpl());
      if (generator != nullptr) {
        *it = c10::IValue(generator->host);
      }
    }
  }
  at::native::cpu_fallback(op, stack);
}

at::Tensor on_device(const at::Tensor& host, c10::Device device) {
  if (!host.defined()) {
    return host;
  }
  return at::empty(host.sizes(), host.options().device(device)).copy_(host);
}

// PrivateUse1 convolutions dispatch to these rather than to a fallback; compute on the host.
at::Tensor convolution(const at::Tensor& input, const at::Tensor& weight,
                       const std::optional<at::Tensor>& bias, c10::SymIntArrayRef stride,
                       c10::SymIntArrayRef padding, c10::SymIntArrayRef dilation,
                       bool transposed, c10::SymIntArrayRef output_padding,
                       c10::SymInt groups) {
  std::optional<at::Tensor> host_bias;
  if (bias.has_value() && bias->defined()) {
    host_bias = on_host(*bias);
  }
  const at::Tensor result =
      at::convolution_symint(on_host(input), on_host(weight), host_bias, stride, padding,
                             dilation, transposed, output_padding, std::move(groups));
  return on_device(result, input.device());
}

std::tuple<at::Tensor, at::Tensor, at::Tensor> convolution_backward(
    const at::Tensor& grad_output, const at::Tensor& input, const at::Tensor& weight,
    c10::SymIntArrayRef stride, c10::SymIntArrayRef padding, c10::SymIntArrayRef dilation,
    bool transposed, c10::SymIntArrayRef output_padding, c10::SymInt groups,
    std::array<bool, 3> output_mask) {
  const c10::SymInt channels = transposed ? weight.sym_size(1) * groups : weight.sym_size(0);
  const std::vector<c10::SymInt> bias_sizes{channels};
  auto [grad_input, grad_weight, grad_bias] = at::convolution_backward_symint(
      on_host(grad_output), on_host(input), on_host(weight), bias_sizes, stride, padding,
      dilation, transposed, output_padding, std::move(groups), output_mask);
  return {on_device(grad_input, input.device()), on_device(grad_weight, input.device()),
          on_device(grad_bias, input.device())};
}

}  // namespace

// Handle storage operations directly: CPU fallback would copy views instead of aliasing.
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
  m.impl("set_.source_Storage_storage_offset", at::native::set_storage_cpu_);
  m.impl("_copy_from", copy_from);
  m.impl("_copy_from_and_resize", copy_from_and_resize);
  m.impl("convolution_overrideable", convolution);
  m.impl("convolution_backward_overrideable", convolution_backward);
}

// on_host handles math bits; their fallbacks would recursively invoke copy.
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
