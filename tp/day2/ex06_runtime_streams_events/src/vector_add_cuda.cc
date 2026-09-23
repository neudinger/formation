#include "src/vector_add.h"

#include "src/cuda_runtime_helpers.h"
#include "src/vector_add_device.h"

#include <cuda_runtime.h>

#include <limits>
#include <stdexcept>
#include <vector>

namespace tp::gpu {
namespace {

// Keep event ownership local to this timing example.
struct Event {
  cudaEvent_t handle{};
  Event() { TP_CUDA_CHECK(cudaEventCreate(&handle)); }
  ~Event() { (void)cudaEventDestroy(handle); }
  Event(const Event &) = delete;
  Event &operator=(const Event &) = delete;
};

template <typename T> class DeviceBuffer {
public:
  explicit DeviceBuffer(std::size_t count) : count_(count) {
    if (count_ > 0) {
      TP_CUDA_CHECK(cudaMalloc(&ptr_, count_ * sizeof(T)));
    }
  }
  DeviceBuffer(const DeviceBuffer &) = delete;
  DeviceBuffer &operator=(const DeviceBuffer &) = delete;
  ~DeviceBuffer() { (void)cudaFree(ptr_); }
  [[nodiscard]] T *get() const { return ptr_; }
  [[nodiscard]] std::size_t bytes() const { return count_ * sizeof(T); }
private:
  T *ptr_ = nullptr;
  std::size_t count_ = 0;
};

} // namespace

std::vector<float> vector_add_cuda(const std::vector<float> &a,
                                   const std::vector<float> &b, float alpha,
                                   LaunchApi launch_api, float *kernel_ms) {
  if (kernel_ms) {
    *kernel_ms = 0.0f;
  }
  if (a.size() != b.size()) {
    throw std::invalid_argument("vector_add_cuda: size mismatch");
  }
  if (a.empty()) {
    return {};
  }
  if (a.size() > static_cast<std::size_t>(std::numeric_limits<int>::max())) {
    throw std::invalid_argument("vector_add_cuda: input too large");
  }

  tp::runtime::Stream stream;
  DeviceBuffer<float> d_a(a.size());
  DeviceBuffer<float> d_b(b.size());
  DeviceBuffer<float> d_out(a.size());
  std::vector<float> out(a.size());
  Event start, stop;

  TP_CUDA_CHECK(cudaMemcpyAsync(d_a.get(), a.data(), d_a.bytes(), cudaMemcpyHostToDevice, stream.get()));
  TP_CUDA_CHECK(cudaMemcpyAsync(d_b.get(), b.data(), d_b.bytes(), cudaMemcpyHostToDevice, stream.get()));
  // Both events use the kernel's stream and bracket only the kernel.
  TP_CUDA_CHECK(cudaEventRecord(start.handle, stream.get()));
  launch_vector_add_kernel(d_a.get(), d_b.get(), d_out.get(), alpha,
                           static_cast<int>(a.size()), stream.get(), launch_api);
  TP_CUDA_CHECK(cudaEventRecord(stop.handle, stream.get()));
  TP_CUDA_CHECK(cudaMemcpyAsync(out.data(), d_out.get(), d_out.bytes(), cudaMemcpyDeviceToHost, stream.get()));
  TP_CUDA_CHECK(cudaStreamSynchronize(stream.get()));
  // Stream synchronization also guarantees that the stop event has completed.
  if (kernel_ms) {
    TP_CUDA_CHECK(cudaEventElapsedTime(kernel_ms, start.handle, stop.handle));
  }
  return out;
}

} // namespace tp::gpu
