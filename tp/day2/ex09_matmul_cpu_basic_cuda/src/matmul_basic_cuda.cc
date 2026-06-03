#include "src/matmul_basic.h"

#include "src/cuda_runtime_helpers.h"
#include "src/matmul_basic_device.h"

#include <cuda_runtime.h>

#include <stdexcept>
#include <vector>

namespace tp::la {
namespace {

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

void validate(const std::vector<float> &a, const std::vector<float> &b,
              const std::vector<float> &c, int m, int n, int k) {
  if (m < 0 || n < 0 || k < 0 ||
      a.size() != static_cast<std::size_t>(m * k) ||
      b.size() != static_cast<std::size_t>(k * n) ||
      c.size() != static_cast<std::size_t>(m * n)) {
    throw std::invalid_argument("matmul_cuda: size mismatch");
  }
}

} // namespace

std::vector<float> matmul_basic_cuda(const std::vector<float> &a,
                              const std::vector<float> &b,
                              const std::vector<float> &c, int m, int n, int k,
                              float alpha, float beta) {
  validate(a, b, c, m, n, k);
  std::vector<float> out(static_cast<std::size_t>(m * n));
  if (out.empty()) {
    return out;
  }
  tp::runtime::Stream stream;
  DeviceBuffer<float> d_a(a.size());
  DeviceBuffer<float> d_b(b.size());
  DeviceBuffer<float> d_c(c.size());
  DeviceBuffer<float> d_out(out.size());
  TP_CUDA_CHECK(cudaMemcpyAsync(d_a.get(), a.data(), d_a.bytes(), cudaMemcpyHostToDevice, stream.get()));
  TP_CUDA_CHECK(cudaMemcpyAsync(d_b.get(), b.data(), d_b.bytes(), cudaMemcpyHostToDevice, stream.get()));
  TP_CUDA_CHECK(cudaMemcpyAsync(d_c.get(), c.data(), d_c.bytes(), cudaMemcpyHostToDevice, stream.get()));
  launch_matmul_basic_kernel(d_a.get(), d_b.get(), d_c.get(), d_out.get(), m, n, k, alpha, beta, stream.get());
  TP_CUDA_CHECK(cudaMemcpyAsync(out.data(), d_out.get(), d_out.bytes(), cudaMemcpyDeviceToHost, stream.get()));
  TP_CUDA_CHECK(cudaStreamSynchronize(stream.get()));
  return out;
}

} // namespace tp::la
