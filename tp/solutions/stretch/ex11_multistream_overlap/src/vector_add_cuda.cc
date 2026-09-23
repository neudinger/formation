#include "src/vector_add.h"

#include "src/chunk_plan.h"
#include "src/cuda_runtime_helpers.h"
#include "src/vector_add_device.h"

#include <cuda_runtime.h>

#include <algorithm>
#include <chrono>
#include <limits>
#include <stdexcept>
#include <vector>

namespace tp::gpu {
namespace {

// Pinned host storage enables asynchronous DMA transfers.
class PinnedBuffer {
public:
  explicit PinnedBuffer(std::size_t count) {
    TP_CUDA_CHECK(cudaMallocHost(&ptr_, count * sizeof(float)));
  }
  PinnedBuffer(const PinnedBuffer &) = delete;
  PinnedBuffer &operator=(const PinnedBuffer &) = delete;
  ~PinnedBuffer() { (void)cudaFreeHost(ptr_); }
  float *get() const { return ptr_; }
private:
  float *ptr_ = nullptr;
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
                                   LaunchApi launch_api, std::size_t stream_count,
                                   double *pipeline_ms) {
  if (pipeline_ms) {
    *pipeline_ms = 0.0;
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

  const auto chunks = tp::make_chunks(a.size(), stream_count);
  PinnedBuffer h_a(a.size()), h_b(b.size()), h_out(a.size());
  std::copy(a.begin(), a.end(), h_a.get());
  std::copy(b.begin(), b.end(), h_b.get());
  DeviceBuffer<float> d_a(a.size()), d_b(b.size()), d_out(a.size());
  std::vector<tp::runtime::Stream> streams(chunks.size());

  const auto begin = std::chrono::steady_clock::now();
  for (std::size_t i = 0; i < chunks.size(); ++i) {
    const auto offset = chunks[i].begin;
    const auto count = chunks[i].end - offset;
    const auto bytes = count * sizeof(float);
    const auto stream = streams[i].get();
    // Ordering within a stream: H2D inputs -> kernel -> D2H output.
    // Disjoint chunks in different streams may overlap.
    TP_CUDA_CHECK(cudaMemcpyAsync(d_a.get() + offset, h_a.get() + offset,
                                 bytes, cudaMemcpyHostToDevice, stream));
    TP_CUDA_CHECK(cudaMemcpyAsync(d_b.get() + offset, h_b.get() + offset,
                                 bytes, cudaMemcpyHostToDevice, stream));
    launch_vector_add_kernel(d_a.get() + offset, d_b.get() + offset,
                             d_out.get() + offset, alpha,
                             static_cast<int>(count), stream, launch_api);
    TP_CUDA_CHECK(cudaMemcpyAsync(h_out.get() + offset, d_out.get() + offset,
                                 bytes, cudaMemcpyDeviceToHost, stream));
  }
  // Submit ALL chunks before waiting; waiting inside the loop would serialize them.
  for (const auto &stream : streams) {
    TP_CUDA_CHECK(cudaStreamSynchronize(stream.get()));
  }
  if (pipeline_ms) {
    *pipeline_ms = std::chrono::duration<double, std::milli>(
        std::chrono::steady_clock::now() - begin).count();
  }
  return std::vector<float>(h_out.get(), h_out.get() + a.size());
}

} // namespace tp::gpu
