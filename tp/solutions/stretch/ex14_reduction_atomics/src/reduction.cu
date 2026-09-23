#include "src/reduction.h"

#include <cuda_runtime.h>

#include <limits>
#include <stdexcept>
#include <string>

namespace tp::gpu {
namespace {

constexpr int kBlockSize = 256; // Power of two for the shared-memory tree.

void check(cudaError_t error, const char *call) {
  if (error != cudaSuccess) {
    throw std::runtime_error(std::string(call) + ": " + cudaGetErrorString(error));
  }
}
#define CUDA_CHECK(call) check((call), #call)

struct DeviceBuffer {
  float *data = nullptr;
  explicit DeviceBuffer(std::size_t count) {
    CUDA_CHECK(cudaMalloc(&data, count * sizeof(float)));
  }
  ~DeviceBuffer() { (void)cudaFree(data); }
  DeviceBuffer(const DeviceBuffer &) = delete;
  DeviceBuffer &operator=(const DeviceBuffer &) = delete;
};

struct Event {
  cudaEvent_t handle{};
  Event() { CUDA_CHECK(cudaEventCreate(&handle)); }
  ~Event() { (void)cudaEventDestroy(handle); }
  Event(const Event &) = delete;
  Event &operator=(const Event &) = delete;
};

__global__ void sum_atomic(const float *input, float *sum, int size) {
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < size) {
    atomicAdd(sum, input[i]);
  }
}

__global__ void sum_block(const float *input, float *sum, int size) {
  __shared__ float partial[kBlockSize];
  const int lane = threadIdx.x;
  const int i = blockIdx.x * blockDim.x + lane;
  // Inactive lanes still participate in every barrier.
  partial[lane] = i < size ? input[i] : 0.0f;
  __syncthreads();
  for (int stride = kBlockSize / 2; stride > 0; stride /= 2) {
    if (lane < stride) {
      partial[lane] += partial[lane + stride];
    }
    __syncthreads();
  }
  if (lane == 0) {
    // Different blocks still update the same global result.
    atomicAdd(sum, partial[0]);
  }
}

} // namespace

float sum_cuda(const std::vector<float> &input, Reduction method,
               float *kernel_ms) {
  if (kernel_ms) {
    *kernel_ms = 0.0f;
  }
  if (input.empty()) {
    return 0.0f;
  }
  if (input.size() > static_cast<std::size_t>(std::numeric_limits<int>::max())) {
    throw std::invalid_argument("sum_cuda: input too large");
  }
  const int size = static_cast<int>(input.size());
  const int blocks = size / kBlockSize + (size % kBlockSize != 0);
  DeviceBuffer device(input.size() + 1);
  float *sum = device.data + input.size();
  Event start, stop;
  CUDA_CHECK(cudaMemcpy(device.data, input.data(), input.size() * sizeof(float),
                        cudaMemcpyHostToDevice));
  // Reset on every call; an atomic addition does not initialize its destination.
  CUDA_CHECK(cudaMemset(sum, 0, sizeof(float)));
  CUDA_CHECK(cudaEventRecord(start.handle));
  if (method == Reduction::per_element_atomic) {
    sum_atomic<<<blocks, kBlockSize>>>(device.data, sum, size);
  } else {
    sum_block<<<blocks, kBlockSize>>>(device.data, sum, size);
  }
  CUDA_CHECK(cudaGetLastError());
  CUDA_CHECK(cudaEventRecord(stop.handle));
  CUDA_CHECK(cudaEventSynchronize(stop.handle));
  if (kernel_ms) {
    CUDA_CHECK(cudaEventElapsedTime(kernel_ms, start.handle, stop.handle));
  }
  float result = 0.0f;
  CUDA_CHECK(cudaMemcpy(&result, sum, sizeof(float), cudaMemcpyDeviceToHost));
  return result;
}

} // namespace tp::gpu
