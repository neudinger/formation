#include "src/vector_add.h"

#include <cuda_runtime.h>

#include <algorithm>
#include <limits>
#include <stdexcept>
#include <string>

namespace tp::gpu {
namespace {

void check(cudaError_t error, const char *call) {
  if (error != cudaSuccess) {
    throw std::runtime_error(std::string(call) + ": " + cudaGetErrorString(error));
  }
}
#define CUDA_CHECK(call) check((call), #call)

struct Buffer {
  float *data = nullptr;
  Buffer(std::size_t count, MemoryMode mode) {
    if (mode == MemoryMode::managed) {
      CUDA_CHECK(cudaMallocManaged(&data, count * sizeof(float)));
    } else {
      CUDA_CHECK(cudaMalloc(&data, count * sizeof(float)));
    }
  }
  ~Buffer() { (void)cudaFree(data); }
  Buffer(const Buffer &) = delete;
  Buffer &operator=(const Buffer &) = delete;
};

// Same arithmetic and launch geometry as Ex04 for both allocation strategies.
__global__ void vector_add_kernel(const float *a, const float *b, float *out,
                                  float alpha, int size) {
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < size) {
    out[i] = alpha * a[i] + b[i];
  }
}

} // namespace

std::vector<float> vector_add_cuda(const std::vector<float> &a,
                                   const std::vector<float> &b, float alpha,
                                   MemoryMode mode) {
  if (a.size() != b.size()) {
    throw std::invalid_argument("vector_add_cuda: size mismatch");
  }
  if (a.empty()) {
    return {};
  }
  if (a.size() > static_cast<std::size_t>(std::numeric_limits<int>::max())) {
    throw std::invalid_argument("vector_add_cuda: input too large");
  }
  const std::size_t bytes = a.size() * sizeof(float);
  Buffer input_a(a.size(), mode), input_b(b.size(), mode), output(a.size(), mode);
  std::vector<float> result(a.size());
  if (mode == MemoryMode::managed) {
    std::copy(a.begin(), a.end(), input_a.data);
    std::copy(b.begin(), b.end(), input_b.data);
  } else {
    CUDA_CHECK(cudaMemcpy(input_a.data, a.data(), bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(input_b.data, b.data(), bytes, cudaMemcpyHostToDevice));
  }
  constexpr int block_size = 256;
  const int size = static_cast<int>(a.size());
  const int blocks = size / block_size + (size % block_size != 0);
  vector_add_kernel<<<blocks, block_size>>>(input_a.data, input_b.data, output.data,
                                           alpha, size);
  CUDA_CHECK(cudaGetLastError());
  // Wait before the CPU touches managed memory, including on limited-UM systems.
  CUDA_CHECK(cudaDeviceSynchronize());
  if (mode == MemoryMode::managed) {
    std::copy(output.data, output.data + a.size(), result.begin());
  } else {
    CUDA_CHECK(cudaMemcpy(result.data(), output.data, bytes, cudaMemcpyDeviceToHost));
  }
  return result;
}

} // namespace tp::gpu
