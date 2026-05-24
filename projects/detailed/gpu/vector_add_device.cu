#include "gpu/vector_add_device.h"

#include "runtime/cuda_errors.h"

#include <cuda_runtime.h>

namespace course::gpu {
namespace {

__global__ void vector_add_kernel(const float *a, const float *b, float *out,
                                  float alpha, int n) {
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) {
    out[i] = alpha * a[i] + b[i];
  }
}

} // namespace

void launch_vector_add_kernel(const float *a, const float *b, float *out,
                              float alpha, int n, cudaStream_t stream,
                              LaunchApi launch_api) {
  constexpr int block_size = 256;
  const dim3 block(block_size);
  const dim3 grid((n + block_size - 1) / block_size);

  if (launch_api == LaunchApi::kTripleChevron) {
    vector_add_kernel<<<grid, block, 0, stream>>>(a, b, out, alpha, n);
    CUDA_CHECK(cudaGetLastError());
    return;
  }

  void *args[] = {
      const_cast<float **>(&a), const_cast<float **>(&b), &out, &alpha, &n,
  };
  CUDA_CHECK(
      cudaLaunchKernel(reinterpret_cast<const void *>(&vector_add_kernel), grid,
                       block, args, 0, stream));
}

} // namespace course::gpu
