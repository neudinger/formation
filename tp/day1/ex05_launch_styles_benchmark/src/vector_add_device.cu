#include "src/vector_add_device.h"

#include "src/cuda_runtime_helpers.h"

namespace tp::gpu {
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
    TP_CUDA_CHECK(cudaGetLastError());
    return;
  }
  // TODO: Launch the same kernel with cudaLaunchKernel for the Runtime API path.
  (void)a;
  (void)b;
  (void)out;
  (void)alpha;
  (void)n;
  TP_CUDA_CHECK(cudaGetLastError());
}

} // namespace tp::gpu
