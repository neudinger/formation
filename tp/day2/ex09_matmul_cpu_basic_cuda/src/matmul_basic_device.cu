#include "src/matmul_basic_device.h"

#include "src/cuda_runtime_helpers.h"

namespace tp::la {
namespace {

__global__ void matmul_basic_kernel(const float *a, const float *b,
                                    const float *c, float *out, int m, int n,
                                    int k, float alpha, float beta) {
  const int row = blockIdx.y * blockDim.y + threadIdx.y;
  const int col = blockIdx.x * blockDim.x + threadIdx.x;
  if (row < m && col < n) {
    // TODO: Accumulate over K for this output cell.
    (void)a;
    (void)b;
    (void)k;
    (void)alpha;
    (void)beta;
    out[row * n + col] = c[row * n + col];
  }
}

} // namespace

void launch_matmul_basic_kernel(const float *a, const float *b, const float *c,
                                float *out, int m, int n, int k, float alpha,
                                float beta, cudaStream_t stream) {
  const dim3 block(16, 16);
  const dim3 grid((n + block.x - 1) / block.x, (m + block.y - 1) / block.y);
  matmul_basic_kernel<<<grid, block, 0, stream>>>(a, b, c, out, m, n, k, alpha, beta);
  TP_CUDA_CHECK(cudaGetLastError());
}

} // namespace tp::la
