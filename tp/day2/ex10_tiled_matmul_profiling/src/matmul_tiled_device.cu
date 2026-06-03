#include "src/matmul_tiled_device.h"

#include "src/cuda_runtime_helpers.h"

namespace tp::la {
namespace {

constexpr int kTile = 16;

__global__ void matmul_tiled_kernel(const float *a, const float *b,
                                    const float *c, float *out, int m, int n,
                                    int k, float alpha, float beta) {

  // TODO: Cooperatively load A and B tiles into shared memory and reuse them.
  const int row = blockIdx.y * kTile + threadIdx.y;
  const int col = blockIdx.x * kTile + threadIdx.x;
  if (row < m && col < n) {
    (void)a;
    (void)b;
    (void)k;
    (void)alpha;
    (void)beta;
    out[row * n + col] = c[row * n + col];
  }
}

} // namespace

void launch_matmul_tiled_kernel(const float *a, const float *b, const float *c,
                                float *out, int m, int n, int k, float alpha,
                                float beta, cudaStream_t stream) {
  const dim3 block(kTile, kTile);
  const dim3 grid((n + kTile - 1) / kTile, (m + kTile - 1) / kTile);
  matmul_tiled_kernel<<<grid, block, 0, stream>>>(a, b, c, out, m, n, k, alpha, beta);
  TP_CUDA_CHECK(cudaGetLastError());
}

} // namespace tp::la
