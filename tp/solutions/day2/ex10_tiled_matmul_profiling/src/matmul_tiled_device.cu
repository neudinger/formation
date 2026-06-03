#include "src/matmul_tiled_device.h"

#include "src/cuda_runtime_helpers.h"

namespace tp::la {
namespace {

constexpr int kTile = 16;

__global__ void matmul_tiled_kernel(const float *a, const float *b,
                                    const float *c, float *out, int m, int n,
                                    int k, float alpha, float beta) {

  __shared__ float tile_a[kTile][kTile];
  __shared__ float tile_b[kTile][kTile];
  const int row = blockIdx.y * kTile + threadIdx.y;
  const int col = blockIdx.x * kTile + threadIdx.x;
  float acc = 0.0f;
  for (int tile = 0; tile < (k + kTile - 1) / kTile; ++tile) {
    const int a_col = tile * kTile + threadIdx.x;
    const int b_row = tile * kTile + threadIdx.y;
    tile_a[threadIdx.y][threadIdx.x] = row < m && a_col < k ? a[row * k + a_col] : 0.0f;
    tile_b[threadIdx.y][threadIdx.x] = b_row < k && col < n ? b[b_row * n + col] : 0.0f;
    __syncthreads();
    for (int inner = 0; inner < kTile; ++inner) {
      acc += tile_a[threadIdx.y][inner] * tile_b[inner][threadIdx.x];
    }
    __syncthreads();
  }
  if (row < m && col < n) {
    const int index = row * n + col;
    out[index] = alpha * acc + beta * c[index];
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
