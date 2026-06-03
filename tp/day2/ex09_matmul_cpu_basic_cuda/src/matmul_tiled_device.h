#pragma once

#include <cuda_runtime.h>

namespace tp::la {

void launch_matmul_tiled_kernel(const float *a, const float *b, const float *c,
                                float *out, int m, int n, int k, float alpha,
                                float beta, cudaStream_t stream);

} // namespace tp::la
