#pragma once

#include "src/vector_add.h"

#include <cuda_runtime.h>

namespace tp::gpu {

void launch_vector_add_kernel(const float *a, const float *b, float *out,
                              float alpha, int n, cudaStream_t stream,
                              LaunchApi launch_api);

} // namespace tp::gpu
