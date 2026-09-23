#pragma once

#include <vector>

namespace tp::gpu {

enum class Reduction { per_element_atomic, block_local };

// Empty input returns zero. Optional timing brackets only the kernel.
float sum_cuda(const std::vector<float> &input, Reduction method,
               float *kernel_ms = nullptr);

} // namespace tp::gpu
