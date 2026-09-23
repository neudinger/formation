#pragma once

#include <vector>

namespace tp::gpu {

enum class MemoryMode { explicit_copies, managed };

// Computes alpha * a + b. Empty inputs return an empty vector.
std::vector<float> vector_add_cuda(const std::vector<float> &a,
                                   const std::vector<float> &b, float alpha,
                                   MemoryMode mode);

} // namespace tp::gpu
