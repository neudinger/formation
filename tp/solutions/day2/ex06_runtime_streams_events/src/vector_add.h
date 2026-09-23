#pragma once

#include <vector>

namespace tp::gpu {

enum class LaunchApi {
  kRuntimeApi,
  kTripleChevron,
};

// Optional kernel_ms receives CUDA event time, excluding allocation and copies.
std::vector<float> vector_add_cuda(const std::vector<float> &a,
                                   const std::vector<float> &b, float alpha,
                                   LaunchApi launch_api = LaunchApi::kTripleChevron,
                                   float *kernel_ms = nullptr);

} // namespace tp::gpu
