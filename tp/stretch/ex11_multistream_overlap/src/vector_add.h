#pragma once

#include <vector>

namespace tp::gpu {

enum class LaunchApi {
  kRuntimeApi,
  kTripleChevron,
};

std::vector<float> vector_add_cuda(const std::vector<float> &a,
                                   const std::vector<float> &b, float alpha,
                                   LaunchApi launch_api = LaunchApi::kTripleChevron);

} // namespace tp::gpu
