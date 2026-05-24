#pragma once

#include <vector>

namespace course::gpu {

enum class LaunchApi {
  kTripleChevron,
  kRuntimeApi,
};

std::vector<float>
vector_add_cuda(const std::vector<float> &a, const std::vector<float> &b,
                float alpha, LaunchApi launch_api = LaunchApi::kRuntimeApi);

} // namespace course::gpu
