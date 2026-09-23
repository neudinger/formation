#pragma once

#include <cstddef>
#include <vector>

namespace tp::gpu {

enum class LaunchApi {
  kRuntimeApi,
  kTripleChevron,
};

// pipeline_ms measures submission through completion, excluding allocation/staging.
// A zero stream_count is treated as one; small inputs use at most one stream per element.
std::vector<float> vector_add_cuda(const std::vector<float> &a,
                                   const std::vector<float> &b, float alpha,
                                   LaunchApi launch_api = LaunchApi::kTripleChevron,
                                   std::size_t stream_count = 1,
                                   double *pipeline_ms = nullptr);

} // namespace tp::gpu
