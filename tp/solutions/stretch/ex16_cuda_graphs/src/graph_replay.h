#pragma once

#include <vector>

namespace tp::gpu {

enum class LaunchMode { direct, graph };
inline constexpr int kKernelsPerBatch = 8;

// Each batch adds 1 to every element eight times. Timing excludes setup/copies.
std::vector<float> repeat_add(const std::vector<float> &input, int batches,
                              LaunchMode mode, double *replay_ms = nullptr);

} // namespace tp::gpu
