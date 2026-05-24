#pragma once

#include <vector>

namespace course::linear_algebra {

std::vector<float> matmul_tiled_cuda(const std::vector<float> &a,
                                     const std::vector<float> &b,
                                     const std::vector<float> &c, int m, int n,
                                     int k, float alpha = 1.0f,
                                     float beta = 0.0f);

} // namespace course::linear_algebra
