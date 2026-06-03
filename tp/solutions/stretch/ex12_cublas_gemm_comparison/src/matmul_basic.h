#pragma once

#include <vector>

namespace tp::la {

std::vector<float> matmul_basic_cuda(const std::vector<float> &a,
                                     const std::vector<float> &b,
                                     const std::vector<float> &c, int m, int n,
                                     int k, float alpha, float beta);

} // namespace tp::la
