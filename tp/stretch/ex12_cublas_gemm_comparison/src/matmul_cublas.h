#pragma once

#include <optional>
#include <string_view>
#include <vector>

namespace tp::la {

std::string_view cublas_strategy_name();
std::optional<std::vector<float>> try_matmul_cublas(const std::vector<float> &a,
                                                    const std::vector<float> &b,
                                                    const std::vector<float> &c,
                                                    int m, int n, int k,
                                                    float alpha, float beta);

} // namespace tp::la
