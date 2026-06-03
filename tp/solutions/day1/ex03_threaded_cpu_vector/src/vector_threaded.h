#pragma once

#include <cstddef>
#include <vector>

namespace tp {

std::vector<float> vector_add_threaded(const std::vector<float> &a,
                                       const std::vector<float> &b,
                                       float alpha, std::size_t worker_count);

} // namespace tp
