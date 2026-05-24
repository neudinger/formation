#pragma once

#include <cstddef>
#include <vector>

namespace course {

std::vector<float> vector_add_threaded(const std::vector<float> &a,
                                       const std::vector<float> &b, float alpha,
                                       std::size_t thread_count = 0);

} // namespace course
