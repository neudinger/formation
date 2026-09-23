#include "src/reduction.h"

#include <cmath>
#include <cstdlib>
#include <iostream>
#include <numeric>
#include <vector>

namespace {

void verify(const std::vector<float> &input) {
  const double expected = std::accumulate(input.begin(), input.end(), 0.0);
  for (auto method : {tp::gpu::Reduction::per_element_atomic,
                      tp::gpu::Reduction::block_local}) {
    float kernel_ms = -1.0f;
    const float got = tp::gpu::sum_cuda(input, method, &kernel_ms);
    // Small integer/quarter inputs keep every partial sum exactly representable.
    if (got != expected || !std::isfinite(kernel_ms) || kernel_ms < 0.0f ||
        (input.empty() && kernel_ms != 0.0f)) {
      std::cerr << "size=" << input.size() << " method=" << static_cast<int>(method)
                << " expected=" << expected << " got=" << got
                << " kernel_ms=" << kernel_ms << '\n';
      std::exit(EXIT_FAILURE);
    }
  }
}

} // namespace

int main() {
  for (int size : {0, 1, 255, 256, 257, 1003, 65537}) {
    std::vector<float> input(size, 1.0f);
    verify(input); // Full blocks, partial blocks, and contention on one result.
    for (int i = 0; i < size; ++i) {
      input[i] = static_cast<float>(i % 9 - 4) * 0.25f;
    }
    verify(input); // Mixed signs and a different result on the next call.
    verify(std::vector<float>(size, 0.0f));
  }
}
