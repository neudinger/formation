#include "src/reduction.h"

#include <cstdlib>
#include <iostream>
#include <numeric>
#include <vector>

int main() {
  constexpr int size = 1 << 20;
  constexpr int repeats = 5;
  std::vector<float> input(size);
  for (int i = 0; i < size; ++i) {
    input[i] = static_cast<float>(i % 8) * 0.25f;
  }
  const double expected = std::accumulate(input.begin(), input.end(), 0.0);
  std::cout << "elements=" << size << " repeats=" << repeats << '\n';
  for (auto method : {tp::gpu::Reduction::per_element_atomic,
                      tp::gpu::Reduction::block_local}) {
    double total_ms = 0.0;
    float result = 0.0f;
    for (int run = 0; run <= repeats; ++run) {
      float kernel_ms = 0.0f;
      result = tp::gpu::sum_cuda(input, method, &kernel_ms);
      // These bounded quarter values have exactly representable partial sums.
      if (result != expected) {
        std::cerr << "incorrect sum: " << result << " expected " << expected << '\n';
        return EXIT_FAILURE;
      }
      if (run > 0) { // Exclude one warm-up per kernel.
        total_ms += kernel_ms;
      }
    }
    std::cout << (method == tp::gpu::Reduction::per_element_atomic
                      ? "per_element_atomic" : "block_local")
              << " kernel_ms=" << total_ms / repeats << " sum=" << result << '\n';
  }
}
