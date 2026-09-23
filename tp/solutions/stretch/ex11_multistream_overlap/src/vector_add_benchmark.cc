#include "src/vector_add.h"
#include "src/vector_cpu.h"

#include <chrono>
#include <cmath>
#include <iostream>
#include <vector>

int main() {
  constexpr std::size_t size = 1 << 22;
  constexpr int repeats = 3;
  std::vector<float> a(size), b(size);
  for (std::size_t i = 0; i < size; ++i) {
    a[i] = static_cast<float>(i % 13);
    b[i] = static_cast<float>(i % 17);
  }
  const auto expected = tp::vector_add_cpu(a, b, 3.0f);
  std::cout << "elements=" << size << " repeats=" << repeats << "\n";
  for (const std::size_t streams : {1, 2, 4}) {
    // Same kernel, data, pinned buffers, and timing boundaries for every count.
    auto out = tp::gpu::vector_add_cuda(
        a, b, 3.0f, tp::gpu::LaunchApi::kTripleChevron, streams); // warm-up
    double total_ms = 0.0, pipeline_ms = 0.0;
    for (int repeat = 0; repeat < repeats; ++repeat) {
      double sample_ms = 0.0;
      const auto begin = std::chrono::steady_clock::now();
      out = tp::gpu::vector_add_cuda(
          a, b, 3.0f, tp::gpu::LaunchApi::kTripleChevron, streams, &sample_ms);
      total_ms += std::chrono::duration<double, std::milli>(
          std::chrono::steady_clock::now() - begin).count();
      pipeline_ms += sample_ms;
      for (std::size_t i = 0; i < size; ++i) {
        if (!(std::fabs(out[i] - expected[i]) < 1.0e-5f)) {
          std::cerr << "incorrect result for streams=" << streams << "\n";
          return 1;
        }
      }
    }
    std::cout << "streams=" << streams
              << " total_ms=" << total_ms / repeats
              << " pipeline_ms=" << pipeline_ms / repeats << "\n";
  }
}
