#include "src/vector_add.h"

#include <chrono>
#include <cmath>
#include <cstdlib>
#include <iostream>
#include <vector>

int main() {
  constexpr int size = 1 << 20;
  constexpr int repeats = 3;
  constexpr float alpha = 2.5f;
  std::vector<float> a(size), b(size), expected(size);
  for (int i = 0; i < size; ++i) {
    a[i] = static_cast<float>(i % 13 - 6) * 0.25f;
    b[i] = static_cast<float>(i % 17 - 8) * 0.5f;
    expected[i] = alpha * a[i] + b[i];
  }
  std::cout << "elements=" << size << " repeats=" << repeats << '\n';
  for (auto mode : {tp::gpu::MemoryMode::explicit_copies,
                    tp::gpu::MemoryMode::managed}) {
    double total_ms = 0.0;
    for (int run = 0; run <= repeats; ++run) {
      const auto begin = std::chrono::steady_clock::now();
      const auto out = tp::gpu::vector_add_cuda(a, b, alpha, mode);
      const auto end = std::chrono::steady_clock::now();
      for (int i = 0; i < size; ++i) {
        if (!(std::fabs(out[i] - expected[i]) <= 1.0e-5f)) {
          std::cerr << "incorrect result at index " << i << '\n';
          return EXIT_FAILURE;
        }
      }
      if (run > 0) { // Exclude one warm-up call per mode.
        total_ms += std::chrono::duration<double, std::milli>(end - begin).count();
      }
    }
    std::cout << (mode == tp::gpu::MemoryMode::explicit_copies
                      ? "explicit_copies" : "managed")
              << " total_ms=" << total_ms / repeats << '\n';
  }
}
