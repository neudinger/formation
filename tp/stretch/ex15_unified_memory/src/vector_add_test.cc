#include "src/vector_add.h"

#include <cmath>
#include <cstdlib>
#include <iostream>
#include <stdexcept>
#include <vector>

namespace {

void require(bool condition, const char *message) {
  if (!condition) {
    std::cerr << "test failure: " << message << '\n';
    std::exit(EXIT_FAILURE);
  }
}

} // namespace

int main() {
  for (auto mode : {tp::gpu::MemoryMode::explicit_copies,
                    tp::gpu::MemoryMode::managed}) {
    for (int size : {0, 1, 255, 256, 257, 1003, 65537}) {
      std::vector<float> a(size), b(size);
      for (int i = 0; i < size; ++i) {
        a[i] = static_cast<float>(i % 13 - 6) * 0.25f;
        b[i] = static_cast<float>(i % 17 - 8) * 0.5f;
      }
      for (float alpha : {2.5f, 0.0f, -1.0f}) {
        const auto out = tp::gpu::vector_add_cuda(a, b, alpha, mode);
        require(out.size() == a.size(), "output size");
        for (std::size_t i = 0; i < out.size(); ++i) {
          require(std::fabs(out[i] - (alpha * a[i] + b[i])) <= 1.0e-5f,
                  "CUDA result differs from CPU reference");
        }
      }
    }
    bool rejected = false;
    try {
      (void)tp::gpu::vector_add_cuda({}, {1.0f}, 1.0f, mode);
    } catch (const std::invalid_argument &) {
      rejected = true;
    }
    require(rejected, "mismatched input sizes must be rejected");
  }
}
