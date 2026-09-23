#include <cstdlib>
#include <iostream>

namespace {

void require(bool condition, const char *message) {
  if (!condition) {
    std::cerr << "test failure: " << message << "\n";
    std::exit(EXIT_FAILURE);
  }
}

} // namespace

#include "src/vector_add.h"
#include "src/vector_cpu.h"

#include <cmath>
#include <stdexcept>
#include <vector>

namespace {

void expect_close(const std::vector<float> &got, const std::vector<float> &expected) {
  require(got.size() == expected.size(), "size mismatch");
  for (std::size_t i = 0; i < got.size(); ++i) {
    require(std::fabs(got[i] - expected[i]) < 1.0e-5f, "value mismatch");
  }
}

} // namespace

int main() {
  // Tiny inputs, uneven chunks, and partial CUDA blocks exercise offset handling.
  for (const std::size_t size : {0, 1, 3, 257, 1003, 4097}) {
    std::vector<float> a(size), b(size);
    for (std::size_t i = 0; i < size; ++i) {
      a[i] = static_cast<float>(i % 13);
      b[i] = static_cast<float>(i % 17);
    }
    const auto expected = tp::vector_add_cpu(a, b, 2.5f);
    for (const std::size_t streams : {0, 1, 2, 4}) {
      for (const auto api : {tp::gpu::LaunchApi::kTripleChevron,
                             tp::gpu::LaunchApi::kRuntimeApi}) {
        double pipeline_ms = -1.0;
        expect_close(tp::gpu::vector_add_cuda(a, b, 2.5f, api, streams,
                                              &pipeline_ms), expected);
        require(std::isfinite(pipeline_ms) && pipeline_ms >= 0.0, "valid timing");
        if (size == 0) {
          require(pipeline_ms == 0.0, "empty input has no GPU work");
        }
      }
    }
  }
  bool rejected = false;
  try {
    (void)tp::gpu::vector_add_cuda({1.0f}, {}, 2.5f);
  } catch (const std::invalid_argument &) {
    rejected = true;
  }
  require(rejected, "reject mismatched input sizes");
}
