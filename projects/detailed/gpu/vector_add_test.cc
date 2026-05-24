#include "cpp/vector_cpu.h"
#include "gpu/vector_add.h"

#include <cmath>
#include <cstdlib>
#include <print>
#include <vector>

namespace {

void require(bool condition, const char *message) {
  if (!condition) {
    std::println(stderr, "test failure: {}", message);
    std::exit(EXIT_FAILURE);
  }
}

void expect_close(const std::vector<float> &got,
                  const std::vector<float> &expected) {
  require(got.size() == expected.size(), "size mismatch");
  for (std::size_t i = 0; i < got.size(); ++i) {
    require(std::fabs(got[i] - expected[i]) < 1.0e-5f, "value mismatch");
  }
}

} // namespace

int main() {
  constexpr std::size_t size = 1 << 16;
  std::vector<float> a(size);
  std::vector<float> b(size);
  for (std::size_t i = 0; i < size; ++i) {
    a[i] = static_cast<float>(i % 17);
    b[i] = static_cast<float>(i % 31);
  }

  const auto expected = course::vector_add_cpu(a, b, 2.5f);
  expect_close(course::gpu::vector_add_cuda(
                   a, b, 2.5f, course::gpu::LaunchApi::kRuntimeApi),
               expected);
  expect_close(course::gpu::vector_add_cuda(
                   a, b, 2.5f, course::gpu::LaunchApi::kTripleChevron),
               expected);

  std::println("vector_add_test passed");
}
