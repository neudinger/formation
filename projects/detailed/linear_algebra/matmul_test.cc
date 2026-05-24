#include "linear_algebra/matmul_basic.h"
#include "linear_algebra/matmul_cpu.h"
#include "linear_algebra/matmul_tiled.h"

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
    require(std::fabs(got[i] - expected[i]) < 1.0e-4f, "value mismatch");
  }
}

} // namespace

int main() {
  constexpr int m = 5;
  constexpr int n = 7;
  constexpr int k = 3;

  std::vector<float> a(m * k);
  std::vector<float> b(k * n);
  std::vector<float> c(m * n, 1.0f);
  for (std::size_t i = 0; i < a.size(); ++i) {
    a[i] = static_cast<float>((i % 5) + 1);
  }
  for (std::size_t i = 0; i < b.size(); ++i) {
    b[i] = static_cast<float>((i % 7) - 3);
  }

  const auto expected =
      course::linear_algebra::matmul_cpu(a, b, c, m, n, k, 2.0f, 0.5f);
  expect_close(
      course::linear_algebra::matmul_basic_cuda(a, b, c, m, n, k, 2.0f, 0.5f),
      expected);
  expect_close(
      course::linear_algebra::matmul_tiled_cuda(a, b, c, m, n, k, 2.0f, 0.5f),
      expected);

  std::println("matmul_test passed");
}
