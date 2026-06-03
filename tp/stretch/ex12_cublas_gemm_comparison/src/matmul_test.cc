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

#include "src/matmul_basic.h"
#include "src/matmul_cpu.h"
#include "src/matmul_tiled.h"

#include <cmath>
#include <vector>

namespace {

void expect_close(const std::vector<float> &got, const std::vector<float> &expected) {
  require(got.size() == expected.size(), "size mismatch");
  for (std::size_t i = 0; i < got.size(); ++i) {
    require(std::fabs(got[i] - expected[i]) < 1.0e-4f, "value mismatch");
  }
}

} // namespace

int main() {
  constexpr int m = 2;
  constexpr int n = 3;
  constexpr int k = 4;
  const std::vector<float> a{1, 2, 3, 4, 5, 6, 7, 8};
  const std::vector<float> b{1, 0, 2, 0, 1, 3, 2, 1, 0, 1, 2, 1};
  const std::vector<float> c(static_cast<std::size_t>(m * n), 1.0f);
  const auto expected = tp::la::matmul_cpu(a, b, c, m, n, k, 1.0f, 0.5f);
  const std::vector<float> hand{18.5f, 16.5f, 13.5f, 42.5f, 32.5f, 37.5f};
  expect_close(expected, hand);
  expect_close(tp::la::matmul_basic_cuda(a, b, c, m, n, k, 1.0f, 0.5f), expected);
  expect_close(tp::la::matmul_tiled_cuda(a, b, c, m, n, k, 1.0f, 0.5f), expected);
}
