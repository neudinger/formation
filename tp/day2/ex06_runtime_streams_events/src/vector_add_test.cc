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
  std::vector<float> a(1024);
  std::vector<float> b(1024);
  for (std::size_t i = 0; i < a.size(); ++i) {
    a[i] = static_cast<float>(i % 13);
    b[i] = static_cast<float>(i % 17);
  }
  const auto expected = tp::vector_add_cpu(a, b, 2.5f);
  expect_close(tp::gpu::vector_add_cuda(a, b, 2.5f, tp::gpu::LaunchApi::kTripleChevron), expected);
  expect_close(tp::gpu::vector_add_cuda(a, b, 2.5f, tp::gpu::LaunchApi::kRuntimeApi), expected);
}
