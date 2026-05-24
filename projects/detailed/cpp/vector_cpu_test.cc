#include "cpp/vector_cpu.h"
#include "cpp/vector_threaded.h"

#include <cmath>
#include <cstdlib>
#include <print>
#include <stdexcept>
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
  const std::vector<float> a{1.0f, 2.0f, 3.0f, 4.0f};
  const std::vector<float> b{10.0f, 20.0f, 30.0f, 40.0f};
  const std::vector<float> expected{12.0f, 24.0f, 36.0f, 48.0f};

  expect_close(course::vector_add_cpu(a, b, 2.0f), expected);
  expect_close(course::vector_add_threaded(a, b, 2.0f, 2), expected);

  bool threw = false;
  try {
    (void)course::vector_add_cpu({1.0f}, {}, 2.0f);
  } catch (const std::invalid_argument &) {
    threw = true;
  }
  require(threw, "size mismatch should throw");

  std::println("vector_cpu_test passed");
  return EXIT_SUCCESS;
}
