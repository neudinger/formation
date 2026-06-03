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

#include "src/vector_cpu.h"
#include "src/vector_threaded.h"

#include <cmath>
#include <vector>

int main() {
  std::vector<float> a(257);
  std::vector<float> b(257);
  for (std::size_t i = 0; i < a.size(); ++i) {
    a[i] = static_cast<float>(i % 11);
    b[i] = static_cast<float>(i % 7);
  }
  const auto expected = tp::vector_add_cpu(a, b, 1.75f);
  const auto got = tp::vector_add_threaded(a, b, 1.75f, 4);
  require(got.size() == expected.size(), "output size");
  for (std::size_t i = 0; i < got.size(); ++i) {
    require(std::fabs(got[i] - expected[i]) < 1.0e-6f, "threaded value mismatch");
  }
}
