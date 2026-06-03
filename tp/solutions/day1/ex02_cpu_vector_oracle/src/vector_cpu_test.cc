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

#include <cmath>
#include <stdexcept>
#include <vector>

int main() {
  const std::vector<float> a{1.0f, 2.0f, -4.0f};
  const std::vector<float> b{10.0f, 20.0f, 30.0f};
  const auto out = tp::vector_add_cpu(a, b, 2.5f);
  require(out.size() == 3, "output size");
  require(std::fabs(out[0] - 12.5f) < 1.0e-6f, "element 0");
  require(std::fabs(out[1] - 25.0f) < 1.0e-6f, "element 1");
  require(std::fabs(out[2] - 20.0f) < 1.0e-6f, "element 2");

  bool threw = false;
  try {
    (void)tp::vector_add_cpu({1.0f}, {1.0f, 2.0f}, 1.0f);
  } catch (const std::invalid_argument &) {
    threw = true;
  }
  require(threw, "size mismatch throws");
}
