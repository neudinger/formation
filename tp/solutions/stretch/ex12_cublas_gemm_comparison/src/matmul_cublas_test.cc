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

#include "src/matmul_cublas.h"

int main() {
  require(tp::la::cublas_strategy_name() == "dynamic-cublas", "strategy name");
}
