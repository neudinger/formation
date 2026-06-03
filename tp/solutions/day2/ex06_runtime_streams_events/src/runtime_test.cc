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

#include "src/cuda_runtime_helpers.h"

#include <cuda_runtime.h>
#include <stdexcept>
#include <string>

int main() {
  const std::string message = tp::runtime::cuda_error_message(cudaErrorInvalidValue, "cudaDemo", "demo.cc", 42);
  require(message.find("cudaDemo") != std::string::npos, "message includes call");
  require(message.find("demo.cc:42") != std::string::npos, "message includes location");
  bool threw = false;
  try {
    tp::runtime::cuda_check(cudaErrorInvalidValue, "cudaDemo", "demo.cc", 42);
  } catch (const std::runtime_error &) {
    threw = true;
  }
  require(threw, "cuda_check throws on error");
}
