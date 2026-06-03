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

#include "src/gpu_info.h"

#include <cuda_runtime.h>
#include <cstring>
#include <string>

int main() {
  require(tp::cuda_version_string(12080) == "12.8 (12080)", "CUDA version decode");
  cudaDeviceProp prop{};
  std::strncpy(prop.name, "Training GPU", sizeof(prop.name) - 1);
  prop.major = 8;
  prop.minor = 6;
  prop.multiProcessorCount = 72;
  prop.warpSize = 32;
  const std::string summary = tp::compact_device_summary(prop);
  require(summary.find("Training GPU") != std::string::npos, "summary includes name");
  require(summary.find("sm_86") != std::string::npos, "summary includes compute capability");
  require(summary.find("SMs=72") != std::string::npos, "summary includes SM count");
  require(summary.find("warp=32") != std::string::npos, "summary includes warp size");
}
