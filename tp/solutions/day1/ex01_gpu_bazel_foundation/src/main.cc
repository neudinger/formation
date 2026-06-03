#include "src/gpu_info.h"

#include <cuda_runtime.h>

#include <cstdlib>
#include <iostream>

namespace {

void check(cudaError_t error, const char *call) {
  if (error != cudaSuccess) {
    std::cerr << call << " failed: " << cudaGetErrorString(error) << "\n";
    std::exit(EXIT_FAILURE);
  }
}

} // namespace

int main() {
  int runtime_version = 0;
  int driver_version = 0;
  int device_count = 0;

  check(cudaRuntimeGetVersion(&runtime_version), "cudaRuntimeGetVersion");
  check(cudaDriverGetVersion(&driver_version), "cudaDriverGetVersion");
  check(cudaGetDeviceCount(&device_count), "cudaGetDeviceCount");

  std::cout << "CUDA runtime version: " << tp::cuda_version_string(runtime_version) << "\n";
  std::cout << "CUDA driver version : " << tp::cuda_version_string(driver_version) << "\n";
  std::cout << "CUDA device count   : " << device_count << "\n";

  for (int device = 0; device < device_count; ++device) {
    cudaDeviceProp prop{};
    check(cudaGetDeviceProperties(&prop, device), "cudaGetDeviceProperties");
    std::cout << "Device " << device << ": " << tp::compact_device_summary(prop) << "\n";
  }

  return EXIT_SUCCESS;
}
