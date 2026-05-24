#include "runtime/cuda_errors.h"

#include <cuda_runtime.h>

#include <cstddef>
#include <format>
#include <iterator>
#include <print>
#include <string>
#include <string_view>

namespace {

std::string uuid_string(const cudaUUID_t &uuid) {
  std::string out;
  for (std::size_t i = 0; i < sizeof(uuid.bytes); ++i) {
    if (i != 0) {
      out.push_back(':');
    }
    std::format_to(
        std::back_inserter(out), "{:02x}",
        static_cast<unsigned int>(static_cast<unsigned char>(uuid.bytes[i])));
  }
  return out;
}

std::string cuda_version_string(int encoded_version) {
  const int major = encoded_version / 1000;
  const int minor = (encoded_version % 1000) / 10;
  return std::format("{}.{} ({})", major, minor, encoded_version);
}

std::string_view c_string_view(const char *value) {
  return std::string_view(value);
}

} // namespace

int main() {
  int runtime_version = 0;
  int driver_version = 0;
  int device_count = 0;

  CUDA_CHECK(cudaRuntimeGetVersion(&runtime_version));
  CUDA_CHECK(cudaDriverGetVersion(&driver_version));
  CUDA_CHECK(cudaGetDeviceCount(&device_count));

  std::println("CUDA runtime version: {}",
               cuda_version_string(runtime_version));
  std::println("CUDA driver version : {}", cuda_version_string(driver_version));
  std::println("CUDA device count   : {}", device_count);

  for (int device = 0; device < device_count; ++device) {
    cudaDeviceProp prop{};
    CUDA_CHECK(cudaGetDeviceProperties(&prop, device));

    std::println("\nDevice {}: {}", device, c_string_view(prop.name));
    std::println("  uuid                     : {}", uuid_string(prop.uuid));
    std::println("  compute capability       : {}.{}", prop.major, prop.minor);
    std::println("  global memory bytes      : {}", prop.totalGlobalMem);
    std::println("  multiprocessors          : {}", prop.multiProcessorCount);
    std::println("  warp size                : {}", prop.warpSize);
    std::println("  max threads per block    : {}", prop.maxThreadsPerBlock);
    std::println("  max threads per SM       : {}",
                 prop.maxThreadsPerMultiProcessor);
    std::println("  async engines            : {}", prop.asyncEngineCount);
    std::println("  unified addressing       : {}", prop.unifiedAddressing);
    std::println("  managed memory           : {}", prop.managedMemory);
  }
}
