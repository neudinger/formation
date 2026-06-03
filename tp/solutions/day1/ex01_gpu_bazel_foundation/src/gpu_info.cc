#include "src/gpu_info.h"

#include <string>

namespace tp {

std::string cuda_version_string(int encoded_version) {
  const int major = encoded_version / 1000;
  const int minor = (encoded_version % 1000) / 10;
  return std::to_string(major) + "." + std::to_string(minor) + " (" +
         std::to_string(encoded_version) + ")";
}

std::string compact_device_summary(const cudaDeviceProp &prop) {
  return std::string(prop.name) + " | sm_" + std::to_string(prop.major) +
         std::to_string(prop.minor) + " | SMs=" +
         std::to_string(prop.multiProcessorCount) + " | warp=" +
         std::to_string(prop.warpSize);
}

} // namespace tp
