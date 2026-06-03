#include "src/gpu_info.h"

#include <string>

namespace tp {

std::string cuda_version_string(int encoded_version) {
  // TODO: Decode CUDA's integer version format as major.minor (encoded).
  (void)encoded_version;
  return "TODO";
}

std::string compact_device_summary(const cudaDeviceProp &prop) {
  // TODO: Include the device name, compute capability, SM count, and warp size.
  (void)prop;
  return "TODO";
}

} // namespace tp
