#pragma once

#include <cuda_runtime.h>

#include <string>

namespace tp {

std::string cuda_version_string(int encoded_version);
std::string compact_device_summary(const cudaDeviceProp &prop);

} // namespace tp
