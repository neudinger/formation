#include "src/cuda_runtime_helpers.h"

#include <sstream>
#include <stdexcept>

namespace tp::runtime {

void cuda_check(cudaError_t error, const char *call, const char *file, int line) {
  if (error != cudaSuccess) {
    std::ostringstream message;
    message << call << " failed at " << file << ':' << line << ": "
            << cudaGetErrorString(error) << " (" << static_cast<int>(error) << ')';
    throw std::runtime_error(message.str());
  }
}

Stream::Stream() { TP_CUDA_CHECK(cudaStreamCreate(&stream_)); }
Stream::~Stream() { (void)cudaStreamDestroy(stream_); }

} // namespace tp::runtime
