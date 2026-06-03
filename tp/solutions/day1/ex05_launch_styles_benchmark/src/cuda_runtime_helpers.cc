#include "src/cuda_runtime_helpers.h"

#include <sstream>
#include <stdexcept>

namespace tp::runtime {

std::string cuda_error_message(cudaError_t error, const char *call,
                               const char *file, int line) {
  std::ostringstream out;
  out << call << " failed at " << file << ':' << line << ": "
      << cudaGetErrorString(error) << " (" << static_cast<int>(error) << ')';
  return out.str();
}

void cuda_check(cudaError_t error, const char *call, const char *file, int line) {
  if (error != cudaSuccess) {
    throw std::runtime_error(cuda_error_message(error, call, file, line));
  }
}

Stream::Stream() { TP_CUDA_CHECK(cudaStreamCreate(&stream_)); }
Stream::~Stream() { (void)cudaStreamDestroy(stream_); }

} // namespace tp::runtime
