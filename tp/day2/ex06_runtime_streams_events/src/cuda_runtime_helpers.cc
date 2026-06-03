#include "src/cuda_runtime_helpers.h"

namespace tp::runtime {

std::string cuda_error_message(cudaError_t error, const char *call,
                               const char *file, int line) {
  // TODO: Include the CUDA call, file, line, string error, and numeric code.
  (void)error;
  (void)call;
  (void)file;
  (void)line;
  return "TODO";
}

void cuda_check(cudaError_t error, const char *call, const char *file, int line) {
  // TODO: Throw std::runtime_error when error is not cudaSuccess.
  (void)error;
  (void)call;
  (void)file;
  (void)line;
}

Stream::Stream() {
  // TODO: Create an explicit CUDA stream.
}
Stream::~Stream() {
  // TODO: Destroy the stream.
}

} // namespace tp::runtime
