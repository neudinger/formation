#pragma once

#include <cuda_runtime.h>

#include <source_location>
#include <stdexcept>
#include <string>
#include <string_view>

namespace course::runtime {

class CudaError : public std::runtime_error {
public:
  explicit CudaError(
      cudaError_t error, std::string_view call,
      const std::source_location &location = std::source_location::current())
      : std::runtime_error(message(error, call, location)), error_(error) {}

  [[nodiscard]] cudaError_t error() const { return error_; }

private:
  static std::string message(cudaError_t error, std::string_view call,
                             const std::source_location &location) {
    return std::string(call) + " failed at " + location.file_name() + ":" +
           std::to_string(location.line()) + ": " + cudaGetErrorString(error);
  }

  cudaError_t error_;
};

inline void cuda_check(
    cudaError_t error, std::string_view call,
    const std::source_location &location = std::source_location::current()) {
  if (error != cudaSuccess) {
    throw CudaError(error, call, location);
  }
}

#define CUDA_CHECK(call)                                                       \
  ::course::runtime::cuda_check((call), #call, std::source_location::current())

} // namespace course::runtime
