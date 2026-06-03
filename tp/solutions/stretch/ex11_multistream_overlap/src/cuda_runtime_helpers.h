#pragma once

#include <cuda_runtime.h>

#include <string>

namespace tp::runtime {

std::string cuda_error_message(cudaError_t error, const char *call,
                               const char *file, int line);
void cuda_check(cudaError_t error, const char *call, const char *file, int line);

#define TP_CUDA_CHECK(call) ::tp::runtime::cuda_check((call), #call, __FILE__, __LINE__)

class Stream {
public:
  Stream();
  Stream(const Stream &) = delete;
  Stream &operator=(const Stream &) = delete;
  ~Stream();

  [[nodiscard]] cudaStream_t get() const { return stream_; }

private:
  cudaStream_t stream_{};
};

} // namespace tp::runtime
