#pragma once

#include "runtime/cuda_errors.h"

#include <cuda_runtime.h>

namespace course::runtime {

class Stream {
public:
  Stream() { CUDA_CHECK(cudaStreamCreate(&stream_)); }
  Stream(const Stream &) = delete;
  Stream &operator=(const Stream &) = delete;
  Stream(Stream &&) = delete;
  Stream &operator=(Stream &&) = delete;
  ~Stream() { (void)cudaStreamDestroy(stream_); }

  [[nodiscard]] cudaStream_t get() const { return stream_; }

private:
  cudaStream_t stream_{};
};

class Event {
public:
  Event() { CUDA_CHECK(cudaEventCreate(&event_)); }
  Event(const Event &) = delete;
  Event &operator=(const Event &) = delete;
  Event(Event &&) = delete;
  Event &operator=(Event &&) = delete;
  ~Event() { (void)cudaEventDestroy(event_); }

  [[nodiscard]] cudaEvent_t get() const { return event_; }

private:
  cudaEvent_t event_{};
};

inline float elapsed_ms(const Event &begin, const Event &end) {
  float ms = 0.0f;
  CUDA_CHECK(cudaEventElapsedTime(&ms, begin.get(), end.get()));
  return ms;
}

} // namespace course::runtime
