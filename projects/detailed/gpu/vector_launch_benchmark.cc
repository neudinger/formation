#include "gpu/vector_add.h"
#include "gpu/vector_add_device.h"
#include "runtime/cuda_errors.h"
#include "runtime/cuda_launch.h"

#include <cuda_runtime.h>

#include <algorithm>
#include <chrono>
#include <numeric>
#include <print>
#include <stdexcept>
#include <string_view>
#include <vector>

namespace {

template <typename T> class DeviceBuffer {
public:
  explicit DeviceBuffer(std::size_t count) : count_(count) {
    if (count_ > 0) {
      CUDA_CHECK(cudaMalloc(&ptr_, count_ * sizeof(T)));
    }
  }
  DeviceBuffer(const DeviceBuffer &) = delete;
  DeviceBuffer &operator=(const DeviceBuffer &) = delete;
  ~DeviceBuffer() { (void)cudaFree(ptr_); }

  [[nodiscard]] T *get() const { return ptr_; }
  [[nodiscard]] std::size_t bytes() const { return count_ * sizeof(T); }

private:
  T *ptr_ = nullptr;
  std::size_t count_ = 0;
};

struct Measurement {
  double host_enqueue_ms = 0.0;
  double enqueue_and_sync_ms = 0.0;
};

struct Summary {
  double average_us = 0.0;
  double minimum_us = 0.0;
};

template <typename Fn> double elapsed_ms(Fn &&fn) {
  const auto begin = std::chrono::steady_clock::now();
  fn();
  const auto end = std::chrono::steady_clock::now();
  return std::chrono::duration<double, std::milli>(end - begin).count();
}

Measurement measure_launches(course::gpu::LaunchApi api, int iterations,
                             const float *a, const float *b, float *out, int n,
                             cudaStream_t stream) {
  const auto begin = std::chrono::steady_clock::now();
  for (int iteration = 0; iteration < iterations; ++iteration) {
    course::gpu::launch_vector_add_kernel(a, b, out, 2.0f, n, stream, api);
  }
  const auto after_enqueue = std::chrono::steady_clock::now();
  CUDA_CHECK(cudaStreamSynchronize(stream));
  const auto after_sync = std::chrono::steady_clock::now();

  return {
      std::chrono::duration<double, std::milli>(after_enqueue - begin).count(),
      std::chrono::duration<double, std::milli>(after_sync - begin).count(),
  };
}

Summary summarize(const std::vector<double> &values_us) {
  if (values_us.empty()) {
    throw std::invalid_argument("summarize: no values");
  }
  const double total = std::accumulate(values_us.begin(), values_us.end(), 0.0);
  return {
      total / static_cast<double>(values_us.size()),
      *std::min_element(values_us.begin(), values_us.end()),
  };
}

void print_summary(std::string_view label, const std::vector<double> &values) {
  const Summary summary = summarize(values);
  std::println("{:<38} avg={:8.3f} us  min={:8.3f} us", label,
               summary.average_us, summary.minimum_us);
}

} // namespace

int main() {
  constexpr int n = 1;
  constexpr int warmup_iterations = 100;
  constexpr int measured_iterations = 10'000;
  constexpr int rounds = 8;

  CUDA_CHECK(cudaFree(nullptr));

  course::runtime::Stream stream;
  const std::vector<float> a(n, 1.0f);
  const std::vector<float> b(n, 2.0f);
  std::vector<float> out(n, 0.0f);

  DeviceBuffer<float> d_a(a.size());
  DeviceBuffer<float> d_b(b.size());
  DeviceBuffer<float> d_out(out.size());

  CUDA_CHECK(cudaMemcpyAsync(d_a.get(), a.data(), d_a.bytes(),
                             cudaMemcpyHostToDevice, stream.get()));
  CUDA_CHECK(cudaMemcpyAsync(d_b.get(), b.data(), d_b.bytes(),
                             cudaMemcpyHostToDevice, stream.get()));
  CUDA_CHECK(cudaStreamSynchronize(stream.get()));

  (void)elapsed_ms([&] {
    for (int iteration = 0; iteration < warmup_iterations; ++iteration) {
      course::gpu::launch_vector_add_kernel(
          d_a.get(), d_b.get(), d_out.get(), 2.0f, n, stream.get(),
          course::gpu::LaunchApi::kRuntimeApi);
      course::gpu::launch_vector_add_kernel(
          d_a.get(), d_b.get(), d_out.get(), 2.0f, n, stream.get(),
          course::gpu::LaunchApi::kTripleChevron);
    }
    CUDA_CHECK(cudaStreamSynchronize(stream.get()));
  });

  std::vector<double> runtime_host_us;
  std::vector<double> runtime_total_us;
  std::vector<double> chevron_host_us;
  std::vector<double> chevron_total_us;

  auto record = [&](course::gpu::LaunchApi api) {
    const Measurement measurement =
        measure_launches(api, measured_iterations, d_a.get(), d_b.get(),
                         d_out.get(), n, stream.get());
    std::vector<double> &host =
        api == course::gpu::LaunchApi::kRuntimeApi ? runtime_host_us
                                                   : chevron_host_us;
    std::vector<double> &total =
        api == course::gpu::LaunchApi::kRuntimeApi ? runtime_total_us
                                                   : chevron_total_us;
    host.push_back(measurement.host_enqueue_ms * 1000.0 /
                   static_cast<double>(measured_iterations));
    total.push_back(measurement.enqueue_and_sync_ms * 1000.0 /
                    static_cast<double>(measured_iterations));
  };

  for (int round = 0; round < rounds; ++round) {
    if (round % 2 == 0) {
      record(course::gpu::LaunchApi::kRuntimeApi);
      record(course::gpu::LaunchApi::kTripleChevron);
    } else {
      record(course::gpu::LaunchApi::kTripleChevron);
      record(course::gpu::LaunchApi::kRuntimeApi);
    }
  }

  CUDA_CHECK(cudaMemcpyAsync(out.data(), d_out.get(), d_out.bytes(),
                             cudaMemcpyDeviceToHost, stream.get()));
  CUDA_CHECK(cudaStreamSynchronize(stream.get()));

  std::println("CUDA vector launch benchmark");
  std::println("elements per launch                  : {}", n);
  std::println("warmup launches per API              : {}", warmup_iterations);
  std::println("measured launches per round          : {}",
               measured_iterations);
  std::println("rounds per API                       : {}", rounds);
  std::println("timed region                         : launches only");
  std::println("excluded                             : cudaMalloc/cudaFree/cudaMemcpy");
  std::println("order                                : alternates each round");
  print_summary("cudaLaunchKernel host enqueue", runtime_host_us);
  print_summary("<<< >>> host enqueue", chevron_host_us);
  print_summary("cudaLaunchKernel enqueue + sync", runtime_total_us);
  print_summary("<<< >>> enqueue + sync", chevron_total_us);
  std::println("sample                               : {}", out[0]);
}
