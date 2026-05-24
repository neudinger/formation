#include "cpp/vector_cpu.h"
#include "cpp/vector_threaded.h"
#include "gpu/vector_add.h"

#include <chrono>
#include <print>
#include <vector>

namespace {

template <typename Fn> double time_ms(Fn &&fn) {
  const auto start = std::chrono::steady_clock::now();
  fn();
  const auto stop = std::chrono::steady_clock::now();
  return std::chrono::duration<double, std::milli>(stop - start).count();
}

} // namespace

int main() {
  constexpr std::size_t size = 1 << 24;
  const std::vector<float> a(size, 1.0f);
  const std::vector<float> b(size, 2.0f);

  std::vector<float> cpu;
  std::vector<float> threaded;
  std::vector<float> cuda_runtime;
  std::vector<float> cuda_chevron;

  const double cpu_ms =
      time_ms([&] { cpu = course::vector_add_cpu(a, b, 2.0f); });
  const double threaded_ms =
      time_ms([&] { threaded = course::vector_add_threaded(a, b, 2.0f); });
  const double cuda_runtime_ms = time_ms([&] {
    cuda_runtime = course::gpu::vector_add_cuda(
        a, b, 2.0f, course::gpu::LaunchApi::kRuntimeApi);
  });
  const double cuda_chevron_ms = time_ms([&] {
    cuda_chevron = course::gpu::vector_add_cuda(
        a, b, 2.0f, course::gpu::LaunchApi::kTripleChevron);
  });

  std::println("elements                : {}", size);
  std::println("cpu serial              : {:.3f} ms", cpu_ms);
  std::println("cpu threaded            : {:.3f} ms", threaded_ms);
  std::println("cuda cudaLaunchKernel   : {:.3f} ms", cuda_runtime_ms);
  std::println("cuda <<< >>>            : {:.3f} ms", cuda_chevron_ms);
  std::println("sample                  : cpu={} cuda={}", cpu[123],
               cuda_runtime[123]);
}
