#include "src/vector_add.h"
#include "src/vector_cpu.h"

#include <chrono>
#include <iostream>
#include <vector>

namespace {

template <typename Fn> double time_ms(Fn fn) {
  const auto begin = std::chrono::steady_clock::now();
  fn();
  const auto end = std::chrono::steady_clock::now();
  return std::chrono::duration<double, std::milli>(end - begin).count();
}

} // namespace

int main() {
  constexpr std::size_t size = 1 << 20;
  std::vector<float> a(size, 1.0f);
  std::vector<float> b(size, 2.0f);
  std::vector<float> out;
  const double cpu_ms = time_ms([&] { out = tp::vector_add_cpu(a, b, 3.0f); });
  // Warm up both paths before measuring (runtime initialization / PTX JIT).
  out = tp::gpu::vector_add_cuda(a, b, 3.0f, tp::gpu::LaunchApi::kTripleChevron);
  out = tp::gpu::vector_add_cuda(a, b, 3.0f, tp::gpu::LaunchApi::kRuntimeApi);
  float triple_kernel_ms = 0.0f;
  float api_kernel_ms = 0.0f;
  const double triple_ms = time_ms([&] {
    out = tp::gpu::vector_add_cuda(a, b, 3.0f,
                                 tp::gpu::LaunchApi::kTripleChevron, &triple_kernel_ms);
  });
  const double api_ms = time_ms([&] {
    out = tp::gpu::vector_add_cuda(a, b, 3.0f,
                                 tp::gpu::LaunchApi::kRuntimeApi, &api_kernel_ms);
  });
  std::cout << "cpu_ms=" << cpu_ms << "\n"
            << "triple_total_ms=" << triple_ms
            << " triple_kernel_ms=" << triple_kernel_ms << "\n"
            << "runtime_api_total_ms=" << api_ms
            << " runtime_api_kernel_ms=" << api_kernel_ms << "\n"
            << "sample=" << out[size / 2] << "\n";
}
