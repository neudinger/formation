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
  const double triple_ms = time_ms([&] { out = tp::gpu::vector_add_cuda(a, b, 3.0f, tp::gpu::LaunchApi::kTripleChevron); });
  const double api_ms = time_ms([&] { out = tp::gpu::vector_add_cuda(a, b, 3.0f, tp::gpu::LaunchApi::kRuntimeApi); });
  std::cout << "cpu_ms=" << cpu_ms << " triple_ms=" << triple_ms
            << " runtime_api_ms=" << api_ms << " sample=" << out[size / 2] << "\n";
}
