#include "src/matmul_basic.h"
#include "src/matmul_cpu.h"
#include "src/matmul_tiled.h"

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
  constexpr int m = 4096; // 1024
  constexpr int n = m;
  constexpr int k = m;
  std::vector<float> a(static_cast<std::size_t>(m * k), 1.0f);
  std::vector<float> b(static_cast<std::size_t>(k * n), 2.0f);
  std::vector<float> c(static_cast<std::size_t>(m * n), 0.0f);
  std::vector<float> out;
  // std::cout << "cpu_ms=" << time_ms([&] { out = tp::la::matmul_cpu(a, b, c, m, n, k, 1.0f, 0.0f); }) << "\n";
  std::cout << "basic_cuda_ms=" << time_ms([&] { out = tp::la::matmul_basic_cuda(a, b, c, m, n, k, 1.0f, 0.0f); }) << "\n";
  std::cout << "basic_cuda_ms=" << time_ms([&] { out = tp::la::matmul_basic_cuda(a, b, c, m, n, k, 1.0f, 0.0f); }) << "\n";
  std::cout << "tiled_cuda_ms=" << time_ms([&] { out = tp::la::matmul_tiled_cuda(a, b, c, m, n, k, 1.0f, 0.0f); }) << "\n";
  std::cout << "sample=" << out[0] << "\n";
}
