#include "linear_algebra/matmul_basic.h"
#include "linear_algebra/matmul_cpu.h"
#include "linear_algebra/matmul_tiled.h"

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
  constexpr int m = 256;
  constexpr int n = 256;
  constexpr int k = 256;
  std::vector<float> a(m * k, 1.0f);
  std::vector<float> b(k * n, 2.0f);
  std::vector<float> c(m * n, 0.0f);
  std::vector<float> out;

  const double cpu_ms = time_ms(
      [&] { out = course::linear_algebra::matmul_cpu(a, b, c, m, n, k); });
  const double basic_ms = time_ms([&] {
    out = course::linear_algebra::matmul_basic_cuda(a, b, c, m, n, k);
  });
  const double tiled_ms = time_ms([&] {
    out = course::linear_algebra::matmul_tiled_cuda(a, b, c, m, n, k);
  });

  std::println("shape      : M={} N={} K={}", m, n, k);
  std::println("cpu        : {:.3f} ms", cpu_ms);
  std::println("cuda basic : {:.3f} ms", basic_ms);
  std::println("cuda tiled : {:.3f} ms", tiled_ms);
  std::println("sample     : {}", out[0]);
}
