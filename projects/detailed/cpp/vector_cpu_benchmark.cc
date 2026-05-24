#include "cpp/vector_cpu.h"
#include "cpp/vector_threaded.h"

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

std::vector<float> filled(std::size_t size, float value) {
  return std::vector<float>(size, value);
}

} // namespace

int main() {
  constexpr std::size_t size = 1 << 24;
  const auto a = filled(size, 1.0f);
  const auto b = filled(size, 2.0f);

  std::vector<float> serial;
  std::vector<float> threaded;

  const double serial_ms =
      time_ms([&] { serial = course::vector_add_cpu(a, b, 2.0f); });
  const double threaded_ms =
      time_ms([&] { threaded = course::vector_add_threaded(a, b, 2.0f); });

  std::println("elements: {}", size);
  std::println("cpu serial  : {:.3f} ms", serial_ms);
  std::println("cpu threaded: {:.3f} ms", threaded_ms);
  std::println("sample      : serial={} threaded={}", serial[123],
               threaded[123]);
  return 0;
}
