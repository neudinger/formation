#include "src/vector_cpu.h"
#include "src/vector_threaded.h"

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
  std::vector<float> a(size, 2.0f);
  std::vector<float> b(size, 3.0f);
  std::vector<float> out;
  const double serial = time_ms([&] { out = tp::vector_add_cpu(a, b, 4.0f); });
  const double threaded = time_ms([&] { out = tp::vector_add_threaded(a, b, 4.0f, 8); });
  std::cout << "serial_ms=" << serial << " threaded_ms=" << threaded
            << " sample=" << out[size / 2] << "\n";
}
