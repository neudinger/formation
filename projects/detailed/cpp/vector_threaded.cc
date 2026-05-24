#include "cpp/vector_threaded.h"

#include <algorithm>
#include <stdexcept>
#include <thread>

namespace course {

std::vector<float> vector_add_threaded(const std::vector<float> &a,
                                       const std::vector<float> &b, float alpha,
                                       std::size_t thread_count) {
  if (a.size() != b.size()) {
    throw std::invalid_argument("vector_add_threaded: size mismatch");
  }
  if (a.empty()) {
    return {};
  }

  const std::size_t hardware_threads =
      std::max(1u, std::thread::hardware_concurrency());
  const std::size_t workers = std::max<std::size_t>(
      1, thread_count == 0 ? hardware_threads : thread_count);
  const std::size_t chunk = (a.size() + workers - 1) / workers;

  std::vector<float> out(a.size());
  std::vector<std::thread> threads;
  threads.reserve(workers);

  for (std::size_t worker = 0; worker < workers; ++worker) {
    const std::size_t begin = worker * chunk;
    const std::size_t end = std::min(a.size(), begin + chunk);
    if (begin >= end) {
      break;
    }
    threads.emplace_back([&, begin, end] {
      for (std::size_t i = begin; i < end; ++i) {
        out[i] = alpha * a[i] + b[i];
      }
    });
  }

  for (auto &thread : threads) {
    thread.join();
  }
  return out;
}

} // namespace course
