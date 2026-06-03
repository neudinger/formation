#include "src/vector_threaded.h"

#include <algorithm>
#include <stdexcept>
#include <thread>

namespace tp {

std::vector<float> vector_add_threaded(const std::vector<float> &a,
                                       const std::vector<float> &b,
                                       float alpha, std::size_t worker_count) {
  if (a.size() != b.size()) {
    throw std::invalid_argument("vector_add_threaded: size mismatch");
  }
  if (a.empty()) {
    return {};
  }
  const std::size_t workers = std::max<std::size_t>(1, std::min(worker_count, a.size()));
  const std::size_t chunk = (a.size() + workers - 1) / workers;
  std::vector<float> out(a.size());
  std::vector<std::thread> threads;
  for (std::size_t worker = 0; worker < workers; ++worker) {
    const std::size_t begin = worker * chunk;
    const std::size_t end = std::min(begin + chunk, a.size());
    if (begin >= end) {
      continue;
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

} // namespace tp
