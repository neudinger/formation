#include "src/vector_threaded.h"

namespace tp {

std::vector<float> vector_add_threaded(const std::vector<float> &a,
                                       const std::vector<float> &b,
                                       float alpha, std::size_t worker_count) {
  // TODO: Split the vectors into chunks, launch worker threads, and join them.
  (void)b;
  (void)alpha;
  (void)worker_count;
  return std::vector<float>(a.size(), 0.0f);
}

} // namespace tp
