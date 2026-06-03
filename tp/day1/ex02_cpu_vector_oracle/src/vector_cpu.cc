#include "src/vector_cpu.h"

namespace tp {

std::vector<float> vector_add_cpu(const std::vector<float> &a,
                                  const std::vector<float> &b, float alpha) {
  // TODO: Validate sizes and compute out[i] = alpha * a[i] + b[i].
  (void)b;
  (void)alpha;
  return std::vector<float>(a.size(), 0.0f);
}

} // namespace tp
