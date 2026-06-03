#include "src/vector_cpu.h"

#include <stdexcept>

namespace tp {

std::vector<float> vector_add_cpu(const std::vector<float> &a,
                                  const std::vector<float> &b, float alpha) {
  if (a.size() != b.size()) {
    throw std::invalid_argument("vector_add_cpu: size mismatch");
  }
  std::vector<float> out(a.size());
  for (std::size_t i = 0; i < a.size(); ++i) {
    out[i] = alpha * a[i] + b[i];
  }
  return out;
}

} // namespace tp
