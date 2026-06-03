#include "src/matmul_cpu.h"

namespace tp::la {

std::vector<float> matmul_cpu(const std::vector<float> &a,
                              const std::vector<float> &b,
                              const std::vector<float> &c, int m, int n, int k,
                              float alpha, float beta) {
  // TODO: Validate dimensions and compute alpha * A * B + beta * C.
  (void)a;
  (void)b;
  (void)c;
  (void)k;
  (void)alpha;
  (void)beta;
  return std::vector<float>(static_cast<std::size_t>(m * n), 0.0f);
}

} // namespace tp::la
