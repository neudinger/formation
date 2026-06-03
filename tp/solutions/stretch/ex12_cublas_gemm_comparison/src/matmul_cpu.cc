#include "src/matmul_cpu.h"

#include <stdexcept>

namespace tp::la {

std::vector<float> matmul_cpu(const std::vector<float> &a,
                              const std::vector<float> &b,
                              const std::vector<float> &c, int m, int n, int k,
                              float alpha, float beta) {
  if (m < 0 || n < 0 || k < 0) {
    throw std::invalid_argument("matmul_cpu: negative dimension");
  }
  if (a.size() != static_cast<std::size_t>(m * k) ||
      b.size() != static_cast<std::size_t>(k * n) ||
      c.size() != static_cast<std::size_t>(m * n)) {
    throw std::invalid_argument("matmul_cpu: size mismatch");
  }
  std::vector<float> out(static_cast<std::size_t>(m * n));
  for (int row = 0; row < m; ++row) {
    for (int col = 0; col < n; ++col) {
      float acc = 0.0f;
      for (int inner = 0; inner < k; ++inner) {
        acc += a[row * k + inner] * b[inner * n + col];
      }
      const int index = row * n + col;
      out[index] = alpha * acc + beta * c[index];
    }
  }
  return out;
}

} // namespace tp::la
