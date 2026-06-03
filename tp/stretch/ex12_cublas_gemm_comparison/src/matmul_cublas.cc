#include "src/matmul_cublas.h"

namespace tp::la {

std::string_view cublas_strategy_name() {
  // TODO: Name the production-library strategy used by this exercise.
  return "TODO";
}

std::optional<std::vector<float>> try_matmul_cublas(const std::vector<float> &a,
                                                    const std::vector<float> &b,
                                                    const std::vector<float> &c,
                                                    int m, int n, int k,
                                                    float alpha, float beta) {
  // TODO: Dynamically call cuBLAS when available and return std::nullopt otherwise.
  (void)a;
  (void)b;
  (void)c;
  (void)m;
  (void)n;
  (void)k;
  (void)alpha;
  (void)beta;
  return std::nullopt;
}

} // namespace tp::la
