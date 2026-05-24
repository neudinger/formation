#include "linear_algebra/matmul_cublas.h"

#include <stdexcept>

namespace course::linear_algebra {

std::vector<float> matmul_cublas_unavailable() {
  throw std::runtime_error(
      "cuBLAS is intentionally not linked in this minimal hermetic scaffold; "
      "add the cublas CUDA redistributable component for the production "
      "baseline lesson.");
}

} // namespace course::linear_algebra
