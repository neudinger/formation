#include "src/graph_replay.h"

#include <cmath>
#include <cstdlib>
#include <iostream>
#include <stdexcept>
#include <vector>

void require(bool condition, const char *message) {
  if (!condition) {
    std::cerr << "test failure: " << message << '\n';
    std::exit(EXIT_FAILURE);
  }
}

int main() {
  for (auto mode : {tp::gpu::LaunchMode::direct, tp::gpu::LaunchMode::graph}) {
    for (int size : {0, 1, 256, 1003}) {
      std::vector<float> input(size);
      for (int i = 0; i < size; ++i) input[i] = (i % 13 - 6) * 0.25f;
      for (int batches : {0, 1, 3}) {
        double ms = -1.0;
        const auto out = tp::gpu::repeat_add(input, batches, mode, &ms);
        require(out.size() == input.size(), "output size");
        for (int i = 0; i < size; ++i) {
          require(out[i] == input[i] + tp::gpu::kKernelsPerBatch * batches,
                  "wrong number of additions");
        }
        require(std::isfinite(ms) && ms >= 0.0, "invalid timing");
      }
    }
    bool rejected = false;
    try {
      (void)tp::gpu::repeat_add({1.0f}, -1, mode);
    } catch (const std::invalid_argument &) {
      rejected = true;
    }
    require(rejected, "negative batch count");
  }
}
