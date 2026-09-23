#include "src/graph_replay.h"

#include <cstdlib>
#include <iostream>
#include <vector>

int main() {
  constexpr int size = 4096;
  constexpr int batches = 1000;
  std::vector<float> input(size, 0.0f);
  std::cout << "elements=" << size << " batches=" << batches
            << " kernels_per_batch=" << tp::gpu::kKernelsPerBatch << '\n';
  for (auto mode : {tp::gpu::LaunchMode::direct, tp::gpu::LaunchMode::graph}) {
    double ms = 0.0;
    const auto out = tp::gpu::repeat_add(input, batches, mode, &ms);
    for (float value : out) {
      if (value != tp::gpu::kKernelsPerBatch * batches) {
        std::cerr << "incorrect result\n";
        return EXIT_FAILURE;
      }
    }
    std::cout << (mode == tp::gpu::LaunchMode::direct ? "direct" : "graph")
              << " replay_ms=" << ms << '\n';
  }
}
