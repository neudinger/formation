#include "src/graph_replay.h"

#include <cuda_runtime.h>

#include <chrono>
#include <limits>
#include <stdexcept>
#include <string>

namespace tp::gpu {
namespace {

void check(cudaError_t error, const char *call) {
  if (error != cudaSuccess) {
    throw std::runtime_error(std::string(call) + ": " + cudaGetErrorString(error));
  }
}
#define CUDA_CHECK(call) check((call), #call)

// Supplied cleanup keeps buffers alive until the stream finishes.
struct Resources {
  float *data = nullptr;
  cudaStream_t stream{};
  cudaGraph_t graph{};
  cudaGraphExec_t executable{};
  Resources() = default;
  Resources(const Resources &) = delete;
  Resources &operator=(const Resources &) = delete;
  ~Resources() {
    if (stream) (void)cudaStreamSynchronize(stream);
    if (executable) (void)cudaGraphExecDestroy(executable);
    if (graph) (void)cudaGraphDestroy(graph);
    (void)cudaFree(data);
    if (stream) (void)cudaStreamDestroy(stream);
  }
};

__global__ void add_one(float *values, int size) {
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < size) {
    values[i] += 1.0f;
  }
}

} // namespace

std::vector<float> repeat_add(const std::vector<float> &input, int batches,
                              LaunchMode mode, double *replay_ms) {
  if (replay_ms) *replay_ms = 0.0;
  if (batches < 0 || input.size() > static_cast<std::size_t>(std::numeric_limits<int>::max())) {
    throw std::invalid_argument("repeat_add: invalid size or batch count");
  }
  if (input.empty() || batches == 0) return input;

  Resources work;
  const auto bytes = input.size() * sizeof(float);
  const int size = static_cast<int>(input.size());
  const int blocks = size / 256 + (size % 256 != 0);
  CUDA_CHECK(cudaStreamCreateWithFlags(&work.stream, cudaStreamNonBlocking));
  CUDA_CHECK(cudaMalloc(&work.data, bytes));
  CUDA_CHECK(cudaMemcpyAsync(work.data, input.data(), bytes, cudaMemcpyHostToDevice, work.stream));
  CUDA_CHECK(cudaStreamSynchronize(work.stream));

  // The same batch is used for direct launches and graph capture.
  auto launch_kernels = [&] {
    for (int i = 0; i < kKernelsPerBatch; ++i) {
      add_one<<<blocks, 256, 0, work.stream>>>(work.data, size);
      CUDA_CHECK(cudaGetLastError());
    }
  };
  if (mode == LaunchMode::graph) {
    // TODO 1: Begin capture, call launch_kernels(), then end capture.
    // TODO 2: Instantiate work.graph into work.executable.
    throw std::logic_error("TODO: capture and instantiate the graph");
  }
  auto launch_batch = [&] {
    if (mode == LaunchMode::graph) {
      // TODO 3: Replay work.executable on work.stream.
      throw std::logic_error("TODO: graph replay");
    } else {
      launch_kernels();
    }
  };

  launch_batch(); // Warm up, including the graph's first upload.
  CUDA_CHECK(cudaStreamSynchronize(work.stream));
  // Restore the input so the warm-up does not change the expected answer.
  CUDA_CHECK(cudaMemcpyAsync(work.data, input.data(), bytes, cudaMemcpyHostToDevice, work.stream));
  CUDA_CHECK(cudaStreamSynchronize(work.stream));
  const auto begin = std::chrono::steady_clock::now();
  for (int i = 0; i < batches; ++i) {
    launch_batch();
  }
  CUDA_CHECK(cudaStreamSynchronize(work.stream));
  const auto end = std::chrono::steady_clock::now();
  if (replay_ms) {
    *replay_ms = std::chrono::duration<double, std::milli>(end - begin).count();
  }
  std::vector<float> result(input.size());
  CUDA_CHECK(cudaMemcpy(result.data(), work.data, bytes, cudaMemcpyDeviceToHost));
  return result;
}

} // namespace tp::gpu
