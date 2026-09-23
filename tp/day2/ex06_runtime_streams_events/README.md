# Exercise 06: Runtime Robustness, Streams, and Events

## Goal

Turn the CUDA wrapper into a robust asynchronous baseline with explicit error messages, streams, and event timing.

## What to finish

- Implement `cuda_error_message` and `cuda_check`.
- Create and destroy explicit streams.
- Run the supplied event-timing example and explain total time versus kernel time.

## Commands

```bash
bazel build //...
bazel test //src:runtime_test
bazel run //src:vector_add_benchmark
```

The starter project should build before you edit it. Its tests are expected to
fail until the TODO implementations are complete.

The complete answer is mirrored at `tp/solutions/day2/ex06_runtime_streams_events`.
## Notes

- CUDA tests and benchmarks require an NVIDIA driver and CUDA-capable GPU.

## Reading the timing output

The benchmark warms up both launch styles before measuring:

- `cpu_ms`: serial CPU computation, measured with `std::chrono`.
- `*_total_ms`: the full CUDA wrapper call, including allocation, transfers,
  kernel execution, synchronization, and cleanup, measured with `std::chrono`.
- `*_kernel_ms`: CUDA event time around the kernel in its stream, excluding
  allocation and host/device transfers. The stream is synchronized before
  calling `cudaEventElapsedTime`.

Compare these values to see the cost outside the kernel. Each is a single sample;
small differences between launch styles are not evidence that one is faster.
Reference: CUDA Programming Guide, section 2.5.3 (CUDA Events).
