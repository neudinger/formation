# Exercise 06: Runtime Robustness, Streams, and Events

## Goal

Turn the CUDA wrapper into a robust asynchronous baseline with explicit error messages, streams, and event timing.

## What to finish

- Implement `cuda_error_message` and `cuda_check`.
- Create and destroy explicit streams.
- Use the benchmark to distinguish CPU timing from CUDA timing.

## Commands

```bash
bazel build //...
bazel test //src:runtime_test
bazel run //src:vector_add_benchmark
```

This is the complete reference solution. It should build and pass its tests.

Starter exercise: `tp/day2/ex06_runtime_streams_events`.
## Notes

- CUDA tests and benchmarks require an NVIDIA driver and CUDA-capable GPU.

