# Exercise 04: First CUDA Vector Add

## Goal

Move the CPU vector operation to the GPU with explicit device memory and a CUDA kernel.

## What to finish

- Finish the kernel index calculation and bounds check.
- Launch enough blocks for the full vector.
- Copy inputs to the device and the result back to the host.

## Commands

```bash
bazel build //...
bazel test //src:vector_add_test
bazel run //src:vector_add_benchmark
```

This is the complete reference solution. It should build and pass its tests.

Starter exercise: `tp/day1/ex04_cuda_vector_add`.
## Notes

- CUDA tests and benchmarks require an NVIDIA driver and CUDA-capable GPU.

