# Exercise 09: MatMul CPU Oracle and Basic CUDA

## Goal

Build the MatMul ladder from a CPU oracle to a naive CUDA kernel with one thread per output element.

## What to finish

- Implement row-major CPU MatMul.
- Implement the basic CUDA kernel.
- Compare CUDA output against the CPU oracle.

## Commands

```bash
bazel build //...
bazel test //src:matmul_test
bazel run //src:matmul_benchmark
```

This is the complete reference solution. It should build and pass its tests.

Starter exercise: `tp/day2/ex09_matmul_cpu_basic_cuda`.
## Notes

- CUDA tests and benchmarks require an NVIDIA driver and CUDA-capable GPU.

