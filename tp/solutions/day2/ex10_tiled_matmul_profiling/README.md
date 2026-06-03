# Exercise 10: Tiled MatMul and Profiling Notes

## Goal

Use shared memory tiling to improve MatMul and record benchmark/profiling observations.

## What to finish

- Load A and B tiles into `__shared__` memory.
- Use `__syncthreads()` around tile reuse.
- Fill in the benchmark notes worksheet.

## Commands

```bash
bazel build //...
bazel test //src:matmul_test
bazel run //src:matmul_benchmark
```

This is the complete reference solution. It should build and pass its tests.

Starter exercise: `tp/day2/ex10_tiled_matmul_profiling`.
## Notes

- CUDA tests and benchmarks require an NVIDIA driver and CUDA-capable GPU.

