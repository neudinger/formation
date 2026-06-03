# Stretch 12: cuBLAS GEMM Comparison

## Goal

Add an optional cuBLAS-backed GEMM path and compare it with the custom CPU/basic/tiled implementations.

## What to finish

- Name and expose the cuBLAS strategy.
- Load cuBLAS dynamically when the runtime image provides it.
- Report `unavailable` clearly when cuBLAS is not present.

## Commands

```bash
bazel build //...
bazel test //src:matmul_cublas_test
bazel run //src:matmul_benchmark
```

This is the complete reference solution. It should build and pass its tests.

Starter exercise: `tp/stretch/ex12_cublas_gemm_comparison`.
## Notes

- CUDA tests and benchmarks require an NVIDIA driver and CUDA-capable GPU.

