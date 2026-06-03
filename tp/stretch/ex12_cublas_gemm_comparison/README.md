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

The starter project should build before you edit it. Its tests are expected to
fail until the TODO implementations are complete.

The complete answer is mirrored at `tp/solutions/stretch/ex12_cublas_gemm_comparison`.
## Notes

- CUDA tests and benchmarks require an NVIDIA driver and CUDA-capable GPU.

