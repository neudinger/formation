# Stretch 13: Triton DSL Kernels

## Goal

Use Triton's Python DSL to implement three GPU kernels that mirror the CUDA
course ladder: vector add, SAXPY, and block MatMul.

## What to finish

- Use `tl.program_id`, `tl.arange`, offsets, and masks for vector add.
- Extend the same block pattern to SAXPY: `out[i] = alpha * a[i] + b[i]`.
- Compute one output tile per Triton program for MatMul.
- Compare Triton output with Torch reference output.

## Commands

```bash
bazel build //...
bazel test //python:triton_kernels_test
bazel run //python:benchmark
```

The starter project should build before you edit it. Its tests are expected to
fail until the TODO implementations are complete.

The complete answer is mirrored at `tp/solutions/stretch/ex13_triton_dsl`.

## Notes

- This stretch lab is Linux-only because official Triton wheels are published
  for Linux CPython. Use a machine with an NVIDIA driver and CUDA-capable GPU.
- The first benchmark run includes Triton JIT compilation and cache setup. Run
  the benchmark twice before interpreting timings.
- The Python dependencies are pinned through Bazel/rules_python:
  `torch==2.12.0` and `triton==3.7.0`.
