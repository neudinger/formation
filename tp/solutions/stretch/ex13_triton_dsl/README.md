# Stretch 13: Triton DSL Kernels

## Goal

Use Triton's Python DSL to implement three GPU kernels that mirror the CUDA
course ladder: vector add, SAXPY, and block MatMul.

## What is implemented

- Vector add with one Triton program per vector block.
- SAXPY with the same block/mask structure.
- Block MatMul with one output tile per Triton program and masked edge tiles.
- Torch reference checks and benchmark targets.

## Commands

```bash
bazel build //...
bazel test //python:triton_kernels_test
bazel run //python:benchmark
```

## Notes

- This stretch lab is Linux-only because official Triton wheels are published
  for Linux CPython. Use a machine with an NVIDIA driver and CUDA-capable GPU.
- The first benchmark run includes Triton JIT compilation and cache setup. Run
  the benchmark twice before interpreting timings.
- The Python dependencies are pinned through Bazel/rules_python:
  `torch==2.12.0` and `triton==3.7.0`.
