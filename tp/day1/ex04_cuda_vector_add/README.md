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

The starter project should build before you edit it. Its tests are expected to
fail until the TODO implementations are complete.

The complete answer is mirrored at `tp/solutions/day1/ex04_cuda_vector_add`.
## Notes

- CUDA tests and benchmarks require an NVIDIA driver and CUDA-capable GPU.

