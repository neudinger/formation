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

## Block-size experiment

After completing the kernel, change only `block_size` in
`src/vector_add_device.cu`: try **128**, **256**, and **512** threads per block.
The grid calculation already rounds up to cover the input.

For each value, run:

```bash
bazel test //src:vector_add_test
bazel run //src:vector_add_benchmark
```

The test checks both launch styles with `N = 1024` and `N = 1003`.
For 1003 elements, each of these block sizes launches 1024 threads: the last
21 threads must do nothing because of the `i < n` guard.

The benchmark keeps `N = 1 << 20` fixed. Record the output for each block size:

| Threads per block | Blocks for N = 1,048,576 | Tests pass? | triple_ms | runtime_api_ms |
| ---: | ---: | --- | ---: | ---: |
| 128 | 8,192 | | | |
| 256 | 4,096 | | | |
| 512 | 2,048 | | | |

These timings measure the full CUDA call, including allocation, transfers, and
synchronization; first-use initialization can also affect them. They do not
isolate kernel performance. Revisit this experiment in Ex06 with CUDA events
and warm-up runs before drawing performance conclusions.

Discuss: why does changing the block size leave the result unchanged? Why does
using more threads per block not necessarily make execution faster?
Restore `block_size = 256` when finished.
