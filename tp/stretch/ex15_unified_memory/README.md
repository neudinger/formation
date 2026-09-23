# Exercise 15: Unified Memory

Reuse Ex04's vector addition: `out[i] = alpha * a[i] + b[i]`.
Both memory paths use the same kernel with 256 threads per block.

## Three TODOs

In `src/vector_add.cu`, finish the managed-memory path:

1. Allocate with `cudaMallocManaged` instead of `cudaMalloc`.
2. Initialize the managed input buffers on the CPU with `std::copy`.
3. After `cudaDeviceSynchronize`, read the managed output with `std::copy`.

The explicit-copy path and kernel are supplied. Both paths release memory with
`cudaFree`. Always wait for the GPU before reading its result on the CPU.

## Run

This directory is an independent Bazel project.

```bash
bazel build //...
bazel test //src:vector_add_test
bazel run //src:vector_add_benchmark
```

The starter builds. Tests stop at the TODOs until completed.
Solution: `tp/solutions/stretch/ex15_unified_memory`.

## Compare

The benchmark checks both results and reports mean **total call time** over three
runs after a warm-up. This includes allocation, input preparation, GPU work, CPU
readback, and cleanup. Each call uses fresh buffers.

| Path | How data becomes accessible | Total time (ms) |
| --- | --- | ---: |
| Explicit copies | `cudaMemcpy` to/from device buffers | measure locally |
| Managed memory | CPU/GPU use the same buffers; driver manages movement | measure locally |

Managed memory simplifies access; data movement still has a cost. Compare a small
and a large input. Which path is simpler, and which is faster on your machine?

Requires a CUDA GPU with managed-memory support. See the
[CUDA Programming Guide, §2.6](../../../cuda-programming-guide.pdf).
