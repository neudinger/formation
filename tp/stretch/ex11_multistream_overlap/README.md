# Stretch 11: Multi-Stream Vector Overlap

## Goal

Overlap vector transfers and computation using pinned host memory and one CUDA stream per chunk.

## What to finish

- Implement `make_chunks`: contiguous chunks whose sizes differ by at most one.
  Treat zero streams as one; do not create empty chunks.
- Inspect the supplied pinned-memory pipeline in `vector_add_cuda.cc`: each
  stream copies its inputs, launches its kernel, and copies its output back.
- Explain why synchronization happens after all chunks have been submitted.
- Compare 1, 2, and 4 streams using the benchmark.

## Commands

```bash
bazel build //...
bazel test //...
bazel run //src:vector_add_benchmark
```

The starter project should build before you edit it. Its tests are expected to
fail until the TODO implementations are complete.

The complete answer is mirrored at `tp/solutions/stretch/ex11_multistream_overlap`.
## Notes

- CUDA tests and benchmarks require an NVIDIA driver and CUDA-capable GPU.

## Reading the benchmark

Each stream count uses the same input and triple-chevron kernel launch, with one
warm-up and three measured runs. Every measured result is checked against the CPU.
The starter uses one chunk until its planner TODO is completed; use the solution
for the complete multi-stream demonstration.

- `total_ms`: mean wall time including pinned/device allocation, host staging,
  transfers, kernels, synchronization, output construction, and cleanup.
- `pipeline_ms`: mean wall time from submitting the first transfer until **all**
  streams finish. This includes H2D transfers, kernels, D2H transfers, and host
  submission overhead; it excludes allocation and host staging. It is not a sum
  of per-stream times or a kernel-only measurement.

Pinned memory and independent nonblocking streams permit overlap; more streams
need not be faster. To verify actual overlap, inspect the copy and kernel lanes
in an Nsight Systems timeline (if installed):

```bash
bazel build //src:vector_add_benchmark
nsys profile --trace=cuda --sample=none -o /tmp/ex11_overlap bazel-bin/src/vector_add_benchmark
```

Reference: CUDA Programming Guide, sections 2.5 (asynchronous execution) and
2.6.3 (page-locked host memory).
