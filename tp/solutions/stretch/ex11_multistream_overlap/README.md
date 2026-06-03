# Stretch 11: Multi-Stream Vector Overlap

## Goal

Prepare vector work for pinned-memory, multi-stream chunking and overlap measurement.

## What to finish

- Split a vector into balanced chunks.
- Use the chunks to drive multiple CUDA streams.
- Compare one-stream and multi-stream timing.

## Commands

```bash
bazel build //...
bazel test //src:chunk_plan_test
bazel run //src:vector_add_benchmark
```

This is the complete reference solution. It should build and pass its tests.

Starter exercise: `tp/stretch/ex11_multistream_overlap`.
## Notes

- CUDA tests and benchmarks require an NVIDIA driver and CUDA-capable GPU.

