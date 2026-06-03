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

The starter project should build before you edit it. Its tests are expected to
fail until the TODO implementations are complete.

The complete answer is mirrored at `tp/solutions/stretch/ex11_multistream_overlap`.
## Notes

- CUDA tests and benchmarks require an NVIDIA driver and CUDA-capable GPU.

