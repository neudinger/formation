# Exercise 03: Threaded CPU Vector Add

## Goal

Add a chunked `std::thread` implementation and compare it with the serial CPU oracle.

## What to finish

- Reuse the serial CPU validation contract.
- Split work into contiguous chunks.
- Join every worker before returning.

## Commands

```bash
bazel build //...
bazel test //src:vector_threaded_test
bazel run //src:vector_cpu_benchmark
```

The starter project should build before you edit it. Its tests are expected to
fail until the TODO implementations are complete.

The complete answer is mirrored at `tp/solutions/day1/ex03_threaded_cpu_vector`.

