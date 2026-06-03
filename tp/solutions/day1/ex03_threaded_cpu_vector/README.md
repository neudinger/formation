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

This is the complete reference solution. It should build and pass its tests.

Starter exercise: `tp/day1/ex03_threaded_cpu_vector`.

