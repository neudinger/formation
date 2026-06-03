# Exercise 05: Launch Styles and Benchmarking

## Goal

Support both `cudaLaunchKernel` and triple-chevron launches, then benchmark the paths.

## What to finish

- Route through the selected `LaunchApi`.
- Keep the correctness test shared between launch styles.
- Read benchmark output by checking what the timing includes.

## Commands

```bash
bazel build //...
bazel test //src:vector_add_test
bazel run //src:vector_add_benchmark
```

This is the complete reference solution. It should build and pass its tests.

Starter exercise: `tp/day1/ex05_launch_styles_benchmark`.
## Notes

- CUDA tests and benchmarks require an NVIDIA driver and CUDA-capable GPU.

