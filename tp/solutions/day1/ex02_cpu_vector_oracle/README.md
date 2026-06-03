# Exercise 02: CPU Vector Oracle

## Goal

Implement and test the CPU correctness oracle for `out[i] = alpha * a[i] + b[i]`.

## What to finish

- Validate input sizes.
- Return an empty vector for empty input.
- Compute every output element exactly once.

## Commands

```bash
bazel build //...
bazel test //src:vector_cpu_test
```

This is the complete reference solution. It should build and pass its tests.

Starter exercise: `tp/day1/ex02_cpu_vector_oracle`.

