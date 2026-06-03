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

The starter project should build before you edit it. Its tests are expected to
fail until the TODO implementations are complete.

The complete answer is mirrored at `tp/solutions/day1/ex02_cpu_vector_oracle`.

