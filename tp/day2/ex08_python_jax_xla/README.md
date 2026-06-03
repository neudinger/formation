# Exercise 08: Python Benchmark Orchestration and JAX/XLA Hook

## Goal

Use hermetic Python as an orchestration layer and expose a JAX/XLA comparison hook that degrades clearly when JAX is absent.

## What to finish

- Return the benchmark commands in execution order.
- Detect whether JAX is importable.
- Print a clear fallback message when JAX is unavailable.

## Commands

```bash
bazel build //...
bazel test //python:python_hooks_test
bazel run //python:benchmark
bazel run //python:jax_xla_demo
```

The starter project should build before you edit it. Its tests are expected to
fail until the TODO implementations are complete.

The complete answer is mirrored at `tp/solutions/day2/ex08_python_jax_xla`.

